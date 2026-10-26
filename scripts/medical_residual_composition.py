"""Fixed unit-weight residual composition with a same-rank amplitude control."""
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'unit_residual_composition_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
PARENT = ROOT / 'artifacts/medical/continual_acquisition_bank_v1'
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from slake_probe import query_view
read, write = common.read, common.write


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def prepare():
    import numpy as np
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    assert read(ROOT / 'artifacts/medical/dynamic_acquisition_tools_v1/RESULT.json')['status'] == 'COMPLETE_DYNAMIC_TOOL_NOT_ESTABLISHED'
    m = read(COHORT / 'private_manifest.json')
    c = read(COHORT / 'private_contracts.json')
    items = {r['qid']: r for r in m['new_validation'] + m['old_background']}
    flags = {str(q): c['flags'][str(q)] for q in items}
    admitted = [q for q in sorted(items) if flags[str(q)]['C']]
    assert len(items) == 323 and len(admitted) == 272
    assert all(not c['flags'][str(r['qid'])]['C'] for k in ('old_source', 'old_validation', 'old_acquisition') for r in m[k])
    configs = []
    for seed in p['training_seeds']:
        for folder in (PARENT / f'I_{seed}', COHORT / f'F_{seed}'):
            cfg = read(folder / 'adapter/adapter_config.json')
            assert cfg['r'] == 16 and cfg['lora_alpha'] == 32 and not cfg['modules_to_save']
            configs.append(cfg)
    assert all(set(c['target_modules']) == set(configs[0]['target_modules']) and c['base_model_name_or_path'] == configs[0]['base_model_name_or_path'] for c in configs)
    rng = np.random.default_rng(p['selection_seed'])
    a, b, u, v = rng.normal(size=(2, 4)), rng.normal(size=(3, 2)), rng.normal(size=(2, 4)), rng.normal(size=(3, 2))
    combined = np.concatenate([b, v], 1) @ np.concatenate([2 * a, 2 * u], 0)
    assert np.allclose(combined, 2 * b @ a + 2 * v @ u)
    doubled = np.concatenate([v, v], 1) @ np.concatenate([2 * u, 2 * u], 0)
    assert np.allclose(doubled, 2 * (2 * v @ u))
    plan = [{'model': f'{arm}_{seed}', 'qid': q, 'view': view} for seed in p['training_seeds'] for arm in ('SUM', 'DOUBLE') for q in admitted for view in ('N', 'B')]
    random.Random(p['selection_seed']).shuffle(plan)
    order = [f'{arm}_{seed}' for seed in p['training_seeds'] for arm in ('SUM', 'DOUBLE')]
    plan.sort(key=lambda e: order.index(e['model']))
    assert len(plan) == len({(r['model'], r['qid'], r['view']) for r in plan}) == 2176
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    write(OUT / 'private_manifest.json', {'validation': m['new_validation'], 'background': m['old_background'], 'flags': flags})
    write(OUT / 'schedule.json', plan)
    cpu = {'status': 'CPU_READY', 'primary_questions': 254, 'background_questions': 69, 'admitted_questions': 272, 'uncovered_primary_kept': 3, 'unchanged_background_kept': 48, 'parent_ranks': [16, 16], 'constructed_rank': 32, 'unit_weights_only_no_search': True, 'parent_configs_compatible': True, 'concatenation_and_amplitude_toy_fixtures_pass': True, 'formal_generation_calls': 2176, 'maximum_generation_calls': 2276, 'new_optimizer_steps': 0, 'full_model_weights_copied': False}
    write(OUT / 'cpu_check.json', cpu)
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps(cpu), flush=True)


def run():
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY' and not (OUT / 'requests.jsonl').exists()
    check()
    torch.set_num_threads(4)
    torch.manual_seed(p['selection_seed'])
    m = read(OUT / 'private_manifest.json')
    items = {r['qid']: r for r in m['validation'] + m['background']}
    plan = read(OUT / 'schedule.json')
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    names, model = {}, None
    for seed in p['training_seeds']:
        for kind, folder in [('I', PARENT / f'I_{seed}/adapter'), ('F', COHORT / f'F_{seed}/adapter')]:
            if model is None:
                model = PeftModel.from_pretrained(base, folder, is_trainable=False, local_files_only=True)
                name = 'default'
            else:
                name = f'{kind}_{seed}'
                model.load_adapter(folder, adapter_name=name, is_trainable=False, local_files_only=True)
            names[kind, seed] = name
    proofs = {}
    for seed in p['training_seeds']:
        i, f = names['I', seed], names['F', seed]
        for arm, parents in [('SUM', [i, f]), ('DOUBLE', [f, f])]:
            label = f'{arm}_{seed}'
            model.add_weighted_adapter(parents, [1., 1.], label, combination_type='cat')
            cfg = model.peft_config[label]
            assert cfg.r == cfg.lora_alpha == 32
            layers, max_relative = 0, 0.
            with torch.no_grad():
                for module in model.modules():
                    if not hasattr(module, 'lora_A') or label not in module.lora_A:
                        continue
                    # Explicit FP32 factors avoid quantizing the constructed adapter to the base BF16 dtype.
                    aa = torch.cat([module.lora_A[x].weight.detach().float() * module.scaling[x] for x in parents], 0)
                    bb = torch.cat([module.lora_B[x].weight.detach().float() for x in parents], 1)
                    module.lora_A[label].to(dtype=torch.float32)
                    module.lora_B[label].to(dtype=torch.float32)
                    module.lora_A[label].weight.copy_(aa)
                    module.lora_B[label].weight.copy_(bb)
                    assert module.scaling[label] == 1
                    x = torch.randn(2, aa.shape[1], device=aa.device, dtype=torch.float32)
                    actual = F.linear(F.linear(x, aa), bb)
                    expected = sum(F.linear(F.linear(x, module.lora_A[z].weight.float()), module.lora_B[z].weight.float()) * module.scaling[z] for z in parents)
                    relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
                    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)
                    max_relative = max(max_relative, relative)
                    layers += 1
            assert layers == 64
            proofs[label] = {'layers_checked': layers, 'constructed_rank': cfg.r, 'actual_delta_is_exact_sum_within_fp32_tolerance': True, 'maximum_probe_relative_error': max_relative, 'parent_factors_unchanged': True, 'optimizer_steps': 0}
    write(OUT / 'COMPOSITION_CHECK.json', proofs)
    calls = 0
    for label in [f'{arm}_{s}' for s in p['training_seeds'] for arm in ('SUM', 'DOUBLE')]:
        model.set_adapter(label)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval()
        for e in (x for x in plan if x['model'] == label):
            check()
            row = items[e['qid']]
            inputs = processor(text=[common.prompt(processor, query_view(row))], images=[common.image_view(row, e['view'])], return_tensors='pt').to('cuda')
            calls += 1
            assert calls <= p['maximum_generation_calls']
            with (OUT / 'requests.jsonl').open('a') as out:
                out.write(json.dumps(dict(e, number=calls, event='intent', started_unix=time.time())) + '\n')
            with torch.inference_mode():
                result = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            tokens = result[0, inputs['input_ids'].shape[1]:].tolist()
            record = dict(e, number=calls, raw=processor.tokenizer.decode(tokens, skip_special_tokens=True), output_ids=tokens, completed_unix=time.time())
            write(OUT / 'records' / f'{calls:04d}.json', record)
            with (OUT / 'requests.jsonl').open('a') as out: out.write(json.dumps(dict(record, event='result')) + '\n')
            write(OUT / 'progress.json', {'phase': 'evaluation', 'model': label, 'new_generation_calls': calls, 'planned_calls': len(plan), 'updated_unix': time.time()})
        write(OUT / (label + '_EVALUATED.json'), {'model': label, 'calls': sum(e['model'] == label for e in plan), 'completed_unix': time.time()})
    assert calls == p['formal_generation_calls'] == 2176
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_UNIT_RESIDUAL_COMPOSITION', 'new_generation_calls': calls, 'new_optimizer_steps': 0, 'constructed_adapters': 4, 'saved_model_checkpoints': 0, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__': {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
