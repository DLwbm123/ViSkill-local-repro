"""Learn a language readout of immutable visual experience with a common null anchor."""
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'frozen_visual_readout_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
PARENT = ROOT / 'artifacts/medical/continual_acquisition_bank_v1'
SOURCE = ROOT / 'artifacts/medical/visual_consensus_guard_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from medical_grounded_reader import supervised
from slake_probe import query_view
read, write = common.read, common.write


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def prepare():
    import torch
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    c = read(COHORT / 'private_manifest.json')
    flags = read(COHORT / 'private_contracts.json')['flags']
    primary = read(SOURCE / 'private_manifest.json')['primary']
    validation, background = c['new_validation'], c['old_background']
    rows = {r['qid']: r for r in primary + validation + background}
    active = [r for r in rows.values() if r in primary or flags[str(r['qid'])]['C']]
    assert len(primary) == 85 and len(validation) == 254 and len(background) == 69 and len(active) == 357
    assert len(c['train']) == 100 and not {r['img_name'] for r in c['train']} & {r['img_name'] for r in rows.values()}
    assert read(SOURCE / 'cpu_check.json')['train_and_previous_panel_image_disjoint']
    old_rows = {r['qid']: r for k in ('new_validation', 'old_source', 'old_background', 'old_validation', 'old_acquisition', 'all_validation') for r in c[k]}
    qold = {i: q for i, q in enumerate(sorted(old_rows))}
    signatures, exact = {}, {}
    for r in sorted(read(COHORT / 'public_bundle/physical_branch_numeric.json'), key=lambda x: (x['source_experiment'], x['source_request_number'])):
        if r['model'] not in ['I_42', 'I_43']: continue
        q = qold[r['question_group']]
        exact[r['model'], q, r['view']] = r
        signatures.setdefault((r['model'], 'BLACK' if r['view'] == 'B' else old_rows[q]['img_name'], old_rows[q]['question']), r)
    order = [x for s in p['training_seeds'] for x in [f'I_{s}', f'F_{s}', f'R_{s}']]
    bindings, physical, physical_sig, cache, cache_idx = {}, [], {}, [], {}
    for label in order:
        for row in (primary if label.startswith('I_') else active):
            for view in ['N', 'B']:
                q = row['qid']
                sig = (label, 'BLACK' if view == 'B' else row['img_name'], row['question'])
                old = exact.get((label, q, view)) or signatures.get(sig)
                if old is not None:
                    origin = (old['source_experiment'], old['source_request_number'])
                    if origin not in cache_idx:
                        rr = read(ROOT / 'artifacts/medical' / origin[0] / 'records' / f'{origin[1]:04d}.json')
                        assert rr['model'] == label
                        cache_idx[origin] = len(cache)
                        cache.append({'model': label, 'source_experiment': origin[0], 'source_request_number': origin[1], 'raw': rr['raw']})
                    ref = {'cache_index': cache_idx[origin]}
                else:
                    if sig not in physical_sig:
                        physical_sig[sig] = len(physical)
                        physical.append({'model': label, 'qid': q, 'view': view, 'physical_index': len(physical)})
                    ref = {'physical_index': physical_sig[sig]}
                bindings[f'{label}|{q}|{view}'] = ref
    assert len(bindings) == 3196 and len(physical) <= 3196
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    write(OUT / 'private_manifest.json', {'train': c['train'], 'primary': primary, 'validation': validation, 'background': background, 'active_ids': sorted(r['qid'] for r in active), 'flags': {str(q): {'C': True} if q in {r['qid'] for r in primary} else flags[str(q)] for q in rows}})
    write(OUT / 'private_bindings.json', bindings)
    write(OUT / 'private_cached_records.json', cache)
    write(OUT / 'physical_plan.json', physical)
    for i in range(100): assert (COHORT / 'teacher_cache' / f'base_null_{i:03d}.pt').exists()
    assert not torch.cuda.is_initialized()
    cpu = {'status': 'CPU_READY', 'training_questions': 100, 'primary_questions': 85, 'secondary_validation_questions': 254, 'old_background_questions': 69, 'active_questions_per_readout': 357, 'logical_predictor_readouts': len(bindings), 'new_physical_generations': len(physical), 'cached_parent_physical_records': len(cache), 'teacher_cache_records_reused': 100, 'new_teacher_forwards': 0, 'train_evaluation_image_disjoint': True, 'primary_parent_update_image_isolated': True, 'readout_adapters': 4, 'optimizer_steps_planned': 208, 'gpu_initialized': False}
    write(OUT / 'cpu_check.json', cpu)
    p.update(status='FROZEN_CPU_READY', formal_generation_calls=len(physical), maximum_generation_calls=len(physical) + 100)
    write(CONFIG, p)
    print(json.dumps(cpu), flush=True)


def run():
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.tuners.lora.layer import LoraLayer
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY' and not (OUT / 'requests.jsonl').exists()
    check()
    torch.set_num_threads(4)
    m = read(OUT / 'private_manifest.json')
    rows = {r['qid']: r for k in ['primary', 'validation', 'background'] for r in m[k]}
    plan = read(OUT / 'physical_plan.json')
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    calls = 0
    teachers = [torch.load(COHORT / 'teacher_cache' / f'base_null_{i:03d}.pt', map_location='cpu', weights_only=True) for i in range(100)]
    positions = {r['qid']: i for i, r in enumerate(m['train'])}

    def load(seed, parent):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
        if parent: model = PeftModel.from_pretrained(model, PARENT / f'I_{seed}/adapter', is_trainable=False, local_files_only=True)
        return model

    def attach(model, seed, parent):
        adapter = LoraConfig(r=16, lora_alpha=32, lora_dropout=0, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if parent:
            model.add_adapter('readout', adapter)
            model.base_model.set_adapter(['default', 'readout'])
        else:
            model = get_peft_model(model, adapter, adapter_name='readout')
        for name, value in model.named_parameters(): value.requires_grad_('.readout.' in name and 'lora_' in name)
        assert all('visual' not in name and '.readout.' in name for name, value in model.named_parameters() if value.requires_grad)
        layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, LoraLayer)]
        visual = [(name, layer) for name, layer in layers if 'visual' in name]
        language = [(name, layer) for name, layer in layers if 'visual' not in name]
        assert len(language) > 0 and all('readout' in layer.lora_A for _, layer in language)
        assert len(visual) == (64 if parent else 0)
        if parent: assert all('default' in layer.active_adapters and not layer.disable_adapters for _, layer in visual)
        return model

    def kl(model, row):
        batch = supervised(processor, row, 'B').to('cuda')
        logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
        active = batch['labels'][:, 1:] != -100
        logp = logits[:, :-1][active].float().log_softmax(-1)
        reference = teachers[positions[row['qid']]]
        assert torch.equal(batch['labels'][:, 1:][active].cpu(), reference['targets'])
        return F.kl_div(logp, reference['logp'].to('cuda'), log_target=True, reduction='none').sum(-1).mean()

    def evaluate(model, label):
        nonlocal calls
        model.eval()
        model.config.use_cache = True
        for entry in [e for e in plan if e['model'] == label]:
            check()
            calls += 1
            assert calls == entry['physical_index'] + 1 and calls <= p['maximum_generation_calls']
            row = rows[entry['qid']]
            inputs = processor(text=[common.prompt(processor, query_view(row))], images=[common.image_view(row, entry['view'])], return_tensors='pt').to('cuda')
            with (OUT / 'requests.jsonl').open('a') as f: f.write(json.dumps(dict(entry, event='intent', number=calls, started_unix=time.time())) + '\n')
            with torch.inference_mode(): generated = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
            record = dict(entry, number=calls, raw=processor.tokenizer.decode(tokens, skip_special_tokens=True), output_ids=tokens, completed_unix=time.time())
            write(OUT / 'records' / f'{calls:04d}.json', record)
            with (OUT / 'requests.jsonl').open('a') as f: f.write(json.dumps(dict(record, event='result')) + '\n')
            write(OUT / 'progress.json', {'phase': 'evaluate', 'model': label, 'new_generation_calls': calls, 'planned_calls': len(plan), 'updated_unix': time.time()})
        write(OUT / (label + '_EVALUATED.json'), {'calls': sum(e['model'] == label for e in plan), 'completed_unix': time.time()})

    fixture = attach(load(42, True), 42, True)
    fixture.enable_input_require_grads()
    fixture.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    fixture.config.use_cache = False
    fixture.train()
    ce = fixture(**common.train_pair(processor, m['train'][0], 'A').to('cuda')).loss
    null = kl(fixture, m['train'][0])
    (ce + null).backward()
    grads = [(n, v.grad) for n, v in fixture.named_parameters() if v.grad is not None]
    assert grads and all('.readout.' in n and 'visual' not in n and torch.isfinite(g).all() for n, g in grads)
    assert torch.isfinite(ce) and torch.isfinite(null)
    write(OUT / 'production_backward_check.json', {'status': 'PASS', 'language_readout_only_finite_gradients': True, 'frozen_visual_I_layers': 64, 'optimizer_steps': 0, 'new_generation_calls': 0, 'loss': float((ce + null).detach()), 'gradient_tensors': len(grads)})
    del fixture, ce, null, grads
    gc.collect()
    torch.cuda.empty_cache()
    for seed in p['training_seeds']:
        parent = load(seed, True)
        assert all(not v.requires_grad for v in parent.parameters())
        evaluate(parent, f'I_{seed}')
        del parent
        gc.collect()
        torch.cuda.empty_cache()
        initial_control = None
        for arm in ['F', 'R']:
            check()
            label = f'{arm}_{seed}'
            folder = OUT / label
            folder.mkdir(exist_ok=False)
            model = attach(load(seed, arm == 'R'), seed, arm == 'R')
            initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
            canonical = {name.split('.readout.')[0].split('model.model.')[-1] + '.' + name.split('.readout.')[1]: value for name, value in initial.items()}
            if arm == 'F': initial_control = canonical
            else:
                assert canonical.keys() == initial_control.keys() and all(torch.equal(value, initial_control[name]) for name, value in canonical.items())
            visual_initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if 'visual' in name and 'lora_' in name}
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.config.use_cache = False
            model.train()
            parameters = [v for v in model.parameters() if v.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
            steps = 0
            try:
                for epoch in range(4):
                    order = m['train'].copy()
                    random.Random(seed + 1000 * epoch).shuffle(order)
                    for offset in range(0, 100, 8):
                        group = order[offset:offset + 8]
                        optimizer.zero_grad(set_to_none=True)
                        losses = []
                        for row in group:
                            check()
                            ce = model(**common.train_pair(processor, row, 'A').to('cuda')).loss
                            null = kl(model, row)
                            loss = ce + null
                            assert torch.isfinite(loss)
                            (loss / len(group)).backward()
                            losses.append([float(x.detach()) for x in [loss, ce, null]])
                        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                        assert torch.isfinite(norm)
                        optimizer.step()
                        steps += 1
                        event = {'model': label, 'seed': seed, 'arm': arm, 'epoch': epoch, 'step': steps, 'rows': len(group), 'supervised_views': 2 * len(group), 'null_anchor_views': len(group), 'loss': sum(x[0] for x in losses) / len(losses), 'CE': sum(x[1] for x in losses) / len(losses), 'null_KL': sum(x[2] for x in losses) / len(losses), 'gradient_norm': float(norm), 'updated_unix': time.time()}
                        with (folder / 'updates.jsonl').open('a') as f: f.write(json.dumps(event) + '\n')
                        write(OUT / 'progress.json', dict(event, phase='train', new_generation_calls=calls))
                    model.save_pretrained(folder / 'adapter', selected_adapters=['readout'], safe_serialization=True)
                    torch.save({'optimizer': optimizer.state_dict(), 'epoch_completed': epoch, 'steps': steps}, folder / 'recovery.pt')
                assert steps == 52
                assert all(torch.equal(value, dict(model.named_parameters())[name].detach().cpu()) for name, value in visual_initial.items())
                delta = sum(float((value.detach().float().cpu() - initial[name].float()).square().sum()) for name, value in model.named_parameters() if value.requires_grad)
                assert math.isfinite(delta) and delta > 0
                write(folder / 'TRAIN_COMPLETED.json', {'optimizer_steps': steps, 'supervised_views': 800, 'null_anchor_views': 400, 'language_readout_parameters': sum(v.numel() for v in parameters), 'parameter_delta_square_norm': delta, 'all_visual_parameters_frozen': True, 'visual_parent_factors_exactly_unchanged': True, 'paired_readout_initialization_equal': arm == 'R', 'completed_unix': time.time()})
            except BaseException:
                model.save_pretrained(folder / 'recovery_adapter', selected_adapters=['readout'], safe_serialization=True)
                torch.save({'optimizer': optimizer.state_dict(), 'steps': steps}, folder / 'interrupted_recovery.pt')
                raise
            del optimizer, parameters, initial, visual_initial, canonical
            model.gradient_checkpointing_disable()
            evaluate(model, label)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del initial_control
    assert calls == p['formal_generation_calls']
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_FROZEN_VISUAL_READOUT', 'new_generation_calls': calls, 'trained_readout_adapters': 4, 'optimizer_steps': 208, 'new_teacher_forwards': 0, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__': {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
