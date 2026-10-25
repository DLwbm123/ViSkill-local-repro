"""Second acquisition capability: shared-parameter consolidation versus an explicit module bank."""
from collections import Counter, defaultdict
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
DATA = Path(os.environ['SLAKE_ROOT'])
NAME = 'old_loss_update_projection_v1'
PARENT = ROOT / 'artifacts/medical/continual_acquisition_bank_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
VISUAL = ROOT / 'artifacts/medical/grounded_reader_anchor_v1'
SUPPORT = ROOT / 'artifacts/medical/task_supported_adapter_v1'
NATURAL = ROOT / 'artifacts/medical/natural_scope_validation_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from medical_grounded_reader import supervised
from slake_probe import normalize, score, query_view
read, write = common.read, common.write


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def project_delta(delta, old_gradient):
    # Project the actual Adam parameter displacement, not its unpreconditioned gradient.
    dot = (delta * old_gradient).sum()
    squared_norm = old_gradient.square().sum()
    active = bool(dot > 0 and squared_norm > 0)
    return delta - (dot / squared_norm) * old_gradient if active else delta, active


def prepare():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    old = read(PARENT / 'RESULT.json')
    assert not old['object_gate_pass']['parameter_consolidation_K']
    assert old['object_gate_pass']['external_module_bank']
    m = read(PARENT / 'private_manifest.json')
    kinds = ('new_validation', 'old_source', 'old_background', 'old_validation')
    items = {r['qid']: r for kind in kinds for r in m[kind]}
    assert len(items) == 349 and len(m['train']) == len(m['old_train_replay']) == 100
    assert not {r['img_name'] for r in m['train'] + m['old_train_replay']} & {r['img_name'] for r in items.values()}
    plan = [{'model': f'P_{seed}', 'qid': q, 'view': view} for seed in p['training_seeds'] for q in sorted(items) for view in ('N', 'B')]
    random.Random(p['selection_seed']).shuffle(plan)
    plan.sort(key=lambda r: r['model'])
    assert len(plan) == p['formal_generation_calls'] == 1396
    assert len({(r['model'], r['qid'], r['view']) for r in plan}) == 1396
    for d, h, expected, active in [([1., 2.], [1., 0.], [0., 2.], True), ([-1., 2.], [1., 0.], [-1., 2.], False), ([1., 2.], [0., 0.], [1., 2.], False)]:
        result, flag = project_delta(torch.tensor(d, dtype=torch.float64), torch.tensor(h, dtype=torch.float64))
        assert flag == active and torch.allclose(result, torch.tensor(expected, dtype=torch.float64))
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    write(OUT / 'private_manifest.json', m)
    write(OUT / 'private_contracts.json', read(PARENT / 'private_contracts.json'))
    write(OUT / 'schedule.json', plan)
    for seed in p['training_seeds']:
        assert (PARENT / f'K_{seed}/TRAIN_COMPLETED.json').exists()
        for kind in ('null', 'old'):
            assert all((PARENT / 'teacher_cache' / f'A_{seed}_{kind}_{i:03d}.pt').exists() for i in range(100))
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    batch = supervised(processor, m['old_train_replay'][0], 'N')
    assert (batch['labels'] != -100).sum() > 0
    assert not torch.cuda.is_initialized()
    cpu = {'status': 'CPU_READY', 'same_frozen_train_and_evaluation_as_parent': True, 'training_questions': 100, 'old_gradient_questions': 100, 'evaluation_questions': 349, 'train_evaluation_image_disjoint': True, 'formal_generations': 1396, 'reused_teacher_distributions': 400, 'new_teacher_forwards': 0, 'projection_positive_negative_zero_fixtures_pass': True, 'gpu_initialized': False, 'unchanged_parent_K_comparison_no_retraining': True}
    write(OUT / 'cpu_check.json', cpu)
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps(cpu), flush=True)


def run():
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, PeftModel
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY' and not (OUT / 'requests.jsonl').exists()
    check()
    torch.set_num_threads(4)
    m = read(OUT / 'private_manifest.json')
    eval_kinds = ('new_validation', 'old_source', 'old_background', 'old_validation')
    items = {r['qid']: r for kind in eval_kinds for r in m[kind]}
    plan = read(OUT / 'schedule.json')
    position = {r['qid']: i for i, r in enumerate(m['train'])}
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    calls, cache = 0, {}

    def load(family, seed, trainable=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
        if family != 'base':
            return PeftModel.from_pretrained(model, VISUAL / family / 'adapter', is_trainable=trainable, local_files_only=True)
        if trainable:
            targets = [name for name, module in model.named_modules() if 'visual' in name and name.endswith(('.attn.qkv', '.attn.proj')) and isinstance(module, torch.nn.Linear)]
            assert targets
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0, bias='none', task_type='CAUSAL_LM', target_modules=targets))
        return model

    def evaluate(model, label):
        nonlocal calls
        model.eval()
        model.config.use_cache = True
        for entry in (e for e in plan if e['model'] == label):
            check()
            calls += 1
            assert calls <= p['maximum_generation_calls']
            row = items[entry['qid']]
            text = common.prompt(processor, query_view(row))
            inputs = processor(text=[text], images=[common.image_view(row, entry['view'])], return_tensors='pt').to('cuda')
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(entry, event='intent', number=calls, started_unix=time.time())) + '\n')
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
            raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            result = dict(entry, number=calls, raw=raw, output_ids=tokens, correct=bool(score(raw, row['answer'])), completed_unix=time.time())
            write(OUT / 'records' / f'{calls:04d}.json', result)
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(result, event='result')) + '\n')
            write(OUT / 'progress.json', {'phase': 'evaluation', 'model': label, 'new_generation_calls': calls, 'planned_calls': len(plan), 'updated_unix': time.time()})
        write(OUT / f'{label}_EVALUATED.json', {'model': label, 'calls': sum(e['model'] == label for e in plan), 'completed_unix': time.time()})

    def selected_logp(model, batch):
        logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
        active = batch['labels'][:, 1:] != -100
        return logits[:, :-1][active].float().log_softmax(-1), batch['labels'][:, 1:][active]

    for seed in p['training_seeds']:
        for kind in ('null', 'old'):
            for i in range(100):
                cache[f'A_{seed}', kind, i] = torch.load(PARENT / 'teacher_cache' / f'A_{seed}_{kind}_{i:03d}.pt', map_location='cpu', weights_only=True)
    assert len(cache) == 400
    write(OUT / 'TEACHER_COMPLETED.json', {'forward_calls': 0, 'reused_distributions': 400, 'validation_distilled': False, 'generation_calls': 0})

    def distill(model, row, view, reference):
        batch = supervised(processor, row, view).to('cuda')
        logp, targets = selected_logp(model, batch)
        assert torch.equal(targets.cpu(), reference['targets'])
        loss = F.kl_div(logp, reference['logp'].to('cuda'), log_target=True, reduction='none').sum(-1).mean()
        assert torch.isfinite(loss)
        return loss

    for seed in p['training_seeds']:
        for arm in p['arms']:
            check()
            label = f'{arm}_{seed}'
            family = f'A_{seed}'
            folder = OUT / label
            folder.mkdir(exist_ok=False)
            model = load(family, seed, trainable=True)
            assert all('visual' in name for name, v in model.named_parameters() if v.requires_grad)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.config.use_cache = False
            model.train()
            parameters = [v for v in model.parameters() if v.requires_grad]
            assert all(v.dtype == torch.float32 for v in parameters)
            initial = {name: v.detach().cpu().clone() for name, v in model.named_parameters() if v.requires_grad}
            old_coefficient = 1.0
            ce = model(**common.train_pair(processor, m['train'][0], 'A').to('cuda')).loss
            null = distill(model, m['train'][0], 'B', cache[family, 'null', 0])
            old = distill(model, m['old_train_replay'][0], 'N', cache[family, 'old', 0])
            assert abs(float(null.detach())) < 2e-4 and abs(float(old.detach())) < 2e-4
            (ce + null + old_coefficient * old).backward()
            grads = [v.grad for v in parameters if v.grad is not None]
            assert grads and all(torch.isfinite(v).all() for v in grads)
            write(folder / 'production_backward_check.json', {'status': 'PASS', 'parent': family, 'visual_trainable_only': True, 'initial_null_KL': float(null.detach()), 'initial_old_KL': float(old.detach()), 'finite_gradient_tensors': len(grads), 'optimizer_steps': 0})
            model.zero_grad(set_to_none=True)
            del ce, null, old, grads
            optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
            steps = 0
            try:
                for epoch in range(p['epochs']):
                    order = m['train'].copy()
                    random.Random(seed + 1000 * epoch).shuffle(order)
                    for offset in range(0, len(order), p['rows_per_update']):
                        group = order[offset:offset + p['rows_per_update']]
                        old_gradient = torch.zeros(sum(v.numel() for v in parameters), device='cuda', dtype=torch.float64)
                        for row in group:
                            replay = m['old_train_replay'][position[row['qid']]]
                            old_ce = model(**supervised(processor, replay, 'N').to('cuda')).loss
                            grads = torch.autograd.grad(old_ce / len(group), parameters, allow_unused=False)
                            old_gradient.add_(torch.cat([v.detach().reshape(-1).double() for v in grads]))
                            del old_ce, grads
                        assert torch.isfinite(old_gradient).all()
                        before_update = torch.cat([v.detach().reshape(-1).double() for v in parameters])
                        optimizer.zero_grad(set_to_none=True)
                        losses = []
                        for row in group:
                            check()
                            i = position[row['qid']]
                            ce = model(**common.train_pair(processor, row, 'A').to('cuda')).loss
                            null = distill(model, row, 'B', cache[family, 'null', i])
                            old = distill(model, m['old_train_replay'][i], 'N', cache[family, 'old', i])
                            loss = ce + null + old_coefficient * old
                            assert torch.isfinite(loss)
                            (loss / len(group)).backward()
                            losses.append([float(v.detach()) for v in (loss, ce, null, old)])
                        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                        assert torch.isfinite(norm)
                        optimizer.step()
                        with torch.no_grad():
                            proposed = torch.cat([v.detach().reshape(-1).double() for v in parameters]) - before_update
                            actual, projected = project_delta(proposed, old_gradient)
                            assert torch.isfinite(actual).all()
                            cursor = 0
                            for parameter in parameters:
                                length = parameter.numel()
                                parameter.copy_((before_update[cursor:cursor + length] + actual[cursor:cursor + length]).reshape_as(parameter))
                                cursor += length
                            applied = torch.cat([v.detach().reshape(-1).double() for v in parameters]) - before_update
                            dot_before = float((proposed * old_gradient).sum())
                            dot_after = float((applied * old_gradient).sum())
                            tolerance = max(1e-7, 1e-4 * float(proposed.norm() * old_gradient.norm()))
                            assert dot_after <= tolerance
                        steps += 1
                        event = {'arm': arm, 'seed': seed, 'parent': family, 'epoch': epoch, 'step': steps, 'rows': len(group), 'supervised_views': 2 * len(group), 'anchor_forward_views': 2 * len(group), 'loss': sum(x[0] for x in losses) / len(losses), 'supervised_loss': sum(x[1] for x in losses) / len(losses), 'null_KL': sum(x[2] for x in losses) / len(losses), 'old_KL': sum(x[3] for x in losses) / len(losses), 'old_coefficient': old_coefficient, 'preclip_gradient_norm': float(norm), 'updated_unix': time.time(), 'old_gradient_views': len(group), 'projection_active': projected, 'old_CE_linear_change_before': dot_before, 'old_CE_linear_change_after': dot_after, 'projection_rounding_tolerance': tolerance, 'old_gradient_norm': float(old_gradient.norm()), 'proposed_delta_norm': float(proposed.norm()), 'applied_delta_norm': float(applied.norm())}
                        with (folder / 'updates.jsonl').open('a') as f:
                            f.write(json.dumps(event) + '\n')
                        write(OUT / 'progress.json', dict(event, phase='train', model=label, new_generation_calls=calls))
                    model.save_pretrained(folder / 'adapter', safe_serialization=True)
                    torch.save({'optimizer': optimizer.state_dict(), 'epoch_completed': epoch, 'steps': steps}, folder / 'recovery.pt')
                delta = sum(float((v.detach().float().cpu() - initial[name].float()).square().sum()) for name, v in model.named_parameters() if v.requires_grad)
                assert delta > 0 and math.isfinite(delta) and steps == 52
                write(folder / 'TRAIN_COMPLETED.json', {'arm': arm, 'seed': seed, 'parent': family, 'teacher_family': family, 'optimizer_steps': steps, 'supervised_views': 800, 'anchor_forward_views': 800, 'old_gradient_views': 400, 'old_coefficient': old_coefficient, 'trainable_parameters': sum(v.numel() for v in parameters), 'parameter_delta_square_norm': delta, 'language_parameters_updated': False, 'visual_adapter_updated': True, 'completed_unix': time.time()})
            except BaseException:
                model.save_pretrained(folder / 'recovery_adapter', safe_serialization=True)
                raise
            del optimizer, parameters, initial
            model.gradient_checkpointing_disable()
            evaluate(model, label)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert calls == p['formal_generation_calls']
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_OLD_LOSS_UPDATE_PROJECTION', 'new_generation_calls': calls, 'trained_adapters': 2, 'optimizer_steps': 104, 'teacher_forward_calls': 0, 'teacher_distributions_reused': 400, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
