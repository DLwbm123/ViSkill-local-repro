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
NAME = 'acquisition_to_organ_transfer_v1'
PARENT = ROOT / 'artifacts/medical/continual_acquisition_bank_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
VISUAL = ROOT / 'artifacts/medical/grounded_reader_anchor_v1'
SUPPORT = ROOT / 'artifacts/medical/task_supported_adapter_v1'
NATURAL = ROOT / 'artifacts/medical/natural_scope_validation_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from medical_grounded_reader import supervised
from medical_natural_scope import legacy_records
from medical_roi_task_audit import task, match_target
from slake_probe import normalize, score, query_view
read, write = common.read, common.write


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def prepare():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    parent_result = read(PARENT / 'RESULT.json')
    assert parent_result['object_gate_pass']['external_module_bank']
    m = read(PARENT / 'private_manifest.json')
    c = read(PARENT / 'private_contracts.json')
    pairs, b = {tuple(x) for x in c['A_pairs']}, set(c['B_normalized_question_templates'])
    reserved = {r['img_name'] for kind in ('old_source', 'old_background') for r in m[kind]}
    def old_flags(row):
        return {'A': (task(row['question']), match_target(row['question'], c['A_lexicon'])) in pairs, 'B': normalize(row['question']) in b}
    candidates = [r for r in read(DATA / 'raw/train.json') if r['q_lang'] == 'en' and r['base_type'] == 'vqa' and r['content_type'] == 'Organ' and r['img_name'] not in reserved and not any(old_flags(r).values())]
    by_template = defaultdict(list)
    for row in candidates:
        by_template[normalize(row['question'])].append(row)
    rng = random.Random(p['selection_seed'])
    train = []
    for template in sorted(by_template, key=lambda t: (len(by_template[t]), t)):
        pool = sorted(by_template[template], key=lambda r: r['qid'])
        rng.shuffle(pool)
        train.append(pool[0])
    selected = {r['qid'] for r in train}
    pool = sorted((r for r in candidates if r['qid'] not in selected), key=lambda r: r['qid'])
    rng.shuffle(pool)
    train += pool[:100 - len(train)]
    assert len(train) == len({r['qid'] for r in train}) == 100
    templates = {normalize(r['question']) for r in train}
    assert len(templates) == len(by_template) == 95
    new_val = [r for r in m['all_validation'] if r['base_type'] == 'vqa' and r['content_type'] == 'Organ' and not any(old_flags(r).values())]
    assert len(new_val) == 254
    def flags(row):
        return dict(old_flags(row), C=normalize(row['question']) in templates and not any(old_flags(row).values()))
    background = [r for r in m['old_background'] if flags(r)['C']]
    items = {r['qid']: r for r in new_val + background}
    assert not {r['img_name'] for r in train} & {r['img_name'] for r in items.values()}
    assert not {r['img_name'] for r in train} & {r['img_name'] for r in m['all_validation'] + m['old_source'] + m['old_background']}
    plan = [{'phase': 'before', 'model': f'I_{seed}', 'qid': r['qid'], 'view': view} for seed in p['training_seeds'] for r in new_val for view in ('N', 'B')]
    plan += [{'phase': 'after', 'model': f'{arm}_{seed}', 'qid': q, 'view': view} for seed in p['training_seeds'] for arm in p['arms'] for q in sorted(items) for view in ('N', 'B')]
    order = [f'I_{s}' for s in p['training_seeds']] + [f'{a}_{s}' for s in p['training_seeds'] for a in p['arms']]
    rng.shuffle(plan)
    plan.sort(key=lambda e: order.index(e['model']))
    assert len({(r['model'], r['qid'], r['view']) for r in plan}) == len(plan)
    assert len(plan) <= p['maximum_generation_calls']
    prior_keys = {(r['model'], r['qid'], r['view']) for r in read(PARENT / 'schedule.json')}
    assert not prior_keys & {(r['model'], r['qid'], r['view']) for r in plan}
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    (OUT / 'teacher_cache').mkdir()
    write(OUT / 'private_manifest.json', {'train': train, 'new_validation': new_val, 'changed_background': background, 'old_source': m['old_source'], 'old_background': m['old_background'], 'old_validation': m['old_validation'], 'old_acquisition': m['new_validation'], 'all_validation': m['all_validation']})
    all_items = {r['qid']: r for r in m['all_validation'] + m['old_source'] + m['old_background']}
    write(OUT / 'private_contracts.json', {'C_question_templates': sorted(templates), 'flags': {str(q): flags(r) for q, r in all_items.items()}, 'A_B_priority_unchanged': True})
    write(OUT / 'schedule.json', plan)
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    batch = supervised(processor, train[0], 'B')
    assert (batch['labels'] != -100).sum() > 0
    n, blank = [processor(text=[common.prompt(processor, query_view(new_val[0]))], images=[common.image_view(new_val[0], v)], return_tensors='pt') for v in ('N', 'B')]
    assert torch.equal(n['input_ids'], blank['input_ids']) and torch.equal(n['image_grid_thw'], blank['image_grid_thw'])
    assert not torch.cuda.is_initialized()
    cpu = {'status': 'CPU_READY', 'training_candidates': len(candidates), 'training_questions': 100, 'training_images': len({r['img_name'] for r in train}), 'training_templates': len(templates), 'training_modality': dict(Counter(r['modality'] for r in train)), 'training_answer_classes': len({normalize(str(r['answer'])) for r in train}), 'new_validation_questions': 254, 'new_validation_images': len({r['img_name'] for r in new_val}), 'C_template_covered_validation': sum(flags(r)['C'] for r in new_val), 'changed_background_questions': len(background), 'new_eval_questions_per_adapter': len(items), 'before_I_generations': 1016, 'formal_new_generations': len(plan), 'train_evaluation_image_disjoint': True, 'training_images_shared_with_prior_B_training': len({r['img_name'] for r in train} & {r['img_name'] for r in m['train']}), 'N_B_equal_input_ids_grid': True, 'gpu_initialized': False}
    write(OUT / 'cpu_check.json', cpu)
    p.update(status='FROZEN_CPU_READY', formal_generation_calls=len(plan), maximum_generation_calls=len(plan) + 100)
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
    eval_kinds = ('new_validation', 'changed_background')
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
            return PeftModel.from_pretrained(model, PARENT / family / 'adapter', is_trainable=trainable, local_files_only=True)
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

    teacher = load('base', p['training_seeds'][0])
    teacher.eval()
    for i, row in enumerate(m['train']):
        check()
        batch = supervised(processor, row, 'B').to('cuda')
        with torch.inference_mode():
            logp, targets = selected_logp(teacher, batch)
        assert torch.isfinite(logp).all()
        cache['base', 'null', i] = {'logp': logp.cpu(), 'targets': targets.cpu()}
        torch.save(cache['base', 'null', i], OUT / 'teacher_cache' / f'base_null_{i:03d}.pt')
    del teacher, batch, logp, targets
    gc.collect()
    torch.cuda.empty_cache()
    write(OUT / 'TEACHER_COMPLETED.json', {'forward_calls': 100, 'generation_calls': 0, 'validation_distilled': False, 'same_base_null_teacher_all_arms': True})
    for seed in p['training_seeds']:
        parent = load(f'I_{seed}', seed)
        evaluate(parent, f'I_{seed}')
        del parent
        gc.collect()
        torch.cuda.empty_cache()

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
            family = 'base' if arm == 'F' else f'I_{seed}'
            folder = OUT / label
            folder.mkdir(exist_ok=False)
            model = load(family, seed, trainable=True)
            assert all('visual' in name for name, v in model.named_parameters() if v.requires_grad)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.config.use_cache = False
            model.train()
            parameters = [v for v in model.parameters() if v.requires_grad]
            initial = {name: v.detach().cpu().clone() for name, v in model.named_parameters() if v.requires_grad}
            null_coefficient = 1.0
            ce = model(**common.train_pair(processor, m['train'][0], 'A').to('cuda')).loss
            null = distill(model, m['train'][0], 'B', cache['base', 'null', 0])
            assert torch.isfinite(null)
            if arm == 'F':
                assert abs(float(null.detach())) < 2e-4
            (ce + null).backward()
            grads = [v.grad for v in parameters if v.grad is not None]
            assert grads and all(torch.isfinite(v).all() for v in grads)
            write(folder / 'production_backward_check.json', {'status': 'PASS', 'parent': family, 'visual_trainable_only': True, 'initial_null_KL': float(null.detach()), 'finite_gradient_tensors': len(grads), 'optimizer_steps': 0})
            model.zero_grad(set_to_none=True)
            del ce, null, grads
            optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
            steps = 0
            try:
                for epoch in range(p['epochs']):
                    order = m['train'].copy()
                    random.Random(seed + 1000 * epoch).shuffle(order)
                    for offset in range(0, len(order), p['rows_per_update']):
                        group = order[offset:offset + p['rows_per_update']]
                        optimizer.zero_grad(set_to_none=True)
                        losses = []
                        for row in group:
                            check()
                            i = position[row['qid']]
                            ce = model(**common.train_pair(processor, row, 'A').to('cuda')).loss
                            null = distill(model, row, 'B', cache['base', 'null', i])
                            loss = ce + null
                            assert torch.isfinite(loss)
                            (loss / len(group)).backward()
                            losses.append([float(v.detach()) for v in (loss, ce, null)])
                        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                        assert torch.isfinite(norm)
                        optimizer.step()
                        steps += 1
                        event = {'arm': arm, 'seed': seed, 'parent': family, 'epoch': epoch, 'step': steps, 'rows': len(group), 'supervised_views': 2 * len(group), 'anchor_forward_views': len(group), 'loss': sum(x[0] for x in losses) / len(losses), 'supervised_loss': sum(x[1] for x in losses) / len(losses), 'null_KL': sum(x[2] for x in losses) / len(losses), 'null_coefficient': null_coefficient, 'preclip_gradient_norm': float(norm), 'updated_unix': time.time()}
                        with (folder / 'updates.jsonl').open('a') as f:
                            f.write(json.dumps(event) + '\n')
                        write(OUT / 'progress.json', dict(event, phase='train', model=label, new_generation_calls=calls))
                    model.save_pretrained(folder / 'adapter', safe_serialization=True)
                    torch.save({'optimizer': optimizer.state_dict(), 'epoch_completed': epoch, 'steps': steps}, folder / 'recovery.pt')
                delta = sum(float((v.detach().float().cpu() - initial[name].float()).square().sum()) for name, v in model.named_parameters() if v.requires_grad)
                assert delta > 0 and math.isfinite(delta) and steps == 52
                write(folder / 'TRAIN_COMPLETED.json', {'arm': arm, 'seed': seed, 'parent': family, 'teacher_family': 'base', 'optimizer_steps': steps, 'supervised_views': 800, 'anchor_forward_views': 400, 'null_coefficient': null_coefficient, 'trainable_parameters': sum(v.numel() for v in parameters), 'parameter_delta_square_norm': delta, 'language_parameters_updated': False, 'visual_adapter_updated': True, 'completed_unix': time.time()})
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
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_ACQUISITION_TO_ORGAN_TRANSFER', 'new_generation_calls': calls, 'trained_adapters': 4, 'optimizer_steps': 208, 'teacher_forward_calls': 100, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
