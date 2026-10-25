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
NAME = 'continual_acquisition_bank_v1'
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


def reference_records():
    rows = [dict(r, reference_origin=r['legacy_experiment']) for r in legacy_records()]
    for folder in (NATURAL, SUPPORT):
        rows += [dict(read(f), reference_origin=folder.name) for f in sorted((folder / 'records').glob('*.json'))]
    index = {(r['model'], r['qid'], r['view']): r for r in rows}
    assert len(index) == len(rows), 'reference physical branches must be unique'
    return index


def prepare():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    assert read(NATURAL / 'RESULT.json')['status'] == 'COMPLETE_NATURAL_SCOPE_NOT_ESTABLISHED'
    old_eval = read(SUPPORT / 'private_manifest.json')
    reserved = {r['img_name'] for kind in ('primary', 'retention') for r in old_eval[kind]}
    raw = [r for r in read(DATA / 'raw/train.json') if r['q_lang'] == 'en' and r['base_type'] == 'vqa' and r['content_type'] in ('Modality', 'Plane') and r['img_name'] not in reserved]
    allowed = {'Plane': {normalize(x) for x in ('Transverse Plane', 'Coronal Plane', 'Sagittal Plane', 'Yes', 'No')}, 'Modality': {normalize(x) for x in ('CT', 'MRI', 'T1', 'T2', 'X-Ray', 'Yes', 'No')}}
    valid = [r for r in raw if normalize(str(r['answer'])) in allowed[r['content_type']]]
    invalid = [r for r in raw if normalize(str(r['answer'])) not in allowed[r['content_type']]]
    assert len(raw) == 530 and len(valid) == 529 and len(invalid) == 1
    by_template = defaultdict(list)
    for row in valid:
        by_template[normalize(row['question'])].append(row)
    rng = random.Random(p['selection_seed'])
    train, used, counts = [], set(), Counter()
    for template in sorted(by_template, key=lambda t: (len(by_template[t]), t)):
        pool = sorted(by_template[template], key=lambda r: r['qid'])
        rng.shuffle(pool)
        row = next(r for r in pool if r['img_name'] not in used)
        train.append(row)
        used.add(row['img_name'])
        counts[row['content_type']] += 1
    for kind in ('Modality', 'Plane'):
        pool = sorted((r for r in valid if r['content_type'] == kind), key=lambda r: r['qid'])
        rng.shuffle(pool)
        for row in pool:
            if counts[kind] >= 50:
                break
            if row['img_name'] not in used:
                train.append(row)
                used.add(row['img_name'])
                counts[kind] += 1
    assert len(train) == len(used) == 100 and counts == {'Modality': 50, 'Plane': 50}
    templates = sorted({normalize(r['question']) for r in train})
    all_val = read(NATURAL / 'private_manifest.json')['validation']
    new_val = [r for r in all_val if r['base_type'] == 'vqa' and r['content_type'] in ('Modality', 'Plane')]
    old_val = [r for r in all_val if r['weak_scope']]
    assert len(new_val) == 172 and len(old_val) == 39
    support = read(SUPPORT / 'private_support.json')
    pairs = {tuple(x) for x in support['task_target_pairs']}
    def flags(row):
        a = (task(row['question']), match_target(row['question'], support['lexicon'])) in pairs
        b = normalize(row['question']) in templates
        return {'A': a, 'B': b}
    assert all(flags(r)['B'] and not flags(r)['A'] for r in new_val)
    assert all(flags(r)['A'] and not flags(r)['B'] for r in old_eval['primary'] + old_val)
    assert sum(flags(r)['B'] for r in all_val) == 172
    assert all(not (flags(r)['A'] and flags(r)['B']) for r in all_val + old_eval['primary'] + old_eval['retention'])
    old_train = read(VISUAL / 'private_manifest.json')['train']
    evaluation = {'new_validation': new_val, 'old_source': old_eval['primary'], 'old_background': old_eval['retention'], 'old_validation': old_val}
    items = {r['qid']: r for pool in evaluation.values() for r in pool}
    assert len(items) == 349
    assert not {r['img_name'] for r in train + old_train} & {r['img_name'] for r in items.values()}
    refs = reference_records()
    assert all(('base', q, view) in refs for q in items for view in ('N', 'B'))
    before = []
    for seed in p['training_seeds']:
        for row in new_val:
            for view in ('N', 'B'):
                key = (f'A_{seed}', row['qid'], view)
                if key not in refs:
                    before.append({'phase': 'before', 'model': key[0], 'qid': key[1], 'view': key[2]})
    after = []
    for seed in p['training_seeds']:
        for arm in p['arms']:
            pool = list(items.values()) if arm != 'I' else [r for r in items.values() if flags(r)['B'] and not flags(r)['A']]
            after += [{'phase': 'after', 'model': f'{arm}_{seed}', 'qid': row['qid'], 'view': view} for row in pool for view in ('N', 'B')]
    plan = before + after
    order = [f'A_{s}' for s in p['training_seeds']] + [f'{a}_{s}' for s in p['training_seeds'] for a in p['arms']]
    rng.shuffle(plan)
    plan.sort(key=lambda r: order.index(r['model']))
    assert len(plan) <= p['maximum_generation_calls']
    assert len({(r['model'], r['qid'], r['view']) for r in plan}) == len(plan)
    assert not set(refs) & {(r['model'], r['qid'], r['view']) for r in plan}
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    (OUT / 'teacher_cache').mkdir()
    write(OUT / 'private_manifest.json', {'train': train, 'old_train_replay': old_train, **evaluation, 'all_validation': all_val})
    write(OUT / 'private_contracts.json', {'A_pairs': support['task_target_pairs'], 'A_lexicon': support['lexicon'], 'B_normalized_question_templates': templates, 'evaluation_flags': {str(q): flags(r) for q, r in items.items()}, 'validation_flags': {str(r['qid']): flags(r) for r in all_val}})
    write(OUT / 'private_training_exclusion.json', [{'qid': r['qid'], 'reason': 'answer_outside_acquisition_value_schema', 'content_type': r['content_type']} for r in invalid])
    write(OUT / 'schedule.json', plan)
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    for row in (train[0], old_train[0]):
        for view in ('N', 'B'):
            batch = supervised(processor, row, view)
            assert (batch['labels'] != -100).sum() > 0
    pair = common.train_pair(processor, train[0], 'A')
    assert torch.equal(pair['input_ids'][0], pair['input_ids'][1]) and torch.equal(pair['labels'][0], pair['labels'][1])
    assert not torch.cuda.is_initialized()
    check_record = {'status': 'CPU_READY', 'training_candidates': 530, 'training_valid_candidates': 529, 'training_schema_excluded': 1, 'training_questions': 100, 'training_images': 100, 'training_types': dict(counts), 'training_modality': dict(Counter(r['modality'] for r in train)), 'B_question_templates': len(templates), 'new_validation_questions': 172, 'new_validation_images': len({r['img_name'] for r in new_val}), 'B_validation_template_coverage': 172, 'old_source_questions': 69, 'old_background_questions': 69, 'old_validation_questions': 39, 'registry_A_B_overlap': 0, 'train_evaluation_image_disjoint': True, 'old_background_B_activations': sum(flags(r)['B'] for r in old_eval['retention']), 'formal_new_generations': len(plan), 'before_A_missing_generations': len(before), 'after_by_model': {f'{a}_{s}': sum(e['model'] == f'{a}_{s}' for e in after) for s in p['training_seeds'] for a in p['arms']}, 'answer_only_labels_and_full_full_pair': True, 'gpu_initialized': False}
    write(OUT / 'cpu_check.json', check_record)
    p['formal_generation_calls'] = len(plan)
    p['maximum_generation_calls'] = min(p['maximum_generation_calls'], len(plan) + 100)
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps(check_record), flush=True)


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

    families = ['base'] + [f'A_{seed}' for seed in p['training_seeds']]
    for family in families:
        seed = p['training_seeds'][0] if family == 'base' else int(family.split('_')[1])
        teacher = load(family, seed)
        teacher.eval()
        if family != 'base':
            evaluate(teacher, family)
        for kind, pool, view in (('null', m['train'], 'B'), ('old', m['old_train_replay'], 'N')):
            for i, row in enumerate(pool):
                check()
                batch = supervised(processor, row, view).to('cuda')
                with torch.inference_mode():
                    logp, targets = selected_logp(teacher, batch)
                assert torch.isfinite(logp).all()
                cache[family, kind, i] = {'logp': logp.cpu(), 'targets': targets.cpu()}
                torch.save(cache[family, kind, i], OUT / 'teacher_cache' / f'{family}_{kind}_{i:03d}.pt')
                write(OUT / 'progress.json', {'phase': 'teacher_cache', 'family': family, 'forwards_completed': len(cache), 'planned_forwards': 600, 'new_generation_calls': calls, 'updated_unix': time.time()})
        del teacher, batch, logp, targets
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT / 'TEACHER_COMPLETED.json', {'forward_calls': len(cache), 'families': families, 'validation_distilled': False, 'generation_calls': 0})

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
            family = 'base' if arm == 'I' else f'A_{seed}'
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
            old_coefficient = 0.0 if arm == 'S' else 1.0
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
                        steps += 1
                        event = {'arm': arm, 'seed': seed, 'parent': family, 'epoch': epoch, 'step': steps, 'rows': len(group), 'supervised_views': 2 * len(group), 'anchor_forward_views': 2 * len(group), 'loss': sum(x[0] for x in losses) / len(losses), 'supervised_loss': sum(x[1] for x in losses) / len(losses), 'null_KL': sum(x[2] for x in losses) / len(losses), 'old_KL': sum(x[3] for x in losses) / len(losses), 'old_coefficient': old_coefficient, 'preclip_gradient_norm': float(norm), 'updated_unix': time.time()}
                        with (folder / 'updates.jsonl').open('a') as f:
                            f.write(json.dumps(event) + '\n')
                        write(OUT / 'progress.json', dict(event, phase='train', model=label, new_generation_calls=calls))
                    model.save_pretrained(folder / 'adapter', safe_serialization=True)
                    torch.save({'optimizer': optimizer.state_dict(), 'epoch_completed': epoch, 'steps': steps}, folder / 'recovery.pt')
                delta = sum(float((v.detach().float().cpu() - initial[name].float()).square().sum()) for name, v in model.named_parameters() if v.requires_grad)
                assert delta > 0 and math.isfinite(delta) and steps == 52
                write(folder / 'TRAIN_COMPLETED.json', {'arm': arm, 'seed': seed, 'parent': family, 'teacher_family': family, 'optimizer_steps': steps, 'supervised_views': 800, 'anchor_forward_views': 800, 'old_coefficient': old_coefficient, 'trainable_parameters': sum(v.numel() for v in parameters), 'parameter_delta_square_norm': delta, 'language_parameters_updated': False, 'visual_adapter_updated': True, 'completed_unix': time.time()})
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
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_CONTINUAL_ACQUISITION_BANK', 'new_generation_calls': calls, 'trained_adapters': 6, 'optimizer_steps': 312, 'teacher_forward_calls': 600, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
