"""Visual-only region adaptation with train-only null and old-task distillation."""
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
NAME = 'grounded_reader_anchor_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from slake_probe import score, query_view
read, write, image_view, prompt = common.read, common.write, common.image_view, common.prompt


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def supervised(processor, row, view):
    import torch
    text = prompt(processor, query_view(row))
    image = image_view(row, view)
    prefix = processor(text=[text], images=[image], return_tensors='pt')['input_ids'][0]
    batch = processor(text=[text + str(row['answer']) + processor.tokenizer.eos_token], images=[image], return_tensors='pt')
    assert torch.equal(batch['input_ids'][0, :len(prefix)], prefix)
    labels = batch['input_ids'].clone()
    labels[:, :len(prefix)] = -100
    assert (labels != -100).sum() > 0 and labels.shape[1] < 1024
    batch['labels'] = labels
    return batch


def prepare():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    previous = ROOT / 'artifacts/medical/region_reader_learning_v1'
    assert read(previous / 'RESULT.json')['status'] == 'COMPLETE_REGION_READER_LEARNING_NOT_ESTABLISHED'
    m = read(previous / 'private_manifest.json')
    raw = [r for r in read(DATA / 'raw/train.json') if r['q_lang'] == 'en' and r['base_type'] == 'vqa' and r['content_type'] in ('Organ', 'Position', 'Size', 'Quantity')]
    pools = defaultdict(list)
    for row in raw:
        pools[row['img_name']].append(row)
    rng = random.Random(p['selection_seed'])
    one = [rng.choice(sorted(rows, key=lambda r: r['qid'])) for name, rows in sorted(pools.items())]
    by_modality = defaultdict(list)
    for row in one:
        by_modality[row['modality']].append(row)
    for rows in by_modality.values():
        rng.shuffle(rows)
    replay = []
    while len(replay) < 100:
        for kind in sorted(by_modality):
            if by_modality[kind] and len(replay) < 100:
                replay.append(by_modality[kind].pop())
        assert any(by_modality.values()) or len(replay) == 100
    assert len(replay) == len({r['img_name'] for r in replay}) == 100
    assert not {r['img_name'] for r in m['train'] + replay} & {r['img_name'] for r in m['primary'] + m['retention']}
    assert len(m['train']) == 100 and len(m['primary']) == 71 and len(m['retention']) == 69
    m['replay'] = replay
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    (OUT / 'teacher_cache').mkdir()
    write(OUT / 'private_manifest.json', m)
    write(OUT / 'schedule.json', read(previous / 'schedule.json'))
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    for row in (m['train'][0], replay[0]):
        for view in ('N', 'B'):
            batch = supervised(processor, row, view)
            assert (batch['labels'] != -100).sum() > 0
    a, b = [supervised(processor, m['train'][0], v) for v in ('N', 'O')]
    assert torch.equal(a['input_ids'], b['input_ids']) and torch.equal(a['labels'], b['labels'])
    assert not torch.cuda.is_initialized()
    write(OUT / 'cpu_check.json', {'status': 'CPU_READY', 'same_original_train_validation_manifest': True, 'train_only_replay_questions': 100, 'replay_images': 100, 'replay_modality_counts': dict(Counter(r['modality'] for r in replay)), 'replay_content_counts': dict(Counter(r['content_type'] for r in replay)), 'train_validation_image_disjoint': True, 'answer_only_labels_and_matched_region_ids': True, 'gpu_initialized': False, 'new_generation_calls': 1908, 'reused_base_generation_calls': 318})
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps(read(OUT / 'cpu_check.json')), flush=True)


def run():
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY' and not (OUT / 'requests.jsonl').exists()
    check()
    m = read(OUT / 'private_manifest.json')
    plan = read(OUT / 'schedule.json')
    items = {r['qid']: r for pool in m.values() for r in pool}
    position = {r['qid']: i for i, r in enumerate(m['train'])}
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    torch.set_num_threads(4)

    def load(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')

    def selected_logp(model, batch):
        logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
        active = batch['labels'][:, 1:] != -100
        return logits[:, :-1][active].float().log_softmax(-1), batch['labels'][:, 1:][active]

    teacher = load(p['training_seeds'][0])
    teacher.eval()
    cache = {}
    for kind, pool, view in (('null', m['train'], 'B'), ('old', m['replay'], 'N')):
        for i, row in enumerate(pool):
            check()
            batch = supervised(processor, row, view).to('cuda')
            with torch.inference_mode():
                logp, targets = selected_logp(teacher, batch)
            assert torch.isfinite(logp).all()
            cache[kind, i] = {'logp': logp.cpu(), 'targets': targets.cpu()}
            torch.save(cache[kind, i], OUT / 'teacher_cache' / f'{kind}_{i:03d}.pt')
            write(OUT / 'progress.json', {'phase': 'teacher_cache', 'forwards_completed': len(cache), 'planned_forwards': 200, 'new_generation_calls': 0, 'updated_unix': time.time()})
    write(OUT / 'TEACHER_COMPLETED.json', {'forward_calls': len(cache), 'generation_calls': 0, 'validation_queries_used': False})
    del teacher, batch, logp, targets
    gc.collect()
    torch.cuda.empty_cache()
    calls = 0

    def distill(model, row, view, reference):
        batch = supervised(processor, row, view).to('cuda')
        logp, targets = selected_logp(model, batch)
        assert torch.equal(targets.cpu(), reference['targets'])
        value = F.kl_div(logp, reference['logp'].to('cuda'), log_target=True, reduction='none').sum(-1).mean()
        assert torch.isfinite(value)
        return value

    def evaluate(model, label):
        nonlocal calls
        model.eval()
        model.config.use_cache = True
        for entry in plan:
            check()
            calls += 1
            assert calls <= p['maximum_generation_calls']
            row = items[entry['qid']]
            inputs = processor(text=[prompt(processor, query_view(row))], images=[image_view(row, entry['view'])], return_tensors='pt').to('cuda')
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(entry, event='intent', number=calls, model=label, started_unix=time.time())) + '\n')
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
            raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            record = dict(entry, number=calls, model=label, raw=raw, output_ids=tokens, correct=bool(score(raw, row['answer'])), completed_unix=time.time())
            write(OUT / 'records' / f'{calls:04d}.json', record)
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(record, event='result')) + '\n')
            write(OUT / 'progress.json', {'phase': 'evaluate', 'model': label, 'new_generation_calls': calls, 'planned_calls': 1908, 'updated_unix': time.time()})
        write(OUT / f'{label}_EVALUATED.json', {'formal_calls': 318, 'completed_unix': time.time()})

    checked = False
    for seed in p['training_seeds']:
        for arm in p['arms']:
            check()
            label = f'{arm}_{seed}'
            folder = OUT / label
            folder.mkdir(exist_ok=False)
            model = load(seed)
            targets = [name for name, module in model.named_modules() if 'visual' in name and name.endswith(('.attn.qkv', '.attn.proj')) and isinstance(module, torch.nn.Linear)]
            assert targets and all('visual' in name for name in targets)
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0, bias='none', task_type='CAUSAL_LM', target_modules=targets))
            assert all('visual' in name for name, v in model.named_parameters() if v.requires_grad)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.config.use_cache = False
            model.train()
            parameters = [v for v in model.parameters() if v.requires_grad]
            if not checked:
                row = m['train'][0]
                ce = model(**common.train_pair(processor, row, 'B').to('cuda')).loss
                null = distill(model, row, 'B', cache['null', 0])
                old = distill(model, m['replay'][0], 'N', cache['old', 0])
                assert abs(float(null.detach())) < 2e-4 and abs(float(old.detach())) < 2e-4
                (ce + null + old).backward()
                grads = [v.grad for v in parameters if v.grad is not None]
                assert grads and all(torch.isfinite(v).all() for v in grads)
                write(OUT / 'production_backward_check.json', {'status': 'PASS', 'language_parameters_frozen': True, 'visual_adapter_gradient_tensors': len(grads), 'finite_full_objective_backward': True, 'initial_null_KL': float(null.detach()), 'initial_old_KL': float(old.detach()), 'optimizer_steps': 0, 'generation_calls': 0})
                model.zero_grad(set_to_none=True)
                del ce, null, old, grads
                checked = True
            optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
            steps = 0
            try:
                for epoch in range(p['epochs']):
                    order = m['train'].copy()
                    random.Random(seed + 1000 * epoch).shuffle(order)
                    for offset in range(0, len(order), p['rows_per_update']):
                        group = order[offset:offset + p['rows_per_update']]
                        optimizer.zero_grad(set_to_none=True)
                        metrics = []
                        for row in group:
                            check()
                            index = position[row['qid']]
                            ce = model(**common.train_pair(processor, row, 'A' if arm == 'A' else 'B').to('cuda')).loss
                            null = distill(model, row, 'B', cache['null', index])
                            old = distill(model, m['replay'][index], 'N', cache['old', index])
                            coefficient = 0.0 if arm == 'U' else 1.0
                            loss = ce + coefficient * (null + old)
                            assert torch.isfinite(loss)
                            (loss / len(group)).backward()
                            metrics.append([float(v.detach()) for v in (loss, ce, null, old)])
                        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                        assert torch.isfinite(norm)
                        optimizer.step()
                        steps += 1
                        event = {'arm': arm, 'seed': seed, 'epoch': epoch, 'step': steps, 'rows': len(group), 'supervised_views': 2 * len(group), 'anchor_forward_views': 2 * len(group), 'loss': sum(v[0] for v in metrics) / len(metrics), 'supervised_loss': sum(v[1] for v in metrics) / len(metrics), 'null_KL': sum(v[2] for v in metrics) / len(metrics), 'old_KL': sum(v[3] for v in metrics) / len(metrics), 'anchor_coefficient': coefficient, 'preclip_gradient_norm': float(norm), 'updated_unix': time.time()}
                        with (folder / 'updates.jsonl').open('a') as f:
                            f.write(json.dumps(event) + '\n')
                        write(OUT / 'progress.json', dict(event, phase='train', model=label, new_generation_calls=calls))
                    model.save_pretrained(folder / 'adapter', safe_serialization=True)
                    torch.save({'optimizer': optimizer.state_dict(), 'epoch_completed': epoch, 'steps': steps}, folder / 'recovery.pt')
                norm_B = sum(float(v.detach().float().square().sum()) for name, v in model.named_parameters() if 'lora_B' in name)
                assert norm_B > 0 and math.isfinite(norm_B) and steps == 52
                write(folder / 'TRAIN_COMPLETED.json', {'arm': arm, 'seed': seed, 'optimizer_steps': steps, 'supervised_image_exposures': 800, 'anchor_forward_image_exposures': 800, 'trainable_parameters': sum(v.numel() for v in parameters), 'lora_B_initial_square_norm': 0, 'lora_B_final_square_norm': norm_B, 'visual_encoder_updated': True, 'language_parameters_updated': False, 'anchor_coefficient': coefficient, 'completed_unix': time.time()})
            except BaseException:
                model.save_pretrained(folder / 'recovery_adapter', safe_serialization=True)
                raise
            del optimizer, parameters
            model.gradient_checkpointing_disable()
            evaluate(model, label)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert calls == 1908
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_GROUNDED_READER_ANCHOR', 'generation_calls': calls, 'reused_base_generation_calls': 318, 'trained_adapters': 6, 'teacher_forward_calls': 200, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
