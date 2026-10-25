"""Paired region-experience adaptation; private medical inputs never enter public bundles."""
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from PIL import Image, ImageChops, ImageOps

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
NAME = 'region_reader_learning_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
sys.path.insert(0, str(SRC / 'scripts'))
from slake_probe import normalize, score, query_view


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def check():
    p = read(CONFIG)
    assert time.time() < p['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def image_view(row, view):
    assert view in ('N', 'O', 'W', 'B')
    if view == 'B':
        return Image.new('RGB', (336, 336), 'black')
    image = Image.open(DATA / 'extracted/imgs' / row['img_name']).convert('RGB')
    if view != 'N':
        image = image.crop(row['oracle_crop_xyxy' if view == 'O' else 'wrong_crop_xyxy'])
    return ImageOps.pad(image, (336, 336), color='black')


def prompt(processor, query):
    assert set(query) == {'qid', 'img_name', 'question'}
    content = [{'type': 'text', 'text': 'Current medical image:\n'}, {'type': 'image'}, {'type': 'text', 'text': '\nQuestion: ' + query['question'] + '\nAnswer:'}]
    messages = [{'role': 'system', 'content': 'Answer the current medical image question. Give only the shortest correct answer, without explanation.'}, {'role': 'user', 'content': content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def train_pair(processor, row, arm):
    import torch
    assert arm in ('A', 'B', 'C')
    second = {'A': 'N', 'B': 'O', 'C': 'W'}[arm]
    images = [image_view(row, 'N'), image_view(row, second)]
    text = prompt(processor, query_view(row))
    prefix = processor(text=[text], images=[images[0]], return_tensors='pt')['input_ids'][0]
    full = text + str(row['answer']) + processor.tokenizer.eos_token
    batch = processor(text=[full, full], images=images, padding=True, return_tensors='pt')
    assert all(torch.equal(ids[:len(prefix)], prefix) for ids in batch['input_ids'])
    labels = batch['input_ids'].clone()
    labels[:, :len(prefix)] = -100
    labels[batch['attention_mask'] == 0] = -100
    assert (labels != -100).sum().item() > 0
    assert torch.equal(labels[0], labels[1])
    assert batch['input_ids'].shape[1] < 1024
    batch['labels'] = labels
    return batch


def prepare():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    eligible = read(ROOT / 'artifacts/medical/roi_mask_m0_v1/private_eligible.json')
    train = sorted((r for r in eligible if r['split'] == 'train'), key=lambda r: r['qid'])
    ledger = read(ROOT / 'artifacts/medical/roi_mask_m0_v1/private_ledger.json')
    primary_ids = {r['qid'] for r in ledger if r['split'] == 'validation' and r['task'] and r['target']}
    validation = [r for r in read(DATA / 'raw/validation.json') if r['q_lang'] == 'en' and r['base_type'] == 'vqa']
    primary = sorted((r for r in validation if r['qid'] in primary_ids), key=lambda r: r['qid'])
    annotations = {r['qid']: r for r in ledger if r['split'] == 'validation'}
    primary = [dict(r, task=annotations[r['qid']]['task'], target=annotations[r['qid']]['target']) for r in primary]
    pilot = read(ROOT / 'artifacts/medical/slake_en_pilot_v1/private_manifest.json')['records']['validation']
    retention = sorted((r for r in pilot if r['content_type'] in ('Organ', 'Position', 'Size', 'Quantity')), key=lambda r: r['qid'])
    regional = sorted((r for r in eligible if r['split'] == 'validation'), key=lambda r: r['qid'])
    assert len(train) == 100 and len({r['img_name'] for r in train}) == 75
    assert len(primary) == 71 and len({r['img_name'] for r in primary}) == 46
    assert len(retention) == 69 and len(regional) == 19
    assert not {r['qid'] for r in primary} & {r['qid'] for r in retention}
    assert not {r['img_name'] for r in train} & {r['img_name'] for r in primary + retention + regional}
    assert {r['qid'] for r in regional} <= {r['qid'] for r in primary}
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    manifest = {'train': train, 'primary': primary, 'retention': retention, 'regional': regional}
    write(OUT / 'private_manifest.json', manifest)
    plan = [{'panel': kind, 'qid': r['qid'], 'view': view} for kind, pool in (('primary', primary), ('retention', retention)) for r in pool for view in ('N', 'B')]
    plan += [{'panel': 'regional', 'qid': r['qid'], 'view': view} for r in regional for view in ('O', 'W')]
    assert len(plan) == 318
    random.Random(p['selection_seed']).shuffle(plan)
    write(OUT / 'schedule.json', plan)
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    for row in (min(train, key=lambda r: r['target_area_fraction']), max(train, key=lambda r: r['target_area_fraction'])):
        pairs = {a: train_pair(processor, row, a) for a in ('A', 'B', 'C')}
        assert all(torch.equal(pairs['A']['input_ids'], b['input_ids']) and torch.equal(pairs['A']['labels'], b['labels']) for b in pairs.values())
        assert ImageChops.difference(image_view(row, 'O'), image_view(row, 'W')).getbbox() is not None
        assert image_view(row, 'B').getbbox() is None
        try:
            prompt(processor, dict(query_view(row), answer='forbidden'))
        except AssertionError:
            pass
        else:
            raise AssertionError('gold entered generation interface')
    assert not torch.cuda.is_initialized()
    counts = {k: {'questions': len(v), 'images': len({r['img_name'] for r in v}), 'tasks': dict(Counter(r.get('task', r['content_type']) for r in v))} for k, v in manifest.items()}
    write(OUT / 'cpu_check.json', {'status': 'CPU_READY', 'generation_gold_field_rejected': True, 'answer_only_labels_nonempty': True, 'arms_have_equal_ids_and_labels': True, 'same_example_order_and_update_budget': True, 'train_validation_image_disjoint': True, 'gpu_initialized': False, 'counts': counts, 'generation_calls_per_model': len(plan), 'formal_generation_calls': 7 * len(plan)})
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps({'status': p['status'], 'counts': counts, 'formal_generation_calls': 7 * len(plan)}), flush=True)


def run():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY'
    check()
    manifest = read(OUT / 'private_manifest.json')
    all_rows = {r['qid']: r for pool in manifest.values() for r in pool}
    plan = read(OUT / 'schedule.json')
    assert not (OUT / 'requests.jsonl').exists(), 'no blind replay of a partially dispatched experiment'
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    torch.set_num_threads(4)
    calls = 0

    def load(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')

    def evaluate(model, label):
        nonlocal calls
        model.eval()
        model.config.use_cache = True
        for entry in plan:
            check()
            calls += 1
            assert calls <= p['maximum_generation_calls']
            row = all_rows[entry['qid']]
            text = prompt(processor, query_view(row))
            inputs = processor(text=[text], images=[image_view(row, entry['view'])], return_tensors='pt').to('cuda')
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
            write(OUT / 'progress.json', {'phase': 'evaluate', 'model': label, 'calls': calls, 'planned_calls': p['formal_generation_calls'], 'updated_unix': time.time()})
        write(OUT / f'{label}_EVALUATED.json', {'model': label, 'formal_calls': len(plan), 'completed_unix': time.time()})

    fixture = load(p['training_seeds'][0])
    targets = [name for name, module in fixture.named_modules() if '.layers.' in name and '.self_attn.' in name and name.rsplit('.', 1)[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    assert targets and not any('visual' in name for name in targets)
    fixture = get_peft_model(fixture, LoraConfig(r=16, lora_alpha=32, lora_dropout=0, bias='none', task_type='CAUSAL_LM', target_modules=targets))
    fixture.enable_input_require_grads()
    fixture.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    fixture.config.use_cache = False
    fixture.train()
    loss = fixture(**train_pair(processor, manifest['train'][0], 'B').to('cuda')).loss
    assert torch.isfinite(loss)
    loss.backward()
    grads = [v.grad for v in fixture.parameters() if v.requires_grad and v.grad is not None]
    assert grads and all(torch.isfinite(v).all() for v in grads)
    write(OUT / 'production_backward_check.json', {'status': 'PASS', 'finite_loss_and_adapter_gradients': True, 'optimizer_steps': 0, 'generation_calls': 0, 'loss': float(loss.detach()), 'gradient_tensors': len(grads), 'completed_unix': time.time()})
    del fixture, loss, grads
    gc.collect()
    torch.cuda.empty_cache()
    base = load(p['training_seeds'][0])
    evaluate(base, 'base')
    del base
    gc.collect()
    torch.cuda.empty_cache()
    for seed in p['training_seeds']:
        for arm in p['arms']:
            check()
            label = f'{arm}_{seed}'
            folder = OUT / label
            folder.mkdir(exist_ok=False)
            model = load(seed)
            # Scope LoRA to language layers; the visual encoder stays frozen.
            targets = [name for name, module in model.named_modules() if '.layers.' in name and '.self_attn.' in name and name.rsplit('.', 1)[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
            assert targets and not any('visual' in name for name in targets)
            adapter = LoraConfig(r=16, lora_alpha=32, lora_dropout=0, bias='none', task_type='CAUSAL_LM', target_modules=targets)
            model = get_peft_model(model, adapter)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.config.use_cache = False
            parameters = [v for v in model.parameters() if v.requires_grad]
            before = sum(float(v.detach().float().square().sum()) for n, v in model.named_parameters() if 'lora_B' in n)
            assert before == 0
            optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
            steps = 0
            model.train()
            try:
                for epoch in range(p['epochs']):
                    order = manifest['train'].copy()
                    random.Random(seed + 1000 * epoch).shuffle(order)
                    for offset in range(0, len(order), p['rows_per_update']):
                        group = order[offset:offset + p['rows_per_update']]
                        optimizer.zero_grad(set_to_none=True)
                        losses = []
                        for row in group:
                            check()
                            batch = train_pair(processor, row, arm).to('cuda')
                            loss = model(**batch).loss
                            assert torch.isfinite(loss)
                            (loss / len(group)).backward()
                            losses.append(float(loss.detach()))
                        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                        assert torch.isfinite(norm)
                        optimizer.step()
                        steps += 1
                        event = {'arm': arm, 'seed': seed, 'epoch': epoch, 'step': steps, 'rows': len(group), 'supervised_views': 2 * len(group), 'loss': sum(losses) / len(losses), 'preclip_gradient_norm': float(norm), 'updated_unix': time.time()}
                        with (folder / 'updates.jsonl').open('a') as f:
                            f.write(json.dumps(event) + '\n')
                        write(OUT / 'progress.json', dict(event, phase='train', model=label, calls=calls))
                    model.save_pretrained(folder / 'adapter', safe_serialization=True)
                    torch.save({'optimizer': optimizer.state_dict(), 'epoch_completed': epoch, 'steps': steps}, folder / 'recovery.pt')
                after = sum(float(v.detach().float().square().sum()) for n, v in model.named_parameters() if 'lora_B' in n)
                assert after > 0 and math.isfinite(after)
                assert steps == p['epochs'] * math.ceil(len(order) / p['rows_per_update'])
                write(folder / 'TRAIN_COMPLETED.json', {'arm': arm, 'seed': seed, 'optimizer_steps': steps, 'supervised_image_exposures': 2 * len(order) * p['epochs'], 'trainable_parameters': sum(v.numel() for v in parameters), 'lora_B_initial_square_norm': before, 'lora_B_final_square_norm': after, 'visual_encoder_updated': False, 'completed_unix': time.time()})
            except BaseException:
                model.save_pretrained(folder / 'recovery_adapter', safe_serialization=True)
                raise
            del optimizer, parameters
            model.gradient_checkpointing_disable()
            evaluate(model, label)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert calls == p['formal_generation_calls'] == 2226
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_PAIRED_READER_LEARNING', 'generation_calls': calls, 'trained_adapters': 6, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
