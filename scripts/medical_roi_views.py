"""Frozen current-image local-view diagnosis with no reference-case context."""
import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
from types import SimpleNamespace

from PIL import Image, ImageChops, ImageOps

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
NAME = os.environ.get('TASK_EXPERIMENT', 'roi_mask_m1_v1')
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
sys.path.insert(0, str(SRC / 'scripts'))
from slake_probe import normalize, score, query_view


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def check_clock(p):
    assert time.time() < p['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def encode(processor, query, condition, region):
    assert set(query) == {'qid', 'img_name', 'question'}
    assert set(region) == {'oracle_crop_xyxy', 'wrong_crop_xyxy'}
    p = read(CONFIG)
    assert condition in p['conditions']
    original = Image.open(DATA / 'extracted/imgs' / query['img_name']).convert('RGB')
    if p.get('view_mode') == 'isolated':
        if condition == 'B':
            selected = Image.new('RGB', (336, 336), 'black')
        elif condition == 'N':
            selected = original
        else:
            selected = original.crop(region['oracle_crop_xyxy' if condition == 'O' else 'wrong_crop_xyxy'])
        selected = ImageOps.pad(selected, (336, 336), color='black')
        content = [{'type': 'text', 'text': 'Current medical image:\n'}, {'type': 'image', 'image': selected}, {'type': 'text', 'text': '\nQuestion: ' + query['question'] + '\nAnswer:'}]
        messages = [{'role': 'system', 'content': 'Answer the current medical image question. Give only the shortest correct answer, without explanation.'}, {'role': 'user', 'content': content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[selected], return_tensors='pt')
        return inputs['input_ids'][0].tolist(), [selected], prompt, inputs['image_grid_thw'].tolist()
    current = original.copy()
    current.thumbnail((512, 512), Image.Resampling.LANCZOS)
    images = [current]
    content = [{'type': 'text', 'text': 'Current medical image:\n'}, {'type': 'image', 'image': current}]
    if condition != 'N':
        additional = original if condition == 'D' else original.crop(region['oracle_crop_xyxy' if condition == 'O' else 'wrong_crop_xyxy'])
        additional = ImageOps.pad(additional, (336, 336), color='black')
        images.append(additional)
        content.extend([{'type': 'text', 'text': '\nAdditional view:\n'}, {'type': 'image', 'image': additional}])
    content.append({'type': 'text', 'text': '\nQuestion: ' + query['question'] + '\nAnswer:'})
    messages = [{'role': 'system', 'content': 'Answer the current medical image question. Give only the shortest correct answer, without explanation.'}, {'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, return_tensors='pt')
    return inputs['input_ids'][0].tolist(), images, prompt, inputs['image_grid_thw'].tolist()


def prepare():
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check_clock(p)
    assert read(ROOT / 'artifacts/medical/roi_mask_m0_v1/RESULT.json')['status'] == 'M0_MASK_PASS'
    panel = read(ROOT / 'artifacts/medical/roi_mask_m0_v1/private_panel.json')
    assert len(panel) == len({r['img_name'] for r in panel}) == 64
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    (OUT / 'inputs').mkdir()
    write(OUT / 'private_panel.json', panel)
    order = panel.copy()
    rng = random.Random(p['selection_seed'] + 1)
    rng.shuffle(order)
    schedule = []
    for repeat in (0, 1):
        for i, row in enumerate(order):
            offset = i % 4
            conditions = p['conditions'][offset:] + p['conditions'][:offset]
            if repeat:
                conditions.reverse()
            schedule.extend({'repeat': repeat, 'qid': row['qid'], 'condition': c} for c in conditions)
    write(OUT / 'schedule.json', schedule)
    cpu_check()
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps({'status': 'FROZEN_CPU_READY', 'images': 64, 'conditions': 4, 'repeats': 2, 'planned_calls': len(schedule)}), flush=True)


def cpu_check():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=p['maximum_image_pixels'])
    panel = read(OUT / 'private_panel.json')
    smallest = min(panel, key=lambda r: r['target_area_fraction'])
    largest = max(panel, key=lambda r: r['target_area_fraction'])
    for row in (smallest, largest):
        query = query_view(row)
        region = {k: row[k] for k in ('oracle_crop_xyxy', 'wrong_crop_xyxy')}
        enc = {c: encode(processor, query, c, region) for c in p['conditions']}
        if p.get('view_mode') == 'isolated':
            assert len({json.dumps((e[0], e[2], e[3])) for e in enc.values()}) == 1
            assert all(len(e[1]) == 1 and e[1][0].size == (336, 336) for e in enc.values())
            assert enc['B'][1][0].getbbox() is None
            assert ImageChops.difference(enc['O'][1][0], enc['W'][1][0]).getbbox() is not None
        else:
            assert enc['D'][0] == enc['O'][0] == enc['W'][0]
            assert enc['D'][2] == enc['O'][2] == enc['W'][2]
            assert enc['D'][3] == enc['O'][3] == enc['W'][3]
            assert len(enc['N'][1]) == 1 and all(len(enc[c][1]) == 2 for c in ('D', 'O', 'W'))
            assert all(ImageChops.difference(enc['N'][1][0], e[1][0]).getbbox() is None for e in enc.values())
            assert ImageChops.difference(enc['O'][1][1], enc['W'][1][1]).getbbox() is not None
        assert all(len(e[0]) + 64 < 9000 for e in enc.values())
        try:
            encode(processor, dict(query, answer='forbidden'), 'O', region)
        except AssertionError:
            pass
        else:
            raise AssertionError('gold entered query view')
    assert not torch.cuda.is_initialized()
    write(OUT / 'cpu_check.json', {'status': 'CPU_READY', 'view_mode': p.get('view_mode', 'additional'), 'small_and_large_region_fixtures': True, 'matched_condition_prompts_ids_grids': True, 'current_full_pixels_identical': p.get('view_mode') != 'isolated', 'crop_pixels_are_raw_source_not_masks': True, 'gold_field_rejected': True, 'gpu_initialized': False, 'completed_unix': time.time()})


async def inference_loop(engine, processor, p, repeat):
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    panel = {r['qid']: r for r in read(OUT / 'private_panel.json')}
    plan = [r for r in read(OUT / 'schedule.json') if r['repeat'] == repeat]
    assert len(plan) == 256
    events = [json.loads(line) for line in (OUT / 'requests.jsonl').read_text().splitlines()] if (OUT / 'requests.jsonl').exists() else []
    attempts = sum(e['event'] == 'intent' for e in events)
    matched = defaultdict(dict)
    for i, entry in enumerate(plan):
        check_clock(p)
        row, condition = panel[entry['qid']], entry['condition']
        region = {k: row[k] for k in ('oracle_crop_xyxy', 'wrong_crop_xyxy')}
        ids, images, prompt, grids = encode(processor, query_view(row), condition, region)
        number = repeat * 256 + i + 1
        path = OUT / 'records' / f'{number:04d}.json'
        if path.exists():
            previous = read(path)
            assert previous['qid'] == row['qid'] and previous['repeat'] == repeat and previous['condition'] == condition
        else:
            attempts += 1
            assert attempts <= p['maximum_generation_calls'] and len(ids) + 64 < 9000
            intent = {'event': 'intent', 'number': number, 'attempt': attempts, 'repeat': repeat, 'qid': row['qid'], 'condition': condition, 'prompt_ids': ids, 'prompt_text': prompt, 'image_grids': grids, 'input_images': [], 'started_unix': time.time()}
            for j, image in enumerate(images):
                name = f'inputs/{number:04d}_{j}.png'
                image.save(OUT / name)
                intent['input_images'].append(name)
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(intent) + '\n')
            write(OUT / 'progress.json', {'repeat': repeat, 'dispatched_sample': number, 'planned_samples': 512, 'actual_attempts': attempts, 'updated_unix': time.time()})
            handle = SimpleNamespace(config=SimpleNamespace(max_model_len=9000, response_length=64), tokenizer_manager=engine.tokenizer_manager)
            try:
                output = await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle, ids, p['sampling'].copy(), f'p{attempts}', images)
            except BaseException as exc:
                with (OUT / 'requests.jsonl').open('a') as f:
                    f.write(json.dumps(dict(intent, event='error', error=repr(exc))) + '\n')
                raise
            raw = processor.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            result = {'event': 'result', 'number': number, 'attempt': attempts, 'repeat': repeat, 'qid': row['qid'], 'condition': condition, 'raw': raw, 'output_ids': output.token_ids, 'completed_unix': time.time()}
            write(path, result)
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(result) + '\n')
        if condition in (p['conditions'] if p.get('view_mode') == 'isolated' else ('D', 'O', 'W')):
            matched[row['qid']][condition] = ids, grids
            if len(matched[row['qid']]) == (4 if p.get('view_mode') == 'isolated' else 3):
                assert len({json.dumps(v) for v in matched[row['qid']].values()}) == 1
    assert len(matched) == 64 and all(len(v) == (4 if p.get('view_mode') == 'isolated' else 3) for v in matched.values())
    write(OUT / f'REPEAT_{repeat}_COMPLETED.json', {'repeat': repeat, 'formal_samples': 256, 'actual_attempts_so_far': attempts, 'matched_input_triples': len(matched), 'gold_used_for_generation': False, 'completed_unix': time.time()})


def infer():
    import sglang
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY'
    repeat = int(os.environ['TASK_REPEAT'])
    assert repeat in (0, 1) and not (OUT / f'REPEAT_{repeat}_COMPLETED.json').exists()
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=p['maximum_image_pixels'])
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    engine = None
    try:
        engine = sglang.Engine(model_path=str(ROOT / 'models/b0'), host='127.0.0.1', port=port, dtype='bfloat16', tp_size=1, mem_fraction_static=.30, context_length=9000, max_running_requests=1, disable_cuda_graph=True, attention_backend='triton', mm_attention_backend='sdpa', trust_remote_code=False, random_seed=p['engine_seeds'][repeat], log_level='info')
        asyncio.set_event_loop(loop)
        loop.run_until_complete(inference_loop(engine, processor, p, repeat))
    finally:
        if engine is not None:
            engine.shutdown()
        loop.close()


if __name__ == '__main__':
    {'prepare': prepare, 'cpu_check': cpu_check, 'infer': infer}[os.environ['TASK_STAGE']]()
