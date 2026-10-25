"""Small, frozen English medical-card retrieval screen; source data stays read only."""
import asyncio
import itertools
import json
import os
import random
import re
import socket
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageOps

NAME = 'slake_en_pilot_v1'
ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def read(path):
    return json.loads(path.read_text())


def check_clock(p):
    assert time.time() < p['cooperative_stop_unix'] and not (OUT / 'STOP_REQUESTED.json').exists(), 'bounded stop'


def normalize(text):
    # The backend may emit a ChatML assistant header before the answer.
    text = re.sub(r'^\s*assistant\s*\n', '', text, flags=re.I)
    text = re.sub(r'^\s*answer\s*:\s*', '', text.strip(), flags=re.I)
    if re.search(r'<answer>(.*?)</answer>', text, re.S | re.I):
        text = re.search(r'<answer>(.*?)</answer>', text, re.S | re.I).group(1)
    text = re.sub(r'[^\w\s]', ' ', text.casefold())
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    numbers = dict(zip('zero one two three four five six seven eight nine ten'.split(), map(str, range(11))))
    return ' '.join(numbers.get(t, t) for t in text.split())


def score(raw, answer):
    pred, gold = normalize(raw), normalize(answer)
    if gold in ('yes', 'no') and pred.split()[:1] in (['yes'], ['no']):
        pred = pred.split()[0]
    return pred == gold


def query_view(item):
    return {k: item[k] for k in ('qid', 'img_name', 'question')}


def make_card(item):
    image = Image.open(DATA / 'extracted/imgs' / item['img_name']).convert('RGB')
    annotations = read(DATA / 'extracted/imgs' / Path(item['img_name']).parent / 'detection.json')
    source_text = ' ' + normalize(item['question'] + ' ' + item['answer']) + ' '
    selected = []
    for annotation in annotations:
        for label, box in annotation.items():
            assert len(box) == 4
            if ' ' + normalize(label) + ' ' in source_text:
                x, y, w, h = map(float, box)
                assert w > 0 and h > 0
                bounded = [max(0, x), max(0, y), min(image.width - 1, x + w), min(image.height - 1, y + h)]
                if bounded[2] > bounded[0] and bounded[3] > bounded[1]:
                    ImageDraw.Draw(image).rectangle(bounded, outline='#ffbe00', width=3)
                    selected.append({'source_label': label, 'source_xywh': box, 'rendered_xyxy': bounded})
    return ImageOps.pad(image, (336, 336), color='black'), selected


def encode(processor, query, condition, card=None):
    assert set(query) == {'qid', 'img_name', 'question'}
    content, images = [], []
    if condition != 'N':
        assert card is not None
        content.append({'type': 'text', 'text': 'Reference case from a different image. Question: ' + card['question'] + '\nVerified reference answer: ' + card['answer'] + '\nUse only applicable visual cues; the current case may have a different answer.\n'})
        if condition in ('V', 'Blank'):
            images.append(Image.open(OUT / card['card_path']).convert('RGB') if condition == 'V' else Image.new('RGB', (336, 336), 'white'))
            content.extend([{'type': 'text', 'text': 'Reference visual evidence card:\n'}, {'type': 'image', 'image': images[-1]}])
    current_image = Image.open(DATA / 'extracted/imgs' / query['img_name']).convert('RGB')
    # Bound pixels before both HF tokenization and the backend's own image processing.
    current_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    images.append(current_image)
    content.extend([{'type': 'text', 'text': '\nCurrent medical image:\n'}, {'type': 'image', 'image': images[-1]}, {'type': 'text', 'text': '\nQuestion: ' + query['question'] + '\nAnswer:'}])
    messages = [{'role': 'system', 'content': 'Answer the current medical image question. Give only the shortest correct answer, without explanation.'}, {'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[prompt], images=images, return_tensors='pt')
    ids = encoded['input_ids'][0].tolist()
    return ids, images, prompt, encoded['image_grid_thw'].tolist()


def prepare():
    import torch
    from transformers import AutoProcessor
    assert not OUT.exists(), 'never overwrite a previous attempt'
    OUT.mkdir(parents=True)
    for name in ('cards', 'records', 'inputs'):
        (OUT / name).mkdir()
    p = read(CONFIG)
    check_clock(p)
    rows, groups = {}, {}
    for split in ('train', 'validation'):
        raw = [r for r in read(DATA / 'raw' / (split + '.json')) if r['q_lang'] == 'en']
        assert len(raw) == {'train': 4919, 'validation': 1053}[split]
        grouped = {}
        for row in raw:
            if row['content_type'] in p['eligible_content_types']:
                grouped.setdefault(row['img_name'], []).append(row)
        rng = random.Random(p['selection_seed'] + (split == 'validation'))
        chosen = [rng.choice(sorted(v, key=lambda r: r['qid'])) for _, v in sorted(grouped.items())]
        rows[split] = chosen
        groups[split] = set(grouped)
        assert len(chosen) == {'train': 450, 'validation': 96}[split]
        assert all((DATA / 'extracted/imgs' / r['img_name']).is_file() for r in chosen)
    assert not groups['train'] & groups['validation']
    order = list(itertools.permutations(p['conditions']))
    rng = random.Random(p['selection_seed'] + 2)
    shuffled = rows['validation'].copy()
    rng.shuffle(shuffled)
    schedule = [{'qid': r['qid'], 'condition': c} for i, r in enumerate(shuffled) for c in order[i % len(order)]]
    write(OUT / 'private_manifest.json', {'records': rows, 'schedule': schedule, 'test_annotation_opened': False})
    cpu_check()
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)


def cpu_check():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    rows = read(OUT / 'private_manifest.json')['records']
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=p['maximum_image_pixels'])
    card, regions = make_card(rows['train'][0])
    card.save(OUT / 'cards/cpu_fixture.png')
    source = dict(rows['train'][0], card_path='cards/cpu_fixture.png')
    q = query_view(rows['validation'][0])
    v = encode(processor, q, 'V', source)
    blank = encode(processor, q, 'Blank', source)
    n = encode(processor, q, 'N')
    text = encode(processor, q, 'Text', source)
    assert v[0] == blank[0] and v[3] == blank[3] and v[2] == blank[2]
    assert len(v[1]) == len(blank[1]) == 2 and len(n[1]) == len(text[1]) == 1
    assert all(e == (255, 255) for e in blank[1][0].getextrema())
    assert normalize('Answer: The Liver.') == normalize('liver')
    assert score('Yes, it is.', 'yes') and not score('No.', 'yes')
    assert score('two', '2') and not score('liver cancer', 'liver')
    assert score('\nassistant\nNo', 'No') and score('\nassistant\nSpinal Cord', 'Spinal Cord')
    assert not score('\nassistant\nPneumonia', 'Lung Cancer')
    try:
        encode(processor, dict(q, answer='forbidden'), 'N')
    except AssertionError:
        pass
    else:
        raise AssertionError('query labels crossed the input boundary')
    assert not torch.cuda.is_initialized()
    largest = max(rows['validation'], key=lambda r: Image.open(DATA / 'extracted/imgs' / r['img_name']).width)
    large_v = encode(processor, query_view(largest), 'V', source)
    large_blank = encode(processor, query_view(largest), 'Blank', source)
    assert large_v[0] == large_blank[0] and large_v[3] == large_blank[3]
    assert max(large_v[1][-1].size) <= 512
    roi_fixture = next((selected for row in rows['train'][:30] if (selected := make_card(row)[1])), None)
    assert roi_fixture is not None
    fixture = {'status': 'CPU_READY', 'completed_unix': time.time(), 'train_images': 450, 'validation_images': 96, 'validation_episodes': 384, 'selection_categories': {s: dict(Counter(r['content_type'] for r in a)) for s, a in rows.items()}, 'source_fixture_regions': regions, 'source_roi_rendering_fixture_count': len(roi_fixture), 'V_Blank_identical_prompt_ids_and_image_grids': True, 'query_answer_field_rejected': True, 'query_long_side_limit': 512, 'largest_query_fixture_size': list(large_v[1][-1].size), 'gpu_initialized': False, 'model_calls': 0, 'test_annotation_opened': False}
    write(OUT / 'cpu_check.json', fixture)
    print(json.dumps(fixture), flush=True)


def features():
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY'
    manifest = read(OUT / 'private_manifest.json')
    rows = manifest['records']['train'] + manifest['records']['validation']
    processor = CLIPProcessor.from_pretrained(os.environ['CLIP_MODEL_ROOT'], local_files_only=True)
    model = CLIPModel.from_pretrained(os.environ['CLIP_MODEL_ROOT'], local_files_only=True, torch_dtype=torch.float16).to('cuda').eval().requires_grad_(False)
    image_features, text_features = [], []
    with torch.inference_mode():
        for offset in range(0, len(rows), 16):
            check_clock(p)
            batch = rows[offset:offset + 16]
            images = [Image.open(DATA / 'extracted/imgs' / r['img_name']).convert('RGB') for r in batch]
            pixels = processor(images=images, return_tensors='pt')['pixel_values'].to('cuda', dtype=torch.float16)
            texts = processor(text=[r['question'] for r in batch], padding=True, truncation=True, max_length=77, return_tensors='pt')
            image_feature = model.get_image_features(pixel_values=pixels).float()
            text_feature = model.get_text_features(**{k: v.to('cuda') for k, v in texts.items()}).float()
            for tensor, dest in [(image_feature, image_features), (text_feature, text_features)]:
                tensor = torch.nn.functional.normalize(tensor, dim=-1)
                assert torch.isfinite(tensor).all()
                dest.append(tensor.cpu().numpy())
    np.savez(OUT / 'features.npz', image=np.concatenate(image_features), text=np.concatenate(text_features))
    write(OUT / 'FEATURES_COMPLETED.json', {'images': len(rows), 'question_embeddings': len(rows), 'shape': list(np.concatenate(image_features).shape), 'completed_unix': time.time(), 'optimizer_steps': 0})


async def inference_loop(engine, processor, p):
    import numpy as np
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    manifest = read(OUT / 'private_manifest.json')
    train = manifest['records']['train']
    validation = manifest['records']['validation']
    resumed = os.environ.get('TASK_RESUME_VALIDATION') == '1'
    previous = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    if resumed:
        assert len(previous) == 450 and all(r['phase'] == 'train_bank' for r in previous)
        assert [r['qid'] for r in previous] == [r['qid'] for r in train]
        events = [json.loads(s) for s in (OUT / 'requests.jsonl').read_text().splitlines()]
        assert Counter(r['event'] for r in events) == {'intent': 450, 'result': 450}
        assert all(r['phase'] == 'train_bank' for r in events)
    else:
        assert not previous and not (OUT / 'requests.jsonl').exists()
    number = len(previous)

    async def generate(item, condition, source=None, phase='validation'):
        nonlocal number
        check_clock(p)
        q = query_view(item)
        ids, images, prompt, grids = encode(processor, q, condition, source)
        assert len(ids) + p['sampling']['max_new_tokens'] < 9000
        number += 1
        assert number <= p['maximum_generation_calls']
        record = {'event': 'intent', 'number': number, 'phase': phase, 'qid': item['qid'], 'condition': condition, 'source_qid': source['qid'] if source else None, 'prompt_ids': ids, 'prompt_text': prompt, 'image_grids': grids, 'input_image_sizes': [list(im.size) for im in images], 'sampling': p['sampling'].copy(), 'started_unix': time.time()}
        paths = []
        for i, im in enumerate(images):
            path = OUT / 'inputs' / f'{number:04d}_{i}.png'
            im.save(path)
            paths.append(str(path.relative_to(OUT)))
        record['input_images'] = paths
        with (OUT / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        write(OUT / 'progress.json', {'phase': phase, 'dispatched': number, 'current_qid': item['qid'], 'condition': condition, 'updated_unix': time.time()})
        handle = SimpleNamespace(config=SimpleNamespace(max_model_len=9000, response_length=p['sampling']['max_new_tokens']), tokenizer_manager=engine.tokenizer_manager)
        try:
            output = await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle, ids, p['sampling'].copy(), f'p{number}', images)
        except BaseException as exc:
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(record, event='error', error=repr(exc), completed_unix=time.time())) + '\n')
            raise
        raw = processor.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        result = {'event': 'result', 'number': number, 'phase': phase, 'qid': item['qid'], 'condition': condition, 'source_qid': source['qid'] if source else None, 'raw': raw, 'output_ids': output.token_ids, 'correct': score(raw, item['answer']), 'content_type': item['content_type'], 'answer_type': item['answer_type'], 'completed_unix': time.time()}
        with (OUT / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        write(OUT / 'records' / f'{number:04d}.json', result)
        return result, (ids, grids)

    bank, accepted_indices = [], []
    rescored = []
    for i, item in enumerate(train):
        if resumed:
            result = dict(previous[i], original_correct=previous[i]['correct'], correct=score(previous[i]['raw'], item['answer']))
            rescored.append({k: result[k] for k in ('number', 'qid', 'original_correct', 'correct')})
        else:
            result, _ = await generate(item, 'N', phase='train_bank')
        if result['correct']:
            image, regions = make_card(item)
            path = 'cards/' + str(item['qid']) + '.png'
            image.save(OUT / path)
            bank.append(dict(item, card_path=path, source_regions=regions))
            accepted_indices.append(i)
    if resumed:
        write(OUT / 'source_rescore.json', rescored)
    write(OUT / 'private_bank.json', bank)
    write(OUT / 'BANK_COMPLETED.json', {'attempted': len(train), 'accepted': len(bank), 'completed_unix': time.time(), 'test_or_validation_admissions': 0, 'source_generation_replayed': False, 'source_header_rescore_applied': resumed})
    assert bank, 'no successful source examples; preserve negative result'
    feature = np.load(OUT / 'features.npz')
    assert feature['image'].shape == feature['text'].shape == (546, 768)
    retrieval = {}
    for j, item in enumerate(validation):
        q_image, q_text = feature['image'][450 + j], feature['text'][450 + j]
        similarity = .7 * (feature['image'][accepted_indices] @ q_image) + .3 * (feature['text'][accepted_indices] @ q_text)
        k = int(similarity.argmax())
        retrieval[item['qid']] = {'bank_index': k, 'score': float(similarity[k]), 'source_qid': bank[k]['qid']}
    write(OUT / 'retrieval.json', retrieval)
    validation_by_qid = {r['qid']: r for r in validation}
    matched = {}
    for entry in manifest['schedule']:
        item = validation_by_qid[entry['qid']]
        source = bank[retrieval[item['qid']]['bank_index']]
        assert source['img_name'] != item['img_name']
        result, encoding = await generate(item, entry['condition'], source if entry['condition'] != 'N' else None)
        if entry['condition'] in ('V', 'Blank'):
            other = matched.setdefault(item['qid'], {})
            other[entry['condition']] = encoding
            if len(other) == 2:
                assert other['V'] == other['Blank'], 'V/Blank token or image-grid confound'
    assert number == 834 and len(matched) == 96 and all(len(v) == 2 for v in matched.values())
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_450_BANK_384_VALIDATION', 'completed_unix': time.time(), 'generation_calls': number, 'optimizer_steps': 0, 'validation_admissions': 0, 'test_annotation_opened': False, 'V_Blank_prompt_and_grid_pairs_checked': 96})


def infer():
    import sglang
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert read(OUT / 'FEATURES_COMPLETED.json')['images'] == 546
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=p['maximum_image_pixels'])
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    args = dict(model_path=str(ROOT / 'models/b0'), host='127.0.0.1', port=port, dtype='bfloat16', tp_size=1, mem_fraction_static=.30, context_length=9000, max_running_requests=1, disable_cuda_graph=True, attention_backend='triton', mm_attention_backend='sdpa', trust_remote_code=False, random_seed=p['engine_seed'], log_level='info')
    write(OUT / 'private_engine_args.json', args)
    engine = None
    try:
        engine = sglang.Engine(**args)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(inference_loop(engine, processor, p))
    finally:
        if engine is not None:
            engine.shutdown()
        loop.close()


def close():
    import numpy as np
    p = read(CONFIG)
    done = read(OUT / 'WORKER_COMPLETED.json')
    exit_record = read(STATE / (NAME + '_EXIT.json'))
    assert exit_record['exit_code'] == 0 and exit_record['reason'] is None
    rows = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(rows) == done['generation_calls'] == 834
    if (OUT / 'source_rescore.json').exists():
        rescored = {r['number']: r for r in read(OUT / 'source_rescore.json')}
        assert len(rescored) == 450
        rows = [dict(r, correct=rescored[r['number']]['correct']) if r['number'] in rescored else r for r in rows]
    events = [json.loads(s) for s in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(x['event'] for x in events) == {'intent': 834, 'result': 834}
    assert {r['number'] for r in rows} == set(range(1, 835))
    val = [r for r in rows if r['phase'] == 'validation']
    indexed = {(r['qid'], r['condition']): r for r in val}
    qids = sorted({r['qid'] for r in val})
    assert len(qids) == 96 and len(indexed) == len(val) == 384
    summary, differences = {}, {}
    for condition in p['conditions']:
        crows = [indexed[qid, condition] for qid in qids]
        summary[condition] = {'correct': sum(r['correct'] for r in crows), 'n': 96, 'accuracy': sum(r['correct'] for r in crows) / 96, 'truncated_at_token_cap': sum(len(r['output_ids']) == p['sampling']['max_new_tokens'] for r in crows), 'by_content_type': {kind: {'correct': sum(r['correct'] for r in crows if r['content_type'] == kind), 'n': sum(r['content_type'] == kind for r in crows)} for kind in sorted({r['content_type'] for r in val})}}
    rng = np.random.default_rng(p['selection_seed'])
    bootstrap_indices = rng.integers(0, 96, size=(10000, 96))
    for other in ('N', 'Blank', 'Text'):
        values = np.array([int(indexed[q, 'V']['correct']) - int(indexed[q, other]['correct']) for q in qids])
        differences['V_minus_' + other] = {'accuracy_difference': float(values.mean()), 'paired_image_bootstrap_95': np.quantile(values[bootstrap_indices].mean(axis=1), [.025, .975]).tolist(), 'improved': int((values == 1).sum()), 'harmed': int((values == -1).sum())}
    screen = differences['V_minus_N']['improved'] - differences['V_minus_N']['harmed'] >= 5 and differences['V_minus_Blank']['improved'] - differences['V_minus_Blank']['harmed'] >= 3
    result = {'status': 'COMPLETE_BOUNDED_ENGLISH_SLAKE_SCREEN', 'source_bank': read(OUT / 'BANK_COMPLETED.json'), 'summary': summary, 'paired_differences': differences, 'predeclared_promising_screen': bool(screen), 'screen_rule': p['screen_rule'], 'worker_completed': done, 'exit': exit_record, 'output_tokens_all_calls': sum(len(r['output_ids']) for r in rows), 'sealed_test_used': False, 'optimizer_steps': 0, 'limitations': ['One question per validation image from six visual content types; not the full 1053-question validation benchmark.', 'One greedy model initialization; no training-seed replication.', 'Exact normalized answers; open-answer synonyms can be undercounted.', 'Reference examples and source ROIs are an initial medical card adapter, not validated clinical procedures or the complete native game skill loop.', 'Source correctness is answer verification; it does not verify a clinical reasoning trajectory.']}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir()
    write(bundle / 'result.json', result)
    write(bundle / 'protocol.json', p)
    cpu = read(OUT / 'cpu_check.json')
    cpu['source_fixture_region_count'] = len(cpu.pop('source_fixture_regions'))
    write(bundle / 'cpu_check.json', cpu)
    write(bundle / 'features_receipt.json', read(OUT / 'FEATURES_COMPLETED.json'))
    write(bundle / 'complete_numeric_records.json', [{k: r[k] for k in ('number', 'phase', 'qid', 'condition', 'source_qid', 'correct', 'content_type', 'answer_type', 'completed_unix')} | {'output_tokens': len(r['output_ids'])} for r in rows])
    write(bundle / 'retrieval.json', read(OUT / 'retrieval.json'))
    write(bundle / 'worker_completed.json', done)
    write(bundle / 'exit.json', exit_record)
    if (OUT / 'source_rescore.json').exists():
        write(bundle / 'source_rescore.json', read(OUT / 'source_rescore.json'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'cpu_check': cpu_check, 'features': features, 'infer': infer, 'close': close}[os.environ['TASK_STAGE']]()
