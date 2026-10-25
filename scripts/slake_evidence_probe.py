"""Answer-free, region-grounded reference cards with task-compatible retrieval."""
import asyncio
from collections import Counter, defaultdict
import itertools
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
from types import SimpleNamespace

from PIL import Image, ImageChops, ImageDraw, ImageOps

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
NAME = 'slake_evidence_v2'
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
OLD = ROOT / 'artifacts/medical/slake_en_pilot_v1'
sys.path.insert(0, str(SRC / 'scripts'))
from slake_probe import normalize, score, query_view


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def check_clock(p):
    assert time.time() < p['cooperative_stop_unix'] and not (OUT / 'STOP_REQUESTED.json').exists()


def entity(label):
    return re.sub(r'^(left|right) ', '', label.lower())


def parse_question(question, lexicon):
    text = question.lower()
    hits = [s for s in lexicon if re.search(r'\b' + re.escape(s) + r'(?:s|es)?\b', text)]
    hits = [s for s in hits if not any(s != t and s in t for t in hits)]
    if len(hits) != 1:
        return None
    if 'how many' in text:
        role = 'count'
    elif 'shape' in text:
        role = 'shape'
    elif re.search(r'largest|smallest|biggest|bigger|larger|smaller|size', text):
        role = 'size'
    elif re.search(r'normal|abnormal|healthy', text):
        role = 'normality'
    elif re.search(r'where|which side|which hemisphere|left or right|which part', text):
        role = 'location'
    elif re.match(r'(is|are|does|do|can|has|have)\b', text):
        role = 'presence'
    elif text.startswith('which is '):
        role = 'appearance'
    else:
        return None
    return role, hits[0]


def xyxy(box):
    x, y, w, h = map(float, box)
    assert w > 0 and h > 0
    return x, y, x + w, y + h


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / (area(a) + area(b) - overlap)


def tokens(question):
    return set(normalize(question).split())


def lexical(a, b):
    a, b = tokens(a), tokens(b)
    return len(a & b) / len(a | b) if a | b else 0


def render(item, wrong=False):
    original = Image.open(DATA / 'extracted/imgs' / item['img_name']).convert('RGB')
    box = item['decoy_xyxy'] if wrong else item['target_xyxy']
    x0, y0, x1, y1 = box
    bounds = (max(0, x0), max(0, y0), min(original.width - 1, x1), min(original.height - 1, y1))
    assert bounds[2] > bounds[0] and bounds[3] > bounds[1]
    marked = original.copy()
    ImageDraw.Draw(marked).rectangle(bounds, outline='#ffbe00', width=3)
    pad = max(x1 - x0, y1 - y0) * .15
    crop = original.crop((max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(original.width, int(x1 + pad + 1)), min(original.height, int(y1 + pad + 1))))
    image = Image.new('RGB', (672, 336), 'black')
    image.paste(ImageOps.pad(marked, (336, 336), color='black'), (0, 0))
    image.paste(ImageOps.pad(crop, (336, 336), color='black'), (336, 0))
    return image


PROCEDURES = {
    'presence': 'Recognize the named target by its outline and neighboring anatomy. Search the CURRENT image and decide whether it is actually visible; an example does not establish current presence.',
    'count': 'Recognize the named structure, then count separate instances in the CURRENT image only. The boxed region is one source example; do not transfer the source instance count.',
    'location': 'Use the example to recognize the target, then locate it in the CURRENT image and answer the requested location there. Do not transfer the source position or side.',
    'shape': 'Locate the target in the CURRENT image using outline and neighbors, then describe its CURRENT shape. The source shape is not the current answer.',
    'size': 'Locate the named structure in the CURRENT image and make the requested CURRENT size comparison. Do not infer size from the source example.',
    'normality': 'Use the box only as a target localization example. Inspect the CURRENT target for the requested finding; this source is not a verified normality standard and cannot establish the current diagnosis.',
    'appearance': 'Use the boxed structure and close-up as a recognition example. Find its corresponding appearance in the CURRENT image; describe the current evidence, not the source answer.'
}


def encode(processor, query, condition, card=None):
    assert set(query) == {'qid', 'img_name', 'question'}
    assert condition in ('N', 'V', 'Blank', 'Text', 'Wrong')
    content, images = [], []
    if condition != 'N' and card is not None:
        assert set(card) == {'source_qid', 'role', 'entity', 'card_path', 'wrong_path'}
        content.append({'type': 'text', 'text': 'Visual recognition procedure for target: ' + card['entity'] + '.\n' + PROCEDURES[card['role']] + '\nThe reference is a DIFFERENT training image. It contains no source question or answer. Use current-image evidence for the final answer.\n'})
        if condition in ('V', 'Blank', 'Wrong'):
            image = Image.new('RGB', (672, 336), 'white') if condition == 'Blank' else Image.open(OUT / card['wrong_path' if condition == 'Wrong' else 'card_path']).convert('RGB')
            images.append(image)
            content.extend([{'type': 'text', 'text': 'Reference: boxed source context on the left, detail on the right.\n'}, {'type': 'image', 'image': image}])
    current = Image.open(DATA / 'extracted/imgs' / query['img_name']).convert('RGB')
    current.thumbnail((512, 512), Image.Resampling.LANCZOS)
    images.append(current)
    content.extend([{'type': 'text', 'text': '\nCurrent medical image:\n'}, {'type': 'image', 'image': current}, {'type': 'text', 'text': '\nQuestion: ' + query['question'] + '\nAnswer:'}])
    messages = [{'role': 'system', 'content': 'Answer the current medical image question. Give only the shortest correct answer, without explanation.'}, {'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[prompt], images=images, return_tensors='pt')
    return encoded['input_ids'][0].tolist(), images, prompt, encoded['image_grid_thw'].tolist()


def prepare():
    import numpy as np
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    OUT.mkdir(parents=True)
    for name in ('cards', 'records', 'inputs'):
        (OUT / name).mkdir()
    check_clock(p)
    train = [r for r in read(DATA / 'raw/train.json') if r['q_lang'] == 'en']
    validation = [r for r in read(DATA / 'raw/validation.json') if r['q_lang'] == 'en']
    assert len(train) == 4919 and len(validation) == 1053
    old = read(OLD / 'private_manifest.json')['records']
    prior_qids = {r['qid'] for r in old['validation']}
    features = np.load(OLD / 'features.npz')['image']
    assert features.shape == (546, 768) and np.isfinite(features).all()
    feature_by_image = {r['img_name']: features[i] for i, r in enumerate(old['train'] + old['validation'])}
    annotations = {}
    for r in train:
        if r['img_name'] not in annotations:
            annotations[r['img_name']] = [(label, xyxy(box)) for annotation in read(DATA / 'extracted/imgs' / Path(r['img_name']).parent / 'detection.json') for label, box in annotation.items()]
    lexicon = sorted({entity(label) for a in annotations.values() for label, _ in a}, key=lambda s: (-len(s), s))
    # Train-only local evidence; a disjoint annotated decoy makes Wrong a matched control.
    bank, seen = [], set()
    for r in sorted(train, key=lambda r: r['qid']):
        parsed = parse_question(r['question'], lexicon)
        if parsed is None:
            continue
        role, target = parsed
        regions = annotations[r['img_name']]
        matches = [box for label, box in regions if entity(label) == target]
        if not matches:
            continue
        main = max(matches, key=area)
        decoys = [box for label, box in regions if entity(label) != target and iou(main, box) <= .1]
        key = r['img_name'], role, target
        if not decoys or key in seen:
            continue
        seen.add(key)
        bank.append(dict(r, role=role, entity=target, target_xyxy=main, decoy_xyxy=max(decoys, key=area)))
    assert bank
    by_key = defaultdict(list)
    for i, row in enumerate(bank):
        by_key[row['role'], row['entity']].append(i)

    def ranked(query):
        assert set(query) == {'qid', 'img_name', 'question'}
        key = parse_question(query['question'], lexicon)
        indices = [i for i in by_key.get(key, []) if bank[i]['img_name'] != query['img_name']]
        if len({bank[i]['img_name'] for i in indices}) < 2:
            return []
        return sorted(((.7 * float(feature_by_image[query['img_name']] @ feature_by_image[bank[i]['img_name']]) + .3 * lexical(query['question'], bank[i]['question']), i) for i in indices), key=lambda v: (-v[0], bank[v[1]]['qid']))

    loo = [ranked(query_view(r)) for r in bank]
    top_scores = [r[0][0] for r in loo if r]
    assert len(top_scores) >= 30
    threshold = max(.75, float(np.quantile(top_scores, .1)))
    groups = defaultdict(list)
    for row in validation:
        if row['content_type'] in p['eligible_content_types'] and row['qid'] not in prior_qids:
            groups[row['img_name']].append(row)
    assert len(groups) == 96
    coverage, mechanism, mapping = [], [], {}
    rng = random.Random(p['selection_seed'])
    for image, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r['qid'])
        coverage.append(rng.choice(rows))
        eligible = []
        for row in rows:
            ranking = ranked(query_view(row))
            if ranking and ranking[0][0] >= threshold:
                mapping[row['qid']] = {'score': ranking[0][0], 'bank_index': ranking[0][1], 'source_qid': bank[ranking[0][1]]['qid']}
                eligible.append(row)
        rng.shuffle(eligible)
        mechanism.extend(eligible[:2])
    chosen = {r['qid']: r for r in coverage + mechanism}
    assert len(chosen) <= p['maximum_questions'] and not set(chosen) & prior_qids
    assert len(mechanism) >= 40 and len({r['img_name'] for r in mechanism}) >= 30, 'insufficient answer-free support; do not launch'
    # Render only selected sources, with no embedded words, answers, IDs or coordinates.
    cards = {}
    for qid in sorted(chosen):
        if qid not in mapping:
            continue
        item = bank[mapping[qid]['bank_index']]
        card_path, wrong_path = 'cards/' + str(item['qid']) + '.png', 'cards/' + str(item['qid']) + '_wrong.png'
        if not (OUT / card_path).exists():
            render(item).save(OUT / card_path)
            render(item, True).save(OUT / wrong_path)
        cards[str(qid)] = dict(source_qid=item['qid'], role=item['role'], entity=item['entity'], card_path=card_path, wrong_path=wrong_path)
    inputs = [query_view(chosen[q]) for q in sorted(chosen)]
    labels = {str(q): {k: r[k] for k in ('answer', 'answer_type', 'content_type')} for q, r in chosen.items()}
    write(OUT / 'private_inputs.json', inputs)
    write(OUT / 'private_labels.json', labels)
    write(OUT / 'private_bank.json', bank)
    write(OUT / 'private_cards.json', cards)
    write(OUT / 'private_retrieval.json', {str(q): mapping.get(q) for q in sorted(chosen)})
    panels = {'coverage': sorted(r['qid'] for r in coverage), 'mechanism': sorted(r['qid'] for r in mechanism), 'union': sorted(chosen)}
    write(OUT / 'panels.json', panels)
    schedule = []
    rotations = [p['conditions'][i:] + p['conditions'][:i] for i in range(5)]
    order = inputs.copy()
    rng.shuffle(order)
    for repeat in range(2):
        for i, query in enumerate(order):
            conditions = rotations[i % 5] if repeat == 0 else list(reversed(rotations[i % 5]))
            schedule.extend({'repeat': repeat, 'qid': query['qid'], 'condition': c} for c in conditions)
    write(OUT / 'schedule.json', schedule)
    write(OUT / 'retrieval_frozen.json', {'training_bank_entries': len(bank), 'training_images': len({r['img_name'] for r in bank}), 'train_loo_count': len(top_scores), 'threshold': threshold, 'coverage_questions': len(coverage), 'coverage_retrieved': sum(str(r['qid']) in cards for r in coverage), 'mechanism_questions': len(mechanism), 'mechanism_images': len({r['img_name'] for r in mechanism}), 'union_questions': len(chosen), 'union_images': len(groups), 'model_calls_planned': len(schedule), 'role_counts': dict(Counter(cards[str(r['qid'])]['role'] for r in mechanism)), 'test_opened': False, 'query_answers_used_for_retrieval': False})
    cpu_check()
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps(read(OUT / 'retrieval_frozen.json')), flush=True)


def cpu_check():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=p['maximum_image_pixels'])
    inputs = {str(q['qid']): q for q in read(OUT / 'private_inputs.json')}
    cards = read(OUT / 'private_cards.json')
    assert parse_question('How many kidneys are there?', ['kidney']) == ('count', 'kidney')
    assert parse_question('Where is the liver?', ['liver']) == ('location', 'liver')
    assert parse_question('Which is bigger, liver or heart?', ['liver', 'heart']) is None
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert score('\nassistant\nNo', 'No') and not score('2', '0')
    keys = list(cards)
    largest = max(keys, key=lambda key: Image.open(DATA / 'extracted/imgs' / inputs[key]['img_name']).width)
    for key in dict.fromkeys([keys[0], largest]):
        query, card = inputs[key], cards[key]
        enc = {c: encode(processor, query, c, card) for c in p['conditions']}
        assert enc['V'][0] == enc['Blank'][0] == enc['Wrong'][0]
        assert enc['V'][2] == enc['Blank'][2] == enc['Wrong'][2]
        assert enc['V'][3] == enc['Blank'][3] == enc['Wrong'][3]
        assert len(enc['V'][1]) == 2 and len(enc['Text'][1]) == len(enc['N'][1]) == 1
        assert ImageChops.difference(enc['V'][1][0], enc['Wrong'][1][0]).getbbox() is not None
        assert all(ImageChops.difference(enc['N'][1][-1], e[1][-1]).getbbox() is None for e in enc.values())
        assert all(e == (255, 255) for e in enc['Blank'][1][0].getextrema())
        for c in ('V', 'Blank', 'Wrong'):
            assert max(enc[c][1][-1].size) <= 512 and len(enc[c][0]) + 64 < 9000
        try:
            encode(processor, dict(query, answer='forbidden'), 'V', card)
        except AssertionError:
            pass
        else:
            raise AssertionError('query answer input boundary failed')
        try:
            encode(processor, query, 'V', dict(card, answer='forbidden'))
        except AssertionError:
            pass
        else:
            raise AssertionError('source answer input boundary failed')
    assert not torch.cuda.is_initialized()
    write(OUT / 'cpu_check.json', {'status': 'CPU_READY', 'gpu_initialized': False, 'query_and_source_answer_fields_rejected': True, 'V_Blank_Wrong_prompt_and_grid_equal': True, 'current_pixels_equal': True, 'different_target_and_decoy_pixels': True, 'test_opened': False, 'completed_unix': time.time()})


async def inference_loop(engine, processor, p, repeat):
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    inputs = {q['qid']: q for q in read(OUT / 'private_inputs.json')}
    labels = read(OUT / 'private_labels.json')
    cards = read(OUT / 'private_cards.json')
    schedule = read(OUT / 'schedule.json')
    plan = [e for e in schedule if e['repeat'] == repeat]
    offset = repeat * len(plan)
    assert len(schedule) <= p['maximum_generation_calls']
    matched = defaultdict(dict)
    for index, entry in enumerate(plan):
        check_clock(p)
        query, condition = inputs[entry['qid']], entry['condition']
        card = cards.get(str(query['qid']))
        ids, images, prompt, grids = encode(processor, query, condition, card)
        number = offset + index + 1
        assert not (OUT / 'records' / f'{number:04d}.json').exists(), 'do not replay completed requests'
        assert len(ids) + p['sampling']['max_new_tokens'] < 9000
        record = {'event': 'intent', 'number': number, 'repeat': repeat, 'qid': query['qid'], 'condition': condition, 'source_qid': card['source_qid'] if card else None, 'prompt_ids': ids, 'prompt_text': prompt, 'image_grids': grids, 'input_images': [], 'started_unix': time.time()}
        for i, image in enumerate(images):
            path = 'inputs/' + f'{number:04d}_{i}.png'
            image.save(OUT / path)
            record['input_images'].append(path)
        with (OUT / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        write(OUT / 'progress.json', {'repeat': repeat, 'dispatched': number, 'planned': len(schedule), 'updated_unix': time.time()})
        handle = SimpleNamespace(config=SimpleNamespace(max_model_len=9000, response_length=64), tokenizer_manager=engine.tokenizer_manager)
        try:
            response = await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle, ids, p['sampling'].copy(), f'p{number}', images)
        except BaseException as exc:
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(record, event='error', error=repr(exc))) + '\n')
            raise
        raw = processor.tokenizer.decode(response.token_ids, skip_special_tokens=True)
        label = labels[str(query['qid'])]
        result = {'event': 'result', 'number': number, 'repeat': repeat, 'qid': query['qid'], 'condition': condition, 'source_qid': record['source_qid'], 'raw': raw, 'output_ids': response.token_ids, 'correct': score(raw, label['answer']), 'answer_type': label['answer_type'], 'content_type': label['content_type'], 'completed_unix': time.time()}
        with (OUT / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        write(OUT / 'records' / f'{number:04d}.json', result)
        if card is not None and condition in ('V', 'Blank', 'Wrong'):
            matched[query['qid']][condition] = (ids, grids)
            if len(matched[query['qid']]) == 3:
                assert len({json.dumps(v) for v in matched[query['qid']].values()}) == 1
    assert all(len(v) == 3 for v in matched.values())
    write(OUT / f'REPEAT_{repeat}_COMPLETED.json', {'repeat': repeat, 'requests': len(plan), 'matched_input_triples': len(matched), 'completed_unix': time.time()})


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
