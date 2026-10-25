"""Final bounded paired-reference screen; confirmation gold is evaluated only at close."""
import asyncio
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
NAME = 'slake_balanced_final_v1'
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
OLD = ROOT / 'artifacts/medical/slake_en_pilot_v1'
sys.path.insert(0, str(SRC / 'scripts'))
from slake_evidence_probe import entity, xyxy, area, iou, render, normalize, score, query_view


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def check_clock():
    assert time.time() < read(CONFIG)['cooperative_stop_unix'] and not (OUT / 'STOP_REQUESTED.json').exists()


def presence(question, lexicon):
    q = question.lower()
    if re.search(r'normal|abnormal|healthy|where|which|shape|many|larger|bigger|smaller|size|color|\b(left|right)\b', q):
        return None
    if not re.match(r'(does|do)\b.*\bcontain\b|(?:is|are)\s+there\b|can\b.*\b(?:observed|seen)\b', q):
        return None
    hits = [s for s in lexicon if re.search(r'\b' + re.escape(s) + r'(?:s|es)?\b', q)]
    hits = [s for s in hits if not any(s != t and s in t for t in hits)]
    return hits[0] if len(hits) == 1 else None


def old_features():
    records = read(OLD / 'private_manifest.json')['records']
    data = np.load(OLD / 'features.npz')['image']
    assert data.shape == (546, 768) and np.isfinite(data).all()
    return {r['img_name']: data[i] for i, r in enumerate(records['train'] + records['validation'])}


def encode(processor, query, condition, card=None, positive_first=True):
    assert set(query) == {'qid', 'img_name', 'question'}
    assert condition in ('N', 'V', 'Blank', 'Text', 'Wrong', 'Swap')
    content, images = [], []
    if condition != 'N' and card is not None:
        assert set(card) == {'positive_qid', 'negative_qid', 'entity', 'positive_path', 'negative_path', 'wrong_positive_path', 'wrong_negative_path'}
        content.append({'type': 'text', 'text': 'Reference examples for ' + card['entity'] + ':\n'})
        for i, positive in enumerate([positive_first, not positive_first]):
            content.append({'type': 'text', 'text': 'Example ' + str(i + 1) + ': target ' + ('present' if positive else 'absent') + '.\n'})
            if condition != 'Text':
                if condition == 'Blank':
                    image = Image.new('RGB', (672, 336), 'white')
                else:
                    use_positive = not positive if condition == 'Swap' else positive
                    key = ('wrong_' if condition == 'Wrong' else '') + ('positive_path' if use_positive else 'negative_path')
                    image = Image.open(OUT / card[key]).convert('RGB')
                images.append(image)
                content.append({'type': 'image', 'image': image})
    current = Image.open(DATA / 'extracted/imgs' / query['img_name']).convert('RGB')
    current.thumbnail((512, 512), Image.Resampling.LANCZOS)
    images.append(current)
    content.extend([{'type': 'text', 'text': '\nCurrent image:\n'}, {'type': 'image', 'image': current}, {'type': 'text', 'text': '\nQuestion: ' + query['question'] + '\nAnswer:'}])
    messages = [{'role': 'system', 'content': 'Answer the current medical image question. Give only the shortest correct answer, without explanation.'}, {'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[prompt], images=images, return_tensors='pt')
    return encoded['input_ids'][0].tolist(), images, prompt, encoded['image_grid_thw'].tolist()


def prepare_sources():
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    OUT.mkdir(parents=True)
    for sub in ('cards', 'records', 'inputs'):
        (OUT / sub).mkdir()
    check_clock()
    rows = [r for r in read(DATA / 'raw/train.json') if r['q_lang'] == 'en']
    assert len(rows) == 4919
    annotations = {}
    for r in rows:
        if r['img_name'] not in annotations:
            annotations[r['img_name']] = [(label, xyxy(box)) for d in read(DATA / 'extracted/imgs' / Path(r['img_name']).parent / 'detection.json') for label, box in d.items()]
    lexicon = sorted({entity(label) for a in annotations.values() for label, _ in a}, key=lambda s: (-len(s), s))
    votes = defaultdict(set)
    for r in rows:
        target = presence(r['question'], lexicon)
        if target and normalize(r['answer']) in ('yes', 'no'):
            votes[r['img_name'], target].add(normalize(r['answer']))
    groups = defaultdict(lambda: defaultdict(list))
    seen, excluded = set(), Counter()
    for r in sorted(rows, key=lambda r: r['qid']):
        target = presence(r['question'], lexicon)
        gold = normalize(r['answer'])
        key = r['img_name'], target
        if target is None or gold not in ('yes', 'no') or key in seen:
            continue
        seen.add(key)
        if len(votes[key]) != 1:
            excluded['contradictory_presence_labels'] += 1
            continue
        regions = annotations[r['img_name']]
        matches = [box for label, box in regions if entity(label) == target]
        if gold == 'yes':
            if not matches:
                excluded['positive_without_target_box'] += 1
                continue
            main = max(matches, key=area)
            decoys = [box for label, box in regions if entity(label) != target and iou(main, box) <= .1]
            if not decoys:
                excluded['positive_without_decoy'] += 1
                continue
            item = dict(r, target=target, target_xyxy=main, decoy_xyxy=max(decoys, key=area))
        else:
            if matches:
                excluded['explicit_negative_conflicts_with_target_box'] += 1
                continue
            item = dict(r, target=target)
        groups[target, r['modality'], r['location']][gold].append(item)
    rng = random.Random(p['selection_seed'])
    candidates, selected_groups = [], []
    for group, classes in sorted(groups.items()):
        if min(len(classes.get('yes', [])), len(classes.get('no', []))) < 2:
            continue
        selected_groups.append(group)
        for gold in ('yes', 'no'):
            pool = classes[gold].copy()
            rng.shuffle(pool)
            candidates.extend(pool[:p['source_candidate_cap_per_class_group']])
    assert len(selected_groups) >= 3 and len(candidates) >= 24
    write(OUT / 'private_source_candidates.json', candidates)
    write(OUT / 'lexicon.json', lexicon)
    write(OUT / 'source_preparation.json', {'candidate_count': len(candidates), 'candidate_groups': len(selected_groups), 'candidate_class_counts': dict(Counter(normalize(r['answer']) for r in candidates)), 'exclusions': dict(excluded), 'test_inputs_opened': False, 'test_answers_accessed': False})
    cpu_check(source_only=True)
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps(read(OUT / 'source_preparation.json')), flush=True)


def freeze_sources():
    candidates = read(OUT / 'private_source_candidates.json')
    rows = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(rows) == len(candidates) and all(r['phase'] == 'source' for r in rows)
    assert [r['qid'] for r in rows] == [r['qid'] for r in candidates]
    groups = defaultdict(lambda: defaultdict(list))
    for item, row in zip(candidates, rows):
        if row['correct']:
            groups[item['target'], item['modality'], item['location']][normalize(item['answer'])].append(item)
    bank = []
    for group, classes in sorted(groups.items()):
        k = min(len(classes.get('yes', [])), len(classes.get('no', [])))
        if k >= 2:
            for gold in ('yes', 'no'):
                bank.extend(classes[gold][:k])
    write(OUT / 'private_source_bank.json', bank)
    features = old_features()
    grouped = defaultdict(lambda: defaultdict(list))
    for row in bank:
        grouped[row['target'], row['modality'], row['location']][normalize(row['answer'])].append(row)
    calibration = []
    for row in bank:
        match = retrieve(query_view(row), bank, features, threshold=-1)
        if match:
            calibration.append(match['score'])
    counts = Counter(normalize(r['answer']) for r in bank)
    class_images = {gold: len({r['img_name'] for r in bank if normalize(r['answer']) == gold}) for gold in ('yes', 'no')}
    gate = len(grouped) >= 3 and min(class_images.values()) >= 12 and len(calibration) >= 20
    obj = {'status': 'SOURCE_GATE_PASS' if gate else 'SOURCE_GATE_STOP', 'groups': len(grouped), 'bank_entries': len(bank), 'bank_images': len({r['img_name'] for r in bank}), 'class_counts': dict(counts), 'class_image_counts': class_images, 'calibration_queries': len(calibration), 'threshold': max(.85, float(np.quantile(calibration, .1))) if calibration else None, 'source_calls': len(candidates), 'test_inputs_opened': False, 'test_answers_accessed': False, 'independent_clinical_verification': False}
    write(OUT / 'SOURCE_FROZEN.json', obj)
    print(json.dumps(obj), flush=True)


def retrieve(query, bank, features, threshold):
    assert set(query) == {'qid', 'img_name', 'question'}
    target = presence(query['question'], read(OUT / 'lexicon.json'))
    if target is None:
        return None
    groups = defaultdict(lambda: defaultdict(list))
    for row in bank:
        if row['target'] == target and row['img_name'] != query['img_name']:
            groups[row['modality'], row['location']][normalize(row['answer'])].append(row)
    choices = []
    for group, classes in groups.items():
        if not classes.get('yes') or not classes.get('no'):
            continue
        selected, scores = {}, []
        for gold in ('yes', 'no'):
            ranked = sorted(((float(features[query['img_name']] @ features[r['img_name']]), r) for r in classes[gold]), key=lambda t: (-t[0], t[1]['qid']))
            scores.append(ranked[0][0])
            selected[gold] = ranked[0][1]['qid']
        choices.append({'score': min(scores), 'positive_qid': selected['yes'], 'negative_qid': selected['no'], 'entity': target})
    choices.sort(key=lambda c: (-c['score'], c['positive_qid'], c['negative_qid']))
    return choices[0] if choices and choices[0]['score'] >= threshold else None


def features():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    check_clock()
    assert read(OUT / 'SOURCE_FROZEN.json')['status'] == 'SOURCE_GATE_PASS'
    # Access only input fields; the answer field is first used in the separate evaluator.
    raw = read(DATA / 'raw/test.json')
    rows = [{'qid': r['qid'], 'img_name': r['img_name'], 'question': r['question'], 'content_type': r['content_type']} for r in raw if r['q_lang'] == 'en' and r['content_type'] in ('Organ', 'Position', 'Abnormality', 'Size', 'Shape', 'Quantity')]
    by_image = {}
    for row in rows:
        by_image[row['img_name']] = row
    assert len(by_image) == 96 and not set(by_image) & set(old_features())
    write(OUT / 'private_confirmation_pool.json', rows)
    images = sorted(by_image)
    processor = CLIPProcessor.from_pretrained(os.environ['CLIP_MODEL_ROOT'], local_files_only=True)
    model = CLIPModel.from_pretrained(os.environ['CLIP_MODEL_ROOT'], local_files_only=True, torch_dtype=torch.float16).to('cuda').eval().requires_grad_(False)
    embeddings = []
    with torch.inference_mode():
        for offset in range(0, len(images), 16):
            check_clock()
            batch = [Image.open(DATA / 'extracted/imgs' / name).convert('RGB') for name in images[offset:offset + 16]]
            pixels = processor(images=batch, return_tensors='pt')['pixel_values'].to('cuda', dtype=torch.float16)
            result = torch.nn.functional.normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1)
            assert torch.isfinite(result).all()
            embeddings.append(result.cpu().numpy())
    np.savez(OUT / 'confirmation_features.npz', image=np.concatenate(embeddings))
    write(OUT / 'private_feature_images.json', images)
    write(OUT / 'FEATURES_COMPLETED.json', {'new_images': len(images), 'shape': [96, 768], 'test_answers_accessed': False, 'query_roi_opened': False, 'optimizer_steps': 0, 'completed_unix': time.time()})


def mapped_box(box, source_size, destination_size):
    return [v / source_size[i % 2] * destination_size[i % 2] for i, v in enumerate(box)]


def freeze_confirmation():
    p = read(CONFIG)
    check_clock()
    bank = read(OUT / 'private_source_bank.json')
    indexed_bank = {r['qid']: r for r in bank}
    source = read(OUT / 'SOURCE_FROZEN.json')
    feature = old_features()
    names = read(OUT / 'private_feature_images.json')
    array = np.load(OUT / 'confirmation_features.npz')['image']
    feature.update(dict(zip(names, array)))
    rows = read(OUT / 'private_confirmation_pool.json')
    lexicon = read(OUT / 'lexicon.json')
    groups = defaultdict(list)
    for row in rows:
        groups[row['img_name']].append(row)
    rng = random.Random(p['selection_seed'] + 1)
    coverage, mechanism, mapping = [], [], {}
    for image, pool in sorted(groups.items()):
        pool = sorted(pool, key=lambda r: r['qid'])
        presence_rows = [r for r in pool if presence(r['question'], lexicon)]
        coverage.append(rng.choice(presence_rows or pool))
        supported = []
        for row in presence_rows:
            match = retrieve(query_view(row), bank, feature, source['threshold'])
            if match:
                mapping[str(row['qid'])] = match
                supported.append(row)
        rng.shuffle(supported)
        mechanism.extend(supported[:2])
    chosen = {r['qid']: r for r in coverage + mechanism}
    panels = {'coverage': sorted(r['qid'] for r in coverage), 'mechanism': sorted(r['qid'] for r in mechanism), 'union': sorted(chosen)}
    assert len(chosen) <= p['maximum_questions']
    mechanism_images = len({r['img_name'] for r in mechanism})
    covered = sum(str(r['qid']) in mapping for r in coverage)
    targets = {mapping[str(r['qid'])]['entity'] for r in mechanism}
    gate = len(mechanism) >= 48 and mechanism_images >= 32 and len(targets) >= 3 and covered / len(coverage) >= .30
    write(OUT / 'private_query_inputs.json', [query_view(chosen[q]) for q in sorted(chosen)])
    write(OUT / 'private_retrieval.json', {str(q): mapping.get(str(q)) for q in sorted(chosen)})
    write(OUT / 'panels.json', panels)
    frozen = {'status': 'CONFIRMATION_GATE_PASS' if gate else 'CONFIRMATION_GATE_STOP', 'coverage_questions': len(coverage), 'coverage_retrieved': covered, 'coverage_fraction': covered / len(coverage), 'mechanism_questions': len(mechanism), 'mechanism_images': mechanism_images, 'mechanism_targets': len(targets), 'union_questions': len(chosen), 'union_images': len(groups), 'source_calls': source['source_calls'], 'confirmation_calls_planned': len(chosen) * 12 if gate else 0, 'test_answers_accessed': False, 'query_roi_opened': False, 'completed_unix': time.time()}
    write(OUT / 'CONFIRMATION_FROZEN.json', frozen)
    if not gate:
        print(json.dumps(frozen), flush=True)
        return
    cards = {}
    for qid in sorted(chosen):
        match = mapping.get(str(qid))
        if match is None:
            continue
        positive = indexed_bank[match['positive_qid']]
        negative = indexed_bank[match['negative_qid']]
        a = Image.open(DATA / 'extracted/imgs' / positive['img_name']).size
        b = Image.open(DATA / 'extracted/imgs' / negative['img_name']).size
        negative_render = dict(negative, target_xyxy=mapped_box(positive['target_xyxy'], a, b), decoy_xyxy=mapped_box(positive['decoy_xyxy'], a, b))
        prefix = 'cards/' + str(positive['qid']) + '_' + str(negative['qid'])
        paths = {'positive_path': prefix + '_p.png', 'negative_path': prefix + '_n.png', 'wrong_positive_path': prefix + '_wp.png', 'wrong_negative_path': prefix + '_wn.png'}
        for key, row, wrong in [('positive_path', positive, False), ('negative_path', negative_render, False), ('wrong_positive_path', positive, True), ('wrong_negative_path', negative_render, True)]:
            if not (OUT / paths[key]).exists():
                render(row, wrong).save(OUT / paths[key])
        cards[str(qid)] = dict(positive_qid=positive['qid'], negative_qid=negative['qid'], entity=match['entity'], **paths)
    write(OUT / 'private_cards.json', cards)
    order = panels['union'].copy()
    rng.shuffle(order)
    schedule = []
    for repeat in (0, 1):
        for i, qid in enumerate(order):
            offset = i % 6
            conditions = p['conditions'][offset:] + p['conditions'][:offset]
            if repeat:
                conditions.reverse()
            positive_first = (i % 2 == 0) != bool(repeat)
            schedule.extend({'repeat': repeat, 'qid': qid, 'condition': c, 'positive_first': positive_first} for c in conditions)
    assert len(schedule) + source['source_calls'] <= p['maximum_generation_calls']
    write(OUT / 'schedule.json', schedule)
    cpu_check(source_only=False)
    print(json.dumps(frozen), flush=True)


def cpu_check(source_only):
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert presence('Does the picture contain kidneys?', ['kidney']) == 'kidney'
    assert presence('Does the left kidney look normal?', ['kidney']) is None
    assert presence('How many kidneys are there?', ['kidney']) is None
    assert score('\nassistant\nNo', 'No') and not score('Yes', 'No')
    assert mapped_box((10, 20, 30, 40), (100, 100), (200, 300)) == [20, 60, 60, 120]
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=p['maximum_image_pixels'])
    if source_only:
        q = query_view(read(OUT / 'private_source_candidates.json')[0])
        encoded = encode(processor, q, 'N')
        assert len(encoded[1]) == 1 and len(encoded[0]) + 64 < 9000
    else:
        inputs = {str(q['qid']): q for q in read(OUT / 'private_query_inputs.json')}
        cards = read(OUT / 'private_cards.json')
        keys = list(cards)
        key = max(keys, key=lambda k: Image.open(DATA / 'extracted/imgs' / inputs[k]['img_name']).width)
        for key in dict.fromkeys([keys[0], key]):
            for order in (True, False):
                q, card = inputs[key], cards[key]
                encoded = {c: encode(processor, q, c, card, order) for c in p['conditions']}
                controls = [encoded[c] for c in ('V', 'Blank', 'Wrong', 'Swap')]
                assert all(e[0] == controls[0][0] and e[3] == controls[0][3] for e in controls)
                assert all(len(e[1]) == 3 for e in controls)
                assert len(encoded['N'][1]) == len(encoded['Text'][1]) == 1
                assert all(ImageChops.difference(encoded['N'][1][-1], e[1][-1]).getbbox() is None for e in encoded.values())
                assert ImageChops.difference(encoded['V'][1][0], encoded['Swap'][1][1]).getbbox() is None
                assert ImageChops.difference(encoded['V'][1][1], encoded['Swap'][1][0]).getbbox() is None
                assert ImageChops.difference(encoded['V'][1][0], encoded['Wrong'][1][0]).getbbox() is not None
                assert all(len(e[0]) + 64 < 9000 for e in encoded.values())
                try:
                    encode(processor, dict(q, answer='forbidden'), 'V', card, order)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('query gold boundary')
                try:
                    encode(processor, q, 'V', dict(card, answer='forbidden'), order)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('source QA boundary')
    assert not torch.cuda.is_initialized()
    write(OUT / ('source_cpu_check.json' if source_only else 'confirmation_cpu_check.json'), {'status': 'CPU_READY', 'source_only': source_only, 'gpu_initialized': False, 'test_answers_accessed': False, 'query_roi_opened': False, 'completed_unix': time.time()})


async def inference_loop(engine, processor, p, phase, repeat):
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    if phase == 'source':
        candidates = read(OUT / 'private_source_candidates.json')
        inputs = {r['qid']: query_view(r) for r in candidates}
        labels = {r['qid']: r['answer'] for r in candidates}
        plan = [{'qid': r['qid'], 'condition': 'N', 'positive_first': True} for r in candidates]
        cards, offset = {}, 0
    else:
        inputs = {r['qid']: r for r in read(OUT / 'private_query_inputs.json')}
        cards = read(OUT / 'private_cards.json')
        plan = [r for r in read(OUT / 'schedule.json') if r['repeat'] == repeat]
        offset = read(OUT / 'SOURCE_FROZEN.json')['source_calls'] + repeat * len(plan)
    matched = defaultdict(dict)
    for index, entry in enumerate(plan):
        check_clock()
        q, condition = inputs[entry['qid']], entry['condition']
        card = cards.get(str(q['qid']))
        ids, images, prompt, grids = encode(processor, q, condition, card, entry['positive_first'])
        number = offset + index + 1
        assert number <= p['maximum_generation_calls'] and len(ids) + 64 < 9000
        path = OUT / 'records' / f'{number:04d}.json'
        assert not path.exists(), 'never replay a completed response'
        record = {'event': 'intent', 'number': number, 'phase': phase, 'repeat': repeat, 'qid': q['qid'], 'condition': condition, 'positive_first': entry['positive_first'], 'positive_qid': card['positive_qid'] if card else None, 'negative_qid': card['negative_qid'] if card else None, 'prompt_ids': ids, 'prompt_text': prompt, 'image_grids': grids, 'input_images': [], 'started_unix': time.time()}
        for i, im in enumerate(images):
            name = f'inputs/{number:04d}_{i}.png'
            im.save(OUT / name)
            record['input_images'].append(name)
        with (OUT / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        write(OUT / 'progress.json', {'phase': phase, 'repeat': repeat, 'dispatched': number, 'maximum': p['maximum_generation_calls'], 'updated_unix': time.time()})
        handle = SimpleNamespace(config=SimpleNamespace(max_model_len=9000, response_length=64), tokenizer_manager=engine.tokenizer_manager)
        try:
            output = await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle, ids, p['sampling'].copy(), f'p{number}', images)
        except BaseException as exc:
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(record, event='error', error=repr(exc))) + '\n')
            raise
        raw = processor.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        result = {k: record[k] for k in ('number', 'phase', 'repeat', 'qid', 'condition', 'positive_first', 'positive_qid', 'negative_qid')}
        result.update(event='result', raw=raw, output_ids=output.token_ids, completed_unix=time.time())
        if phase == 'source':
            result['correct'] = score(raw, labels[q['qid']])
        else:
            assert 'answer' not in q and 'correct' not in result
        with (OUT / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        write(path, result)
        if card is not None and condition in ('V', 'Blank', 'Wrong', 'Swap'):
            matched[q['qid']][condition] = ids, grids
            if len(matched[q['qid']]) == 4:
                assert len({json.dumps(v) for v in matched[q['qid']].values()}) == 1
    assert all(len(v) == 4 for v in matched.values())
    write(OUT / ('SOURCE_COMPLETED.json' if phase == 'source' else f'REPEAT_{repeat}_COMPLETED.json'), {'phase': phase, 'repeat': repeat, 'requests': len(plan), 'matched_input_quadruples': len(matched), 'test_answers_accessed': False, 'completed_unix': time.time()})


def infer():
    import sglang
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY'
    phase = os.environ['TASK_PHASE']
    repeat = int(os.environ.get('TASK_REPEAT', '0'))
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
        loop.run_until_complete(inference_loop(engine, processor, p, phase, repeat))
    finally:
        if engine is not None:
            engine.shutdown()
        loop.close()


if __name__ == '__main__':
    {'prepare_sources': prepare_sources, 'freeze_sources': freeze_sources, 'features': features, 'freeze_confirmation': freeze_confirmation, 'infer': infer}[os.environ['TASK_STAGE']]()
