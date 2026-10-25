"""M0: answer-independent anatomical target mapping and matched local-view support."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image

ROOT = Path(os.environ['TASK_ROOT'])
DATA = Path(os.environ['SLAKE_ROOT'])
OUT = ROOT / 'artifacts/medical/roi_task_m0_v1'
PATHOLOGY = re.compile(r'cancer|edema|tumor|mass|nodule|effusion|pneumonia|atelectasis|pneumothorax|cardiomegaly|infiltrate', re.I)
SEED = 20261028


def read(p):
    return json.loads(p.read_text())


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False))


def canonical(label):
    return re.sub(r'^(left|right) ', '', label.lower())


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b):
    z = overlap(a, b)
    return z / (area(a) + area(b) - z)


def task(question):
    q = question.lower()
    if re.search(r'\b(where|which|many|largest|biggest|bigger|larger|smaller|smallest|contain|exist|observed|seen)\b', q):
        return None
    if re.match(r'(is|are|does|do)\b', q) and re.search(r'normal|abnormal|healthy|unhealthy', q):
        return 'normality'
    if 'shape' in q:
        return 'shape'
    if re.search(r'color|colour|density|appearance', q):
        return 'appearance'
    return None


def match_target(question, lexicon):
    q = question.lower()
    hits = [s for s in lexicon if re.search(r'\b' + re.escape(s) + r'(?:s|es)?\b', q)]
    hits = [s for s in hits if not any(s != t and s in t for t in hits)]
    return hits[0] if len(hits) == 1 else None


def geometry(boxes, size, question, target):
    width, height = size
    clipped = []
    for raw in boxes:
        x, y, w, h = map(float, raw)
        b = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
        if min(w, h) <= 0 or b[2] <= b[0] or b[3] <= b[1]:
            return None, 'invalid_box'
        clipped.append(b)
    # Near-overlapping annotations are one region; disjoint singular targets stay ambiguous.
    clusters = []
    for b in clipped:
        for cluster in clusters:
            if any(iou(b, a) >= .5 for a in cluster):
                cluster.append(b)
                break
        else:
            clusters.append([b])
    plural = bool(re.search(r'\b' + re.escape(target) + r'(?:s|es)\b|\bboth\b', question.lower()))
    if len(clusters) > 1 and not plural:
        return None, 'singular_target_multiple_disjoint_regions'
    target_box = [min(b[0] for b in clipped), min(b[1] for b in clipped), max(b[2] for b in clipped), max(b[3] for b in clipped)]
    fraction = area(target_box) / (width * height)
    if not .01 <= fraction <= .60:
        return None, 'nonlocal_or_tiny_target'
    margin = .15 * max(target_box[2] - target_box[0], target_box[3] - target_box[1])
    crop = [max(0, int(target_box[0] - margin)), max(0, int(target_box[1] - margin)), min(width, int(target_box[2] + margin + 1)), min(height, int(target_box[3] + margin + 1))]
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    choices = []
    for x in np.linspace(0, width - w, 9).round().astype(int):
        for y in np.linspace(0, height - h, 9).round().astype(int):
            wrong = [int(x), int(y), int(x + w), int(y + h)]
            target_overlap = overlap(wrong, target_box) / area(target_box)
            crop_overlap = iou(wrong, crop)
            if target_overlap <= .05 and crop_overlap <= .10:
                distance = (wrong[0] + wrong[2] - target_box[0] - target_box[2]) ** 2 + (wrong[1] + wrong[3] - target_box[1] - target_box[3]) ** 2
                choices.append((target_overlap, -distance, wrong))
    if not choices:
        return None, 'no_same_size_off_target_control'
    choices.sort(key=lambda x: (x[0], x[1], x[2]))
    return {'target_xyxy': target_box, 'oracle_crop_xyxy': crop, 'wrong_crop_xyxy': choices[0][2], 'target_area_fraction': fraction, 'target_instances': len(clusters), 'wrong_target_overlap_fraction': choices[0][0]}, None


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    rows = []
    for split in ('train', 'validation'):
        rows.extend(dict(r, split=split) for r in read(DATA / 'raw' / (split + '.json')) if r['q_lang'] == 'en' and r['base_type'] == 'vqa')
    annotations, sizes = {}, {}
    for r in rows:
        name = r['img_name']
        if name not in annotations:
            annotations[name] = [(k, b) for item in read(DATA / 'extracted/imgs' / Path(name).parent / 'detection.json') for k, b in item.items() if not PATHOLOGY.search(k)]
            sizes[name] = Image.open(DATA / 'extracted/imgs' / name).size
    lexicon = sorted({canonical(k) for a in annotations.values() for k, b in a}, key=lambda x: (-len(x), x))
    eligible, ledger = [], []
    for r in rows:
        reason, g = None, None
        kind = task(r['question'])
        if kind is None:
            reason = 'not_explicit_visual_attribute_task'
            target = None
        else:
            target = match_target(r['question'], lexicon)
            if target is None:
                reason = 'no_single_named_anatomical_target'
            else:
                matching = [(k, b) for k, b in annotations[r['img_name']] if canonical(k) == target]
                for side in ('left', 'right'):
                    if re.search(r'\b' + side + r'\s+' + re.escape(target), r['question'].lower()):
                        matching = [(k, b) for k, b in matching if k.lower().startswith(side + ' ')]
                if not matching:
                    reason = 'no_matching_anatomical_region'
                else:
                    g, reason = geometry([b for k, b in matching], sizes[r['img_name']], r['question'], target)
        ledger.append({'qid': r['qid'], 'image': r['img_name'], 'split': r['split'], 'target': target, 'task': kind, 'accepted': reason is None, 'reason': reason})
        if reason is None:
            eligible.append(dict(r, target=target, task=kind, **g))
    grouped = defaultdict(list)
    for r in eligible:
        grouped[r['img_name']].append(r)
    rng = random.Random(SEED)
    one_per_image = [rng.choice(sorted(rs, key=lambda x: x['qid'])) for name, rs in sorted(grouped.items())]
    target_groups = defaultdict(list)
    for r in one_per_image:
        target_groups[r['target']].append(r)
    for pool in target_groups.values():
        rng.shuffle(pool)
    panel = []
    while len(panel) < 64 and any(target_groups.values()):
        for target in sorted(target_groups):
            if target_groups[target] and len(panel) < 64:
                panel.append(target_groups[target].pop())
    gate = len(one_per_image) >= 64 and len({r['target'] for r in panel}) >= 3
    image_index = {n: i for i, n in enumerate(sorted(sizes))}
    write(OUT / 'private_eligible.json', eligible)
    write(OUT / 'private_panel.json', panel)
    write(OUT / 'private_ledger.json', ledger)
    # Reuse previous fixed embeddings only to flag similar content, not to select the panel.
    old = ROOT / 'artifacts/medical/slake_en_pilot_v1'
    oldrows = read(old / 'private_manifest.json')['records']
    names = [r['img_name'] for r in oldrows['train'] + oldrows['validation']]
    features = np.load(old / 'features.npz')['image']
    similarity = features @ features.T
    suspected = [(i, j, float(similarity[i, j])) for i in range(len(names)) for j in range(i + 1, len(names)) if similarity[i, j] >= .9995]
    public = OUT / 'public_bundle'
    public.mkdir()
    numeric_ledger = [{k: r[k] for k in ('qid', 'split', 'target', 'task', 'accepted', 'reason')} | {'image_group': image_index[r['image']]} for r in ledger]
    stats = {'status': 'M0_PASS' if gate else 'M0_STOP', 'model_calls': 0, 'new_image_encodings': 0, 'test_accessed': False, 'rows_screened': len(rows), 'eligible_questions': len(eligible), 'eligible_images': len(one_per_image), 'eligible_by_split': dict(Counter(r['split'] for r in eligible)), 'eligible_by_target': dict(Counter(r['target'] for r in eligible)), 'eligible_by_task': dict(Counter(r['task'] for r in eligible)), 'exclusions': dict(Counter(r['reason'] for r in ledger if not r['accepted'])), 'panel_images': len(panel), 'panel_targets': dict(Counter(r['target'] for r in panel)), 'panel_tasks': dict(Counter(r['task'] for r in panel)), 'panel_split': dict(Counter(r['split'] for r in panel)), 'remaining_train_images': len({r['img_name'] for r in eligible if r['split'] == 'train'} - {r['img_name'] for r in panel}), 'panel_majority_answer_fraction': max(Counter(r['answer'].strip().casefold() for r in panel).values()) / len(panel) if panel else None, 'suspected_near_content_pairs_existing_CLIP_cosine_ge_0_9995': len(suspected), 'patient_independence': 'unknown', 'pixel_duplicates_or_clinical_validity_certified': False, 'selection_uses_model_correctness': False, 'roi_selection_uses_answers': False, 'mapping_rule': 'Exact question-named anatomical target, side-aware label match, no disease-label ROI, all unambiguous target regions; no answer-based region choice.', 'oracle_interpretation': 'Privileged anatomical localization diagnosis only; neither autonomous selection nor proof of clinical evidence sufficiency.', 'completed_unix': time.time()}
    write(OUT / 'RESULT.json', stats)
    write(public / 'result.json', stats)
    write(public / 'numeric_eligibility_ledger.json', numeric_ledger)
    write(public / 'panel_numeric.json', [{'qid': r['qid'], 'image_group': image_index[r['img_name']], 'target': r['target'], 'task': r['task'], 'split': r['split'], 'target_area_fraction': r['target_area_fraction']} for r in panel])
    write(public / 'suspected_content_pairs.json', [{'image_group_a': image_index[names[i]], 'image_group_b': image_index[names[j]], 'cosine': s} for i, j, s in suspected])
    assert geometry([[0, 0, -1, 10]], (100, 100), 'What shape is the liver?', 'liver')[1] == 'invalid_box'
    assert task('Does the image contain liver?') is None
    assert task('Where is the abnormality in the liver?') is None
    assert task('Is the liver healthy?') == 'normality'
    assert task('What is the shape of the liver?') == 'shape'
    assert overlap((0, 0, 5, 5), (6, 6, 9, 9)) == 0
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
