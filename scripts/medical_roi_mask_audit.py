"""Separate M0 mask-grounded task audit; retain the stopped box-based audit unchanged."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
OUT = ROOT / 'artifacts/medical/roi_mask_m0_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_roi_task_audit as base


def crop_geometry(box, size):
    width, height = size
    if min(box[2] - box[0], box[3] - box[1]) < 6:
        return None, 'under_six_native_pixels'
    if base.area(box) / (width * height) > .6:
        return None, 'nonlocal_target'
    margin = .15 * max(box[2] - box[0], box[3] - box[1])
    crop = [max(0, int(box[0] - margin)), max(0, int(box[1] - margin)), min(width, int(box[2] + margin + 1)), min(height, int(box[3] + margin + 1))]
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    choices = []
    for x in np.linspace(0, width - w, 9).round().astype(int):
        for y in np.linspace(0, height - h, 9).round().astype(int):
            wrong = [int(x), int(y), int(x + w), int(y + h)]
            fraction = base.overlap(wrong, box) / base.area(box)
            if fraction <= .05 and base.iou(wrong, crop) <= .1:
                distance = (wrong[0] + wrong[2] - box[0] - box[2]) ** 2 + (wrong[1] + wrong[3] - box[1] - box[3]) ** 2
                choices.append((fraction, -distance, wrong))
    if not choices:
        return None, 'no_same_size_off_target_control'
    choices.sort(key=lambda x: (x[0], x[1], x[2]))
    return {'target_xyxy': box, 'oracle_crop_xyxy': crop, 'wrong_crop_xyxy': choices[0][2], 'target_area_fraction': base.area(box) / (width * height), 'wrong_target_overlap_fraction': choices[0][0]}, None


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    mapping = {int(a): b for a, b in (line.split(':', 1) for line in (DATA / 'raw/mask.txt').read_text().splitlines())}
    anatomical = {i: name for i, name in mapping.items() if not base.PATHOLOGY.search(name)}
    lexicon = sorted({base.canonical(s) for s in anatomical.values()}, key=lambda x: (-len(x), x))
    rows = []
    for split in ('train', 'validation'):
        rows.extend(dict(r, split=split) for r in base.read(DATA / 'raw' / (split + '.json')) if r['q_lang'] == 'en' and r['base_type'] == 'vqa')
    masks, sizes, eligible, ledger = {}, {}, [], []
    for row in rows:
        kind, reason = base.task(row['question']), None
        target = base.match_target(row['question'], lexicon) if kind else None
        g = None
        if kind is None:
            reason = 'not_explicit_visual_attribute_task'
        elif target is None:
            reason = 'no_single_named_anatomical_target'
        else:
            name = row['img_name']
            if name not in masks:
                image = Image.open(DATA / 'extracted/imgs' / name)
                array = np.asarray(Image.open(DATA / 'extracted/imgs' / name.replace('source.jpg', 'mask.png')).convert('RGB'))
                valid = np.array_equal(array[:, :, 0], array[:, :, 1]) and np.array_equal(array[:, :, 0], array[:, :, 2]) and set(np.unique(array[:, :, 0])) <= ({0} | set(mapping)) and (array.shape[1], array.shape[0]) == image.size
                masks[name] = array[:, :, 0] if valid else None
                sizes[name] = image.size
            mask = masks[name]
            ids = [i for i, label in anatomical.items() if base.canonical(label) == target]
            for side in ('left', 'right'):
                if re.search(r'\b' + side + r'\s+' + re.escape(target), row['question'].lower()):
                    ids = [i for i in ids if anatomical[i].lower().startswith(side + ' ')]
            if mask is None:
                reason = 'non_discrete_overlay_or_unaligned_mask'
            else:
                active = [i for i in ids if np.any(mask == i)]
                plural = bool(re.search(r'\b' + re.escape(target) + r'(?:s|es)\b|\bboth\b', row['question'].lower()))
                if not active:
                    reason = 'no_matching_anatomical_mask'
                elif len(active) > 1 and not plural:
                    reason = 'singular_target_ambiguous_bilateral_masks'
                else:
                    yy, xx = np.where(np.isin(mask, active))
                    box = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
                    g, reason = crop_geometry(box, sizes[name])
        ledger.append({'qid': row['qid'], 'image': row['img_name'], 'split': row['split'], 'target': target, 'task': kind, 'accepted': reason is None, 'reason': reason})
        if reason is None:
            eligible.append(dict(row, target=target, task=kind, **g))
    grouped = defaultdict(list)
    for row in eligible:
        grouped[row['img_name']].append(row)
    rng = random.Random(20261029)
    one_per_image = [rng.choice(sorted(v, key=lambda x: x['qid'])) for k, v in sorted(grouped.items())]
    by_target = defaultdict(list)
    for row in one_per_image:
        by_target[row['target']].append(row)
    for pool in by_target.values():
        rng.shuffle(pool)
    panel = []
    while len(panel) < 64 and any(by_target.values()):
        for target in sorted(by_target):
            if by_target[target] and len(panel) < 64:
                panel.append(by_target[target].pop())
    passed = len(panel) == 64 and len({r['target'] for r in panel}) >= 3
    base.write(OUT / 'private_eligible.json', eligible)
    base.write(OUT / 'private_panel.json', panel)
    base.write(OUT / 'private_ledger.json', ledger)
    image_groups = {name: i for i, name in enumerate(sorted({r['img_name'] for r in rows}))}
    result = {'status': 'M0_MASK_PASS' if passed else 'M0_MASK_STOP', 'model_calls': 0, 'new_image_encodings': 0, 'test_accessed': False, 'prior_box_audit_preserved': 'roi_task_m0_v1', 'anatomical_mask_mapping_from_author_file': True, 'mask_quality_rule': 'Exact grayscale code from author mask.txt, aligned dimensions; non-discrete overlays rejected.', 'roi_rule': 'Question-named anatomical code only, side-aware, no disease code, no gold-based target selection; >=6 native pixels per side and <=60% image area. No 1%-of-image floor because valid small structures are the stated task.', 'eligible_questions': len(eligible), 'eligible_images': len(one_per_image), 'eligible_by_target': dict(Counter(r['target'] for r in eligible)), 'eligible_by_task': dict(Counter(r['task'] for r in eligible)), 'eligible_by_split': dict(Counter(r['split'] for r in eligible)), 'exclusions': dict(Counter(r['reason'] for r in ledger if not r['accepted'])), 'panel_images': len(panel), 'panel_targets': dict(Counter(r['target'] for r in panel)), 'panel_tasks': dict(Counter(r['task'] for r in panel)), 'panel_split': dict(Counter(r['split'] for r in panel)), 'panel_under_one_percent_target_area': sum(r['target_area_fraction'] < .01 for r in panel), 'remaining_train_images': len({r['img_name'] for r in eligible if r['split'] == 'train'} - {r['img_name'] for r in panel}), 'panel_majority_answer_fraction': max(Counter(r['answer'].strip().casefold() for r in panel).values()) / len(panel) if panel else None, 'patient_independence': 'unknown', 'independent_clinical_truth_certified': False, 'roi_selection_uses_gold_answers': False, 'model_correctness_used_for_selection': False, 'completed_unix': time.time()}
    base.write(OUT / 'RESULT.json', result)
    public = OUT / 'public_bundle'
    public.mkdir()
    base.write(public / 'result.json', result)
    base.write(public / 'numeric_eligibility_ledger.json', [{k: r[k] for k in ('qid', 'split', 'target', 'task', 'accepted', 'reason')} | {'image_group': image_groups[r['image']]} for r in ledger])
    base.write(public / 'panel_numeric.json', [{'qid': r['qid'], 'image_group': image_groups[r['img_name']], 'target': r['target'], 'task': r['task'], 'split': r['split'], 'target_area_fraction': r['target_area_fraction']} for r in panel])
    assert crop_geometry([0, 0, 4, 4], (100, 100))[1] == 'under_six_native_pixels'
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
