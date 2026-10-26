"""One locked confirmation of immutable A/B experience with an answer-free predictor."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageOps
ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
WINDOW = Path(os.environ['TASK_WINDOW_STATE'])
NAME = 'ab_bank_independent_confirmation_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from medical_roi_task_audit import task, match_target
from slake_probe import normalize, query_view
read, write = common.read, common.write
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
ACQUISITION = ROOT / 'artifacts/medical/continual_acquisition_bank_v1'
VISUAL = ROOT / 'artifacts/medical/grounded_reader_anchor_v1'


def check():
    p = read(CONFIG)
    assert time.time() < p['cooperative_stop_unix']
    assert not (WINDOW / 'STOP_NEW_WORK.json').exists() and not (OUT / 'STOP_REQUESTED.json').exists()


def prepare():
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    # Discard gold at the JSON decoding boundary; selectors and predictors never receive it.
    records = json.loads((DATA / 'raw/test.json').read_text(), object_pairs_hook=lambda pairs: {k: v for k, v in pairs if k != 'answer'})
    assert all('answer' not in r for r in records)
    rows = sorted([r for r in records if r['q_lang'] == 'en'], key=lambda r: r['qid'])
    assert len(rows) == len({r['qid'] for r in rows})
    c = read(ACQUISITION / 'private_contracts.json')
    pairs, lexicon, templates = {tuple(x) for x in c['A_pairs']}, c['A_lexicon'], set(c['B_normalized_question_templates'])
    routing = {}
    for row in rows:
        kind, target = task(row['question']), match_target(row['question'], lexicon)
        a, b = (kind, target) in pairs, normalize(row['question']) in templates
        routing[str(row['qid'])] = {'branch': 'A' if a else ('I' if b else 'base'), 'A_contract': a, 'B_contract': b, 'lung_appearance': bool(a and kind == 'appearance' and target == 'lung'), 'Plane': row['base_type'] == 'vqa' and row['content_type'] == 'Plane', 'Modality': row['base_type'] == 'vqa' and row['content_type'] == 'Modality'}
    history = {r['img_name'] for split in ['train', 'validation'] for r in read(DATA / 'raw' / (split + '.json')) if r['q_lang'] == 'en'}
    # The experiment used English rows; unused Chinese annotations can share test image IDs.
    for name in ['region_reader_learning_v1', 'grounded_reader_anchor_v1', 'continual_acquisition_bank_v1', 'acquisition_to_organ_transfer_v1', 'visual_consensus_guard_v1']:
        manifest = read(ROOT / 'artifacts/medical' / name / 'private_manifest.json')
        for pool in manifest.values():
            if isinstance(pool, list): history.update(r['img_name'] for r in pool if isinstance(r, dict) and 'img_name' in r)
    candidate_images = sorted({r['img_name'] for r in rows})
    def descriptor(name):
        with Image.open(DATA / 'extracted/imgs' / name) as im:
            im = im.convert('RGB')
            size = im.size
            thumb = np.asarray(im.resize((64, 64), Image.Resampling.BILINEAR).convert('L'), dtype=np.float32) / 255
        centered = thumb.ravel() - thumb.mean()
        norm = np.linalg.norm(centered)
        return size, thumb.ravel(), centered / max(float(norm), 1e-8)
    old_names = sorted(history)
    old_descriptors = [descriptor(n) for n in old_names]
    old_gray = np.stack([v[1] for v in old_descriptors])
    old_norm = np.stack([v[2] for v in old_descriptors])
    collisions = []
    for name in candidate_images:
        size, gray, norm = descriptor(name)
        corr = old_norm @ norm
        mae = np.abs(old_gray - gray).mean(1)
        for index in np.flatnonzero((corr >= .9995) & (mae <= .02)):
            other = old_names[index]
            old_size = old_descriptors[index][0]
            if abs((size[0] / size[1]) / (old_size[0] / old_size[1]) - 1) > .01: continue
            exact = False
            if size == old_size and np.array_equal(gray, old_gray[index]):
                with Image.open(DATA / 'extracted/imgs' / name) as a, Image.open(DATA / 'extracted/imgs' / other) as b:
                    exact = np.array_equal(np.asarray(a.convert('RGB')), np.asarray(b.convert('RGB')))
            collisions.append({'test_image': name, 'historical_image': other, 'exact_pixels': bool(exact), 'correlation': float(corr[index]), 'thumbnail_MAE': float(mae[index])})
    quarantined = {r['test_image'] for r in collisions} | (set(candidate_images) & history)
    independent = {r['qid'] for r in rows if r['img_name'] not in quarantined}
    clean = [r for r in rows if r['qid'] in independent]
    assert len(clean) >= 500 and len({r['img_name'] for r in clean}) >= 50, 'insufficient independent input panel; do not relax'
    assert len({r['img_name'] for r in clean if routing[str(r['qid'])]['Plane']}) >= 20, 'insufficient independent Plane images'
    cohort = read(COHORT / 'private_manifest.json')
    old_rows = {r['qid']: r for k in ['new_validation', 'old_source', 'old_background', 'old_validation', 'old_acquisition', 'all_validation'] for r in cohort[k]}
    old_q = {i: q for i, q in enumerate(sorted(old_rows))}
    sig = {}
    for r in sorted(read(COHORT / 'public_bundle/physical_branch_numeric.json'), key=lambda r: (r['source_experiment'], r['source_request_number'])):
        if r['model'] not in ['base', 'A_42', 'A_43', 'I_42', 'I_43']: continue
        q = old_q[r['question_group']]
        sig.setdefault((r['model'], 'BLACK' if r['view'] == 'B' else old_rows[q]['img_name'], old_rows[q]['question']), r)
    physical, unique, cache, cache_index, bindings = [], {}, [], {}, {}
    for model in ['base', 'A_42', 'A_43', 'I_42', 'I_43']:
        required = rows if model == 'base' else [r for r in rows if routing[str(r['qid'])]['branch'] == model.split('_')[0]]
        for row in required:
            for view in ['N', 'B']:
                key = model, 'BLACK' if view == 'B' else row['img_name'], row['question']
                old = sig.get(key)
                if old is not None:
                    origin = old['source_experiment'], old['source_request_number']
                    if origin not in cache_index:
                        raw = read(ROOT / 'artifacts/medical' / origin[0] / 'records' / f'{origin[1]:04d}.json')
                        assert raw['model'] == model
                        cache_index[origin] = len(cache)
                        cache.append({'model': model, 'source_experiment': origin[0], 'source_request_number': origin[1], 'raw': raw['raw']})
                    ref = {'cache_index': cache_index[origin]}
                else:
                    if key not in unique:
                        unique[key] = len(physical)
                        physical.append({'model': model, 'qid': row['qid'], 'view': view, 'physical_index': len(physical)})
                    ref = {'physical_index': unique[key]}
                bindings[f'{model}|{row["qid"]}|{view}'] = ref
    assert len(physical) + 100 <= 12000
    parents = {}
    for label in ['A_42', 'A_43', 'I_42', 'I_43']:
        folder = (VISUAL if label.startswith('A') else ACQUISITION) / label / 'adapter'
        assert (folder / 'adapter_config.json').exists()
        files = [f for f in folder.iterdir() if f.is_file()]
        parents[label] = {'private_folder': str(folder), 'files': [{'name': f.name, 'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns} for f in files]}
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    write(OUT / 'private_input_manifest.json', {'records': rows, 'independent_qids': sorted(independent), 'quarantined_images': sorted(quarantined)})
    write(OUT / 'private_routing.json', routing)
    write(OUT / 'private_parents.json', parents)
    write(OUT / 'private_cached_records.json', cache)
    write(OUT / 'private_bindings.json', bindings)
    write(OUT / 'private_duplicate_flags.json', collisions)
    write(OUT / 'physical_plan.json', physical)
    cpu = {'status': 'CPU_READY', 'all_English_questions': len(rows), 'all_images': len(candidate_images), 'independent_questions': len(clean), 'independent_images': len({r['img_name'] for r in clean}), 'image_ID_overlap_train_validation': len(set(candidate_images) & history), 'suspected_near_duplicate_pairs': len(collisions), 'exact_pixel_duplicate_pairs': sum(r['exact_pixels'] for r in collisions), 'quarantined_images': len(quarantined), 'history_train_validation_images_screened': len(history), 'independent_by_scope': {k: {'questions': sum(routing[str(r['qid'])][k] for r in clean), 'images': len({r['img_name'] for r in clean if routing[str(r['qid'])][k]})} for k in ['Plane', 'Modality', 'lung_appearance']}, 'branch_questions': dict(Counter(r['branch'] for r in routing.values())), 'logical_predictor_reads': len(bindings), 'new_physical_generations': len(physical), 'cached_physical_records': len(cache), 'optimizer_steps': 0, 'gold_excluded_at_decoder_boundary': True, 'candidate_and_routing_frozen_before_predictions': True, 'gold_scoring_after_all_predictions_only': True, 'pixel_screen_is_not_patient_or_exhaustive_near_duplicate_proof': True}
    write(OUT / 'cpu_check.json', cpu)
    p.update(status='FROZEN_CPU_READY', formal_generation_calls=len(physical), maximum_generation_calls=len(physical) + 100)
    write(CONFIG, p)
    print(json.dumps(cpu), flush=True)


def run():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY' and not (OUT / 'requests.jsonl').exists()
    check()
    torch.set_num_threads(4)
    rows = {r['qid']: r for r in read(OUT / 'private_input_manifest.json')['records']}
    assert all('answer' not in r for r in rows.values())
    plan, parents = read(OUT / 'physical_plan.json'), read(OUT / 'private_parents.json')
    for v in parents.values():
        for f in v['files']:
            actual = Path(v['private_folder']) / f['name']
            assert actual.stat().st_size == f['bytes'] and actual.stat().st_mtime_ns == f['mtime_ns']
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(base, parents['A_42']['private_folder'], adapter_name='A_42', is_trainable=False, local_files_only=True)
    for label in ['A_43', 'I_42', 'I_43']: model.load_adapter(parents[label]['private_folder'], adapter_name=label, is_trainable=False, local_files_only=True)
    for v in model.parameters(): v.requires_grad_(False)
    calls = 0
    def evaluate(label):
        nonlocal calls
        model.eval()
        for entry in [r for r in plan if r['model'] == label]:
            check()
            calls += 1
            assert calls == entry['physical_index'] + 1 and calls <= p['maximum_generation_calls']
            row = rows[entry['qid']]
            assert 'answer' not in row
            inputs = processor(text=[common.prompt(processor, query_view(row))], images=[common.image_view(row, entry['view'])], return_tensors='pt').to('cuda')
            with (OUT / 'requests.jsonl').open('a') as f: f.write(json.dumps(dict(entry, event='intent', number=calls, started_unix=time.time())) + '\n')
            with torch.inference_mode(): generated = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
            record = dict(entry, number=calls, raw=processor.tokenizer.decode(tokens, skip_special_tokens=True), output_ids=tokens, completed_unix=time.time())
            write(OUT / 'records' / f'{calls:04d}.json', record)
            with (OUT / 'requests.jsonl').open('a') as f: f.write(json.dumps(dict(record, event='result')) + '\n')
            write(OUT / 'progress.json', {'model': label, 'new_generations': calls, 'planned_generations': len(plan), 'updated_unix': time.time()})
        write(OUT / (label + '_EVALUATED.json'), {'calls': sum(r['model'] == label for r in plan), 'completed_unix': time.time()})
    with model.disable_adapter():
        layers = [x for x in model.modules() if isinstance(x, LoraLayer)]
        assert len(layers) == 64 and all(x.disable_adapters for x in layers)
        write(OUT / 'BASE_READOUT_CHECK.json', {'layers': 64, 'all_disabled': True})
        evaluate('base')
    for label in ['A_42', 'A_43', 'I_42', 'I_43']:
        model.set_adapter(label)
        for v in model.parameters(): v.requires_grad_(False)
        layers = [x for x in model.modules() if isinstance(x, LoraLayer)]
        assert len(layers) == 64 and all(x.active_adapters == [label] and not x.disable_adapters for x in layers)
        write(OUT / (label + '_READOUT_CHECK.json'), {'layers': 64, 'correct_adapter_active': True, 'all_parameters_frozen': all(not v.requires_grad for v in model.parameters())})
        evaluate(label)
    assert calls == p['formal_generation_calls']
    write(OUT / 'PREDICTIONS_LOCKED.json', {'status': 'ALL_PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING', 'new_generations': calls, 'locked_unix': time.time(), 'new_optimizer_steps': 0, 'test_answer_used_by_predictor': False})
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_AB_BANK_CONFIRMATION', 'new_generation_calls': calls, 'optimizer_steps': 0, 'test_answer_used_by_predictor': False, 'gold_scored': False, 'completed_unix': time.time()})


if __name__ == '__main__': {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
