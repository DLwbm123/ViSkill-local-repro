"""Frozen counterfactual-consensus activation on update-isolated source images."""
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'visual_consensus_guard_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from medical_roi_task_audit import task, match_target
from slake_probe import normalize, query_view
read, write = common.read, common.write


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def eligible_candidate(base, n42, n43, b42, b43, w42, w43):
    a = normalize(n42)
    agreement = bool(a and a == normalize(n43) and a != normalize(base))
    guard = agreement and all(a != normalize(x) for x in (b42, b43, w42, w43))
    return agreement, guard


def prepare():
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    excluded = set()
    for experiment in ('region_reader_learning_v1', 'grounded_reader_anchor_v1', 'continual_acquisition_bank_v1', 'acquisition_to_organ_transfer_v1'):
        manifest = read(ROOT / 'artifacts/medical' / experiment / 'private_manifest.json')
        for kind in ('train', 'old_train_replay'):
            excluded.update(r['img_name'] for r in manifest.get(kind, []))
    cohort = read(COHORT / 'private_manifest.json')
    for kind in ('old_source', 'old_background', 'all_validation'):
        excluded.update(r['img_name'] for r in cohort[kind])
    support = read(ROOT / 'artifacts/medical/continual_acquisition_bank_v1/private_contracts.json')
    pairs = {tuple(x) for x in support['A_pairs']}
    btemplates = set(support['B_normalized_question_templates'])
    ctemplates = set(read(COHORT / 'private_contracts.json')['C_question_templates'])
    def in_scope(row):
        return normalize(row['question']) in ctemplates and normalize(row['question']) not in btemplates and (task(row['question']), match_target(row['question'], support['A_lexicon'])) not in pairs
    candidates = [r for r in read(common.DATA / 'raw/train.json') if r['q_lang'] == 'en' and r['base_type'] == 'vqa' and r['content_type'] == 'Organ' and r['img_name'] not in excluded and in_scope(r)]
    by_image = {}
    for row in candidates: by_image.setdefault(row['img_name'], []).append(row)
    rng = random.Random(p['selection_seed'])
    primary = [rng.choice(sorted(by_image[n], key=lambda r: r['qid'])) for n in sorted(by_image)]
    assert len(primary) == len({r['img_name'] for r in primary}) == 85 >= 50
    background = cohort['old_background']
    flags = read(COHORT / 'private_contracts.json')['flags']
    items = {r['qid']: r for r in primary + background}
    admit = {str(q): (q in {r['qid'] for r in primary} or flags[str(q)]['C']) for q in items}
    active = [r for r in items.values() if admit[str(r['qid'])]]
    assert len(active) == 106
    images = [r['img_name'] for r in active]
    assert len(set(images)) == 106 and not {r['img_name'] for r in primary} & excluded
    shuffled = sorted(images)
    rng.shuffle(shuffled)
    wrong = {n: shuffled[(i + 1) % len(shuffled)] for i, n in enumerate(shuffled)}
    assert all(a != b for a, b in wrong.items())
    priorities = {}
    for panel, pool in [('primary', primary), ('background', [r for r in background if admit[str(r['qid'])]])]:
        priorities[panel] = sorted(r['qid'] for r in pool)
        rng.shuffle(priorities[panel])
    old_items = {r['qid']: r for k in ('new_validation', 'old_source', 'old_background', 'old_validation', 'old_acquisition', 'all_validation') for r in cohort[k]}
    qold = {i: q for i, q in enumerate(sorted(old_items))}
    catalog = read(COHORT / 'public_bundle/physical_branch_numeric.json')
    exact, signatures = {}, {}
    for row in sorted(catalog, key=lambda r: (r['source_experiment'], r['source_request_number'])):
        if row['model'] not in ('base', 'T_42', 'T_43'): continue
        q = qold[row['question_group']]
        exact[row['model'], q, row['view']] = row
        sig = (row['model'], 'BLACK' if row['view'] == 'B' else old_items[q]['img_name'], old_items[q]['question'])
        signatures.setdefault(sig, row)
    cache, cache_indices, physical, new_signatures, bindings = [], {}, [], {}, {}
    logical = [(model, r['qid'], v) for r in active for model in ('base', 'T_42', 'T_43') for v in (('N', 'B') if model == 'base' else ('N', 'B', 'W'))]
    for model, q, view in logical:
        row = items[q]
        image = 'BLACK' if view == 'B' else (wrong[row['img_name']] if view == 'W' else row['img_name'])
        sig = (model, image, row['question'])
        old = exact.get((model, q, view)) if view != 'W' else None
        old = old or signatures.get(sig)
        if old is not None:
            origin, number = old['source_experiment'], old['source_request_number']
            assert re.fullmatch('[a-z0-9_]+', origin)
            key = (origin, number)
            if key not in cache_indices:
                raw = read(ROOT / 'artifacts/medical' / origin / 'records' / f'{number:04d}.json')
                assert raw['number'] == number and raw['model'] == model
                cache_indices[key] = len(cache)
                cache.append({'model': model, 'source_experiment': origin, 'source_request_number': number, 'raw': raw['raw']})
            ref = {'cache_index': cache_indices[key]}
        else:
            if sig not in new_signatures:
                new_signatures[sig] = len(physical)
                physical.append({'model': model, 'qid': q, 'question': row['question'], 'image_identity': image, 'physical_index': len(physical)})
            ref = {'physical_index': new_signatures[sig]}
        bindings[f'{model}|{q}|{view}'] = ref
    assert len(logical) == 848 and len(physical) <= 722
    physical.sort(key=lambda e: ('base', 'T_42', 'T_43').index(e['model']))
    remap = {e['physical_index']: i for i, e in enumerate(physical)}
    for e in physical: e['physical_index'] = remap[e['physical_index']]
    for ref in bindings.values():
        if 'physical_index' in ref: ref['physical_index'] = remap[ref['physical_index']]
    assert eligible_candidate('kidney', 'heart', 'heart', 'lung', 'lung', 'brain', 'brain') == (True, True)
    assert eligible_candidate('kidney', 'heart', 'heart', 'heart', 'lung', 'brain', 'brain') == (True, False)
    assert eligible_candidate('kidney', 'heart', 'lung', 'liver', 'liver', 'brain', 'brain') == (False, False)
    assert eligible_candidate('heart', 'heart', 'heart', 'lung', 'lung', 'brain', 'brain') == (False, False)
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    write(OUT / 'private_manifest.json', {'primary': primary, 'background': background, 'admit': admit, 'wrong_image': wrong, 'sham_priority': priorities, 'background_flags': {str(r['qid']): flags[str(r['qid'])] for r in background}})
    write(OUT / 'private_cached_records.json', cache)
    write(OUT / 'private_bindings.json', bindings)
    write(OUT / 'physical_plan.json', physical)
    p.update(status='FROZEN_CPU_READY', formal_generation_calls=len(physical), maximum_generation_calls=len(physical) + 100)
    write(CONFIG, p)
    cpu = {'status': 'CPU_READY', 'candidate_questions': len(candidates), 'candidate_images': len(by_image), 'primary_questions': 85, 'primary_images': 85, 'primary_image_modality': {x: sum(r['modality'] == x for r in primary) for x in sorted({r['modality'] for r in primary})}, 'excluded_images': len(excluded), 'background_questions': 69, 'changed_background_questions': 21, 'train_and_previous_panel_image_disjoint': True, 'logical_predictor_requests': len(logical), 'cached_physical_records_reused': len(cache), 'new_physical_generations': len(physical), 'generation_cap': len(physical) + 100, 'guard_positive_negative_and_agreement_fixtures': True, 'exact_input_cache_reuse_without_correctness_selection': True, 'new_optimizer_steps': 0}
    write(OUT / 'cpu_check.json', cpu)
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
    m = read(OUT / 'private_manifest.json')
    rows = {r['qid']: r for r in m['primary'] + m['background']}
    plan = read(OUT / 'physical_plan.json')
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(base, COHORT / 'T_42/adapter', is_trainable=False, local_files_only=True)
    model.load_adapter(COHORT / 'T_43/adapter', adapter_name='second', is_trainable=False, local_files_only=True)
    new, calls = {}, 0
    def evaluate(label):
        nonlocal calls
        model.eval()
        for e in (x for x in plan if x['model'] == label):
            check()
            query = {'qid': e['qid'], 'img_name': rows[e['qid']]['img_name'] if e['image_identity'] == 'BLACK' else e['image_identity'], 'question': e['question']}
            image = common.image_view(query, 'B' if e['image_identity'] == 'BLACK' else 'N')
            inputs = processor(text=[common.prompt(processor, query)], images=[image], return_tensors='pt').to('cuda')
            calls += 1
            assert calls <= p['maximum_generation_calls']
            meta = {'physical_index': e['physical_index'], 'model': label, 'qid': e['qid'], 'number': calls}
            with (OUT / 'requests.jsonl').open('a') as f: f.write(json.dumps(dict(meta, event='intent', started_unix=time.time())) + '\n')
            with torch.inference_mode(): output = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
            r = dict(meta, raw=processor.tokenizer.decode(tokens, skip_special_tokens=True), output_ids=tokens, completed_unix=time.time())
            new[e['physical_index']] = r
            write(OUT / 'records' / f'{calls:04d}.json', r)
            with (OUT / 'requests.jsonl').open('a') as f: f.write(json.dumps(dict(r, event='result')) + '\n')
            write(OUT / 'progress.json', {'model': label, 'new_generation_calls': calls, 'planned_calls': len(plan), 'updated_unix': time.time()})
    for v in model.parameters(): v.requires_grad_(False)
    with model.disable_adapter():
        layers = [x for x in model.modules() if isinstance(x, LoraLayer)]
        assert len(layers) == 64 and all(x.disable_adapters for x in layers)
        write(OUT / 'BASE_READOUT_CHECK.json', {'all_lora_layers_disabled': True, 'layers': len(layers)})
        evaluate('base')
    for label, adapter in [('T_42', 'default'), ('T_43', 'second')]:
        model.set_adapter(adapter)
        for v in model.parameters(): v.requires_grad_(False)
        layers = [x for x in model.modules() if isinstance(x, LoraLayer)]
        assert len(layers) == 64 and all(x.active_adapters == [adapter] and not x.disable_adapters for x in layers)
        write(OUT / (label + '_READOUT_CHECK.json'), {'layers': len(layers), 'correct_adapter_active': True, 'all_parameters_frozen': all(not v.requires_grad for v in model.parameters())})
        evaluate(label)
    assert calls == p['formal_generation_calls'] == len(plan)
    cache, bindings = read(OUT / 'private_cached_records.json'), read(OUT / 'private_bindings.json')
    def raw(label, q, view):
        ref = bindings[f'{label}|{q}|{view}']
        return cache[ref['cache_index']]['raw'] if 'cache_index' in ref else new[ref['physical_index']]['raw']
    decisions = {}
    for q in rows:
        if not m['admit'][str(q)]:
            decisions[str(q)] = {'consensus': False, 'guard': False, 'sham': False}
            continue
        args = [raw('base', q, 'N')] + [raw(label, q, view) for view in ('N', 'B', 'W') for label in ('T_42', 'T_43')]
        consensus, guard = eligible_candidate(*args)
        decisions[str(q)] = {'consensus': consensus, 'guard': guard, 'sham': False}
    for panel, priority in m['sham_priority'].items():
        eligible = [q for q in priority if decisions[str(q)]['consensus']]
        for i, q in enumerate(eligible): decisions[str(q)]['sham'] = decisions[str(eligible[(i + 1) % len(eligible)])]['guard']
        assert sum(decisions[str(q)]['guard'] for q in priority) == sum(decisions[str(q)]['sham'] for q in priority)
    write(OUT / 'private_decisions.json', decisions)
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_VISUAL_CONSENSUS_GUARD', 'new_generation_calls': calls, 'cached_records_reused': len(cache), 'primary_questions': 85, 'old_background_questions': 69, 'optimizer_steps': 0, 'policy_uses_one_frozen_parent_pair': True, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__': {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
