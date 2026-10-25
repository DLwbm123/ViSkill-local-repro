"""Frozen natural-prevalence admission audit with demand-driven branch generation."""
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
NAME = 'natural_scope_validation_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
SUPPORT = ROOT / 'artifacts/medical/task_supported_adapter_v1'
VISUAL = ROOT / 'artifacts/medical/grounded_reader_anchor_v1'
READER = ROOT / 'artifacts/medical/region_reader_learning_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from medical_supported_adapter import vectorize
from medical_roi_task_audit import task, match_target
from slake_probe import score, query_view
read, write = common.read, common.write


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def legacy_records():
    rows = []
    for folder, allowed in ((READER, ('base',)), (VISUAL, ('A_42', 'A_43'))):
        m = read(folder / 'private_manifest.json')
        metadata = {r['qid']: r for kind in ('primary', 'retention') for r in m[kind]}
        for file in sorted((folder / 'records').glob('*.json')):
            row = read(file)
            if row['model'] in allowed and row['view'] in ('N', 'B') and row['panel'] in ('primary', 'retention'):
                rows.append(dict(row, legacy_image=metadata[row['qid']]['img_name'], legacy_experiment=folder.name))
    assert len(rows) == 840
    return rows


def prepare():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    assert read(SUPPORT / 'RESULT.json')['all_gates_pass']
    rows = sorted((r for r in read(DATA / 'raw/validation.json') if r['q_lang'] == 'en'), key=lambda r: r['qid'])
    assert len(rows) == 1053 and len({r['img_name'] for r in rows}) == 96
    support = read(SUPPORT / 'private_support.json')
    pairs = {tuple(x) for x in support['task_target_pairs']}
    known = torch.load(SUPPORT / 'private_router.pt', map_location='cpu', weights_only=True)
    matrix = torch.from_numpy(vectorize([r['question'] for r in rows], known['vocabulary'], known['idf'].numpy()))
    with torch.no_grad():
        probability = torch.sigmoid(matrix @ known['coefficient'][:-1] + known['coefficient'][-1]).tolist()
    admitted = [value >= .90 for value in probability]
    sham = admitted.copy()
    random.Random(p['selection_seed'] + 1).shuffle(sham)
    routing = {}
    for row, value, admit, control in zip(rows, probability, admitted, sham):
        kind = task(row['question'])
        target = match_target(row['question'], support['lexicon'])
        rule = (kind, target) in pairs
        row.update(task=kind, target=target, weak_scope=row['base_type'] == 'vqa' and rule)
        routing[str(row['qid'])] = {'probability': value, 'admit': admit, 'sham_admit': control, 'rule_admit': rule}
    assert sum(r['weak_scope'] for r in rows) == 39
    assert not any(r['base_type'] != 'vqa' and routing[str(r['qid'])]['rule_admit'] for r in rows)
    # Check saved router compatibility without replaying any reader generation.
    old = read(SUPPORT / 'private_manifest.json')['primary'][0]
    saved = read(SUPPORT / 'private_routing_frozen.json')['queries'][str(old['qid'])]
    feature = torch.from_numpy(vectorize([old['question']], known['vocabulary'], known['idf'].numpy()))
    value = float(torch.sigmoid(feature @ known['coefficient'][:-1] + known['coefficient'][-1]))
    assert abs(value - saved['probability']) < 1e-12
    old_rows = legacy_records()
    by_q = {r['qid']: r for r in rows}
    assert all(by_q[r['qid']]['img_name'] == r['legacy_image'] for r in old_rows)
    old_index = {(r['model'], r['qid'], r['view']) for r in old_rows}
    needed, required = [], set()
    for row in rows:
        route = routing[str(row['qid'])]
        models = ['base'] + (['A_42', 'A_43'] if route['admit'] or route['sham_admit'] or route['rule_admit'] else [])
        for model in models:
            for view in ('N', 'B'):
                required.add((model, row['qid'], view))
                if (model, row['qid'], view) not in old_index:
                    needed.append({'model': model, 'qid': row['qid'], 'view': view})
    model_order = {'base': 0, 'A_42': 1, 'A_43': 2}
    rng = random.Random(p['selection_seed'])
    rng.shuffle(needed)
    needed.sort(key=lambda e: model_order[e['model']])
    assert len(needed) <= p['maximum_generation_calls']
    assert len({(r['model'], r['qid'], r['view']) for r in needed}) == len(needed)
    assert not old_index & {(r['model'], r['qid'], r['view']) for r in needed}
    assert required == (required & old_index) | {(r['model'], r['qid'], r['view']) for r in needed}
    old_qids = {r['qid'] for r in old_rows if r['model'] == 'base'}
    assert len(old_qids) == 140 and all(r['qid'] in old_qids for r in rows if r['weak_scope'])
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    write(OUT / 'private_manifest.json', {'validation': rows, 'legacy_qids': sorted(old_qids), 'new_qids': sorted(set(by_q) - old_qids)})
    write(OUT / 'private_routing_frozen.json', {'queries': routing, 'admitted': sum(admitted), 'sham_admitted': sum(sham), 'frozen_before_new_generations': True, 'router_refitted': False, 'threshold': .90})
    write(OUT / 'schedule.json', needed)
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    fixture = rows[0]
    text = common.prompt(processor, query_view(fixture))
    inputs = [processor(text=[text], images=[common.image_view(fixture, view)], return_tensors='pt') for view in ('N', 'B')]
    assert torch.equal(inputs[0]['input_ids'], inputs[1]['input_ids'])
    assert torch.equal(inputs[0]['image_grid_thw'], inputs[1]['image_grid_thw'])
    assert not torch.cuda.is_initialized()
    write(OUT / 'cpu_check.json', {'status': 'CPU_READY', 'questions': 1053, 'images': 96, 'VQA_questions': 898, 'knowledge_questions': 155, 'weak_supported_questions': 39, 'weak_offscope_questions': 1014, 'previously_evaluated_questions': 140, 'new_questions_on_observed_images': 913, 'all_supported_questions_previously_evaluated': True, 'legacy_catalog_records': 840, 'legacy_records_required': len(required & old_index), 'required_physical_branch_records': len(required), 'new_generations_planned': len(needed), 'new_by_model': {model: sum(e['model'] == model for e in needed) for model in model_order}, 'all_required_policy_branches_present_or_scheduled': True, 'no_duplicate_reader_generations': True, 'saved_router_probability_fixture_matches': True, 'router_refitted': False, 'reader_updated': False, 'matched_N_Blank_ids_grids': True, 'gpu_initialized': False})
    p['formal_generation_calls'] = len(needed)
    p['maximum_generation_calls'] = min(p['maximum_generation_calls'], len(needed) + 100)
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps(read(OUT / 'cpu_check.json')), flush=True)


def run():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY' and not (OUT / 'requests.jsonl').exists()
    check()
    torch.set_num_threads(4)
    rows = read(OUT / 'private_manifest.json')['validation']
    items = {r['qid']: r for r in rows}
    plan = read(OUT / 'schedule.json')
    assert len(plan) == p['formal_generation_calls']
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    calls = 0
    for label in ('base', 'A_42', 'A_43'):
        entries = [e for e in plan if e['model'] == label]
        if not entries:
            continue
        check()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
        if label != 'base':
            model = PeftModel.from_pretrained(model, VISUAL / label / 'adapter', is_trainable=False, local_files_only=True)
        model.eval()
        for entry in entries:
            check()
            calls += 1
            assert calls <= p['maximum_generation_calls']
            row = items[entry['qid']]
            text = common.prompt(processor, query_view(row))
            inputs = processor(text=[text], images=[common.image_view(row, entry['view'])], return_tensors='pt').to('cuda')
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(entry, event='intent', number=calls, started_unix=time.time())) + '\n')
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
            raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            record = dict(entry, number=calls, raw=raw, output_ids=tokens, correct=bool(score(raw, row['answer'])), completed_unix=time.time())
            write(OUT / 'records' / f'{calls:04d}.json', record)
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(record, event='result')) + '\n')
            write(OUT / 'progress.json', {'phase': 'missing_branch_evaluation', 'model': label, 'new_generation_calls': calls, 'planned_calls': len(plan), 'updated_unix': time.time()})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert calls == len(plan)
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_NATURAL_SCOPE_VALIDATION', 'new_generation_calls': calls, 'legacy_catalog_records': 840, 'reused_required_records': read(OUT / 'cpu_check.json')['legacy_records_required'], 'questions': 1053, 'new_reader_optimizer_steps': 0, 'new_router_optimizer_steps': 0, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
