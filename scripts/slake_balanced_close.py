"""Close a predeclared feasibility stop without reading confirmation answers."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
OUT = ROOT / 'artifacts/medical/slake_balanced_final_v1'
STATE = ROOT / 'state/reproduction'
sys.path.insert(0, str(SRC / 'scripts'))
from slake_balanced_probe import read, write, retrieve, old_features, presence, query_view, normalize

import numpy as np


def close():
    p = read(SRC / 'configs/reproduction/slake_balanced_final_v1.json')
    receipt = read(STATE / 'slake_balanced_final_v1_receipt.json')
    exit_record = read(STATE / 'slake_balanced_final_v1_EXIT.json')
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert exit_record['exit_code'] == 0 and exit_record['reason'] is None
    assert done['status'] in ('SOURCE_GATE_STOP', 'CONFIRMATION_GATE_STOP')
    assert exit_record['end_unix'] < p['deadline_unix'] and exit_record['incremental_bytes'] < p['maximum_incremental_bytes']
    assert not exit_record['argv_violations']
    rows = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    candidates = read(OUT / 'private_source_candidates.json')
    assert len(rows) == len(candidates) and all(r['phase'] == 'source' for r in rows)
    assert [r['number'] for r in rows] == list(range(1, len(rows) + 1))
    assert [r['qid'] for r in rows] == [r['qid'] for r in candidates]
    events = [json.loads(line) for line in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(e['event'] for e in events) == {'intent': len(rows), 'result': len(rows)}
    source = read(OUT / 'SOURCE_FROZEN.json')
    bank = read(OUT / 'private_source_bank.json')
    admitted = {r['qid'] for r in bank}
    group = defaultdict(Counter)
    for row in bank:
        group[row['target'], row['modality'], row['location']][normalize(row['answer'])] += 1
    assert all(v['yes'] == v['no'] and v['yes'] >= 2 for v in group.values())
    source_groups = [{'target': k[0], 'modality': k[1], 'anatomical_location': k[2], 'positive_entries': v['yes'], 'negative_entries': v['no']} for k, v in sorted(group.items())]
    by_class = {gold: {'attempted': sum(normalize(r['answer']) == gold for r in candidates), 'model_agrees': sum(row['correct'] for item, row in zip(candidates, rows) if normalize(item['answer']) == gold), 'admitted': sum(normalize(r['answer']) == gold for r in bank)} for gold in ('yes', 'no')}
    numeric = [{k: row[k] for k in ('number', 'phase', 'repeat', 'qid', 'condition', 'correct', 'completed_unix')} | {'output_tokens': len(row['output_ids']), 'admitted': row['qid'] in admitted} for row in rows]
    confirmation, descriptive = None, None
    if done['status'] == 'CONFIRMATION_GATE_STOP':
        confirmation = read(OUT / 'CONFIRMATION_FROZEN.json')
        assert confirmation['confirmation_calls_planned'] == 0 and not (OUT / 'schedule.json').exists()
        assert not (OUT / 'private_cards.json').exists()
        pool = read(OUT / 'private_confirmation_pool.json')
        assert all(set(r) == {'qid', 'img_name', 'question', 'content_type'} for r in pool)
        features = old_features()
        features.update(dict(zip(read(OUT / 'private_feature_images.json'), np.load(OUT / 'confirmation_features.npz')['image'])))
        lexicon = read(OUT / 'lexicon.json')
        groups_before, groups_after = set(), set()
        before_count, after_count, parsed_count = 0, 0, 0
        targets = Counter()
        for row in pool:
            if presence(row['question'], lexicon):
                parsed_count += 1
            match = retrieve(query_view(row), bank, features, threshold=-1)
            if match:
                before_count += 1
                groups_before.add(row['img_name'])
                if match['score'] >= source['threshold']:
                    after_count += 1
                    groups_after.add(row['img_name'])
                    targets[match['entity']] += 1
        descriptive = {'analysis_type': 'posthoc_input_only_feasibility_description', 'test_gold_used': False, 'pool_visual_questions': len(pool), 'parsed_presence_questions': parsed_count, 'matched_pair_questions_before_similarity_threshold': before_count, 'matched_pair_images_before_threshold': len(groups_before), 'matched_pair_questions_after_frozen_threshold': after_count, 'matched_pair_images_after_threshold': len(groups_after), 'supported_question_targets_after_threshold': dict(targets), 'threshold_changed': False, 'additional_model_calls': 0}
    warnings = {'source_exit_code': exit_record['stage_costs'][0]['exit_code'], 'source_async_generator_shutdown_warning': 'Task was destroyed but it is pending!' in (OUT / 'source.log').read_text(), 'source_cutlass_atexit_warning': 'CACHE_FILE' in (OUT / 'source.log').read_text(), 'interpretation': 'Shutdown warnings retained; all source requests returned and stage exit was zero. No restart was performed.'}
    result = {'status': 'COMPLETE_FEASIBILITY_STOP', 'stop_stage': done['status'], 'source': source, 'source_groups': source_groups, 'source_verification_by_class': by_class, 'confirmation_precheck': confirmation, 'posthoc_input_feasibility': descriptive, 'completed_generation_calls': len(rows), 'confirmation_generation_calls': 0, 'request_errors': 0, 'output_token_ids': sum(len(r['output_ids']) for r in rows), 'source_call_replays': 0, 'optimizer_steps': 0, 'confirmation_effect_measured': False, 'confirmation_answer_fields_used': False, 'query_roi_opened': False, 'source_cards_rendered': 0, 'production_three_image_conditions_executed': False, 'route_decision': 'End this final bounded static medical reference configuration; no rule lowering, candidate replenishment, extra confirmation, model search or training.', 'scientific_interpretation': 'Balanced train sources were feasible, but the frozen configuration did not support enough questions/images/coverage for the planned confirmation. This is not a new accuracy result and does not establish zero or negative visual effect.', 'data_boundary': 'Test input fields and frozen CLIP image features were used for feasibility. Test answer fields were not consulted or scored. Future rule changes using these input coverage observations must not describe this input distribution as wholly untouched.', 'cost': {'supervisor_wall_seconds': exit_record['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': exit_record['end_unix'] - p['first_preparation_unix'], 'new_CLIP_images': read(OUT / 'FEATURES_COMPLETED.json')['new_images'] if (OUT / 'FEATURES_COMPLETED.json').exists() else 0, 'stage_costs': exit_record['stage_costs'], 'peak_incremental_bytes_at_exit': exit_record['incremental_bytes']}, 'shutdown_warnings': warnings, 'exit': exit_record, 'completed_unix': time.time()}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    assert not bundle.exists()
    bundle.mkdir()
    for name, obj in [('result.json', result), ('complete_numeric_source_records.json', numeric), ('protocol.json', p), ('source_preparation.json', read(OUT / 'source_preparation.json')), ('source_frozen.json', source), ('source_completed.json', read(OUT / 'SOURCE_COMPLETED.json')), ('source_cpu_check.json', read(OUT / 'source_cpu_check.json')), ('execution_source.json', read(OUT / 'execution_source.json')), ('worker_completed.json', done)]:
        write(bundle / name, obj)
    if confirmation:
        write(bundle / 'confirmation_precheck.json', confirmation)
        write(bundle / 'features_receipt.json', read(OUT / 'FEATURES_COMPLETED.json'))
        write(bundle / 'input_feasibility_diagnostics.json', descriptive)
        write(bundle / 'panels.json', read(OUT / 'panels.json'))
        retrieval = read(OUT / 'private_retrieval.json')
        pairs = sorted({(v['positive_qid'], v['negative_qid']) for v in retrieval.values() if v})
        pair_indices = {pair: i for i, pair in enumerate(pairs)}
        write(bundle / 'retrieval_numeric.json', {q: {'source_pair_index': pair_indices[v['positive_qid'], v['negative_qid']], 'score': v['score']} if v else None for q, v in retrieval.items()})
    print(json.dumps({'status': result['status'], 'source_by_class': by_class, 'source_groups': source_groups, 'confirmation_precheck': confirmation, 'posthoc_input_feasibility': descriptive, 'cost': result['cost'], 'shutdown_warnings': warnings}), flush=True)


if __name__ == '__main__':
    close()
