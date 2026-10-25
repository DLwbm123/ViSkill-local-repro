"""Close the frozen regional-card screen from saved outputs; no inference or relabeling."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
OUT = ROOT / 'artifacts/medical/slake_evidence_v2'
STATE = ROOT / 'state/reproduction'
sys.path.insert(0, str(SRC / 'scripts'))
from slake_probe import normalize, score


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def difference(qids, other, indexed, inputs, seed, reference='V'):
    by_repeat, values = [], []
    for repeat in (0, 1):
        diff = [int(indexed[repeat, q, reference]['correct']) - int(indexed[repeat, q, other]['correct']) for q in qids]
        by_repeat.append({'repeat': repeat, 'mean': float(np.mean(diff)), 'improved_qids': [q for q, d in zip(qids, diff) if d == 1], 'harmed_qids': [q for q, d in zip(qids, diff) if d == -1]})
        values.append(diff)
    averaged = np.mean(values, axis=0)
    groups = defaultdict(list)
    for i, q in enumerate(qids):
        groups[inputs[q]['img_name']].append(i)
    groups = list(groups.values())
    numerator = np.array([averaged[g].sum() for g in groups])
    denominator = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(groups), size=(10000, len(groups)))
    bootstrap = numerator[samples].sum(axis=1) / denominator[samples].sum(axis=1)
    return {'mean': float(averaged.mean()), 'image_cluster_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(), 'questions': len(qids), 'images': len(groups), 'by_repeat': by_repeat}


def close():
    p = read(SRC / 'configs/reproduction/slake_evidence_v2.json')
    assert p['status'] == 'FROZEN_CPU_READY'
    receipt = read(STATE / 'slake_evidence_v2_receipt.json')
    exit_record = read(STATE / 'slake_evidence_v2_EXIT.json')
    incident = read(OUT / 'runtime_incident.json')
    initial_receipt = read(OUT / 'failures/runtime_attempt0/slake_evidence_v2_receipt.json')
    failed_wall = incident['first_attempt_exit']['end_unix'] - initial_receipt['start_unix']
    assert exit_record['exit_code'] == 0 and exit_record['reason'] is None
    assert exit_record['end_unix'] < p['deadline_unix']
    assert exit_record['incremental_bytes'] < p['maximum_incremental_bytes']
    assert not exit_record['argv_violations']
    done = read(OUT / 'WORKER_COMPLETED.json')
    panels = read(OUT / 'panels.json')
    inputs = {q['qid']: q for q in read(OUT / 'private_inputs.json')}
    labels = read(OUT / 'private_labels.json')
    cards = read(OUT / 'private_cards.json')
    bank = {r['qid']: r for r in read(OUT / 'private_bank.json')}
    schedule = read(OUT / 'schedule.json')
    rows = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(rows) == len(schedule) <= p['maximum_generation_calls']
    assert [r['number'] for r in rows] == list(range(1, len(rows) + 1))
    assert [(r['repeat'], r['qid'], r['condition']) for r in rows] == [(r['repeat'], r['qid'], r['condition']) for r in schedule]
    events = [json.loads(line) for line in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(r['event'] for r in events) == {'intent': len(rows), 'result': len(rows)}
    indexed = {(r['repeat'], r['qid'], r['condition']): r for r in rows}
    assert len(indexed) == len(rows) == len(inputs) * 5 * 2
    assert all(score(r['raw'], labels[str(r['qid'])]['answer']) == r['correct'] for r in rows)
    image_groups = {name: i for i, name in enumerate(sorted({q['img_name'] for q in inputs.values()}))}
    summary, paired, copying, splits = {}, {}, {}, {}
    for panel, qids in panels.items():
        splits[panel] = {
            'hidden_source_answer_same': [q for q in qids if str(q) in cards and normalize(bank[cards[str(q)]['source_qid']]['answer']) == normalize(labels[str(q)]['answer'])],
            'hidden_source_answer_different': [q for q in qids if str(q) in cards and normalize(bank[cards[str(q)]['source_qid']]['answer']) != normalize(labels[str(q)]['answer'])]
        }
        summary[panel], paired[panel], copying[panel] = {}, {}, {}
        for condition in p['conditions']:
            selected = [indexed[repeat, q, condition] for repeat in (0, 1) for q in qids]
            summary[panel][condition] = {
                'correct': sum(r['correct'] for r in selected), 'executions': len(selected), 'questions': len(qids), 'images': len({inputs[q]['img_name'] for q in qids}),
                'accuracy': float(np.mean([r['correct'] for r in selected])),
                'per_repeat_correct': [sum(indexed[repeat, q, condition]['correct'] for q in qids) for repeat in (0, 1)],
                'role': {role: {'correct': sum(r['correct'] for r in selected if cards.get(str(r['qid']), {}).get('role', 'unsupported') == role), 'executions': sum(cards.get(str(r['qid']), {}).get('role', 'unsupported') == role for r in selected)} for role in sorted({cards.get(str(q), {}).get('role', 'unsupported') for q in qids})},
                'answer_type': {kind: {'correct': sum(r['correct'] for r in selected if r['answer_type'] == kind), 'executions': sum(r['answer_type'] == kind for r in selected)} for kind in sorted({r['answer_type'] for r in selected})}
            }
            available = [r for r in selected if str(r['qid']) in cards]
            copying[panel][condition] = {
                'reference_available_executions': len(available),
                'answer_matches_hidden_source': sum(score(r['raw'], bank[cards[str(r['qid'])]['source_qid']]['answer']) for r in available),
                'wrong_answer_matches_hidden_source': sum(not r['correct'] and score(r['raw'], bank[cards[str(r['qid'])]['source_qid']]['answer']) for r in available),
                'different_answer_stratum_wrong_source_matches': sum(not r['correct'] and score(r['raw'], bank[cards[str(r['qid'])]['source_qid']]['answer']) for r in available if r['qid'] in splits[panel]['hidden_source_answer_different'])
            }
        paired[panel] = {other: difference(qids, other, indexed, inputs, p['selection_seed']) for other in ('N', 'Blank', 'Text', 'Wrong')}
        for stratum, subset in splits[panel].items():
            if subset:
                paired[panel][stratum] = {other: difference(subset, other, indexed, inputs, p['selection_seed']) for other in ('N', 'Blank', 'Text', 'Wrong')}
    mechanism = paired['mechanism']
    mismatch = mechanism['hidden_source_answer_different']
    gates = {
        'panel_support': len(panels['mechanism']) >= 40 and mechanism['Blank']['images'] >= 30,
        'V_minus_Blank_positive_each_repeat': all(x['mean'] > 0 for x in mechanism['Blank']['by_repeat']),
        'V_minus_Text_positive_each_repeat': all(x['mean'] > 0 for x in mechanism['Text']['by_repeat']),
        'V_minus_Blank_positive_lower_bound': mechanism['Blank']['image_cluster_bootstrap_95'][0] > 0,
        'V_minus_Text_positive_lower_bound': mechanism['Text']['image_cluster_bootstrap_95'][0] > 0,
        'V_minus_Wrong_positive_each_repeat': all(x['mean'] > 0 for x in mechanism['Wrong']['by_repeat']),
        'overall_V_minus_N_nonnegative': paired['union']['N']['mean'] >= 0,
        'different_answer_questions_at_least_20': len(splits['mechanism']['hidden_source_answer_different']) >= 20,
        'different_answer_V_minus_Blank_positive': mismatch['Blank']['mean'] > 0,
        'different_answer_V_minus_Text_positive': mismatch['Text']['mean'] > 0,
        'different_answer_no_extra_wrong_source_matches': copying['mechanism']['V']['different_answer_stratum_wrong_source_matches'] <= copying['mechanism']['Blank']['different_answer_stratum_wrong_source_matches']
    }
    numeric = []
    for row in rows:
        item = {k: row[k] for k in ('number', 'repeat', 'qid', 'condition', 'source_qid', 'correct', 'answer_type', 'content_type', 'completed_unix')}
        item.update(image_group=image_groups[inputs[row['qid']]['img_name']], output_tokens=len(row['output_ids']), matches_hidden_source_answer=score(row['raw'], bank[row['source_qid']]['answer']) if row['source_qid'] is not None else None)
        numeric.append(item)
    result = {'status': 'COMPLETE_VISUAL_EVIDENCE_GATE_PASS' if all(gates.values()) else 'COMPLETE_VISUAL_EVIDENCE_NOT_ESTABLISHED', 'summary': summary, 'paired_differences': paired, 'copying_diagnostics': copying, 'gates': gates, 'all_gates_pass': all(gates.values()), 'panels': panels, 'hidden_answer_strata': splits, 'total_generation_calls': len(rows), 'total_request_intents_including_failure': len(rows) + incident['incomplete_request_intents'], 'incomplete_first_attempt_requests': incident['incomplete_request_intents'], 'total_output_token_ids': sum(len(r['output_ids']) for r in rows), 'truncated_at_64_tokens': sum(len(r['output_ids']) == 64 for r in rows), 'completed_run_request_errors': 0, 'runtime_initialization_failures': 1, 'optimizer_steps': 0, 'test_annotation_opened': False, 'engine_reproducibility': {c: {'equal_correctness_questions': sum(indexed[0, q, c]['correct'] == indexed[1, q, c]['correct'] for q in inputs), 'equal_normalized_response_questions': sum(normalize(indexed[0, q, c]['raw']) == normalize(indexed[1, q, c]['raw']) for q in inputs), 'questions': len(inputs)} for c in p['conditions']}, 'cost': {'successful_supervisor_wall_seconds': exit_record['end_unix'] - receipt['start_unix'], 'initial_failed_supervisor_wall_seconds': failed_wall, 'all_supervisor_wall_seconds': exit_record['end_unix'] - receipt['start_unix'] + failed_wall, 'preparation_to_exit_seconds': exit_record['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_supervisor_exit': exit_record['incremental_bytes'], 'completed_response_replays': 0, 'feature_recomputations': 0}, 'exit': exit_record, 'worker_completed': done, 'limitations': ['Question holdout on already observed validation images, not blind images or patient-independent evidence.', 'Conditional mechanism panel selected by answer-free compatibility; report 18/96 coverage rather than full-benchmark applicability.', 'Train annotations and reference crops are an evidence-card prototype, not independently verified clinical procedures.', 'Strict exact match remains sensitive to expression; no retrospective scoring change.', 'Two inference-engine initializations are not independent training seeds.', 'Hidden-answer mismatch grouping is evaluation-only; no query gold enters retrieval or prompts.']}
    result['response_agreement'] = {other: {'same_normalized_response_executions': sum(normalize(indexed[repeat, q, 'V']['raw']) == normalize(indexed[repeat, q, other]['raw']) for repeat in (0, 1) for q in panels['mechanism']), 'executions': 2 * len(panels['mechanism'])} for other in ('N', 'Blank', 'Text', 'Wrong')}
    abstained = [q for q in inputs if str(q) not in cards]
    result['abstention_diagnostics'] = {'questions': len(abstained), 'same_correctness_all_conditions_question_repeats': sum(len({indexed[repeat, q, c]['correct'] for c in p['conditions']}) == 1 for repeat in (0, 1) for q in abstained), 'same_normalized_response_all_conditions_question_repeats': sum(len({normalize(indexed[repeat, q, c]['raw']) for c in p['conditions']}) == 1 for repeat in (0, 1) for q in abstained), 'question_repeats': 2 * len(abstained)}
    result['source_bank'] = {'entries': len(bank), 'images': len({r['img_name'] for r in bank.values()}), 'selected_entries': len({c['source_qid'] for c in cards.values()}), 'selected_images': len({bank[c['source_qid']]['img_name'] for c in cards.values()})}
    result['limitations'].append('Matching a hidden source answer measures behavioral concordance; for binary mismatched labels it can simply equal the error count and does not establish copying causally.')
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    assert not bundle.exists()
    bundle.mkdir()
    for name, obj in [('result.json', result), ('complete_numeric_records.json', numeric), ('protocol.json', p), ('panels.json', panels), ('schedule.json', schedule), ('retrieval_frozen.json', read(OUT / 'retrieval_frozen.json')), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('execution_source.json', read(OUT / 'execution_source.json'))]:
        write(bundle / name, obj)
    write(bundle / 'retrieval_numeric.json', {q: {'source_qid': r['source_qid'], 'score': r['score']} if r is not None else None for q, r in read(OUT / 'private_retrieval.json').items()})
    for repeat in (0, 1):
        write(bundle / f'repeat_{repeat}_completed.json', read(OUT / f'REPEAT_{repeat}_COMPLETED.json'))
    write(bundle / 'runtime_incident.json', incident)
    write(bundle / 'summary_cpu_check.json', read(OUT / 'summary_cpu_check.json'))
    write(bundle / 'loading_observation.json', read(OUT / 'loading_observation.json'))
    print(json.dumps({'status': result['status'], 'gates': gates, 'mechanism': summary['mechanism'], 'copying': copying['mechanism'], 'paired': paired['mechanism'], 'cost': result['cost']}, ensure_ascii=False), flush=True)


def diagnostics():
    p = read(SRC / 'configs/reproduction/slake_evidence_v2.json')
    original = read(OUT / 'RESULT.json')
    qids = original['panels']['mechanism']
    inputs = {q['qid']: q for q in read(OUT / 'private_inputs.json')}
    labels = read(OUT / 'private_labels.json')
    cards = read(OUT / 'private_cards.json')
    bank = {r['qid']: r for r in read(OUT / 'private_bank.json')}
    rows = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    indexed = {(r['repeat'], r['qid'], r['condition']): r for r in rows}
    presence = [q for q in qids if cards[str(q)]['role'] == 'presence']
    negative = [q for q in presence if normalize(labels[str(q)]['answer']) == 'no']
    positive = [q for q in presence if normalize(labels[str(q)]['answer']) == 'yes']
    confusion = {}
    for condition in p['conditions']:
        confusion[condition] = {
            'negative_questions': len(negative), 'negative_executions': 2 * len(negative),
            'false_yes_on_gold_no': sum(score(indexed[repeat, q, condition]['raw'], 'yes') for repeat in (0, 1) for q in negative),
            'correct_no': sum(indexed[repeat, q, condition]['correct'] for repeat in (0, 1) for q in negative),
            'positive_questions': len(positive), 'positive_executions': 2 * len(positive),
            'false_no_on_gold_yes': sum(score(indexed[repeat, q, condition]['raw'], 'no') for repeat in (0, 1) for q in positive)
        }
    context = {reference + '_minus_N': difference(qids, 'N', indexed, inputs, p['selection_seed'], reference) for reference in ('Text', 'Blank', 'Wrong')}
    # Compare direct counts with paired differences; this catches reference-index errors.
    assert abs(context['Text_minus_N']['mean'] - (original['summary']['mechanism']['Text']['correct'] - original['summary']['mechanism']['N']['correct']) / (2 * len(qids))) < 1e-12
    selected = {c['source_qid'] for c in cards.values() if c['role'] == 'presence'}
    v_baseline_gains = [set(row['improved_qids']) for row in original['paired_differences']['mechanism']['N']['by_repeat']]
    v_blank_gains = [set(row['improved_qids']) for row in original['paired_differences']['mechanism']['Blank']['by_repeat']]
    obj = {'analysis_type': 'posthoc_descriptive_saved_response_only', 'new_model_calls': 0, 'original_scores_and_gates_unchanged': True, 'context_differences': context, 'presence_confusion': confusion, 'selected_presence_sources': len(selected), 'selected_presence_sources_with_gold_yes': sum(normalize(bank[q]['answer']) == 'yes' for q in selected), 'stable_V_over_N_gain_qids': sorted(set.intersection(*v_baseline_gains)), 'stable_V_over_Blank_gain_qids': sorted(set.intersection(*v_blank_gains)), 'limitations': ['Shared context regressions locate an effect of the added text on this panel, not which phrase caused it.', 'Reference selection supplies positive target exemplars for presence; this may induce a presence prior, but a balanced-reference control was not run.', 'Binary wrong-source matches are not causal proof of copying; source answers were never shown.', 'The single stable V-over-Blank gain is not a V-over-N improvement and cannot establish overall efficacy.']}
    write(OUT / 'POSTHOC_DIAGNOSTICS.json', obj)
    write(OUT / 'public_bundle/posthoc_diagnostics.json', obj)
    print(json.dumps(obj), flush=True)


if __name__ == '__main__':
    diagnostics() if os.environ.get('TASK_STAGE') == 'diagnostics' else close()
