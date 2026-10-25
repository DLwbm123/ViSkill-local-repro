"""Close the frozen anatomical-view diagnosis using saved outputs only."""
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = os.environ.get('TASK_EXPERIMENT', 'roi_mask_m1_v1')
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
sys.path.insert(0, str(SRC / 'scripts'))
from slake_probe import normalize, score
from slake_evidence_summary import difference


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def close():
    p = read(SRC / 'configs/reproduction' / (NAME + '.json'))
    receipt = read(STATE / (NAME + '_receipt.json'))
    exit_record = read(STATE / (NAME + '_EXIT.json'))
    assert p['status'] == 'FROZEN_CPU_READY'
    assert exit_record['exit_code'] == 0 and exit_record['reason'] is None
    assert exit_record['end_unix'] < p['deadline_unix']
    assert exit_record['incremental_bytes'] < p['maximum_incremental_bytes']
    assert not exit_record['argv_violations']
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert done['status'] == 'COMPLETE_TWO_REPEATS'
    assert all(read(OUT / f'REPEAT_{r}_COMPLETED.json')['formal_samples'] == 256 for r in (0, 1))
    panel = {r['qid']: r for r in read(OUT / 'private_panel.json')}
    assert len(panel) == len({r['img_name'] for r in panel.values()}) == 64
    schedule = read(OUT / 'schedule.json')
    rows = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(rows) == len(schedule) == 512
    assert [r['number'] for r in rows] == list(range(1, 513))
    assert [(r['repeat'], r['qid'], r['condition']) for r in rows] == [(r['repeat'], r['qid'], r['condition']) for r in schedule]
    events = [json.loads(line) for line in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(e['event'] for e in events) == {'intent': 512, 'result': 512}
    for r in rows:
        r['correct'] = bool(score(r['raw'], panel[r['qid']]['answer']))
    indexed = {(r['repeat'], r['qid'], r['condition']): r for r in rows}
    assert len(indexed) == len(rows)
    qids = sorted(panel)
    controls = [c for c in p['conditions'] if c != 'O']
    primary_controls = ['N', 'B'] if p.get('view_mode') == 'isolated' else ['N', 'D']
    paired = {c: difference(qids, c, indexed, panel, p['selection_seed'], reference='O') for c in controls}

    def accuracy(subset):
        return {c: {'correct': sum(indexed[r, q, c]['correct'] for r in (0, 1) for q in subset), 'executions': 2 * len(subset), 'accuracy': sum(indexed[r, q, c]['correct'] for r in (0, 1) for q in subset) / (2 * len(subset)), 'per_repeat_correct': [sum(indexed[r, q, c]['correct'] for q in subset) for r in (0, 1)]} for c in p['conditions']}

    summary = accuracy(qids)
    for c in controls:
        assert abs(paired[c]['mean'] - (summary['O']['correct'] - summary[c]['correct']) / 128) < 1e-12
    # Conservative interpretation: the same image improves over both controls in both repeats.
    stable = set(qids)
    for c in primary_controls:
        for r in paired[c]['by_repeat']:
            stable.intersection_update(r['improved_qids'])
    stable_targets = sorted({panel[q]['target'] for q in stable})
    gates = {}
    for c in primary_controls:
        gates['O_minus_' + c + '_at_least_10pp'] = paired[c]['mean'] >= .10
        gates['O_minus_' + c + '_positive_95_lower'] = paired[c]['image_cluster_bootstrap_95'][0] > 0
        gates['O_minus_' + c + '_positive_each_repeat'] = all(r['mean'] > 0 for r in paired[c]['by_repeat'])
    gates['O_minus_W_positive_average'] = paired['W']['mean'] > 0
    gates['O_minus_W_nonnegative_each_repeat'] = all(r['mean'] >= 0 for r in paired['W']['by_repeat'])
    gates['stable_gains_in_at_least_two_targets'] = len(stable_targets) >= 2
    groups = {}
    for field in ('target', 'task', 'split', 'modality'):
        groups[field] = {}
        for value in sorted({r[field] for r in panel.values()}):
            subset = sorted(q for q in panel if panel[q][field] == value)
            groups[field][value] = {'questions': len(subset), 'summary': accuracy(subset), **{'O_minus_' + c: difference(subset, c, indexed, panel, p['selection_seed'], reference='O') for c in primary_controls}}
    groups['target_area'] = {}
    for label, small in (('below_1pct', True), ('at_least_1pct', False)):
        subset = [q for q in qids if (panel[q]['target_area_fraction'] < .01) == small]
        if subset:
            groups['target_area'][label] = {'questions': len(subset), 'summary': accuracy(subset)}
    agreement = {c: {'equal_correctness_questions': sum(indexed[0, q, c]['correct'] == indexed[1, q, c]['correct'] for q in qids), 'equal_normalized_response_questions': sum(normalize(indexed[0, q, c]['raw']) == normalize(indexed[1, q, c]['raw']) for q in qids), 'questions': len(qids)} for c in p['conditions']}
    image_group = {q: i for i, q in enumerate(qids)}
    numeric = [{**{k: r[k] for k in ('number', 'repeat', 'condition', 'correct')}, 'question_group': image_group[r['qid']], 'image_group': image_group[r['qid']], 'target': panel[r['qid']]['target'], 'task': panel[r['qid']]['task'], 'split': panel[r['qid']]['split'], 'target_area_fraction': panel[r['qid']]['target_area_fraction'], 'output_tokens': len(r['output_ids'])} for r in rows]
    # Public identifiers are arbitrary panel indices, not source image names or original QA IDs.
    def anonymize(value):
        if isinstance(value, dict):
            return {k: [image_group[q] for q in v] if k.endswith('_qids') else anonymize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [anonymize(v) for v in value]
        return value
    result = {'status': 'COMPLETE_LOCAL_VIEW_GATE_PASS' if all(gates.values()) else 'COMPLETE_LOCAL_VIEW_NOT_ESTABLISHED', 'all_gates_pass': all(gates.values()), 'gates': gates, 'summary': summary, 'paired_differences': anonymize(paired), 'groups': anonymize(groups), 'stable_gain_question_groups': [image_group[q] for q in sorted(stable)], 'stable_gain_targets': stable_targets, 'engine_reproducibility': agreement, 'calls': {'formal': 512, 'actual_intents': 512, 'errors': 0, 'completed_replays': 0, 'output_tokens': sum(len(r['output_ids']) for r in rows), 'truncated_at_64': sum(len(r['output_ids']) == 64 for r in rows)}, 'cost': {'supervisor_wall_seconds': exit_record['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': exit_record['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': exit_record['incremental_bytes']}, 'optimizer_steps': 0, 'test_annotation_opened': False, 'limitations': ['Train/validation development panel, not blind evaluation.', 'Anatomical annotation supplies privileged localization; no autonomous selection is tested.', 'Exact match can reject valid wording; scoring is unchanged.', 'Patient independence and clinical label validity are unknown.', 'Two engine initializations are not independent training seeds.', 'W can contain useful evidence; it is an off-target geometric control.', 'Small task and target groups are descriptive, with no subgroup selection or adjusted confirmatory claims.']}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    for name, value in [('result.json', result), ('complete_numeric_records.json', numeric), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('execution_source.json', read(OUT / 'execution_source.json'))]:
        write(bundle / name, value)
    print(json.dumps({'status': result['status'], 'gates': gates, 'summary': summary, 'paired': anonymize(paired), 'stable_gain_targets': stable_targets, 'cost': result['cost']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    close()
