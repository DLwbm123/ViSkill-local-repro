"""Close the frozen consensus policy using existing responses; no model operations."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
OUT = ROOT / 'artifacts/medical/visual_consensus_guard_v1'
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
sys.path.insert(0, str(SRC / 'scripts'))
from medical_region_reader import read, write
from medical_visual_consensus import eligible_candidate
from slake_probe import score


def close():
    p = read(SRC / 'configs/reproduction/visual_consensus_guard_v1.json')
    end = read(ROOT / 'state/reproduction/visual_consensus_guard_v1_EXIT.json')
    receipt = read(ROOT / 'state/reproduction/visual_consensus_guard_v1_receipt.json')
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert end['exit_code'] == 0 and end['reason'] is None and not end['argv_violations']
    assert end['end_unix'] < p['deadline_unix'] and end['incremental_bytes'] < p['maximum_incremental_bytes']
    assert done['status'] == 'COMPLETE_VISUAL_CONSENSUS_GUARD' and done['new_generation_calls'] == 476
    assert done['optimizer_steps'] == 0 and not done['test_annotation_opened']
    assert read(OUT / 'BASE_READOUT_CHECK.json') == {'all_lora_layers_disabled': True, 'layers': 64}
    for label in ['T_42', 'T_43']:
        check = read(OUT / (label + '_READOUT_CHECK.json'))
        assert check['layers'] == 64 and check['correct_adapter_active'] and check['all_parameters_frozen']
    m, plan = read(OUT / 'private_manifest.json'), read(OUT / 'physical_plan.json')
    rows = {r['qid']: r for r in m['primary'] + m['background']}
    assert len(m['primary']) == len({r['img_name'] for r in m['primary']}) == 85
    assert len(m['background']) == 69 and sum(m['admit'].values()) == 106
    records = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(records) == len(plan) == 476
    assert [r['number'] for r in records] == list(range(1, 477))
    assert [r['physical_index'] for r in records] == list(range(476))
    assert all((r['model'], r['qid']) == (e['model'], e['qid']) for r, e in zip(records, plan))
    events = [json.loads(x) for x in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(e['event'] for e in events) == {'intent': 476, 'result': 476}
    for event in ('intent', 'result'):
        assert [e['physical_index'] for e in events if e['event'] == event] == list(range(476))
    assert (OUT / 'physical_plan.json').stat().st_mtime < min(e['started_unix'] for e in events if e['event'] == 'intent')
    assert (OUT / 'private_decisions.json').stat().st_mtime >= max(r['completed_unix'] for r in records)
    cache, bindings = read(OUT / 'private_cached_records.json'), read(OUT / 'private_bindings.json')
    assert len(cache) == 193 and len(bindings) == 848
    assert {ref['physical_index'] for ref in bindings.values() if 'physical_index' in ref} == set(range(476))
    assert {ref['cache_index'] for ref in bindings.values() if 'cache_index' in ref} == set(range(193))
    def raw(label, q, view):
        ref = bindings[f'{label}|{q}|{view}']
        return cache[ref['cache_index']]['raw'] if 'cache_index' in ref else records[ref['physical_index']]['raw']
    decisions = {}
    for q in rows:
        consensus, guard = (False, False)
        if m['admit'][str(q)]:
            consensus, guard = eligible_candidate(raw('base', q, 'N'), *[raw(label, q, v) for v in ('N', 'B', 'W') for label in ('T_42', 'T_43')])
        decisions[str(q)] = {'consensus': consensus, 'guard': guard, 'sham': False}
    for priority in m['sham_priority'].values():
        eligible = [q for q in priority if decisions[str(q)]['consensus']]
        for i, q in enumerate(eligible): decisions[str(q)]['sham'] = decisions[str(eligible[(i + 1) % len(eligible)])]['guard']
        assert sum(decisions[str(q)]['guard'] for q in priority) == sum(decisions[str(q)]['sham'] for q in priority)
    assert decisions == read(OUT / 'private_decisions.json')
    cohort = read(COHORT / 'private_manifest.json')
    old_rows = {r['qid']: r for k in ('new_validation', 'old_source', 'old_background', 'old_validation', 'old_acquisition', 'all_validation') for r in cohort[k]}
    old_q = {i: q for i, q in enumerate(sorted(old_rows))}
    catalog = {(r['model'], old_q[r['question_group']], r['view']): r for r in read(COHORT / 'public_bundle/physical_branch_numeric.json')}
    refs = {}
    values = {}
    policies = ['O', 'U42', 'U43', 'Q', 'G', 'S']
    for q, row in rows.items():
        if not m['admit'][str(q)]:
            flags = m['background_flags'][str(q)]
            label = 'A_42' if flags['A'] else ('I_42' if flags['B'] else 'base')
            ref = catalog[label, q, 'N']
            assert old_rows[q]['answer'] == row['answer'] and old_rows[q]['img_name'] == row['img_name']
            refs[q] = ref
            for policy in policies: values[policy, q] = int(ref['correct'])
        else:
            for policy in policies:
                selected = 'T_42' if policy == 'U42' or (policy in ['Q', 'G', 'S'] and decisions[str(q)][{'Q': 'consensus', 'G': 'guard', 'S': 'sham'}[policy]]) else ('T_43' if policy == 'U43' else 'base')
                values[policy, q] = int(score(raw(selected, q, 'N'), row['answer']))
    qmap = {q: i for i, q in enumerate(sorted(rows))}
    def effect(qs, a, b):
        delta = np.array([values[a, q] - values[b, q] for q in qs])
        assert len(qs) == len({rows[q]['img_name'] for q in qs})
        sample = np.random.default_rng(p['selection_seed']).integers(0, len(qs), (10000, len(qs)))
        return {'mean': float(delta.mean()), 'image_bootstrap_95': np.quantile(delta[sample].mean(1), [.025, .975]).tolist(), 'gains': [qmap[q] for q, d in zip(qs, delta) if d > 0], 'harms': [qmap[q] for q, d in zip(qs, delta) if d < 0]}
    summary, effects, coverage = {}, {}, {}
    for panel, rr in [('primary', m['primary']), ('background', m['background'])]:
        qs = [r['qid'] for r in rr]
        summary[panel] = {'questions': len(qs), 'correct': {policy: sum(values[policy, q] for q in qs) for policy in policies}}
        coverage[panel] = {name: sum(decisions[str(q)][name] for q in qs) for name in ['consensus', 'guard', 'sham']}
        for a, b in [('G', 'O'), ('G', 'S'), ('Q', 'O'), ('U42', 'O'), ('U43', 'O')]: effects[f'{panel}_{a}_minus_{b}'] = effect(qs, a, b)
        e = effects[f'{panel}_G_minus_O']
        assert abs(e['mean'] - (len(e['gains']) - len(e['harms'])) / len(qs)) < 1e-12
    primary, sham, bg = [effects[k] for k in ['primary_G_minus_O', 'primary_G_minus_S', 'background_G_minus_O']]
    gates = {'primary_gain_at_least_10pp': primary['mean'] >= .10, 'primary_positive_95_lower': primary['image_bootstrap_95'][0] > 0, 'primary_zero_new_harm': not primary['harms'], 'primary_coverage_at_least_10pct': coverage['primary']['guard'] / 85 >= .10, 'sham_positive_mean': sham['mean'] > 0, 'sham_positive_95_lower': sham['image_bootstrap_95'][0] > 0, 'background_nonnegative_mean': bg['mean'] >= 0, 'background_95_lower_at_least_minus_5pp': bg['image_bootstrap_95'][0] >= -.05, 'background_zero_new_harm': not bg['harms']}
    result = {'status': 'COMPLETE_CONSENSUS_GUARD_PASS' if all(gates.values()) else 'COMPLETE_CONSENSUS_GUARD_NOT_ESTABLISHED', 'gates': gates, 'all_gates_pass': all(gates.values()), 'summary': summary, 'effects': effects, 'coverage': coverage, 'cost': {'new_generations': 476, 'cached_physical_records': 193, 'logical_predictor_readouts': 848, 'new_optimizer_steps': 0, 'errors': 0, 'replays': 0, 'supervisor_seconds': end['end_unix'] - receipt['start_unix'], 'original_preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes': end['incremental_bytes']}, 'test_annotation_opened': False, 'limitations': ['One joint policy from two frozen readers, not two independent G seeds.', 'Update-image-isolated official train source, known templates, unknown pretraining/project exposure and patient independence.', 'Background previously observed; external unchanged branches are not shared-parameter retention.', 'Black fallback follows the rule by construction; wrong-image answers are not clinical negative labels.', 'No full ViSkill/PPO/clinical claim.']}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    for name, obj in [('result.json', result), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('readout_checks.json', {x: read(OUT / (x + '_READOUT_CHECK.json')) for x in ['BASE', 'T_42', 'T_43']}), ('policy_numeric.json', [{'question_group': qmap[q], 'panel': 'primary' if q in {r['qid'] for r in m['primary']} else 'background', **decisions[str(q)], **{policy: values[policy, q] for policy in policies}} for q in sorted(rows)]), ('cache_provenance.json', [{k: r[k] for k in ['model', 'source_experiment', 'source_request_number']} for r in cache]), ('physical_numeric.json', [{'number': r['number'], 'model': r['model'], 'physical_index': r['physical_index'], 'output_tokens': len(r['output_ids'])} for r in records]), ('initialization_incident.json', read(OUT / 'initialization_incident.json'))]: write(bundle / name, obj)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': close()
