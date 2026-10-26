"""Offline closure of a frozen visual representation and paired language readouts."""
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'frozen_visual_readout_v1'
OUT = ROOT / 'artifacts/medical' / NAME
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
SOURCE = ROOT / 'artifacts/medical/visual_consensus_guard_v1'
sys.path.insert(0, str(SRC / 'scripts'))
from medical_region_reader import read, write
from slake_probe import score


def close():
    p = read(SRC / 'configs/reproduction' / (NAME + '.json'))
    end = read(ROOT / 'state/reproduction' / (NAME + '_EXIT.json'))
    receipt = read(ROOT / 'state/reproduction' / (NAME + '_receipt.json'))
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert end['exit_code'] == 0 and end['reason'] is None and not end['argv_violations']
    assert end['end_unix'] < p['deadline_unix'] and end['incremental_bytes'] < p['maximum_incremental_bytes']
    assert done['status'] == 'COMPLETE_FROZEN_VISUAL_READOUT' and done['new_generation_calls'] == p['formal_generation_calls'] == 1852
    assert done['trained_readout_adapters'] == 4 and done['optimizer_steps'] == 208 and done['new_teacher_forwards'] == 0 and not done['test_annotation_opened']
    proof = read(OUT / 'production_backward_check.json')
    assert proof['status'] == 'PASS' and proof['language_readout_only_finite_gradients'] and proof['frozen_visual_I_layers'] == 64
    assert proof['gradient_tensors'] == 288 and proof['optimizer_steps'] == proof['new_generation_calls'] == 0
    training, updates = {}, []
    for seed in p['training_seeds']:
        for arm in ['F', 'R']:
            label = f'{arm}_{seed}'
            training[label] = read(OUT / label / 'TRAIN_COMPLETED.json')
            t = training[label]
            assert t['optimizer_steps'] == 52 and t['supervised_views'] == 800 and t['null_anchor_views'] == 400
            assert t['all_visual_parameters_frozen'] and t['visual_parent_factors_exactly_unchanged']
            assert t['parameter_delta_square_norm'] > 0 and math.isfinite(t['parameter_delta_square_norm'])
            assert t['language_readout_parameters'] == 7372800
            if arm == 'R': assert t['paired_readout_initialization_equal']
            u = [json.loads(s) for s in (OUT / label / 'updates.jsonl').read_text().splitlines()]
            assert len(u) == 52 and [r['step'] for r in u] == list(range(1, 53))
            assert all(math.isfinite(r[k]) for r in u for k in ['loss', 'CE', 'null_KL', 'gradient_norm'])
            assert sum(r['supervised_views'] for r in u) == 800 and sum(r['null_anchor_views'] for r in u) == 400
            updates += u
    m = read(OUT / 'private_manifest.json')
    rows = {r['qid']: r for k in ['primary', 'validation', 'background'] for r in m[k]}
    panels = {k: [r['qid'] for r in m[k]] for k in ['primary', 'validation', 'background']}
    assert len(rows) == 408 and [len(panels[k]) for k in ['primary', 'validation', 'background']] == [85, 254, 69]
    assert len(m['active_ids']) == 357
    cache, bindings, plan = [read(OUT / n) for n in ['private_cached_records.json', 'private_bindings.json', 'physical_plan.json']]
    records = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(cache) == 46 and len(bindings) == 3196 and len(plan) == len(records) == 1852
    assert [r['number'] for r in records] == list(range(1, 1853))
    assert [r['physical_index'] for r in records] == list(range(1852))
    assert all(all(r[k] == e[k] for k in ['physical_index', 'model', 'qid', 'view']) for r, e in zip(records, plan))
    events = [json.loads(s) for s in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(r['event'] for r in events) == {'intent': 1852, 'result': 1852}
    for event in ['intent', 'result']:
        assert [r['physical_index'] for r in events if r['event'] == event] == list(range(1852))
    assert (OUT / 'physical_plan.json').stat().st_mtime < min(r['started_unix'] for r in events if r['event'] == 'intent')
    assert {b['physical_index'] for b in bindings.values() if 'physical_index' in b} == set(range(1852))
    assert {b['cache_index'] for b in bindings.values() if 'cache_index' in b} == set(range(46))
    for label in ['I_42', 'F_42', 'R_42', 'I_43', 'F_43', 'R_43']:
        assert read(OUT / (label + '_EVALUATED.json'))['calls'] == sum(r['model'] == label for r in plan)
    def raw(label, q, view):
        b = bindings[f'{label}|{q}|{view}']
        return cache[b['cache_index']]['raw'] if 'cache_index' in b else records[b['physical_index']]['raw']
    logical = {(label, int(q), v): int(score(raw(label, int(q), v), rows[int(q)]['answer'])) for key in bindings for label, q, v in [key.split('|')]}
    c = read(COHORT / 'private_manifest.json')
    old = {r['qid']: r for k in ['new_validation', 'old_source', 'old_background', 'old_validation', 'old_acquisition', 'all_validation'] for r in c[k]}
    assert len(old) == 1191
    old_q = {i: q for i, q in enumerate(sorted(old))}
    catalog = {(r['model'], old_q[r['question_group']], r['view']): r for r in read(COHORT / 'public_bundle/physical_branch_numeric.json')}
    required = set()
    def before_model(seed, q):
        flags = m['flags'][str(q)]
        return f'A_{seed}' if flags.get('A') else (f'I_{seed}' if flags.get('B') else 'base')
    for q in panels['validation'] + panels['background']:
        assert rows[q]['img_name'] == old[q]['img_name'] and rows[q]['answer'] == old[q]['answer']
        for seed in [42, 43]:
            for view in ['N', 'B']:
                required.update([('base', q, view), (before_model(seed, q), q, view)])
    legacy = {k: catalog[k] for k in required}
    g = read(SOURCE / 'private_manifest.json')
    grow = {r['qid']: r for k in ['primary', 'background'] for r in g[k]}
    gmap = {i: q for i, q in enumerate(sorted(grow))}
    gpolicy = {gmap[r['question_group']]: r for r in read(SOURCE / 'public_bundle/policy_numeric.json')}
    gb, gc = read(SOURCE / 'private_bindings.json'), read(SOURCE / 'private_cached_records.json')
    base_primary, base_provenance = {}, []
    for q in panels['primary']:
        assert rows[q]['img_name'] == grow[q]['img_name'] and rows[q]['answer'] == grow[q]['answer']
        for view in ['N', 'B']:
            ref = gb[f'base|{q}|{view}']
            if 'cache_index' in ref:
                rr = gc[ref['cache_index']]
            else:
                rr = read(SOURCE / 'records' / f"{ref['physical_index'] + 1:04d}.json")
            base_primary[q, view] = int(score(rr['raw'], rows[q]['answer']))
            base_provenance.append({'qid_private': q, 'view': view, 'source_experiment': rr.get('source_experiment', SOURCE.name), 'source_request_number': rr.get('source_request_number', rr.get('number')), 'correct': bool(base_primary[q, view])})
        assert base_primary[q, 'N'] == gpolicy[q]['O']
    assert sum(base_primary[q, 'N'] for q in panels['primary']) == 52
    def value(policy, seed, q, view):
        if policy == 'base':
            return base_primary[q, view] if q in panels['primary'] else int(legacy['base', q, view]['correct'])
        if policy == 'I_parent':
            assert q in panels['primary']
            return logical[f'I_{seed}', q, view]
        if policy in ['F', 'R'] and q in m['active_ids']: return logical[f'{policy}_{seed}', q, view]
        assert q not in panels['primary']
        return int(legacy[before_model(seed, q), q, view]['correct'])
    qmap = {q: i for i, q in enumerate(sorted(rows))}
    imap = {n: i for i, n in enumerate(sorted({r['img_name'] for r in rows.values()}))}
    def summary(qs, policies):
        return {policy: {v: [sum(value(policy, s, q, v) for q in qs) for s in [42, 43]] for v in ['N', 'B']} for policy in policies}
    def effect(qs, terms):
        matrix = np.array([[sum(sign * value(policy, s, q, v) for policy, v, sign in terms) for q in qs] for s in [42, 43]])
        average = matrix.mean(0)
        groups = defaultdict(list)
        for i, q in enumerate(qs): groups[rows[q]['img_name']].append(i)
        groups = list(groups.values())
        num, den = np.array([average[g].sum() for g in groups]), np.array([len(g) for g in groups])
        sampled = np.random.default_rng(p['selection_seed']).integers(0, len(groups), (10000, len(groups)))
        boot = num[sampled].sum(1) / den[sampled].sum(1)
        return {'mean': float(average.mean()), 'image_cluster_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(), 'questions': len(qs), 'images': len(groups), 'paired_parent_seeds': [{'seed': s, 'mean': float(d.mean()), 'gains': [qmap[q] for q, x in zip(qs, d) if x > 0], 'harms': [qmap[q] for q, x in zip(qs, d) if x < 0]} for s, d in zip([42, 43], matrix)]}
    diff = lambda a, b: [(a, 'N', 1), (b, 'N', -1)]
    effects = {f'primary_R_minus_{b}': effect(panels['primary'], diff('R', b)) for b in ['F', 'I_parent', 'base']}
    effects['primary_R_F_visual_interaction'] = effect(panels['primary'], [('R', 'N', 1), ('R', 'B', -1), ('F', 'N', -1), ('F', 'B', 1)])
    effects['validation_R_minus_F'] = effect(panels['validation'], diff('R', 'F'))
    effects['validation_R_minus_before'] = effect(panels['validation'], diff('R', 'before'))
    effects['background_R_minus_before'] = effect(panels['background'], diff('R', 'before'))
    effects['background_F_minus_before'] = effect(panels['background'], diff('F', 'before'))
    gates = {}
    for name, threshold in [('primary_R_minus_F', .05), ('primary_R_minus_I_parent', .10), ('primary_R_minus_base', .10), ('primary_R_F_visual_interaction', 0)]:
        e = effects[name]
        gates[name + '_mean'] = e['mean'] >= threshold if threshold else e['mean'] > 0
        gates[name + '_each_seed_positive'] = all(r['mean'] > 0 for r in e['paired_parent_seeds'])
        gates[name + '_positive_95_lower'] = e['image_cluster_bootstrap_95'][0] > 0
    e = effects['validation_R_minus_F']
    gates.update(validation_R_F_mean_nonnegative=e['mean'] >= 0, validation_R_F_each_seed_nonnegative=all(r['mean'] >= 0 for r in e['paired_parent_seeds']), validation_R_F_lower_at_least_minus_5pp=e['image_cluster_bootstrap_95'][0] >= -.05)
    e = effects['background_R_minus_before']
    gates.update(background_mean_nonnegative=e['mean'] >= 0, background_each_seed_nonnegative=all(r['mean'] >= 0 for r in e['paired_parent_seeds']), background_lower_at_least_minus_5pp=e['image_cluster_bootstrap_95'][0] >= -.05, background_zero_new_harm_each_seed=all(not r['harms'] for r in e['paired_parent_seeds']))
    flags = read(COHORT / 'private_contracts.json')['flags']
    gates['old_A_B_external_branches_unchanged'] = all(not flags[str(r['qid'])]['C'] for k in ['old_source', 'old_validation', 'old_acquisition'] for r in c[k])
    summaries = {k: summary(qs, ['base', 'I_parent', 'F', 'R'] if k == 'primary' else ['base', 'before', 'F', 'R']) for k, qs in panels.items()}
    result = {'status': 'COMPLETE_FROZEN_READOUT_GATE_PASS' if all(gates.values()) else 'COMPLETE_FROZEN_READOUT_NOT_ESTABLISHED', 'all_gates_pass': all(gates.values()), 'gates': gates, 'summary': summaries, 'effects': effects, 'training': training, 'cost': {'new_generations': 1852, 'logical_predictor_readouts': 3196, 'cached_parent_physical_records': 46, 'reused_teacher_records': 100, 'new_teacher_forwards': 0, 'new_optimizer_steps': 208, 'readout_adapters_trained': 4, 'supervisor_wall_seconds': end['end_unix'] - receipt['start_unix'], 'original_preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes': end['incremental_bytes'], 'errors': 0, 'replays': 0}, 'limitations': ['Observed source85 and validation254, known train templates, unknown patient independence and historical exposure.', 'I has extra historical52updates/fixed visual factors.22Ctraining images overlap acquisition supervision;40overlap acquisition supervision or nonzero old KL, per separate exposure audit. Not isolated acquisition semantics or equal lifetime cost.', 'External unchanged old branches are distinct from single-readout retention.', 'Frozen I and a finite language gradient are implementation evidence, not performance or clinical benefit.', 'No full ViSkill, PPO, visual-card or clinical workflow claim.'], 'test_annotation_opened': False}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    numeric = [{'model': label, 'question_group': qmap[q], 'image_group': imap[rows[q]['img_name']], 'view': v, 'correct': bool(value), **bindings[f'{label}|{q}|{v}']} for (label, q, v), value in sorted(logical.items())]
    reference = [dict(rr, question_group=qmap[q], image_group=imap[rows[q]['img_name']]) for (label, q, view), rr in sorted(legacy.items())]
    base_numeric = [{k: v for k, v in rr.items() if k != 'qid_private'} | {'question_group': qmap[rr['qid_private']], 'image_group': imap[rows[rr['qid_private']]['img_name']]} for rr in base_provenance]
    modality = {kind: summary([q for q in panels['primary'] if rows[q]['modality'] == kind], ['base', 'I_parent', 'F', 'R']) for kind in sorted({rows[q]['modality'] for q in panels['primary']})}
    for name, obj in [('result.json', result), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('production_backward_check.json', proof), ('training_updates.json', updates), ('logical_branch_numeric.json', numeric), ('reference_numeric.json', reference), ('base_source_numeric.json', base_numeric), ('cache_provenance.json', [{k: r[k] for k in ['model', 'source_experiment', 'source_request_number']} for r in cache]), ('modality_numeric.json', modality), ('execution_source.json', read(OUT / 'execution_source.json'))]: write(bundle / name, obj)
    print(json.dumps({'status': result['status'], 'gates': gates, 'summary': summaries, 'effects': effects, 'cost': result['cost']}), flush=True)


if __name__ == '__main__': close()
