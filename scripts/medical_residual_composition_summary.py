"""Paired closure for unit residual addition; reuse verified original-prompt references."""
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'unit_residual_composition_v1'
OUT = ROOT / 'artifacts/medical' / NAME
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
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
    assert done['status'] == 'COMPLETE_UNIT_RESIDUAL_COMPOSITION' and done['new_generation_calls'] == p['formal_generation_calls'] == 2176
    assert done['new_optimizer_steps'] == done['saved_model_checkpoints'] == 0 and done['constructed_adapters'] == 4 and not done['test_annotation_opened']
    proofs = read(OUT / 'COMPOSITION_CHECK.json')
    assert set(proofs) == {f'{a}_{s}' for s in p['training_seeds'] for a in ('SUM', 'DOUBLE')}
    assert all(r['layers_checked'] == 64 and r['constructed_rank'] == 32 and r['actual_delta_is_exact_sum_within_fp32_tolerance'] and math.isfinite(r['maximum_probe_relative_error']) and r['optimizer_steps'] == 0 for r in proofs.values())
    m = read(OUT / 'private_manifest.json')
    items = {r['qid']: r for r in m['validation'] + m['background']}
    assert len(items) == 323 and sum(f['C'] for f in m['flags'].values()) == 272
    plan = read(OUT / 'schedule.json')
    records = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    key = lambda r: (r['model'], r['qid'], r['view'])
    assert len(records) == len(plan) == 2176 and [r['number'] for r in records] == list(range(1, 2177))
    assert [key(r) for r in records] == [key(r) for r in plan] and len({key(r) for r in records}) == 2176
    events = [json.loads(s) for s in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(e['event'] for e in events) == {'intent': 2176, 'result': 2176}
    for event in ('intent', 'result'):
        assert [key(r) for r in events if r['event'] == event] == [key(r) for r in plan]
    assert (OUT / 'schedule.json').stat().st_mtime <= min(r['started_unix'] for r in events if r['event'] == 'intent')
    for label in proofs:
        assert read(OUT / (label + '_EVALUATED.json'))['calls'] == 544
    indexed = {key(r): int(score(r['raw'], items[r['qid']]['answer'])) for r in records}
    cohort = read(COHORT / 'private_manifest.json')
    old_items = {r['qid']: r for k in ('new_validation', 'old_source', 'old_background', 'old_validation', 'old_acquisition', 'all_validation') for r in cohort[k]}
    assert len(old_items) == 1191
    assert all(items[q]['answer'] == old_items[q]['answer'] and items[q]['img_name'] == old_items[q]['img_name'] for q in items)
    old_q = {i: q for i, q in enumerate(sorted(old_items))}
    seeds = p['training_seeds']
    def branch(policy, seed, q):
        flag = m['flags'][str(q)]
        before = f'A_{seed}' if flag['A'] else (f'I_{seed}' if flag['B'] else 'base')
        if policy == 'base': return 'base'
        if policy == 'before' or not flag['C']: return before
        assert policy in ('SUM', 'DOUBLE', 'F', 'T', 'I')
        return f'{policy}_{seed}'
    primary = sorted(r['qid'] for r in m['validation'])
    background = sorted(r['qid'] for r in m['background'])
    required = set()
    for q in items:
        for seed in seeds:
            for policy in ('base', 'before', 'F', 'T') + (('I',) if q in primary else ()):
                required.update((branch(policy, seed, q), q, v) for v in ('N', 'B'))
    catalog = read(COHORT / 'public_bundle/physical_branch_numeric.json')
    legacy = {(r['model'], old_q[r['question_group']], r['view']): r for r in catalog if (r['model'], old_q[r['question_group']], r['view']) in required}
    assert set(legacy) == required and len(legacy) == 3870
    indexed.update({k: int(r['correct']) for k, r in legacy.items()})
    qmap = {q: i for i, q in enumerate(sorted(items))}
    imap = {n: i for i, n in enumerate(sorted({r['img_name'] for r in items.values()}))}
    def value(policy, seed, q, view): return indexed[branch(policy, seed, q), q, view]
    def summarize(qs, policies):
        return {policy: {v: {'questions': len(qs), 'correct_by_parent_seed': [sum(value(policy, s, q, v) for q in qs) for s in seeds]} for v in ('N', 'B')} for policy in policies}
    def effect(qs, terms):
        matrix = np.array([[sum(sign * value(policy, s, q, v) for policy, v, sign in terms) for q in qs] for s in seeds])
        averaged = matrix.mean(0)
        groups = defaultdict(list)
        for i, q in enumerate(qs): groups[items[q]['img_name']].append(i)
        groups = list(groups.values())
        num, den = np.array([averaged[g].sum() for g in groups]), np.array([len(g) for g in groups])
        sampled = np.random.default_rng(p['selection_seed']).integers(0, len(groups), size=(10000, len(groups)))
        boot = num[sampled].sum(1) / den[sampled].sum(1)
        return {'mean': float(averaged.mean()), 'image_cluster_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(), 'questions': len(qs), 'images': len(groups), 'by_paired_parent_seed': [{'seed': s, 'mean': float(d.mean()), 'improved_question_groups': [qmap[q] for q, x in zip(qs, d) if x > 0], 'harmed_question_groups': [qmap[q] for q, x in zip(qs, d) if x < 0]} for s, d in zip(seeds, matrix)]}
    diff = lambda a, b: [(a, 'N', 1), (b, 'N', -1)]
    effects = {f'primary_SUM_minus_{control}': effect(primary, diff('SUM', control)) for control in ('before', 'I', 'F', 'DOUBLE', 'T')}
    effects['primary_SUM_F_visual_interaction'] = effect(primary, [('SUM', 'N', 1), ('SUM', 'B', -1), ('F', 'N', -1), ('F', 'B', 1)])
    effects['background_SUM_minus_before'] = effect(background, diff('SUM', 'before'))
    effects['background_DOUBLE_minus_before'] = effect(background, diff('DOUBLE', 'before'))
    gates = {}
    for name, threshold in [('primary_SUM_minus_before', .10), ('primary_SUM_minus_I', .10), ('primary_SUM_minus_F', .05), ('primary_SUM_minus_DOUBLE', 0), ('primary_SUM_F_visual_interaction', 0)]:
        e = effects[name]
        gates[name + '_mean'] = e['mean'] >= threshold if threshold else e['mean'] > 0
        gates[name + '_each_seed_positive'] = all(r['mean'] > 0 for r in e['by_paired_parent_seed'])
        gates[name + '_positive_95_lower'] = e['image_cluster_bootstrap_95'][0] > 0
    e = effects['background_SUM_minus_before']
    gates.update(background_mean_nonnegative=e['mean'] >= 0, background_each_seed_nonnegative=all(r['mean'] >= 0 for r in e['by_paired_parent_seed']), background_95_lower_at_least_minus_5pp=e['image_cluster_bootstrap_95'][0] >= -.05, background_zero_new_harm_each_seed=all(not r['harmed_question_groups'] for r in e['by_paired_parent_seed']))
    flags = read(COHORT / 'private_contracts.json')['flags']
    gates['protected_A_B_branches_unchanged'] = all(not flags[str(r['qid'])]['C'] for k in ('old_source', 'old_validation', 'old_acquisition') for r in cohort[k])
    summaries = {'primary': summarize(primary, ['base', 'before', 'I', 'F', 'T', 'SUM', 'DOUBLE']), 'old_background': summarize(background, ['base', 'before', 'F', 'T', 'SUM', 'DOUBLE'])}
    groups = {kind: {'summary': summarize(qs, ['base', 'I', 'F', 'T', 'SUM', 'DOUBLE']), 'SUM_minus_F': effect(qs, diff('SUM', 'F'))} for kind in sorted({items[q]['modality'] for q in primary}) for qs in [[q for q in primary if items[q]['modality'] == kind]]}
    result = {'status': 'COMPLETE_UNIT_COMPOSITION_GATE_PASS' if all(gates.values()) else 'COMPLETE_UNIT_COMPOSITION_NOT_ESTABLISHED', 'all_gates_pass': all(gates.values()), 'gates': gates, 'summary': summaries, 'paired_effects': effects, 'groups': groups, 'composition_check': proofs, 'calls': {'new_generations': 2176, 'verified_numeric_reference_branches_reused': len(legacy), 'policy_outcomes_new_and_reused': 2584, 'new_optimizer_steps': 0, 'new_teacher_forwards': 0, 'constructed_RAM_adapters': 4, 'saved_model_checkpoints': 0, 'errors': 0, 'completed_replays': 0}, 'cost': {'supervisor_wall_seconds': end['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': end['incremental_bytes']}, 'limitations': ['Original observed development cohort, unknown patient independence and unverified author QA.', 'Exact FP32 residual identity does not imply nonlinear logit additivity or clinical benefit.', 'Rank32 SUM and DOUBLE have equal factor counts but DOUBLE repeats factors and has effective rank at most16.', 'SUM inherits I52+F52 prior updates; T has the same stage count but rank16 inference. No fully equal capacity/history attribution.', 'Old A/B immutable branch retention is external, not shared-parameter forgetting protection.', 'Previous reference correctness was reused as verified numeric evidence, not recomputed or regenerated.', 'No full ViSkill, PPO, clinical workflow or visual-card claim.'], 'test_annotation_opened': False}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    numeric = [{'number': r['number'], 'model': r['model'], 'view': r['view'], 'question_group': qmap[r['qid']], 'image_group': imap[items[r['qid']]['img_name']], 'correct': bool(indexed[key(r)]), 'output_tokens': len(r['output_ids'])} for r in records]
    old_numeric = [dict(r, question_group=qmap[q], image_group=imap[items[q]['img_name']]) for (model, q, view), r in sorted(legacy.items())]
    for name, obj in [('result.json', result), ('physical_numeric.json', numeric), ('reference_numeric.json', old_numeric), ('composition_check.json', proofs), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('execution_source.json', read(OUT / 'execution_source.json'))]: write(bundle / name, obj)
    print(json.dumps({'status': result['status'], 'gates': gates, 'summary': summaries, 'effects': effects, 'cost': result['cost']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': close()
