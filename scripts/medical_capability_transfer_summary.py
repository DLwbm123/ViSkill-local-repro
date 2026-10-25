"""Offline closure for separately frozen shared consolidation and module-bank objects."""
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'acquisition_to_organ_transfer_v1'
PARENT = ROOT / 'artifacts/medical/continual_acquisition_bank_v1'
OUT = ROOT / 'artifacts/medical' / NAME
sys.path.insert(0, str(SRC / 'scripts'))
from medical_continual_acquisition import reference_records
from medical_region_reader import read, write
from slake_probe import score


def close():
    p = read(SRC / 'configs/reproduction' / (NAME + '.json'))
    end = read(ROOT / 'state/reproduction' / (NAME + '_EXIT.json'))
    receipt = read(ROOT / 'state/reproduction' / (NAME + '_receipt.json'))
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert end['exit_code'] == 0 and end['reason'] is None and not end['argv_violations']
    assert end['end_unix'] < p['deadline_unix'] and end['incremental_bytes'] < p['maximum_incremental_bytes']
    assert done['status'] == 'COMPLETE_ACQUISITION_TO_ORGAN_TRANSFER'
    assert done['new_generation_calls'] == p['formal_generation_calls'] == 3216
    assert done['trained_adapters'] == 4 and done['optimizer_steps'] == 208 and not done['test_annotation_opened']
    teacher = read(OUT / 'TEACHER_COMPLETED.json')
    assert teacher['forward_calls'] == 100 and teacher['generation_calls'] == 0 and not teacher['validation_distilled'] and teacher['same_base_null_teacher_all_arms']
    assert len(list((OUT / 'teacher_cache').glob('*.pt'))) == 100
    m = read(OUT / 'private_manifest.json')
    panels = {k: sorted(r['qid'] for r in m[k]) for k in ('new_validation', 'old_source', 'old_background', 'old_validation', 'old_acquisition', 'all_validation')}
    assert [len(panels[k]) for k in panels] == [254, 69, 69, 39, 172, 1053]
    items = {r['qid']: r for k in panels for r in m[k]}
    assert not {r['img_name'] for r in m['train']} & {r['img_name'] for r in items.values()}
    contracts = read(OUT / 'private_contracts.json')
    flags = {int(q): f for q, f in contracts['flags'].items()}
    assert set(flags) == set(items) and all(sum(f.values()) <= 1 for f in flags.values())
    assert sum(flags[q]['C'] for q in panels['old_background']) == 21
    assert sum(flags[q]['C'] for q in panels['new_validation']) == 251
    assert all(not flags[q]['A'] and not flags[q]['B'] for q in panels['new_validation'])
    plan = read(OUT / 'schedule.json')
    new = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    key = lambda r: (r['model'], r['qid'], r['view'])
    assert len(new) == len(plan) == 3216 and [r['number'] for r in new] == list(range(1, 3217))
    assert [key(r) for r in new] == [key(r) for r in plan]
    assert len({key(r) for r in new}) == 3216
    events = [json.loads(s) for s in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(r['event'] for r in events) == {'intent': 3216, 'result': 3216}
    for event in ('intent', 'result'):
        assert [key(r) for r in events if r['event'] == event] == [key(r) for r in plan]
    assert (OUT / 'private_contracts.json').stat().st_mtime <= min(r['started_unix'] for r in events if r['event'] == 'intent')
    training, updates = {}, []
    for seed in p['training_seeds']:
        for arm in p['arms']:
            label = f'{arm}_{seed}'
            folder = OUT / label
            t = read(folder / 'TRAIN_COMPLETED.json')
            b = read(folder / 'production_backward_check.json')
            assert t['optimizer_steps'] == 52 and t['supervised_views'] == 800 and t['anchor_forward_views'] == 400 and t['teacher_family'] == 'base'
            assert t['parameter_delta_square_norm'] > 0 and math.isfinite(t['parameter_delta_square_norm'])
            assert not t['language_parameters_updated'] and t['visual_adapter_updated']
            assert b['status'] == 'PASS' and b['visual_trainable_only'] and b['finite_gradient_tensors'] > 0 and b['optimizer_steps'] == 0
            assert math.isfinite(b['initial_null_KL'])
            if arm == 'F':
                assert abs(b['initial_null_KL']) < 2e-4
            u = [json.loads(s) for s in (folder / 'updates.jsonl').read_text().splitlines()]
            assert [r['step'] for r in u] == list(range(1, 53))
            assert sum(r['supervised_views'] for r in u) == 800 and sum(r['anchor_forward_views'] for r in u) == 400
            assert all(math.isfinite(r[k]) for r in u for k in ('loss', 'supervised_loss', 'null_KL', 'preclip_gradient_norm'))
            assert all(r['null_coefficient'] == 1 for r in u)
            assert read(OUT / (label + '_EVALUATED.json'))['calls'] == 550
            training[label] = {'completion': t, 'production_backward': b}
            updates.extend(u)
    refs = reference_records()
    for f in sorted((PARENT / 'records').glob('*.json')):
        r = read(f)
        assert key(r) not in refs
        refs[key(r)] = dict(r, reference_origin=PARENT.name)
    assert not set(refs) & {key(r) for r in new}
    indexed = {key(r): r for r in new}
    required_refs = set()
    for q in items:
        required_refs.update(('base', q, v) for v in ('N', 'B'))
    for seed in p['training_seeds']:
        assert read(OUT / f'I_{seed}_EVALUATED.json')['calls'] == 508
        for q in items:
            for model in ([f'A_{seed}'] if flags[q]['A'] else ([f'I_{seed}'] if flags[q]['B'] else [])):
                required_refs.update((model, q, v) for v in ('N', 'B'))
    used = [dict(refs[k], reused_reference=True, source_experiment=refs[k]['reference_origin']) for k in sorted(required_refs)]
    records = used + [dict(r, reused_reference=False, source_experiment=NAME) for r in new]
    assert all(bool(score(r['raw'], items[r['qid']]['answer'])) == r['correct'] for r in records)
    indexed = {key(r): int(r['correct']) for r in records}
    assert len(indexed) == len(records)
    qmap = {q: i for i, q in enumerate(sorted(items))}
    imap = {x: i for i, x in enumerate(sorted({r['img_name'] for r in items.values()}))}
    seeds = p['training_seeds']

    def branch(policy, seed, q):
        if policy == 'base':
            return 'base'
        if policy in ('raw_I', 'raw_F', 'raw_T'):
            return f'{policy[-1]}_{seed}'
        if policy == 'parent_I':
            return f'I_{seed}' if flags[q]['C'] else 'base'
        assert policy in ('before', 'F', 'T')
        if flags[q]['A']:
            return f'A_{seed}'
        if flags[q]['B']:
            return f'I_{seed}'
        return f'{policy}_{seed}' if policy != 'before' and flags[q]['C'] else 'base'

    def val(policy, seed, q, view):
        return indexed[branch(policy, seed, q), q, view]

    def summary(qs, policies):
        return {policy: {v: {'questions': len(qs), 'correct_by_seed': [sum(val(policy, s, q, v) for q in qs) for s in seeds], 'mean_accuracy': float(np.mean([val(policy, s, q, v) for s in seeds for q in qs]))} for v in ('N', 'B')} for policy in policies}

    def effect(qs, terms):
        matrix = np.array([[sum(sign * val(policy, s, q, v) for policy, v, sign in terms) for q in qs] for s in seeds])
        averaged = matrix.mean(0)
        groups = defaultdict(list)
        for i, q in enumerate(qs):
            groups[items[q]['img_name']].append(i)
        groups = list(groups.values())
        num, den = np.array([averaged[g].sum() for g in groups]), np.array([len(g) for g in groups])
        sampled = np.random.default_rng(p['selection_seed']).integers(0, len(groups), size=(10000, len(groups)))
        boot = num[sampled].sum(1) / den[sampled].sum(1)
        return {'mean': float(averaged.mean()), 'image_cluster_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(), 'questions': len(qs), 'images': len(groups), 'by_training_seed': [{'seed': s, 'mean': float(d.mean()), 'improved_question_groups': [qmap[q] for q, x in zip(qs, d) if x > 0], 'harmed_question_groups': [qmap[q] for q, x in zip(qs, d) if x < 0]} for s, d in zip(seeds, matrix)]}

    diff = lambda a, b: [(a, 'N', 1), (b, 'N', -1)]
    visual = lambda a, b: [(a, 'N', 1), (a, 'B', -1), (b, 'N', -1), (b, 'B', 1)]
    effects = {}
    for name, terms in [('T_minus_F', diff('T', 'F')), ('T_minus_parentI', diff('T', 'parent_I')), ('T_minus_before', diff('T', 'before')), ('F_minus_before', diff('F', 'before')), ('T_F_visual_interaction', visual('T', 'F')), ('raw_I_minus_base', diff('raw_I', 'base'))]:
        effects['new_' + name] = effect(panels['new_validation'], terms)
    for panel in ('old_source', 'old_validation', 'old_acquisition', 'old_background', 'all_validation'):
        for policy in ('T', 'F'):
            effects[panel + '_' + policy + '_minus_before'] = effect(panels[panel], diff(policy, 'before'))
    gates = {}
    for name, e, threshold in [('reuse_gain_at_least_5pp', effects['new_T_minus_F'], .05), ('reuse_visual_interaction', effects['new_T_F_visual_interaction'], 0), ('learning_from_I_at_least_10pp', effects['new_T_minus_parentI'], .10), ('gain_over_prior_bank_at_least_10pp', effects['new_T_minus_before'], .10)]:
        gates[name + '_mean'] = e['mean'] >= threshold if threshold else e['mean'] > 0
        gates[name + '_each_seed_positive'] = all(r['mean'] > 0 for r in e['by_training_seed'])
        gates[name + '_positive_95_lower'] = e['image_cluster_bootstrap_95'][0] > 0
    for panel in ('old_source', 'old_validation', 'old_acquisition'):
        gates[panel + '_exact_immutable_branch'] = all(branch('T', seed, q) == branch('before', seed, q) for seed in seeds for q in panels[panel])
    old = effects['old_background_T_minus_before']
    gates['old_background_mean_nonnegative'] = old['mean'] >= 0
    gates['old_background_each_seed_nonnegative'] = all(r['mean'] >= 0 for r in old['by_training_seed'])
    gates['old_background_95_lower_at_least_minus_5pp'] = old['image_cluster_bootstrap_95'][0] >= -.05
    gates['old_background_zero_harm_each_seed'] = all(not r['harmed_question_groups'] for r in old['by_training_seed'])
    summaries = {panel: summary(qs, ['base', 'before', 'F', 'T'] + (['raw_I', 'raw_F', 'raw_T', 'parent_I'] if panel == 'new_validation' else [])) for panel, qs in panels.items()}
    groups = {}
    for field in ('content_type', 'modality'):
        groups[field] = {}
        for kind in sorted({items[q][field] for q in panels['new_validation']}):
            qs = [q for q in panels['new_validation'] if items[q][field] == kind]
            groups[field][kind] = {'summary': summary(qs, ['base', 'raw_I', 'raw_F', 'raw_T', 'parent_I', 'F', 'T']), 'T_minus_F': effect(qs, diff('T', 'F')), 'T_minus_before': effect(qs, diff('T', 'before'))}
    passed = all(gates.values())
    result = {'status': 'COMPLETE_CAPABILITY_TRANSFER_GATE_PASS' if passed else 'COMPLETE_CAPABILITY_TRANSFER_NOT_ESTABLISHED', 'all_gates_pass': passed, 'gates': gates, 'summary': summaries, 'paired_effects': effects, 'new_capability_groups': groups, 'training': training, 'calls': {'new_generations': 3216, 'actual_intents': 3216, 'required_reference_records_reused': len(used), 'reference_catalog_available': len(refs), 'extra_policy_generations': 0, 'errors': 0, 'completed_replays': 0, 'teacher_forwards': 100, 'teacher_generations': 0, 'actual_optimizer_steps': 208, 'new_truncated_at_64': sum(len(r['output_ids']) == 64 for r in new)}, 'cost': {'supervisor_wall_seconds': end['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': end['incremental_bytes']}, 'limitations': ['Development images/questions and known task templates, not blind/test replication.', 'A/B external immutable retention is not same-adapter retention; all21changed background cases kept.', 'T inherits52priorsteps and22sharedTRAINimages; only new-stage optimization budgets are matched, so unique acquisition semantics are not isolated.', 'All254new questions retained in primary;3uncovered queries use base in deployed policies.', 'Unknown patient independence and unverified author QA labels; image-cluster intervals.', 'No full ViSkill, PPO, visual-card, clinical trajectory or deployment safety claim.'], 'test_annotation_opened': False}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    numeric = [{'source_experiment': r['source_experiment'], 'source_request_number': r['number'], 'reused_reference': r['reused_reference'], 'model': r['model'], 'view': r['view'], 'correct': r['correct'], 'question_group': qmap[r['qid']], 'image_group': imap[items[r['qid']]['img_name']], 'output_tokens': len(r['output_ids'])} for r in records]
    registry = [{'question_group': qmap[q], 'image_group': imap[items[q]['img_name']], 'contract_A': flags[q]['A'], 'contract_B': flags[q]['B'], 'contract_C': flags[q]['C'], 'panels': [k for k, qs in panels.items() if q in qs], 'content_type': items[q]['content_type'], 'modality': items[q]['modality']} for q in sorted(items)]
    for name, obj in [('result.json', result), ('physical_branch_numeric.json', numeric), ('contract_numeric.json', registry), ('updates.json', updates), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('execution_source.json', read(OUT / 'execution_source.json'))]:
        write(bundle / name, obj)
    print(json.dumps({'all_gates_pass': passed, 'gates': gates, 'summary': summaries, 'effects': effects, 'cost': result['cost']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    close()
