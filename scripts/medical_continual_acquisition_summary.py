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
NAME = 'continual_acquisition_bank_v1'
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
    assert done['status'] == 'COMPLETE_CONTINUAL_ACQUISITION_BANK'
    assert done['new_generation_calls'] == p['formal_generation_calls'] == 4188
    assert done['trained_adapters'] == 6 and done['optimizer_steps'] == 312 and not done['test_annotation_opened']
    teacher = read(OUT / 'TEACHER_COMPLETED.json')
    assert teacher['forward_calls'] == 600 and teacher['generation_calls'] == 0 and not teacher['validation_distilled']
    assert len(list((OUT / 'teacher_cache').glob('*.pt'))) == 600
    m = read(OUT / 'private_manifest.json')
    panels = {k: sorted(r['qid'] for r in m[k]) for k in ('new_validation', 'old_source', 'old_background', 'old_validation', 'all_validation')}
    assert [len(panels[k]) for k in panels] == [172, 69, 69, 39, 1053]
    items = {r['qid']: r for k in panels for r in m[k]}
    assert not {r['img_name'] for r in m['train'] + m['old_train_replay']} & {r['img_name'] for r in items.values()}
    contracts = read(OUT / 'private_contracts.json')
    flags = {int(q): f for key in ('evaluation_flags', 'validation_flags') for q, f in contracts[key].items()}
    assert set(flags) == set(items) and all(not (f['A'] and f['B']) for f in flags.values())
    assert sum(flags[q]['B'] for q in panels['old_background']) == 11
    assert all(flags[q]['B'] for q in panels['new_validation'])
    assert all(flags[q]['A'] for k in ('old_source', 'old_validation') for q in panels[k])
    plan = read(OUT / 'schedule.json')
    new = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    key = lambda r: (r['model'], r['qid'], r['view'])
    assert len(new) == len(plan) == 4188 and [r['number'] for r in new] == list(range(1, 4189))
    assert [key(r) for r in new] == [key(r) for r in plan]
    assert len({key(r) for r in new}) == 4188
    events = [json.loads(s) for s in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(r['event'] for r in events) == {'intent': 4188, 'result': 4188}
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
            assert t['optimizer_steps'] == 52 and t['supervised_views'] == t['anchor_forward_views'] == 800
            assert t['parameter_delta_square_norm'] > 0 and math.isfinite(t['parameter_delta_square_norm'])
            assert not t['language_parameters_updated'] and t['visual_adapter_updated']
            assert b['status'] == 'PASS' and b['visual_trainable_only'] and b['finite_gradient_tensors'] > 0 and b['optimizer_steps'] == 0
            assert abs(b['initial_null_KL']) < 2e-4 and abs(b['initial_old_KL']) < 2e-4
            u = [json.loads(s) for s in (folder / 'updates.jsonl').read_text().splitlines()]
            assert [r['step'] for r in u] == list(range(1, 53))
            assert sum(r['supervised_views'] for r in u) == sum(r['anchor_forward_views'] for r in u) == 800
            assert all(math.isfinite(r[k]) for r in u for k in ('loss', 'supervised_loss', 'null_KL', 'old_KL', 'preclip_gradient_norm'))
            assert all(r['old_coefficient'] == (0 if arm == 'S' else 1) for r in u)
            assert read(OUT / (label + '_EVALUATED.json'))['calls'] == (366 if arm == 'I' else 698)
            training[label] = {'completion': t, 'production_backward': b}
            updates.extend(u)
    refs = reference_records()
    assert not set(refs) & {key(r) for r in new}
    indexed = {key(r): r for r in new}
    required_refs = set()
    for q in items:
        required_refs.update(('base', q, v) for v in ('N', 'B'))
    for q in set().union(*(set(panels[k]) for k in panels if k != 'all_validation')):
        for seed in p['training_seeds']:
            required_refs.update((f'A_{seed}', q, v) for v in ('N', 'B') if (f'A_{seed}', q, v) not in indexed)
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
        if policy in ('A', 'I', 'K', 'S'):
            return f'{policy}_{seed}'
        if policy == 'before':
            return f'A_{seed}' if flags[q]['A'] else 'base'
        if policy == 'bank':
            return f'A_{seed}' if flags[q]['A'] else (f'I_{seed}' if flags[q]['B'] else 'base')
        assert policy in ('shared_K', 'shared_S')
        return f'{policy[-1]}_{seed}' if flags[q]['A'] or flags[q]['B'] else 'base'

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
    for label, terms in [('K_minus_A', diff('K', 'A')), ('I_minus_base', diff('I', 'base')), ('K_visual_interaction', visual('K', 'A')), ('I_visual_interaction', visual('I', 'base')), ('K_minus_S', diff('K', 'S')), ('S_minus_A', diff('S', 'A'))]:
        effects['new_' + label] = effect(panels['new_validation'], terms)
    for panel in ('old_source', 'old_validation'):
        for a, b in [('K', 'A'), ('S', 'A'), ('K', 'S'), ('bank', 'A')]:
            effects[panel + '_' + a + '_minus_' + b] = effect(panels[panel], diff(a, b))
    for policy in ('bank', 'shared_K', 'shared_S'):
        for panel in ('old_background', 'all_validation'):
            effects[panel + '_' + policy + '_minus_before'] = effect(panels[panel], diff(policy, 'before'))
    gates = {'parameter_consolidation_K': {}, 'external_module_bank': {}}
    def positive(g, name, e, threshold=0):
        g[name + '_mean'] = e['mean'] >= threshold if threshold else e['mean'] > 0
        g[name + '_each_seed_positive'] = all(s['mean'] > 0 for s in e['by_training_seed'])
        g[name + '_positive_95_lower'] = e['image_cluster_bootstrap_95'][0] > 0
    def retain(g, name, e):
        g[name + '_mean_nonnegative'] = e['mean'] >= 0
        g[name + '_each_seed_nonnegative'] = all(s['mean'] >= 0 for s in e['by_training_seed'])
        g[name + '_95_lower_at_least_minus_5pp'] = e['image_cluster_bootstrap_95'][0] >= -.05
        g[name + '_zero_new_harm_each_seed'] = all(not s['harmed_question_groups'] for s in e['by_training_seed'])
    for obj, prefix in [('parameter_consolidation_K', 'K'), ('external_module_bank', 'I')]:
        g = gates[obj]
        positive(g, 'new_gain_at_least_10pp', effects['new_' + ('K_minus_A' if prefix == 'K' else 'I_minus_base')], .10)
        positive(g, 'new_visual_interaction', effects['new_' + prefix + '_visual_interaction'])
    for panel in ('old_source', 'old_validation'):
        retain(gates['parameter_consolidation_K'], panel, effects[panel + '_K_minus_A'])
        gates['external_module_bank'][panel + '_exact_immutable_A_branch'] = all(branch('bank', s, q) == f'A_{s}' for s in seeds for q in panels[panel])
    retain(gates['external_module_bank'], 'old_background', effects['old_background_bank_minus_before'])
    summaries = {panel: summary(qs, ['base', 'A', 'K', 'S', 'before', 'bank', 'shared_K', 'shared_S'] + (['I'] if panel == 'new_validation' else [])) for panel, qs in panels.items() if panel != 'all_validation'}
    summaries['all_validation'] = summary(panels['all_validation'], ['base', 'before', 'bank', 'shared_K', 'shared_S'])
    groups = {}
    for field in ('content_type', 'modality'):
        groups[field] = {}
        for kind in sorted({items[q][field] for q in panels['new_validation']}):
            qs = [q for q in panels['new_validation'] if items[q][field] == kind]
            groups[field][kind] = {'summary': summary(qs, ['base', 'A', 'I', 'K', 'S']), 'K_minus_A': effect(qs, diff('K', 'A')), 'I_minus_base': effect(qs, diff('I', 'base'))}
    passed = {obj: all(g.values()) for obj, g in gates.items()}
    result = {'status': 'COMPLETE_CONTINUAL_ACQUISITION_BANK', 'object_gate_pass': passed, 'gates': gates, 'summary': summaries, 'paired_effects': effects, 'new_capability_groups': groups, 'training': training, 'calls': {'new_generations': 4188, 'actual_intents': 4188, 'required_reference_records_reused': len(used), 'reference_catalog_available': len(refs), 'extra_policy_generations': 0, 'errors': 0, 'completed_replays': 0, 'teacher_forwards': 600, 'teacher_generations': 0, 'actual_optimizer_steps': 312, 'new_truncated_at_64': sum(len(r['output_ids']) == 64 for r in new)}, 'cost': {'supervisor_wall_seconds': end['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': end['incremental_bytes']}, 'limitations': ['Development images/questions and known task templates, not blind/test replication.', 'External immutable A retention differs from same-parameter retention; two objects judged separately.', 'Unknown patient independence and unverified author QA labels; image-cluster intervals.', 'No full ViSkill, PPO, visual-card, clinical trajectory or deployment safety claim.'], 'test_annotation_opened': False}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    numeric = [{'source_experiment': r['source_experiment'], 'source_request_number': r['number'], 'reused_reference': r['reused_reference'], 'model': r['model'], 'view': r['view'], 'correct': r['correct'], 'question_group': qmap[r['qid']], 'image_group': imap[items[r['qid']]['img_name']], 'output_tokens': len(r['output_ids'])} for r in records]
    registry = [{'question_group': qmap[q], 'image_group': imap[items[q]['img_name']], 'contract_A': flags[q]['A'], 'contract_B': flags[q]['B'], 'panels': [k for k, qs in panels.items() if q in qs], 'content_type': items[q]['content_type'], 'modality': items[q]['modality']} for q in sorted(items)]
    for name, obj in [('result.json', result), ('physical_branch_numeric.json', numeric), ('contract_numeric.json', registry), ('updates.json', updates), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('execution_source.json', read(OUT / 'execution_source.json')), ('training_exclusion.json', [dict(training_exclusion_group=i, reason=r['reason'], content_type=r['content_type']) for i, r in enumerate(read(OUT / 'private_training_exclusion.json'))])]:
        write(bundle / name, obj)
    print(json.dumps({'object_gate_pass': passed, 'gates': gates, 'summary': summaries, 'effects': effects, 'cost': result['cost']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    close()
