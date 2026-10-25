"""Close frozen admission policies from shared deterministic branch records."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'task_supported_adapter_v1'
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
sys.path.insert(0, str(SRC / 'scripts'))
from slake_probe import score
from medical_region_reader import read, write


def close():
    p = read(SRC / 'configs/reproduction' / (NAME + '.json'))
    receipt = read(STATE / (NAME + '_receipt.json'))
    end = read(STATE / (NAME + '_EXIT.json'))
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert end['exit_code'] == 0 and end['reason'] is None and not end['argv_violations']
    assert end['end_unix'] < p['deadline_unix'] and end['incremental_bytes'] < p['maximum_incremental_bytes']
    assert done['status'] == 'COMPLETE_TASK_SUPPORT_ACTIVATION' and done['new_generation_calls'] == 828
    assert done['new_reader_optimizer_steps'] == 0 and done['new_router_fitted']
    m = read(OUT / 'private_manifest.json')
    panels = {kind: sorted(r['qid'] for r in m[kind]) for kind in ('primary', 'retention')}
    panels['union'] = sorted(panels['primary'] + panels['retention'])
    items = {r['qid']: r for kind in ('primary', 'retention') for r in m[kind]}
    assert len(items) == len({r['img_name'] for r in items.values()}) == 138
    assert len(panels['primary']) == len(panels['retention']) == 69
    assert not {r['img_name'] for r in m['fit']} & {r['img_name'] for r in items.values()}
    assert len(m['fit']) == 1569 and all('answer' not in r for r in m['fit'])
    rows = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    plan = read(OUT / 'schedule.json')
    assert len(rows) == p['formal_generation_calls'] == 828
    assert [r['number'] for r in rows] == list(range(1, 829))
    assert [(r['model'], r['panel'], r['qid'], r['view']) for r in rows] == [(model, e['panel'], e['qid'], e['view']) for model in ('base', 'A_42', 'A_43') for e in plan]
    assert all(bool(score(r['raw'], items[r['qid']]['answer'])) == r['correct'] for r in rows)
    events = [json.loads(x) for x in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(e['event'] for e in events) == {'intent': 828, 'result': 828}
    frozen = read(OUT / 'private_routing_frozen.json')
    assert frozen['frozen_before_new_model_generations'] and frozen['threshold'] == p['admission_threshold']
    assert (OUT / 'private_routing_frozen.json').stat().st_mtime <= min(e['started_unix'] for e in events if e['event'] == 'intent')
    routing = {int(q): r for q, r in frozen['queries'].items()}
    assert set(routing) == set(items)
    assert all(r['admit'] == (r['probability'] >= p['admission_threshold']) for r in routing.values())
    assert all(r['rule_admit'] == (q in panels['primary']) for q, r in routing.items())
    assert sum(r['admit'] for r in routing.values()) == sum(r['sham_admit'] for r in routing.values()) == frozen['admitted'] == frozen['sham_admitted']
    trained = read(OUT / 'ROUTER_TRAIN_COMPLETED.json')
    assert trained['fit_questions'] == 1569 and trained['fit_positives'] == 124
    assert trained['finite_gradient'] and trained['coefficient_square_norm'] > 0 and not trained['evaluation_answers_used']
    assert trained['LBFGS_iterations'] <= 100
    indexed = {(r['model'], r['qid'], r['view']): int(r['correct']) for r in rows}
    assert len(indexed) == len(rows)
    public_q = {q: i for i, q in enumerate(panels['union'])}
    public_i = {name: i for i, name in enumerate(sorted({r['img_name'] for r in items.values()}))}
    seeds = p['adapter_seeds']
    policies = ('N', 'U', 'R', 'S', 'F')

    def branch(policy, seed, q):
        flag = policy == 'U' or policy in ('R', 'S', 'F') and routing[q][{'R': 'admit', 'S': 'sham_admit', 'F': 'rule_admit'}[policy]]
        return f'A_{seed}' if flag else 'base'

    def value(policy, seed, q, view):
        return indexed[branch(policy, seed, q), q, view]

    def summary(subset):
        return {policy: {view: {'questions': len(subset), 'correct_by_seed': [sum(value(policy, seed, q, view) for q in subset) for seed in seeds], 'mean_accuracy': float(np.mean([value(policy, seed, q, view) for seed in seeds for q in subset]))} for view in ('N', 'B')} for policy in policies}

    def effect(subset, terms):
        values, by_seed = [], []
        for seed in seeds:
            delta = np.array([sum(sign * value(policy, seed, q, view) for policy, view, sign in terms) for q in subset])
            values.append(delta)
            by_seed.append({'adapter_seed': seed, 'mean': float(delta.mean()), 'improved_question_groups': [public_q[q] for q, d in zip(subset, delta) if d > 0], 'harmed_question_groups': [public_q[q] for q, d in zip(subset, delta) if d < 0]})
        averaged = np.mean(values, axis=0)
        groups = defaultdict(list)
        for i, q in enumerate(subset):
            groups[items[q]['img_name']].append(i)
        groups = list(groups.values())
        numerator = np.array([averaged[g].sum() for g in groups])
        denominator = np.array([len(g) for g in groups])
        rng = np.random.default_rng(p['selection_seed'])
        sample = rng.integers(0, len(groups), size=(10000, len(groups)))
        bootstrap = numerator[sample].sum(1) / denominator[sample].sum(1)
        return {'mean': float(averaged.mean()), 'image_cluster_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(), 'questions': len(subset), 'images': len(groups), 'by_adapter_seed': by_seed}

    R_N = [('R', 'N', 1), ('N', 'N', -1)]
    R_S = [('R', 'N', 1), ('S', 'N', -1)]
    VIS = [('R', 'N', 1), ('R', 'B', -1), ('N', 'N', -1), ('N', 'B', 1)]
    summaries = {panel: summary(subset) for panel, subset in panels.items()}
    effects = {'R_minus_base_primary': effect(panels['primary'], R_N), 'R_minus_sham_union': effect(panels['union'], R_S), 'visual_interaction_primary': effect(panels['primary'], VIS), 'retention_R_minus_base': effect(panels['retention'], R_N)}
    for panel, subset in panels.items():
        for policy in ('U', 'F'):
            effects[policy + '_minus_base_' + panel] = effect(subset, [(policy, 'N', 1), ('N', 'N', -1)])
    for panel, key, terms in [('primary', 'R_minus_base_primary', R_N), ('union', 'R_minus_sham_union', R_S), ('primary', 'visual_interaction_primary', VIS)]:
        assert abs(effects[key]['mean'] - sum(sign * summaries[panel][policy][view]['mean_accuracy'] for policy, view, sign in terms)) < 1e-12
    tp = sum(routing[q]['admit'] for q in panels['primary'])
    fp = sum(routing[q]['admit'] for q in panels['retention'])
    scope = {'true_positive': tp, 'false_positive': fp, 'false_negative': 69 - tp, 'true_negative': 69 - fp, 'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / 69, 'threshold': p['admission_threshold'], 'reference': 'Frozen question-only weak support rule, not clinical applicability.'}
    gates = {'R_minus_base_primary_at_least_10pp': effects['R_minus_base_primary']['mean'] >= .10}
    for key in ('R_minus_base_primary', 'R_minus_sham_union', 'visual_interaction_primary'):
        gates[key + '_average_positive'] = effects[key]['mean'] > 0
        gates[key + '_each_seed_positive'] = all(r['mean'] > 0 for r in effects[key]['by_adapter_seed'])
        gates[key + '_positive_95_lower'] = effects[key]['image_cluster_bootstrap_95'][0] > 0
    retention = effects['retention_R_minus_base']
    gates.update(retention_average_nonnegative=retention['mean'] >= 0, retention_each_seed_nonnegative=all(r['mean'] >= 0 for r in retention['by_adapter_seed']), retention_95_lower_at_least_minus_5pp=retention['image_cluster_bootstrap_95'][0] >= -.05, retention_zero_harmed_each_seed=all(not r['harmed_question_groups'] for r in retention['by_adapter_seed']), weak_scope_precision_at_least_95pct=scope['precision'] >= .95, weak_scope_recall_at_least_70pct=scope['recall'] >= .70)
    groups = {}
    for panel, field in (('primary', 'task'), ('primary', 'target'), ('primary', 'modality'), ('retention', 'content_type')):
        groups[panel + '_' + field] = {}
        for kind in sorted({items[q][field] for q in panels[panel]}):
            subset = [q for q in panels[panel] if items[q][field] == kind]
            groups[panel + '_' + field][kind] = {'questions': len(subset), 'summary': summary(subset), 'R_minus_base': effect(subset, R_N)}
    result = {'status': 'COMPLETE_SUPPORT_ACTIVATION_GATE_PASS' if all(gates.values()) else 'COMPLETE_SUPPORT_ACTIVATION_NOT_ESTABLISHED', 'all_gates_pass': all(gates.values()), 'gates': gates, 'summary': summaries, 'paired_effects': effects, 'weak_scope_prediction': scope, 'groups': groups, 'router_training': trained, 'activity': {'R_admitted_queries': frozen['admitted'], 'S_admitted_queries': frozen['sham_admitted'], 'queries': 138, 'per_policy_reader_calls_if_deployed': 1}, 'calls': {'actual_new_generations': 828, 'actual_intents': 828, 'errors': 0, 'completed_replays': 0, 'extra_R_S_F_generations': 0, 'shared_branch_policy_evaluation': True, 'output_tokens': sum(len(r['output_ids']) for r in rows), 'truncated_at_64': sum(len(r['output_ids']) == 64 for r in rows), 'new_reader_optimizer_steps': 0}, 'cost': {'supervisor_wall_seconds': end['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': end['incremental_bytes']}, 'test_annotation_opened': False, 'limitations': ['Conditional balanced development panels from official train; not natural benchmark prevalence or official test.', 'Evaluation images excluded from current reader/teacher/router updates, but prior project/pretrained exposure may exist.', 'A component was selected using observed development evidence; both frozen adapter seeds are retained.', 'Unknown patient independence and unverified clinical labels.', 'Question-rule scope labels are not independent clinical annotations, and the need for learned rather than rule admission is unestablished.', 'External fallback can preserve policy behavior without proving single-adapter parameter retention.', 'Deterministic policies share exact branch records; derived policy outcomes are not additional independent generations.']}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    numeric = [{**{k: r[k] for k in ('number', 'model', 'panel', 'view', 'correct')}, 'question_group': public_q[r['qid']], 'image_group': public_i[items[r['qid']]['img_name']], 'output_tokens': len(r['output_ids'])} for r in rows]
    routing_numeric = [{'question_group': public_q[q], 'image_group': public_i[items[q]['img_name']], 'weak_support_label': q in panels['primary'], **routing[q]} for q in panels['union']]
    for name, obj in [('result.json', result), ('complete_numeric_records.json', numeric), ('routing_numeric.json', routing_numeric), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('router_training.json', trained), ('execution_source.json', read(OUT / 'execution_source.json'))]:
        write(bundle / name, obj)
    print(json.dumps({'status': result['status'], 'gates': gates, 'scope': scope, 'summary': summaries, 'effects': effects, 'cost': result['cost']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    close()
