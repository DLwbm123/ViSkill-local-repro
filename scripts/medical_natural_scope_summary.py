"""Natural-prevalence closure: only required historical and new physical branches count."""
from collections import Counter, defaultdict
import os
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'natural_scope_validation_v1'
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
sys.path.insert(0, str(SRC / 'scripts'))
from medical_region_reader import read, write
from medical_natural_scope import legacy_records
from slake_probe import score


def close():
    p = read(SRC / 'configs/reproduction' / (NAME + '.json'))
    end = read(STATE / (NAME + '_EXIT.json'))
    receipt = read(STATE / (NAME + '_receipt.json'))
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert end['exit_code'] == 0 and end['reason'] is None and not end['argv_violations']
    assert end['end_unix'] < p['deadline_unix'] and end['incremental_bytes'] < p['maximum_incremental_bytes']
    assert done['status'] == 'COMPLETE_NATURAL_SCOPE_VALIDATION' and done['new_generation_calls'] == p['formal_generation_calls'] == 1962
    assert done['new_reader_optimizer_steps'] == done['new_router_optimizer_steps'] == 0
    m = read(OUT / 'private_manifest.json')
    items = {r['qid']: r for r in m['validation']}
    assert len(items) == 1053 and len({r['img_name'] for r in items.values()}) == 96
    frozen = read(OUT / 'private_routing_frozen.json')
    assert frozen['frozen_before_new_generations'] and not frozen['router_refitted'] and frozen['threshold'] == .90
    routing = {int(q): r for q, r in frozen['queries'].items()}
    assert set(routing) == set(items)
    assert all(r['admit'] == (r['probability'] >= .90) for r in routing.values())
    assert all(routing[q]['rule_admit'] == items[q]['weak_scope'] for q in items)
    assert sum(r['admit'] for r in routing.values()) == sum(r['sham_admit'] for r in routing.values()) == frozen['admitted'] == frozen['sham_admitted']
    required = set()
    for q in items:
        route = routing[q]
        models = ['base'] + (['A_42', 'A_43'] if route['admit'] or route['sham_admit'] or route['rule_admit'] else [])
        required.update((model, q, view) for model in models for view in ('N', 'B'))
    assert len(required) == 2406
    catalog = legacy_records()
    catalog_keys = {(r['model'], r['qid'], r['view']) for r in catalog}
    old = [dict(r, reused_reference=True, source_experiment=r['legacy_experiment']) for r in catalog if (r['model'], r['qid'], r['view']) in required]
    assert len(old) == done['reused_required_records'] == 444 and len(catalog) == 840
    assert all(r['legacy_image'] == items[r['qid']]['img_name'] for r in old)
    new = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    plan = read(OUT / 'schedule.json')
    assert len(new) == len(plan) == 1962 and [r['number'] for r in new] == list(range(1, 1963))
    assert [(r['model'], r['qid'], r['view']) for r in new] == [(r['model'], r['qid'], r['view']) for r in plan]
    assert not catalog_keys & {(r['model'], r['qid'], r['view']) for r in new}
    events = [json.loads(x) for x in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(e['event'] for e in events) == {'intent': 1962, 'result': 1962}
    assert (OUT / 'private_routing_frozen.json').stat().st_mtime <= min(e['started_unix'] for e in events if e['event'] == 'intent')
    records = old + [dict(r, reused_reference=False, source_experiment=NAME) for r in new]
    assert all(bool(score(r['raw'], items[r['qid']]['answer'])) == r['correct'] for r in records)
    indexed = {(r['model'], r['qid'], r['view']): int(r['correct']) for r in records}
    assert len(indexed) == len(records) == 2406 and set(indexed) == required
    panels = {'natural': sorted(items), 'supported_historical': sorted(q for q in items if items[q]['weak_scope']), 'offscope': sorted(q for q in items if not items[q]['weak_scope']), 'old_questions': m['legacy_qids'], 'new_questions': m['new_qids']}
    assert len(panels['supported_historical']) == 39 and len(panels['offscope']) == 1014
    assert len(panels['old_questions']) == 140 and len(panels['new_questions']) == 913
    assert not set(panels['new_questions']) & set(panels['supported_historical'])
    public_q = {q: i for i, q in enumerate(panels['natural'])}
    public_i = {name: i for i, name in enumerate(sorted({r['img_name'] for r in items.values()}))}
    policies, seeds = ('N', 'R', 'S', 'F'), p['adapter_seeds']

    def branch(policy, seed, q):
        flag = policy != 'N' and routing[q][{'R': 'admit', 'S': 'sham_admit', 'F': 'rule_admit'}[policy]]
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
        selected = rng.integers(0, len(groups), size=(10000, len(groups)))
        bootstrap = numerator[selected].sum(1) / denominator[selected].sum(1)
        return {'mean': float(averaged.mean()), 'image_cluster_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(), 'questions': len(subset), 'images': len(groups), 'by_adapter_seed': by_seed}

    R_N = [('R', 'N', 1), ('N', 'N', -1)]
    R_S = [('R', 'N', 1), ('S', 'N', -1)]
    VIS = [('R', 'N', 1), ('R', 'B', -1), ('N', 'N', -1), ('N', 'B', 1)]
    summaries = {name: summary(subset) for name, subset in panels.items()}
    effects = {name + '_R_minus_base': effect(subset, R_N) for name, subset in panels.items()}
    effects['natural_R_minus_sham'] = effect(panels['natural'], R_S)
    effects['natural_visual_interaction'] = effect(panels['natural'], VIS)
    effects['natural_F_minus_base'] = effect(panels['natural'], [('F', 'N', 1), ('N', 'N', -1)])
    for key, terms in [('natural_R_minus_base', R_N), ('natural_R_minus_sham', R_S), ('natural_visual_interaction', VIS)]:
        assert abs(effects[key]['mean'] - sum(sign * summaries['natural'][policy][view]['mean_accuracy'] for policy, view, sign in terms)) < 1e-12
    tp = sum(routing[q]['admit'] for q in panels['supported_historical'])
    fp = sum(routing[q]['admit'] for q in panels['offscope'])
    scope = {'true_positive': tp, 'false_positive': fp, 'false_negative': 39 - tp, 'true_negative': 1014 - fp, 'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / 39, 'weak_support_prevalence': 39 / 1053, 'admitted_queries': frozen['admitted'], 'reference': 'Frozen train support question rule and VQA metadata for evaluation only; not clinical certification.'}
    gates = {}
    for key in ('natural_R_minus_base', 'natural_R_minus_sham', 'natural_visual_interaction'):
        gates[key + '_mean_positive'] = effects[key]['mean'] > 0
        gates[key + '_each_seed_positive'] = all(r['mean'] > 0 for r in effects[key]['by_adapter_seed'])
        gates[key + '_positive_95_lower'] = effects[key]['image_cluster_bootstrap_95'][0] > 0
    off = effects['offscope_R_minus_base']
    gates.update(offscope_mean_nonnegative=off['mean'] >= 0, offscope_each_seed_nonnegative=all(r['mean'] >= 0 for r in off['by_adapter_seed']), offscope_95_lower_at_least_minus_5pp=off['image_cluster_bootstrap_95'][0] >= -.05, offscope_zero_harmed_each_seed=all(not r['harmed_question_groups'] for r in off['by_adapter_seed']), new_questions_zero_harmed_each_seed=all(not r['harmed_question_groups'] for r in effects['new_questions_R_minus_base']['by_adapter_seed']), weak_scope_precision_at_least_95pct=scope['precision'] >= .95, weak_scope_recall_at_least_70pct=scope['recall'] >= .70)
    groups = {}
    for field in ('base_type', 'content_type', 'modality'):
        groups[field] = {}
        for kind in sorted({r[field] for r in items.values()}):
            subset = sorted(q for q, r in items.items() if r[field] == kind)
            groups[field][kind] = {'questions': len(subset), 'summary': summary(subset), 'R_minus_base': effect(subset, R_N)}
    result = {'status': 'COMPLETE_NATURAL_SCOPE_GATE_PASS' if all(gates.values()) else 'COMPLETE_NATURAL_SCOPE_NOT_ESTABLISHED', 'all_gates_pass': all(gates.values()), 'gates': gates, 'summary': summaries, 'paired_effects': effects, 'weak_scope_prediction': scope, 'groups': groups, 'calls': {'new_generations': 1962, 'actual_intents': 1962, 'legacy_catalog_available': 840, 'legacy_required_reused': 444, 'required_physical_branches': 2406, 'unused_legacy_not_counted_as_reused': 396, 'errors': 0, 'completed_replays': 0, 'extra_policy_generations': 0, 'new_output_tokens': sum(len(r['output_ids']) for r in new), 'new_truncated_at_64': sum(len(r['output_ids']) == 64 for r in new), 'new_reader_optimizer_steps': 0, 'new_router_optimizer_steps': 0}, 'cost': {'supervisor_wall_seconds': end['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': end['incremental_bytes']}, 'test_annotation_opened': False, 'limitations': ['All English validation, but observed 96 images; no blind/test claim.', 'All 39 supported questions are historical development observations, not an independent efficacy replication.', '913 additional questions test new offscope behavior on previously observed images.', 'Weak rule scope labels and original QA answers are not independent clinical verification.', 'External base fallback does not show single-adapter parameter retention.', 'Shared exact branch records and two frozen adapter seeds do not create extra independent generations or patients.', 'No unconditional-adapter whole-cohort score is computed for unobserved branches.']}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    physical = sorted(records, key=lambda r: (r['model'], r['qid'], r['view']))
    numeric = [{'branch_group': i, 'source_experiment': r['source_experiment'], 'source_request_number': r['number'], 'reused_reference': r['reused_reference'], **{k: r[k] for k in ('model', 'view', 'correct')}, 'question_group': public_q[r['qid']], 'image_group': public_i[items[r['qid']]['img_name']], 'output_tokens': len(r['output_ids'])} for i, r in enumerate(physical)]
    routes = [{'question_group': public_q[q], 'image_group': public_i[items[q]['img_name']], 'weak_scope_label': items[q]['weak_scope'], 'previously_evaluated_question': q in panels['old_questions'], 'base_type': items[q]['base_type'], **routing[q]} for q in panels['natural']]
    for name, obj in [('result.json', result), ('physical_branch_numeric.json', numeric), ('routing_numeric.json', routes), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('execution_source.json', read(OUT / 'execution_source.json'))]:
        write(bundle / name, obj)
    print(json.dumps({'status': result['status'], 'gates': gates, 'scope': scope, 'summary': summaries, 'effects': effects, 'cost': result['cost']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    close()
