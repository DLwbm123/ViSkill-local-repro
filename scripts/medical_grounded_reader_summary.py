"""Offline visual-anchor closure; reference base responses are reused, never replayed."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'grounded_reader_anchor_v1'
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
sys.path.insert(0, str(SRC / 'scripts'))
from slake_probe import score


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def close():
    p = read(SRC / 'configs/reproduction' / (NAME + '.json'))
    receipt = read(STATE / (NAME + '_receipt.json'))
    end = read(STATE / (NAME + '_EXIT.json'))
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert end['exit_code'] == 0 and end['reason'] is None and not end['argv_violations']
    assert end['end_unix'] < p['deadline_unix'] and end['incremental_bytes'] < p['maximum_incremental_bytes']
    assert done['status'] == 'COMPLETE_GROUNDED_READER_ANCHOR' and done['generation_calls'] == 1908
    assert done['reused_base_generation_calls'] == 318
    teacher = read(OUT / 'TEACHER_COMPLETED.json')
    assert teacher['forward_calls'] == 200 and teacher['generation_calls'] == 0 and not teacher['validation_queries_used']
    backward = read(OUT / 'production_backward_check.json')
    assert backward['status'] == 'PASS' and backward['language_parameters_frozen']
    assert abs(backward['initial_null_KL']) < 2e-4 and abs(backward['initial_old_KL']) < 2e-4
    manifest = read(OUT / 'private_manifest.json')
    items = {r['qid']: r for pool in manifest.values() for r in pool}
    qids = sorted({r['qid'] for pool in ('primary', 'retention') for r in manifest[pool]})
    public_q = {q: i for i, q in enumerate(qids)}
    public_i = {name: i for i, name in enumerate(sorted({items[q]['img_name'] for q in qids}))}
    models = ['base'] + [f'{a}_{s}' for s in p['training_seeds'] for a in p['arms']]
    schedule = read(OUT / 'schedule.json')
    new_rows = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(new_rows) == p['formal_generation_calls'] == 1908
    assert [r['number'] for r in new_rows] == list(range(1, 1909))
    assert [(r['model'], r['panel'], r['qid'], r['view']) for r in new_rows] == [(model, e['panel'], e['qid'], e['view']) for model in models[1:] for e in schedule]
    previous = ROOT / 'artifacts/medical/region_reader_learning_v1'
    assert read(previous / 'RESULT.json')['calls']['formal'] == 2226
    base_rows = [r for f in sorted((previous / 'records').glob('*.json')) if (r := read(f))['model'] == 'base']
    assert len(base_rows) == 318 and [r['number'] for r in base_rows] == list(range(1, 319))
    assert [(r['panel'], r['qid'], r['view']) for r in base_rows] == [(e['panel'], e['qid'], e['view']) for e in schedule]
    rows = base_rows + new_rows
    events = [json.loads(line) for line in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(e['event'] for e in events) == {'intent': 1908, 'result': 1908}
    assert all(bool(score(r['raw'], items[r['qid']]['answer'])) == r['correct'] for r in rows)
    indexed = {(r['model'], r['panel'], r['qid'], r['view']): int(r['correct']) for r in rows}
    assert len(indexed) == len(rows)
    training, update_rows = {}, []
    for model in models[1:]:
        training[model] = read(OUT / model / 'TRAIN_COMPLETED.json')
        t = training[model]
        updates = [json.loads(x) for x in (OUT / model / 'updates.jsonl').read_text().splitlines()]
        assert len(updates) == t['optimizer_steps'] == 52
        assert [u['step'] for u in updates] == list(range(1, 53))
        assert sum(u['supervised_views'] for u in updates) == t['supervised_image_exposures'] == 800
        assert t['lora_B_initial_square_norm'] == 0 and t['lora_B_final_square_norm'] > 0
        assert t['visual_encoder_updated'] and not t['language_parameters_updated']
        assert sum(u['anchor_forward_views'] for u in updates) == t['anchor_forward_image_exposures'] == 800
        assert all(np.isfinite(u['null_KL']) and np.isfinite(u['old_KL']) and np.isfinite(u['supervised_loss']) for u in updates)
        assert t['anchor_coefficient'] == (0.0 if model.startswith('U_') else 1.0)
        assert all(np.isfinite(u['loss']) and np.isfinite(u['preclip_gradient_norm']) for u in updates)
        update_rows.extend(dict(u, model=model) for u in updates)
    assert len({t['trainable_parameters'] for t in training.values()}) == 1

    def correct(model, panel, q, view):
        return indexed[model, 'primary' if panel == 'regional' and view in ('N', 'B') else panel, q, view]

    def counts(panel, subset):
        views = ('N', 'B', 'O', 'W') if panel == 'regional' else ('N', 'B')
        return {m: {v: {'correct': sum(correct(m, panel, q, v) for q in subset), 'questions': len(subset), 'accuracy': sum(correct(m, panel, q, v) for q in subset) / len(subset)} for v in views} for m in models}

    def effect(panel, subset, terms):
        by_seed, values = [], []
        for seed in p['training_seeds']:
            delta = np.array([sum(sign * correct(model.format(seed=seed), panel, q, view) for model, view, sign in terms) for q in subset])
            values.append(delta)
            by_seed.append({'seed': seed, 'mean': float(delta.mean()), 'improved_question_groups': [public_q[q] for q, d in zip(subset, delta) if d > 0], 'harmed_question_groups': [public_q[q] for q, d in zip(subset, delta) if d < 0]})
        averaged = np.mean(values, axis=0)
        groups = defaultdict(list)
        for i, q in enumerate(subset):
            groups[items[q]['img_name']].append(i)
        groups = list(groups.values())
        numerator = np.array([averaged[g].sum() for g in groups])
        denominator = np.array([len(g) for g in groups])
        rng = np.random.default_rng(p['selection_seed'])
        samples = rng.integers(0, len(groups), size=(10000, len(groups)))
        bootstrap = numerator[samples].sum(axis=1) / denominator[samples].sum(axis=1)
        return {'mean': float(averaged.mean()), 'image_cluster_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(), 'questions': len(subset), 'images': len(groups), 'by_seed': by_seed}

    B_A = [('B_{seed}', 'N', 1), ('A_{seed}', 'N', -1)]
    B_U = [('B_{seed}', 'N', 1), ('U_{seed}', 'N', -1)]
    VIS = [('B_{seed}', 'N', 1), ('B_{seed}', 'B', -1), ('A_{seed}', 'N', -1), ('A_{seed}', 'B', 1)]
    BASE = [('B_{seed}', 'N', 1), ('base', 'N', -1)]
    panels = {k: sorted(r['qid'] for r in manifest[k]) for k in ('primary', 'retention', 'regional')}
    summary = {k: counts(k, subset) for k, subset in panels.items()}
    effects = {'B_minus_A': effect('primary', panels['primary'], B_A), 'B_minus_U_primary': effect('primary', panels['primary'], B_U), 'B_minus_U_retention': effect('retention', panels['retention'], B_U), 'visual_interaction': effect('primary', panels['primary'], VIS), 'B_minus_base_primary': effect('primary', panels['primary'], BASE), 'retention_B_minus_base': effect('retention', panels['retention'], BASE)}
    for name, terms in [('A_minus_base', [('A_{seed}', 'N', 1), ('base', 'N', -1)]), ('U_minus_base', [('U_{seed}', 'N', 1), ('base', 'N', -1)])]:
        effects[name] = effect('primary', panels['primary'], terms)
    for key, terms in [('B_minus_A', B_A), ('B_minus_U_primary', B_U), ('visual_interaction', VIS)]:
        expected = np.mean([sum(sign * summary['primary'][model.format(seed=s)][v]['accuracy'] for model, v, sign in terms) for s in p['training_seeds']])
        assert abs(expected - effects[key]['mean']) < 1e-12
    gates = {'B_minus_A_at_least_10pp': effects['B_minus_A']['mean'] >= .10}
    for key in ('B_minus_A', 'visual_interaction'):
        gates[key + '_positive_each_seed'] = all(r['mean'] > 0 for r in effects[key]['by_seed'])
        gates[key + '_positive_95_lower'] = effects[key]['image_cluster_bootstrap_95'][0] > 0
    gates['B_minus_U_primary_average_nonnegative'] = effects['B_minus_U_primary']['mean'] >= 0
    gates['B_minus_U_primary_each_seed_nonnegative'] = all(r['mean'] >= 0 for r in effects['B_minus_U_primary']['by_seed'])
    gates['B_minus_U_retention_average_positive'] = effects['B_minus_U_retention']['mean'] > 0
    gates['B_minus_U_retention_each_seed_positive'] = all(r['mean'] > 0 for r in effects['B_minus_U_retention']['by_seed'])
    gates['B_minus_U_retention_positive_95_lower'] = effects['B_minus_U_retention']['image_cluster_bootstrap_95'][0] > 0
    gates['visual_interaction_positive_average'] = effects['visual_interaction']['mean'] > 0
    gates['retention_average_nonnegative'] = effects['retention_B_minus_base']['mean'] >= 0
    gates['retention_each_seed_nonnegative'] = all(r['mean'] >= 0 for r in effects['retention_B_minus_base']['by_seed'])
    gates['retention_95_lower_at_least_minus_5pp'] = effects['retention_B_minus_base']['image_cluster_bootstrap_95'][0] >= -.05
    groups = {}
    for field in ('task', 'target', 'modality'):
        groups[field] = {}
        for value in sorted({r[field] for r in manifest['primary']}):
            subset = sorted(r['qid'] for r in manifest['primary'] if r[field] == value)
            groups[field][value] = {'questions': len(subset), 'summary': counts('primary', subset), 'B_minus_A': effect('primary', subset, B_A), 'B_minus_base': effect('primary', subset, BASE)}
    groups['retention_type'] = {}
    for kind in sorted({r['content_type'] for r in manifest['retention']}):
        subset = sorted(r['qid'] for r in manifest['retention'] if r['content_type'] == kind)
        groups['retention_type'][kind] = {'questions': len(subset), 'summary': counts('retention', subset), 'B_minus_base': effect('retention', subset, BASE)}
    result = {'status': 'COMPLETE_VISUAL_ANCHOR_GATE_PASS' if all(gates.values()) else 'COMPLETE_VISUAL_ANCHOR_NOT_ESTABLISHED', 'all_gates_pass': all(gates.values()), 'gates': gates, 'summary': summary, 'paired_effects': effects, 'groups': groups, 'training': training, 'calls': {'formal_new': 1908, 'reused_base_reference': 318, 'analytical_responses': 2226, 'actual_intents': 1908, 'errors': 0, 'completed_replays': 0, 'teacher_forwards': 200, 'teacher_generations': 0, 'new_output_tokens': sum(len(r['output_ids']) for r in new_rows), 'new_truncated_at_64': sum(len(r['output_ids']) == 64 for r in new_rows)}, 'cost': {'supervisor_wall_seconds': end['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': end['incremental_bytes'], 'optimizer_steps_all_arms': 312, 'supervised_image_exposures_all_arms': 4800, 'anchor_forward_image_exposures_all_arms': 4800}, 'test_annotation_opened': False, 'limitations': ['Observed validation development data; no blind or test-set claim.', 'Train and validation image IDs are disjoint; patient independence is unknown.', 'Two paired training seeds; question/image bootstrap does not estimate full training-population uncertainty.', 'Train annotations are privileged experience construction, not autonomous localization.', 'Only visual attention adapters were trained; null/old-task targets came from training-only teacher-forced base distributions.', '318 matched historical base responses are reused; no new base generations or validation distillation occurred.', 'Exact match and original labels are unchanged; clinical truth is not independently certified.', 'Small and uneven task/target groups are descriptive, not confirmatory subgroup successes.', 'All base/adapter comparisons use this matched HF backend; earlier SGLang scores are not training baselines.']}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    numeric = [{'number': i + 1, 'source_request_number': r['number'], 'reused_base_reference': r['model'] == 'base', **{k: r[k] for k in ('model', 'panel', 'view', 'correct')}, 'question_group': public_q[r['qid']], 'image_group': public_i[items[r['qid']]['img_name']], 'output_tokens': len(r['output_ids'])} for i, r in enumerate(rows)]
    for name, value in [('result.json', result), ('complete_numeric_records.json', numeric), ('training_updates.json', update_rows), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('teacher_completed.json', teacher), ('production_backward_check.json', read(OUT / 'production_backward_check.json')), ('execution_source.json', read(OUT / 'execution_source.json'))]:
        write(bundle / name, value)
    print(json.dumps({'status': result['status'], 'gates': gates, 'summary': summary, 'effects': effects, 'cost': result['cost']}, ensure_ascii=False), flush=True)



if __name__ == '__main__':
    close()
