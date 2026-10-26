"""Score the one locked A/B confirmation after every answer-free prediction is frozen."""
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
WINDOW = Path(os.environ['TASK_WINDOW_STATE'])
NAME = 'ab_bank_independent_confirmation_v1'
OUT = ROOT / 'artifacts/medical' / NAME
sys.path.insert(0, str(SRC / 'scripts'))
from medical_region_reader import read, write
from slake_probe import score


def close():
    assert not (OUT / 'RESULT.json').exists(), 'one locked evaluation; read existing result instead of rescoring'
    p = read(SRC / 'configs/reproduction' / (NAME + '.json'))
    end = read(ROOT / 'state/reproduction' / (NAME + '_EXIT.json'))
    receipt = read(ROOT / 'state/reproduction' / (NAME + '_receipt.json'))
    done, locked = read(OUT / 'WORKER_COMPLETED.json'), read(OUT / 'PREDICTIONS_LOCKED.json')
    assert end['exit_code'] == 0 and end['reason'] is None and not end['argv_violations']
    assert end['end_unix'] < p['deadline_unix'] and end['incremental_bytes'] < p['maximum_incremental_bytes']
    assert done['status'] == 'COMPLETE_AB_BANK_CONFIRMATION' and done['new_generation_calls'] == p['formal_generation_calls'] == 1614
    assert done['optimizer_steps'] == 0 and not done['test_answer_used_by_predictor'] and not done['gold_scored']
    assert locked['status'] == 'ALL_PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING' and locked['new_generations'] == 1614
    checks = {'base': read(OUT / 'BASE_READOUT_CHECK.json')}
    assert checks['base'] == {'layers': 64, 'all_disabled': True}
    for label in ['A_42', 'A_43', 'I_42', 'I_43']:
        checks[label] = read(OUT / (label + '_READOUT_CHECK.json'))
        assert checks[label]['layers'] == 64 and checks[label]['correct_adapter_active'] and checks[label]['all_parameters_frozen']
    inputs = read(OUT / 'private_input_manifest.json')
    rows = {r['qid']: r for r in inputs['records']}
    assert len(rows) == 1061 and all('answer' not in r for r in rows.values())
    route, bindings = read(OUT / 'private_routing.json'), read(OUT / 'private_bindings.json')
    cache, plan = read(OUT / 'private_cached_records.json'), read(OUT / 'physical_plan.json')
    records = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(records) == len(plan) == 1614 and len(bindings) == 2942 and len(cache) == 274
    assert [r['number'] for r in records] == list(range(1, 1615)) and [r['physical_index'] for r in records] == list(range(1614))
    assert all(all(r[k] == e[k] for k in ['model', 'qid', 'view', 'physical_index']) for r, e in zip(records, plan))
    events = [json.loads(s) for s in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(r['event'] for r in events) == {'intent': 1614, 'result': 1614}
    for event in ['intent', 'result']:
        assert [r['physical_index'] for r in events if r['event'] == event] == list(range(1614))
    first = min(r['started_unix'] for r in events if r['event'] == 'intent')
    for file in ['private_input_manifest.json', 'private_routing.json', 'private_bindings.json', 'physical_plan.json']:
        assert (OUT / file).stat().st_mtime < first
    assert locked['locked_unix'] >= max(r['completed_unix'] for r in records)
    assert {r['physical_index'] for r in bindings.values() if 'physical_index' in r} == set(range(1614))
    assert {r['cache_index'] for r in bindings.values() if 'cache_index' in r} == set(range(274))
    for label in checks:
        assert read(OUT / (label + '_EVALUATED.json'))['calls'] == sum(r['model'] == label for r in plan)
    for parent in read(OUT / 'private_parents.json').values():
        for file in parent['files']:
            actual = Path(parent['private_folder']) / file['name']
            assert actual.stat().st_size == file['bytes'] and actual.stat().st_mtime_ns == file['mtime_ns']
    # This marker is written only after all output-integrity checks, before reading gold.
    marker = OUT / 'GOLD_EVALUATION_OPENED.json'
    if marker.exists():
        assert read(marker)['opened_unix'] >= locked['locked_unix']
    else:
        write(marker, {'opened_unix': time.time(), 'predictions_locked_unix': locked['locked_unix'], 'authorized_once_for_frozen_evaluation': True, 'no_new_predictions_or_training': True})
    assert read(marker)['opened_unix'] >= locked['locked_unix']
    gold = {r['qid']: r for r in read(DATA / 'raw/test.json') if r['q_lang'] == 'en'}
    assert gold.keys() == rows.keys()
    assert all(gold[q]['img_name'] == rows[q]['img_name'] and gold[q]['question'] == rows[q]['question'] for q in rows)
    def origin(label, q, view):
        b = bindings[f'{label}|{q}|{view}']
        if 'cache_index' in b: return cache[b['cache_index']]
        return records[b['physical_index']]
    logical = {}
    for key in bindings:
        label, q, view = key.split('|')
        q = int(q)
        raw = origin(label, q, view)
        assert raw['model'] == label
        if 'physical_index' in raw:
            original = rows[raw['qid']]
            assert original['question'] == rows[q]['question']
            assert raw['view'] == view and (view == 'B' or original['img_name'] == rows[q]['img_name'])
        logical[label, q, view] = int(score(raw['raw'], gold[q]['answer']))
    seeds = [42, 43]
    qs = sorted(rows)
    clean = sorted(inputs['independent_qids'])
    assert len(clean) == 1061 and len({rows[q]['img_name'] for q in clean}) == 96
    def branch(seed, q):
        r = route[str(q)]['branch']
        return 'base' if r == 'base' else f'{r}_{seed}'
    def value(policy, seed, q, view): return logical['base' if policy == 'base' else branch(seed, q), q, view]
    qmap = {q: i for i, q in enumerate(qs)}
    imap = {image: i for i, image in enumerate(sorted({r['img_name'] for r in rows.values()}))}
    def summary(pool):
        return {'questions': len(pool), 'images': len({rows[q]['img_name'] for q in pool}), 'correct': {a: {v: [sum(value(a, s, q, v) for q in pool) for s in seeds] for v in ['N', 'B']} for a in ['base', 'bank']}}
    def groups(pool):
        by_image = defaultdict(list)
        for i, q in enumerate(pool): by_image[rows[q]['img_name']].append(i)
        return list(by_image.values())
    def effect(pool, visual=False):
        matrix = np.array([[value('bank', s, q, 'N') - value('base', s, q, 'N') - (value('bank', s, q, 'B') - value('base', s, q, 'B') if visual else 0) for q in pool] for s in seeds])
        avg = matrix.mean(0)
        gg = groups(pool)
        num, den = np.array([avg[g].sum() for g in gg]), np.array([len(g) for g in gg])
        samples = np.random.default_rng(p['selection_seed']).integers(0, len(gg), (10000, len(gg)))
        boot = num[samples].sum(1) / den[samples].sum(1)
        return {'mean': float(avg.mean()), 'image_cluster_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(), 'questions': len(pool), 'images': len(gg), 'by_parent_seed': [{'seed': s, 'mean': float(d.mean()), 'gains': [qmap[q] for q, x in zip(pool, d) if x > 0], 'harms': [qmap[q] for q, x in zip(pool, d) if x < 0]} for s, d in zip(seeds, matrix)]}
    def harm(pool):
        base = np.array([value('base', 42, q, 'N') for q in pool])
        harmed = np.array([[base[j] and not value('bank', s, q, 'N') for j, q in enumerate(pool)] for s in seeds], dtype=float)
        gg = groups(pool)
        den = np.array([base[g].sum() for g in gg])
        num = np.array([harmed.mean(0)[g].sum() for g in gg])
        samples = np.random.default_rng(p['selection_seed']).integers(0, len(gg), (10000, len(gg)))
        denominators = den[samples].sum(1)
        assert base.sum() > 0 and np.all(denominators > 0)
        boot = num[samples].sum(1) / denominators
        return {'base_correct_denominator': int(base.sum()), 'new_harms_by_seed': harmed.sum(1).astype(int).tolist(), 'rates_by_seed': (harmed.sum(1) / base.sum()).tolist(), 'paired_mean_rate': float(harmed.mean(0).sum() / base.sum()), 'image_cluster_bootstrap_95': np.quantile(boot, [.025, .975]).tolist()}
    panels = {'all_official_English': qs, 'independent_all': clean}
    panels.update({name: [q for q in clean if route[str(q)][name]] for name in ['Plane', 'Modality', 'lung_appearance']})
    panels.update({name: [q for q in clean if route[str(q)]['branch'] == label] for name, label in [('A_all', 'A'), ('B_all', 'I'), ('offscope', 'base')]})
    summaries = {name: summary(pool) for name, pool in panels.items()}
    effects = {name: effect(pool) for name, pool in panels.items()}
    visual = effect(clean, visual=True)
    damage = harm(clean)
    gates = {}
    for name, threshold in [('independent_all', 0), ('Plane', .10), ('lung_appearance', .10)]:
        e = effects[name]
        gates[name + '_mean'] = e['mean'] >= threshold if threshold else e['mean'] > 0
        gates[name + '_each_seed_positive'] = all(v['mean'] > 0 for v in e['by_parent_seed'])
        gates[name + '_positive_95_lower'] = e['image_cluster_bootstrap_95'][0] > 0
        if name != 'independent_all': gates[name + '_at_least_10_images'] = e['images'] >= 10
    e = effects['Modality']
    gates.update(Modality_mean_at_least_minus_2pp=e['mean'] >= -.02, Modality_each_seed_at_least_minus_2pp=all(r['mean'] >= -.02 for r in e['by_parent_seed']), Modality_95_lower_at_least_minus_5pp=e['image_cluster_bootstrap_95'][0] >= -.05, visual_mean_positive=visual['mean'] > 0, visual_each_seed_positive=all(r['mean'] > 0 for r in visual['by_parent_seed']), visual_positive_95_lower=visual['image_cluster_bootstrap_95'][0] > 0, harm_each_seed_at_most_5percent=all(x <= .05 for x in damage['rates_by_seed']), harm_mean_95_upper_at_most_5percent=damage['image_cluster_bootstrap_95'][1] <= .05)
    assert all(value('bank', s, q, v) == value('base', s, q, v) for q in panels['offscope'] for s in seeds for v in ['N', 'B'])
    modalities = {m: summary([q for q in clean if rows[q]['modality'] == m]) for m in sorted({rows[q]['modality'] for q in clean})}
    result = {'status': 'COMPLETE_AB_CONFIRMATION_GATE_PASS' if all(gates.values()) else 'COMPLETE_AB_CONFIRMATION_NOT_ESTABLISHED', 'all_gates_pass': all(gates.values()), 'gates': gates, 'summary': summaries, 'effects': effects, 'visual_interaction': visual, 'new_harm_rate': damage, 'modality_summary': modalities, 'offscope_exact_fallback': True, 'cost': {'new_physical_generations': 1614, 'historical_physical_records_reused': 274, 'logical_predictor_readouts': 2942, 'optimizer_steps': 0, 'new_teacher_forwards': 0, 'errors': 0, 'replays': 0, 'supervisor_seconds': end['end_unix'] - receipt['start_unix'], 'original_preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': end['incremental_bytes']}, 'label_lock': {'all_predictions_locked_before_gold_scoring': True, 'gold_scoring_authorized_for_this_single_confirmation': True, 'test_inputs_previously_prechecked': True}, 'limits': ['Answer-lock confirmation on all English officialtest, not complete input or pretraining nonexposure.', 'English train/development image-ID isolation;84test images also occur in unused Chinese annotation rows.', 'Fixed pixel/near-duplicate screen has no flags but does not establish patient independence or exhaustive near-duplicate exclusion.', 'Two immutable parent-seed policies and image sampling intervals do not capture all training or route-selection uncertainty.', 'Black-image anchoring was used in training; a black comparison alone does not prove semantic causality.', 'Frozen external branches are structural preservation, not shared-parameter or clinical safety.', 'No full ViSkill, PPO or clinical skill claim.']}
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    case_numeric = [{'question_group': qmap[q], 'image_group': imap[rows[q]['img_name']], 'independent': q in set(clean), **route[str(q)], 'base_N': value('base', 42, q, 'N'), 'base_B': value('base', 42, q, 'B'), 'bank_N': [value('bank', s, q, 'N') for s in seeds], 'bank_B': [value('bank', s, q, 'B') for s in seeds]} for q in qs]
    logical_numeric = [{'model': model, 'question_group': qmap[q], 'image_group': imap[rows[q]['img_name']], 'view': v, 'correct': bool(z), **bindings[f'{model}|{q}|{v}']} for (model, q, v), z in sorted(logical.items())]
    for name, obj in [('result.json', result), ('case_numeric.json', case_numeric), ('logical_branch_numeric.json', logical_numeric), ('cache_provenance.json', [{k: c[k] for k in ['model', 'source_experiment', 'source_request_number']} for c in cache]), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('input_boundary_revision.json', read(WINDOW / 'input_boundary_revision.json')), ('readout_checks.json', checks), ('prediction_lock.json', locked), ('gold_evaluation_opened.json', read(marker)), ('execution_source.json', read(OUT / 'execution_source.json'))]: write(bundle / name, obj)
    print(json.dumps({'status': result['status'], 'gates': gates, 'summary': summaries, 'effects': effects, 'visual': visual, 'harm': damage, 'cost': result['cost']}), flush=True)


if __name__ == '__main__': close()
