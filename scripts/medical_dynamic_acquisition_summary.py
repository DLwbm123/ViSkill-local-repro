"""Close the fixed current-case tool experiment without replaying old responses."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'dynamic_acquisition_tools_v1'
OUT = ROOT / 'artifacts/medical' / NAME
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
sys.path.insert(0, str(SRC / 'scripts'))
from medical_dynamic_acquisition import parse
from medical_region_reader import read, write
from slake_probe import score


def close():
    p = read(SRC / 'configs/reproduction' / (NAME + '.json'))
    end = read(ROOT / 'state/reproduction' / (NAME + '_EXIT.json'))
    receipt = read(ROOT / 'state/reproduction' / (NAME + '_receipt.json'))
    done = read(OUT / 'WORKER_COMPLETED.json')
    assert end['exit_code'] == 0 and end['reason'] is None and not end['argv_violations']
    assert end['end_unix'] < p['deadline_unix'] and end['incremental_bytes'] < p['maximum_incremental_bytes']
    assert done['status'] == 'COMPLETE_DYNAMIC_ACQUISITION_TOOLS' and done['optimizer_steps'] == 0 and not done['test_annotation_opened']
    records = [read(f) for f in sorted((OUT / 'records').glob('*.json'))]
    assert len(records) == done['new_generation_calls'] <= p['maximum_generation_calls']
    assert [r['number'] for r in records] == list(range(1, len(records) + 1))
    events = [json.loads(s) for s in (OUT / 'requests.jsonl').read_text().splitlines()]
    assert Counter(r['event'] for r in events) == {'intent': len(records), 'result': len(records)}
    for event in ('intent', 'result'):
        assert [r['number'] for r in events if r['event'] == event] == list(range(1, len(records) + 1))
    m = read(OUT / 'private_manifest.json')
    items = {r['qid']: r for r in m['validation'] + m['background']}
    assert len(items) == 323 and sum(m['admit'].values()) == 272
    assert all(a != b for a, b in m['wrong_image'].items())
    expert = [r for r in records if r['stage'] == 'expert']
    readers = [r for r in records if r['stage'] == 'reader']
    assert len(expert) == done['expert_generations'] == 436
    expected_expert = {(s, i, kind) for s in p['expert_seeds'] for i in range(-1, 108) for kind in ('modality', 'plane')}
    assert {(r['expert_seed'], r['image_group'], r['kind']) for r in expert} == expected_expert
    assert all(len(r['output_ids']) <= 8 for r in expert)
    parsed = {(r['expert_seed'], r['image_group'], r['kind']): parse(r['kind'], r['raw']) for r in expert}
    def tags(seed, name):
        i = -1 if name is None else m['admitted_images'].index(name)
        return (parsed[seed, i, 'modality'], parsed[seed, i, 'plane'])
    feasibility = {}
    collisions = {}
    for seed in p['expert_seeds']:
        primary = m['primary_images']
        joint = sum(tags(seed, n) != tags(seed, None) and tags(seed, n) != tags(seed, m['wrong_image'][n]) for n in primary)
        valid = {tags(seed, n) for n in primary if 'Unknown' not in tags(seed, n)}
        feasibility[str(seed)] = {'joint_contrast_images': joint, 'distinct_fully_parsed_pairs': len(valid), 'unknown_field_count': sum(t == 'Unknown' for n in m['admitted_images'] for t in tags(seed, n)), 'pass': joint >= 20 and len(valid) >= 3}
        collisions[str(seed)] = {'primary_images': len(primary), 'current_equals_blank': sum(tags(seed, n) == tags(seed, None) for n in primary), 'current_equals_wrong': sum(tags(seed, n) == tags(seed, m['wrong_image'][n]) for n in primary)}
    assert read(OUT / 'EXPERT_COMPLETED.json')['feasibility'] == feasibility
    assert done['feasibility_pass'] == all(x['pass'] for x in feasibility.values())
    cost = {'supervisor_wall_seconds': end['end_unix'] - receipt['start_unix'], 'preparation_to_exit_seconds': end['end_unix'] - p['first_preparation_unix'], 'incremental_bytes_at_exit': end['incremental_bytes']}
    result = {'status': 'COMPLETE_TOOL_FEASIBILITY_STOP', 'feasibility_pass': done['feasibility_pass'], 'feasibility': feasibility, 'expert_collisions': collisions, 'calls': {'new_generations': len(records), 'expert_generations': len(expert), 'reader_generations': len(readers), 'planned_logical_outcomes': 2584, 'actual_logical_outcomes': done['logical_reader_outcomes'], 'errors': 0, 'completed_replays': 0, 'optimizer_steps': 0}, 'cost': cost, 'test_annotation_opened': False}
    if not done['feasibility_pass']:
        assert not readers and done['reader_generations'] == done['logical_reader_outcomes'] == 0
        result['accuracy_effect_measured'] = False
    else:
        assert read(OUT / 'BASE_READER_CHECK.json')['all_lora_layers_disabled']
        physical = read(OUT / 'private_physical_reader_plan.json')
        resolved = read(OUT / 'private_resolved_logical_plan.json')
        logical = read(OUT / 'logical_plan.json')
        assert len(resolved) == len(logical) == done['logical_reader_outcomes'] == 2584
        assert len(physical) == len(readers) == done['reader_generations'] <= 1904
        assert [r['physical_index'] for r in readers] == list(range(len(readers)))
        assert [r['number'] for r in readers] == list(range(437, 437 + len(readers)))
        assert (OUT / 'private_physical_reader_plan.json').stat().st_mtime <= min(r['started_unix'] for r in events if r['event'] == 'intent' and r['stage'] == 'reader')
        keys = []
        for i, e in enumerate(physical):
            row = items[e['qid']]
            keys.append((row['img_name'], row['question'], tuple(e['tags'])))
            assert readers[i]['qid'] == e['qid']
        assert len(set(keys)) == len(keys)
        cohort = read(COHORT / 'private_manifest.json')
        old_items = {r['qid']: r for k in ('new_validation', 'old_source', 'old_background', 'old_validation', 'old_acquisition', 'all_validation') for r in cohort[k]}
        assert len(old_items) == 1191
        qold = {i: q for i, q in enumerate(sorted(old_items))}
        assert all(items[q]['answer'] == old_items[q]['answer'] and items[q]['img_name'] == old_items[q]['img_name'] for q in items)
        catalog = read(COHORT / 'public_bundle/physical_branch_numeric.json')
        required = {('base', q) for q in items}
        def before_model(seed, q):
            flag = m['before_flags'][str(q)]
            return f'A_{seed}' if flag['A'] else (f'I_{seed}' if flag['B'] else 'base')
        required.update((before_model(s, q), q) for s in p['expert_seeds'] for q in items)
        legacy = {(r['model'], qold[r['question_group']]): r for r in catalog if r['view'] == 'N' and (r['model'], qold[r['question_group']]) in required}
        assert set(legacy) == required and len(legacy) == 345
        mapped = {}
        for i, (e, planned) in enumerate(zip(resolved, logical)):
            assert e['logical_index'] == i and all(e[k] == planned[k] for k in ('seed', 'qid', 'condition'))
            q, seed, condition = e['qid'], e['seed'], e['condition']
            if not m['admit'][str(q)]:
                assert e['reference_model'] == before_model(seed, q) and 'physical_index' not in e
            else:
                pair = ('Unknown', 'Unknown')
                if condition == 'D': pair = tags(seed, items[q]['img_name'])
                elif condition == 'B': pair = tags(seed, None)
                elif condition == 'W': pair = tags(seed, m['wrong_image'][items[q]['img_name']])
                assert keys[e['physical_index']] == (items[q]['img_name'], items[q]['question'], pair)
            mapped[seed, q, condition] = e
        assert len(mapped) == 2584 and sum('reference_model' in e for e in resolved) == 408
        qmap = {q: i for i, q in enumerate(sorted(items))}
        imap = {n: i for i, n in enumerate(sorted({r['img_name'] for r in items.values()}))}
        def value(seed, q, condition):
            if condition == 'O': return int(legacy[before_model(seed, q), q]['correct'])
            e = mapped[seed, q, condition]
            if 'reference_model' in e: return int(legacy[e['reference_model'], q]['correct'])
            return int(score(readers[e['physical_index']]['raw'], items[q]['answer']))
        panels = {'primary': sorted(r['qid'] for r in m['validation']), 'old_background': sorted(r['qid'] for r in m['background'])}
        def summarize(qs):
            return {c: {'correct_by_expert_seed': [sum(value(s, q, c) for q in qs) for s in p['expert_seeds']], 'questions': len(qs)} for c in ('O', 'N', 'D', 'B', 'W')}
        def effect(qs, a, b):
            matrix = np.array([[value(s, q, a) - value(s, q, b) for q in qs] for s in p['expert_seeds']])
            averaged = matrix.mean(0)
            groups = defaultdict(list)
            for i, q in enumerate(qs): groups[items[q]['img_name']].append(i)
            groups = list(groups.values())
            num, den = np.array([averaged[g].sum() for g in groups]), np.array([len(g) for g in groups])
            sampled = np.random.default_rng(p['selection_seed']).integers(0, len(groups), size=(10000, len(groups)))
            bootstrap = num[sampled].sum(1) / den[sampled].sum(1)
            return {'mean': float(averaged.mean()), 'image_cluster_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(), 'questions': len(qs), 'images': len(groups), 'by_frozen_expert_seed': [{'seed': s, 'mean': float(d.mean()), 'improved_question_groups': [qmap[q] for q, v in zip(qs, d) if v > 0], 'harmed_question_groups': [qmap[q] for q, v in zip(qs, d) if v < 0]} for s, d in zip(p['expert_seeds'], matrix)]}
        effects = {f'primary_{a}_minus_{b}': effect(panels['primary'], a, b) for a, b in [('D', 'N'), ('D', 'O'), ('D', 'B'), ('D', 'W'), ('N', 'O')]}
        effects['background_D_minus_O'] = effect(panels['old_background'], 'D', 'O')
        gates = {}
        for control, threshold in [('N', .10), ('O', .10), ('B', .05), ('W', .05)]:
            e = effects[f'primary_D_minus_{control}']
            gates[f'D_minus_{control}_mean_threshold'] = e['mean'] >= threshold
            gates[f'D_minus_{control}_each_expert_positive'] = all(s['mean'] > 0 for s in e['by_frozen_expert_seed'])
            gates[f'D_minus_{control}_positive_95_lower'] = e['image_cluster_bootstrap_95'][0] > 0
        e = effects['background_D_minus_O']
        gates.update(background_mean_nonnegative=e['mean'] >= 0, background_each_expert_nonnegative=all(s['mean'] >= 0 for s in e['by_frozen_expert_seed']), background_95_lower_at_least_minus_5pp=e['image_cluster_bootstrap_95'][0] >= -.05, background_zero_new_harm_each_expert=all(not s['harmed_question_groups'] for s in e['by_frozen_expert_seed']))
        cohort_flags = read(COHORT / 'private_contracts.json')['flags']
        gates['protected_A_B_registry_unchanged'] = all(not cohort_flags[str(r['qid'])]['C'] for k in ('old_source', 'old_validation', 'old_acquisition') for r in cohort[k])
        result.update(status='COMPLETE_DYNAMIC_TOOL_GATE_PASS' if all(gates.values()) else 'COMPLETE_DYNAMIC_TOOL_NOT_ESTABLISHED', accuracy_effect_measured=True, all_gates_pass=all(gates.values()), gates=gates, summary={k: summarize(qs) for k, qs in panels.items()}, paired_effects=effects, groups={kind: {'summary': summarize(qs), 'D_minus_O': effect(qs, 'D', 'O')} for kind in sorted({items[q]['modality'] for q in panels['primary']}) for qs in [[q for q in panels['primary'] if items[q]['modality'] == kind]]})
        result['calls'].update(reused_verified_numeric_reference_branches=len(legacy), actual_new_logical_outcomes=2584, logical_reference_outcomes=408, actual_unique_reader_inputs=len(readers), reader_input_deduplication_savings=2176 - len(readers))
        numeric = [{'logical_group': i, 'question_group': qmap[e['qid']], 'image_group': imap[items[e['qid']]['img_name']], 'expert_seed': e['seed'], 'condition': e['condition'], 'correct': bool(value(e['seed'], e['qid'], e['condition'])), **({'new_reader_physical_index': e['physical_index']} if 'physical_index' in e else {'reused_reference_model': e['reference_model']})} for i, e in enumerate(resolved)]
        reference_numeric = [dict(r, question_group=qmap[q], image_group=imap[items[q]['img_name']]) for (model, q), r in sorted(legacy.items())]
        write(OUT / 'private_numeric_ready.json', {'logical': numeric, 'reference': reference_numeric})
    result['limitations'] = ['Previously observed development images and original unverified author QA; unknown patient independence.', 'Two frozen expert weight instances share one base reader; equal physical inputs share exact outputs.', 'Canonical tool predictions are not clinically certified acquisition labels.', 'Fixed two-action scope policy, not learned action selection, PPO, visual cards or full ViSkill.', 'Historical reference correctness was reused from the verified previous closure, not recomputed or regenerated.']
    write(OUT / 'RESULT.json', result)
    bundle = OUT / 'public_bundle'
    bundle.mkdir(exist_ok=False)
    for name, obj in [('result.json', result), ('protocol.json', p), ('cpu_check.json', read(OUT / 'cpu_check.json')), ('execution_source.json', read(OUT / 'execution_source.json'))]:write(bundle / name, obj)
    if done['feasibility_pass']:
        numeric = read(OUT / 'private_numeric_ready.json')
        write(bundle / 'logical_numeric.json', numeric['logical'])
        write(bundle / 'reference_numeric.json', numeric['reference'])
    write(bundle / 'physical_call_numeric.json', [{k: r[k] for k in ('number', 'stage') if k in r} | {'output_tokens': len(r['output_ids'])} | {k: r[k] for k in ('expert_seed', 'kind', 'view', 'physical_index') if k in r} | ({'expert_image_group': r['image_group']} if 'image_group' in r else {}) for r in records])
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': close()
