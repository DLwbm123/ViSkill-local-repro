"""Read-only saved-state audit; no new generation or library writes."""
import json
from pathlib import Path
from vagen.skills.provider import ViSkillProvider

root = Path(__file__).resolve().parents[1]
base = root / 'artifacts/component_tests/skill_capacity_v1'
out = root / 'docs/evidence/components/retrieval_coverage_v1'
p = ViSkillProvider({'library': {'path': str(base / 'diagnostic_library')},
    'cold_start': {'enable': False}, 'distill': {'mode': 'self',
    'representation': 'directional_contract_v1', 'allow_assistant_envelope': True},
    'retrieval': {'similarity_weight': .9, 'min_score': .8, 'contract_guard': True}})
p.init_env_type('FrozenLake')
states = []
for fn in sorted(base.glob('episode_*/episode.json')):
    d = json.loads(fn.read_text())
    states.append(('capacity128', d['seed'], d['initial_env_state']))
for fn in sorted((root / 'artifacts/component_tests/dynamics_wording_v1').glob('*/paired_input_check.json')):
    states.append(('dynamics32', int(fn.parent.name), json.loads(fn.read_text())['state']))
assert len(states) == 160 and len(p.library._skills) == 3
rows = []
for group, seed, state in states:
    state['player_pos'] = tuple(state['player_pos'])
    raw = p.library.retrieve(state)
    current = p.validate_retrieved_skill(raw, state)
    query = p.library.feature_extractor.extract(state)
    candidates = [{**s, 'score': p.library._score(query, s)} for s in p.library._skills]
    valid = [s for s in candidates if p.validate_retrieved_skill(s, state) is not None]
    best = max(valid, key=lambda s: s['score']) if valid else None
    current_ready = bool(current and current['score'] >= .8)
    best_ready = bool(best and best['score'] >= .8)
    assert not current_ready or best_ready
    rows.append({'group': group, 'seed': seed, 'bfs': p.bfs_distance(state),
        'top1_id': raw['id'], 'current_eligible': current_ready, 'any_eligible': best_ready,
        'missed_eligible_alternative': best_ready and not current_ready,
        'best_valid_id': best['id'] if best else None,
        'best_valid_score': best['score'] if best else None})
summary = {}
for group in ['capacity128', 'dynamics32']:
    r = [x for x in rows if x['group'] == group]
    summary[group] = {'states': len(r),
        'current_eligible': sum(x['current_eligible'] for x in r),
        'any_eligible': sum(x['any_eligible'] for x in r),
        'missed_eligible_alternative': sum(x['missed_eligible_alternative'] for x in r),
        'multi_step_states': sum(x['bfs'] is not None and x['bfs'] > 1 for x in r),
        'multi_step_any_eligible': sum(x['any_eligible'] and x['bfs'] is not None and x['bfs'] > 1 for x in r)}
out.mkdir(parents=True, exist_ok=True)
(out / 'result.json').write_text(json.dumps({'summary': summary,
    'reference_action_lengths': [len(s['trajectory']) for s in p.library._skills],
    'model_calls': 0, 'library_modified': False, 'performance_claim': False,
    'incident': 'First audit hit JSON list player_pos in BFS; normalize restored position to tuple, no model calls or saved input edits.',
    'rows': rows}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
