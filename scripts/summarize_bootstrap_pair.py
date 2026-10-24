"""Summarize every planned episode; preserve missing/failed conditions explicitly."""
from collections import Counter
import json
import os
from pathlib import Path

ROOT = Path(os.environ['TASK_ROOT'])
BASE = ROOT / 'results/ppo_smoke/bootstrap_card_pair_v1'


def eligible(row):
    return bool(row['traj_success'] and row['bfs'] and len(row['trajectory_actions']) <= row['bfs'])


def summarize(rows):
    multi = [r for r in rows if r['bfs'] is not None and r['bfs'] >= 2]
    return {'episodes': len(rows), 'success': sum(r['traj_success'] for r in rows),
        'quality_eligible': sum(eligible(r) for r in rows), 'multi_step_tasks': len(multi),
        'multi_step_success': sum(r['traj_success'] for r in multi),
        'multi_step_quality': sum(eligible(r) for r in multi),
        'valid_multi_step_distillations': sum(bool((r.get('distillation_audit') or {}).get('valid')) for r in multi),
        'distillation_calls': sum((r.get('distillation_audit') or {}).get('model_calls', 0) for r in rows),
        'injected': sum(r['retrieved_skill_id'] is not None for r in rows),
        'positive_shadow_bonus': sum(r['reward_support']['shadow_bonus_lambda_0_8'] > 0 for r in rows),
        'terminal_reasons': dict(Counter(r['terminal_reason'] for r in rows))}


def paired(a, b):
    assert set(a) == set(b)
    result = {}
    for metric, fn in [('success', lambda r: bool(r['traj_success'])), ('quality', eligible)]:
        pairs = [(fn(a[s]), fn(b[s])) for s in sorted(a)]
        result[metric] = {'both': sum(x and y for x, y in pairs),
            'only_first': sum(x and not y for x, y in pairs),
            'only_second': sum(y and not x for x, y in pairs),
            'neither': sum(not x and not y for x, y in pairs), 'pairs': len(pairs)}
    return result


arms = {}
dev_after = {}
for arm in ['A', 'B']:
    out = BASE / arm
    rows = [json.loads(p.read_text()) for p in (out / 'episodes').glob('*/episode.json')]
    assert len(rows) == 288, f'{arm}: expected288 complete episodes, got{len(rows)}'
    groups = {'train': [r for r in rows if r['phase'] == 'train'],
              'dev_before': [r for r in rows if r['phase'] == 'dev' and r['policy_version'] == 0],
              'dev_after': [r for r in rows if r['phase'] == 'dev' and r['policy_version'] == 4],
              'probe': [r for r in rows if r['phase'] == 'probe']}
    assert {k: len(v) for k, v in groups.items()} == {'train': 128, 'dev_before': 64, 'dev_after': 64, 'probe': 32}
    for values in groups.values():
        assert len({r['seed'] for r in values}) == len(values)
    updates = [json.loads(p.read_text()) for p in (out / 'updated_batches').glob('*.json')]
    assert len(updates) == 4
    assert sorted(r['rollout_policy_version'] for r in updates) == [0, 1, 2, 3]
    model_calls = int((out / 'model_calls.txt').read_text())
    assert model_calls <= 1600
    deltas = [json.loads(line) for p in (out / 'update_audit').glob('*.jsonl') for line in p.read_text().splitlines()]
    assert deltas and all(r['finite'] for r in deltas)
    syncs = [json.loads(line) for p in (out / 'sync_audit').glob('*.jsonl') for line in p.read_text().splitlines()]
    before = {r['seed']: r for r in groups['dev_before']}
    after = {r['seed']: r for r in groups['dev_after']}
    dev_after[arm] = after
    arms[arm] = {'groups': {k: summarize(v) for k, v in groups.items()},
        'dev_paired_before_after': paired(before, after), 'model_calls': model_calls,
        'saved_request_outputs': len(list((out / 'requests').glob('*.json'))),
        'updated_batches': updates, 'optimizer_audit_records': len(deltas),
        'nonzero_optimizer_records': sum(r['sample_delta_l2'] > 0 for r in deltas),
        'actual_sync_return_records': len(syncs),
        'trainer_completion': json.loads((out / 'completion/trainer.json').read_text()),
        'final_model_retained': False, 'independent_recovery_verified': False}
    if arm == 'A':
        assert all(r['retrieved_skill_id'] is None for r in rows)
    (out / 'summary.json').write_text(json.dumps(arms[arm], indent=2))

result = {'arms': arms, 'dev_paired_A_after_vs_B_after': paired(dev_after['A'], dev_after['B']),
    'scope': 'Single training seed, fixed4-batch bootstrap pilot, not converged performance or complete F recovery.',
    'skill_reward_weight': 0, 'reward_training_verified': False,
    'no_checkpoint': True, 'total_model_calls': sum(r['model_calls'] for r in arms.values())}
assert result['total_model_calls'] <= 3200
(BASE / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'summary_complete': True, 'total_model_calls': result['total_model_calls'],
                  'dev_paired_A_after_vs_B_after': result['dev_paired_A_after_vs_B_after']}))
