"""CPU-only exact map separation and actual dataloader checks, before model calls."""
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.bootstrap_pilot import PilotDataset
from vagen.skills.mechanism_audit import jsonable
from verl.utils.dataset.rl_dataset import collate_fn

ROOT = Path(os.environ['TASK_ROOT'])
SOURCE = ROOT / 'src'
NAME = 'bootstrap_card_pair_v1'
protocol = json.loads((SOURCE / f'configs/smoke/{NAME}.json').read_text())
out = ROOT / 'artifacts/component_tests' / NAME
out.mkdir(parents=True, exist_ok=True)


def key(state):
    return json.dumps(jsonable([state['desc'], state['player_pos']]), sort_keys=True)


def state_for(seed):
    env = FrozenLake(protocol['environment'])
    env.gym_env = env._create_gym_env(seed)
    env.gym_env.reset(seed=seed)
    state = jsonable(env.get_state_for_skill())
    prompt = env.get_system_prompt()
    env.gym_env.close()
    assert 'Movement is deterministic.' in prompt and 'Frozen tiles are slippery' not in prompt
    return state, prompt


old = set()
def collect_states(value):
    if isinstance(value, dict):
        if 'desc' in value and 'player_pos' in value:
            old.add(key(value))
        for child in value.values():
            collect_states(child)
    elif isinstance(value, list):
        for child in value:
            collect_states(child)

for p in (ROOT / 'artifacts/component_tests').rglob('*.json'):
    if out in p.parents:
        continue
    try:
        collect_states(json.loads(p.read_text()))
    except (ValueError, TypeError, KeyError):
        pass
for p in (ROOT / 'artifacts/component_tests/dynamics_wording_v1').glob('*/paired_input_check.json'):
    old.add(key(json.loads(p.read_text())['state']))
# Earlier development seed families remain excluded even if their old runtime file is absent.
for seed in list(range(1, 21)) + list(range(10001, 10021)) + list(range(20001, 20129)) + list(range(40001, 40033)):
    old.add(key(state_for(seed)[0]))

splits = {}
seen = set(old)
rejected = []
counts = {'train': 128, 'dev': 64, 'probe': 32, 'test': 256}
for name, bounds in protocol['candidate_data_not_yet_frozen'].items():
    if not isinstance(bounds, list):
        continue
    phase = {'train_seeds_inclusive': 'train', 'dev_seeds_inclusive': 'dev',
             'final_train_probe_seeds_inclusive': 'probe', 'future_test_seeds_inclusive': 'test'}[name]
    rows = []
    for seed in range(bounds[0], bounds[1] + 1):
        state, prompt = state_for(seed)
        identity = key(state)
        if identity in seen:
            rejected.append({'phase': phase, 'seed': seed, 'reason': 'exact board/position already used'})
            continue
        seen.add(identity)
        rows.append({'seed': seed, 'state': state})
        if len(rows) == counts[phase]:
            break
    assert len(rows) == counts[phase], f'fixed CPU candidate pool exhausted for {phase}'
    splits[phase] = rows
    if phase != 'test':
        spec = {'envs': [{'name': 'FrozenLake', 'n_envs': len(rows), 'data_source': 'frozenlake',
            'seed_list': [r['seed'] for r in rows] + [-1], 'max_turns': 5,
            'response_length_per_turn': 512, 'config': protocol['environment']}]}
        target = SOURCE / f'configs/smoke/bootstrap_{phase}.yaml'
        OmegaConf.save(OmegaConf.create(spec), target)
        dataset = PilotDataset(str(target), OmegaConf.create({}))
        assert [int(r['seed']) for r in dataset.items] == [r['seed'] for r in rows]
        loader = StatefulDataLoader(dataset, batch_size=32 if phase == 'train' else 8,
            shuffle=False, drop_last=phase == 'train', num_workers=0, collate_fn=collate_fn)
        batches = list(loader)
        assert len(batches) == (4 if phase == 'train' else len(rows) // 8)
        assert [int(s) for b in batches for s in b['seed']] == [r['seed'] for r in rows]

manifest = {'status': 'CPU_DATA_PREFLIGHT_PASS', 'historical_exact_states': len(old),
            'splits': splits, 'system_prompt': prompt, 'actual_train_batches': 4,
            'sentinel_excluded': True, 'no_model_calls': True,
            'excluded_duplicate_candidates': rejected,
            'selection': 'Ascending fixed candidate ranges; first fixed count of exact new board/position pairs; no BFS/outcome selection.',
            'map_overlap_rule': 'Exact board and initial player position; no transformations introduced.'}
(out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'status': manifest['status'], 'counts': {k: len(v) for k, v in splits.items()},
                  'historical_exact_states': len(old)}))
