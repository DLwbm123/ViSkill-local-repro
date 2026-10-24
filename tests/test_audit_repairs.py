"""Explicit repair route; original fidelity failures remain in their audit files."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from vagen.skills.reward import compute_state_progress_reward,compute_reward
from vagen.skills.distill_callback import DistillCallback
from vagen.skills.solver import FrozenLakeSolver
from vagen.skills.library import ViSkillLibrary


def test_state_progress_and_extra_step_penalty():
    ref=[(0,0),(0,1),(0,2),(1,2),(2,2)]
    alternative=[(3,4),(4,4),(4,5),(4,6)]
    assert compute_state_progress_reward(ref,alternative)==.75
    assert compute_state_progress_reward(ref,alternative+[(4,5)])==.5
    assert compute_state_progress_reward(ref,[(0,0),(0,-1)])==0


def test_explicit_upper_clip_and_no_success_bonus():
    ref=[(0,0),(1,0),(1,1),(0,1),(0,2)]
    shortcut=[(0,0),(0,1),(0,2)]
    assert compute_state_progress_reward(ref,shortcut,bounded=False)==1.5
    assert compute_state_progress_reward(ref,shortcut)==1
    assert compute_reward([.02],'state_progress_bounded',False,['down','right','up','right'],['right','right'],.8,ref,shortcut)==pytest.approx(.82)
    assert compute_reward([1.02],'state_progress_bounded',True,['right'],['right'],.8)==1.02


def test_missing_state_must_not_fall_back_to_action_prefix():
    with pytest.raises(ValueError,match='reference position'):
        compute_reward([.02],'state_progress_bounded',False,['right'],['right'],.8)


def test_opt_in_batch_novelty_and_position_roundtrip(tmp_path):
    cfg={'library':{'path':str(tmp_path/'library'),'max_size':10},'distill':{'mode':'self','quality_ratio':1,'min_score':.9,'recheck_batch_novelty':True},'render':{'trajectory':{'thumbnail_size':120,'draw_path':True}}}
    cb=DistillCallback(cfg)
    old=json.loads((Path(os.environ['TASK_ROOT'])/'artifacts/component_tests/original_v1/strategy.json').read_text())
    state={'desc':[list(s) for s in ['SFFF','FHHF','FFFF','FFFG']],'player_pos':(0,0)}
    rows=[]
    for seed in [111,222]:
        replay=FrozenLakeSolver().replay(state)
        rows.append({'env_name':'FrozenLake','traj_success':True,'skill_score':None,'seed':seed,
          'initial_env_state':state,'trajectory_actions':replay['actions'],'trajectory_frames':replay['frames'],
          'trajectory_positions':replay['positions'],'distilled_strategy':old['parsed_strategy']})
    novel={'desc':[list(s) for s in ['GFFF','SFFF','FFFF','FFFF']],'player_pos':(1,0)}
    replay=FrozenLakeSolver().replay(novel)
    rows.append(dict(rows[0],seed=333,initial_env_state=novel,trajectory_actions=replay['actions'],trajectory_frames=replay['frames'],trajectory_positions=replay['positions']))
    batch=SimpleNamespace(non_tensor_batch={k:[x[k] for x in rows] for k in rows[0]})
    metrics=cb.step(batch)
    assert metrics['skill/admitted']==2
    loaded=ViSkillLibrary(cb.provider.library.feature_extractor);loaded.load(cfg['library']['path'])
    assert [x['seed'] for x in loaded._skills]==[111,333]
    assert len(loaded._skills[0]['positions'])==len(loaded._skills[0]['trajectory'])+1
    output=Path(os.environ['TASK_ROOT'])/'artifacts/component_tests/repairs_v1'
    output.mkdir(parents=True,exist_ok=True)
    (output/'novelty_metrics.json').write_text(json.dumps({'metrics':metrics,'retained_seeds':[x['seed'] for x in loaded._skills]},indent=2))
