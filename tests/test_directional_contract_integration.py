import json
from types import SimpleNamespace
from vagen.skills.distill_callback import DistillCallback
from vagen.skills.provider import ViSkillProvider


def sample():
    strategy=json.dumps({'when':{'goal_direction':'down_right','goal_manhattan_distance':3,'requires_clear_route':True},
                         'what':[{'action':'down','count':2},{'action':'right','count':1}]})
    return {'traj_success':True,'seed':1,'skill_score':None,'retrieved_skill_id':None,
            'initial_env_state':{'desc':['SFF','FFF','FGF'],'player_pos':(0,0),'is_slippery':False,'slip_prob':0},
            'trajectory_actions':['down','down','right'],'trajectory_positions':[(0,0),(1,0),(2,0),(2,1)],
            'trajectory_frames':[],'distilled_strategy':strategy}


def make_callback(path, enabled):
    cb=DistillCallback({'library':{'path':str(path)},'distill':{'mode':'self','quality_ratio':1,
        'representation':'directional_contract_v1' if enabled else 'free_text','recheck_batch_novelty':True},
        'retrieval':{'contract_guard':enabled}})
    cb.provider.init_env_type('FrozenLake')
    cb.provider.render_trajectory=lambda *a,**k: None  # unit test of admission, not card rendering
    return cb


def step(cb,row):
    return cb.step(SimpleNamespace(non_tensor_batch={**{k:[v] for k,v in row.items()},'env_name':['FrozenLake']}))


def test_invalid_caption_rejected_only_in_explicit_mode(tmp_path):
    row=sample();row['distilled_strategy']=row['distilled_strategy'].replace('down_right','up_right')
    assert step(make_callback(tmp_path/'original',False),row)['skill/admitted']==1
    assert step(make_callback(tmp_path/'guarded',True),row)['skill/admitted']==0


def test_persistent_contract_and_retrieval_guard(tmp_path):
    cb=make_callback(tmp_path/'valid',True);row=sample()
    assert step(cb,row)['skill/admitted']==1
    reloaded=ViSkillProvider(cb.provider.config);reloaded.init_env_type('FrozenLake')
    assert reloaded.retrieve(row['initial_env_state'])['strategy']==row['distilled_strategy']
    bad={**row['initial_env_state'],'desc':['SFF','HFF','FGF']}
    assert reloaded.library.retrieve(bad) is not None
    assert reloaded.retrieve(bad) is None
    unknown={k:v for k,v in row['initial_env_state'].items() if k!='is_slippery'}
    assert reloaded.retrieve(unknown) is None
    assert reloaded.retrieve({**row['initial_env_state'],'slip_prob':.02}) is None


def test_unknown_dynamics_rejects_admission(tmp_path):
    row=sample();row['initial_env_state'].pop('slip_prob')
    assert step(make_callback(tmp_path/'unknown',True),row)['skill/admitted']==0
