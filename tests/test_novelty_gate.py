from types import SimpleNamespace
from conftest import provider_config, save
from vagen.skills.distill_callback import DistillCallback


def test_quality_and_existing_skill_gate(provider):
    assert provider.should_distill(6,6,None)
    assert not provider.should_distill(7,6,None)
    assert not provider.should_distill(6,6,.95)
    assert provider.should_distill(6,6,.5)


def test_same_batch_duplicate_geometry_rejected(tmp_path,trajectory,strategy):
    callback=DistillCallback(provider_config(tmp_path/'callback_library'))
    rows=[{'env_name':'FrozenLake','traj_success':True,'retrieved_skill_id':None,'skill_score':None,
           'initial_env_state':trajectory['state'],'seed':s,'trajectory_actions':trajectory['actions'],
           'trajectory_frames':trajectory['frames'],'trajectory_positions':trajectory['positions'],
           'distilled_strategy':strategy['parsed_strategy']} for s in [111,222]]
    batch=SimpleNamespace(non_tensor_batch={k:[x[k] for x in rows] for k in rows[0]})
    metrics=callback.step(batch)
    save('batch_duplicate.json',{'fixture_only':True,'different_seed_same_geometry':True,'metrics':metrics,
         'expected_admissions':1,'actual_admissions':metrics['skill/admitted']})
    assert metrics['skill/admitted']==1, 'published callback does not recheck novelty between insertions'
