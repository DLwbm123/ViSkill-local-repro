import json
from types import SimpleNamespace
from vagen.skills.reward import compute_reward
from test_directional_contract_integration import sample, make_callback, step


def test_failure_bonus_requires_intermediate_progress():
    args=dict(env_rewards=[0.02], mode='state_progress_bounded', guided_weight=.8)
    for success, positions in [(False, [(0,0),(0,0)]), (True, [(0,0),(1,0)])]:
        assert compute_reward(**args, traj_success=success, skill_trajectory=['Down'],
            model_trajectory=['Down'], reference_positions=[(0,0),(1,0)],model_positions=positions)==.02
    value=compute_reward(**args,traj_success=False,skill_trajectory=['Down','Right'],
        model_trajectory=['Down'],reference_positions=[(0,0),(1,0),(1,1)],model_positions=[(0,0),(1,0)])
    assert abs(value-.42)<1e-9  # Fixture only, not observed model reward activation.


def test_admission_funnel_preserves_gates(tmp_path, monkeypatch):
    monkeypatch.setenv('MECHANISM_AUDIT_DIR',str(tmp_path/'audit'))
    cb=make_callback(tmp_path/'lib',True)
    failed=sample(); failed.update(seed=2,traj_success=False)
    long=sample();long.update(seed=3,trajectory_actions=['down']*4)
    invalid=sample();invalid.update(seed=4,distilled_strategy='bad')
    valid=sample();duplicate=sample()
    rows=[failed,long,invalid,valid,duplicate]
    batch=SimpleNamespace(non_tensor_batch={**{k:[r[k] for r in rows] for k in valid},'env_name':['FrozenLake']*5})
    assert cb.step(batch)['skill/admitted']==1
    records=list((tmp_path/'audit/admissions').glob('*.json'));assert len(records)==1
    data=json.loads(records[0].read_text())
    assert [r['admission_audit'] for r in data['rows']]==['not_success','quality_ratio','invalid_directional_strategy','admitted','duplicate_seed']
    assert data['after']['skills'][0]['id']=='skill_0'


def test_prior_semantic_rejection_is_not_misreported_as_empty_json(tmp_path, monkeypatch):
    monkeypatch.setenv('MECHANISM_AUDIT_DIR',str(tmp_path/'audit'))
    cb=make_callback(tmp_path/'lib',True);row=sample()
    row['distilled_strategy']=''
    row['distillation_audit']={'valid':False,'reason':'relative goal condition mismatch','raw':'saved raw','model_calls':1}
    assert step(cb,row)['skill/admitted']==0
    record=json.loads(next((tmp_path/'audit/admissions').glob('*.json')).read_text())['rows'][0]
    assert record['admission_error']=='relative goal condition mismatch'
    assert record['distillation_audit']['raw']=='saved raw'
