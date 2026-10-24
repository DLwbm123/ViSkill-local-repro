from vagen.envs.frozenlake.frozenlake_env import FrozenLake


def test_explicit_dynamics_correction_matches_both_slip_sources():
    # No reset/model needed: inspect the actual environment prompt builder.
    original=FrozenLake({'is_slippery':False,'slip_prob':0}).get_system_prompt()
    deterministic=FrozenLake({'is_slippery':False,'slip_prob':0,'accurate_dynamics_prompt':True}).get_system_prompt()
    assert 'may move perpendicular' in original
    assert 'Movement is deterministic.' in deterministic
    assert 'may move perpendicular' not in deterministic
    assert original.split('Actions you can take:')[1]==deterministic.split('Actions you can take:')[1]
    for kwargs in [{'is_slippery':True,'slip_prob':0},{'is_slippery':False,'slip_prob':.02}]:
        assert 'may move perpendicular' in FrozenLake({**kwargs,'accurate_dynamics_prompt':True}).get_system_prompt()
