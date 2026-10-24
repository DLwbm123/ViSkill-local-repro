import json
import pytest
from vagen.skills.structured_contract import parse_contract, validate_observed_contract, check_applicability


def test_contract_rejects_wrong_claims_and_unsafe_transfer():
    # Synthetic validator test only, never a model output or E success claim.
    raw=json.dumps({'when': {'goal_offset': [2, 1], 'requires_clear_route': True},
                    'what': [{'action': 'down', 'count': 2}, {'action': 'right', 'count': 1}]})
    row={'traj_success':True,'trajectory_actions':['down','down','right'],
         'trajectory_positions':[(0,0),(1,0),(2,0),(2,1)],
         'initial_env_state':{'desc':['SFF','FFF','FGF'],'player_pos':(0,0)}}
    contract,actions=validate_observed_contract(raw,row)
    assert check_applicability(contract,actions,{'desc':['SFF','HFF','FGF'],'player_pos':(0,0)}) == (False,'hole on route')
    assert not check_applicability(contract,actions,{'desc':['SFG','FFF','FFF'],'player_pos':(0,0)})[0]
    wrong=json.loads(raw);wrong['when']['goal_offset']=[0,2]
    with pytest.raises(ValueError,match='displacement'):validate_observed_contract(json.dumps(wrong),row)
    with pytest.raises(ValueError):parse_contract('```json\n'+raw+'\n```')


@pytest.mark.parametrize('count',[True,0,-1,1.5,1000000])
def test_invalid_action_count_fails_closed(count):
    with pytest.raises(ValueError):
        parse_contract(json.dumps({'when':{'goal_offset':[1,0],'requires_clear_route':True},
                                   'what':[{'action':'down','count':count}]}))
