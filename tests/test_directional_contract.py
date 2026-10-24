import json
import pytest
from vagen.skills.directional_contract import parse_directional_contract, validate_directional_observation, check_directional_applicability


def test_lossless_wrapper_and_rejection_without_correcting_claim():
    obj={'when':{'goal_direction':'down_right','goal_manhattan_distance':3,'requires_clear_route':True},
         'what':[{'action':'down','count':2},{'action':'right','count':1}]}
    raw='```json\n'+json.dumps(obj)+'\n```'
    row={'traj_success':True,'trajectory_actions':['down','down','right'],
         'trajectory_positions':[(0,0),(1,0),(2,0),(2,1)],
         'initial_env_state':{'desc':['SFF','FFF','FGF'],'player_pos':(0,0)}}
    contract,actions=validate_directional_observation(raw,row)
    assert contract==obj
    assert check_directional_applicability(contract,actions,{'desc':['SFF','HFF','FGF'],'player_pos':(0,0)})==(False,'hole on route')
    obj['when']['goal_direction']='up_right'
    with pytest.raises(ValueError,match='condition mismatch'):validate_directional_observation(json.dumps(obj),row)
    with pytest.raises(ValueError):parse_directional_contract(raw+' explanation')


@pytest.mark.parametrize('distance',[True,0,-1,1.5])
def test_distance_type_is_strict(distance):
    with pytest.raises(ValueError):parse_directional_contract(json.dumps({'when':{'goal_direction':'down','goal_manhattan_distance':distance,'requires_clear_route':True},'what':[{'action':'down','count':1}]}))


def test_optional_single_role_envelope_does_not_edit_json():
    obj={'when':{'goal_direction':'left','goal_manhattan_distance':1,'requires_clear_route':True},
         'what':[{'action':'left','count':1}]}
    raw='\nassistant\n```json\n'+json.dumps(obj)+'\n```'
    with pytest.raises(ValueError):parse_directional_contract(raw)
    assert parse_directional_contract(raw,allow_assistant_envelope=True)[0]==obj
    for bad in ['assistant\n'+raw,raw+'\nextra',raw+'\n'+json.dumps(obj)]:
        with pytest.raises(ValueError):parse_directional_contract(bad,allow_assistant_envelope=True)
