"""Freeze unseen board quotas on CPU; this never runs a model or env.step."""
import json,os,time
from pathlib import Path
from collections import Counter
from omegaconf import OmegaConf
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.skills.provider import ViSkillProvider
from vagen.skills.mechanism_audit import jsonable

ROOT=Path(os.environ['TASK_ROOT']);NAME='value_head_single_step_v1'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
OUT=ROOT/f'artifacts/component_tests/{NAME}'
assert not OUT.exists(),'refuse replacing frozen selection'
OUT.mkdir(parents=True)

def board(s):return json.dumps(jsonable(s['desc']),sort_keys=True)
def state_for(seed):
    e=FrozenLake(P['environment'])
    try:
        e.gym_env=e._create_gym_env(seed);e.gym_env.reset(seed=seed)
        return jsonable(e.get_state_for_skill())
    finally:
        if e.gym_env is not None:e.gym_env.close()

old=set();seeds=set();paths=0
def collect(x):
    if isinstance(x,dict):
        if 'desc' in x and 'player_pos' in x:old.add(board(x))
        if isinstance(x.get('seed'),int) and 0<x['seed']<130001:seeds.add(x['seed'])
        for value in x.values():collect(value)
    elif isinstance(x,list):
        for value in x:collect(value)

# Board identity is stricter than old (board,player-position) identity.
# Include reserved Test states and all actual screened seed records, without
# running any policy on Test or checking only successful historical episodes.
for base,pattern in [(ROOT/'artifacts/component_tests','*.json'),(ROOT/'results/ppo_smoke','episode.json')]:
    for path in base.rglob(pattern):
        if OUT in path.parents:continue
        try:collect(json.loads(path.read_text()));paths+=1
        except (ValueError,TypeError,KeyError):pass
seeds.update(range(1,21));seeds.update(range(10001,10021))
seeds.update(range(20001,20129));seeds.update(range(40001,40033))
seeds.update(range(70001,71080));seeds.update(range(80001,82019))
for seed in sorted(seeds):old.add(board(state_for(seed)))
cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system,resolve=True)
cfg['library']={'path':str(ROOT/P['source_library']),'max_size':32};cfg['reward']['guided_weight']=0
provider=ViSkillProvider(cfg);provider.init_env_type('FrozenLake')
assert len(provider.library._skills)==5
library=ROOT/P['source_library']/'skill_index.json';before=library.read_text()
seen=set(old);splits={};screenings={}
for phase,quotas,bounds in [('train',P['training_quotas'],P['train_candidate_bounds']),('dev',P['dev_quotas'],P['dev_candidate_bounds'])]:
    chosen={key:[] for key in quotas};screen=[]
    for seed in range(bounds[0],bounds[1]+1):
        assert time.time()<min(P['absolute_total_deadline'],P['preparation_started_unix']+P['max_seconds_including_preparation'])
        s=state_for(seed);key=board(s);duplicate=key in seen;seen.add(key)
        state=dict(s,player_pos=tuple(s['player_pos']));bfs=provider.bfs_distance(state);skill=provider.retrieve(state)
        target=bool(skill and skill['id']=='skill_4' and skill['score']>=.8 and bfs==3)
        if phase=='train':category='multi_up3_applicable' if target else 'single_BFS1' if bfs==1 else 'medium_BFS2_3_without_multicard' if bfs in [2,3] else None
        else:category='conditional' if target else 'natural_multi' if bfs and bfs>1 else None
        reason='duplicate_board' if duplicate else 'ineligible' if category is None else 'quota_full' if len(chosen[category])>=quotas[category] else 'selected'
        screen.append({'seed':seed,'reason':reason,'category':category,'bfs':bfs,'applicable_skill4':target})
        if reason=='selected':chosen[category].append({'seed':seed,'state':s,'bfs':bfs,'category':category,'group':'train' if phase=='train' else category})
        if all(len(chosen[k])==n for k,n in quotas.items()):break
    assert all(len(chosen[k])==n for k,n in quotas.items()),'declared CPU candidate pool exhausted; no automatic widening'
    # Fixed category round-robin retains all32 and avoids an outcome-dependent order.
    rows=[]
    for i in range(max(quotas.values())):
        for category in quotas:
            if i<len(chosen[category]):rows.append(chosen[category][i])
    splits[phase]=rows;screenings[phase]=screen
    spec={'envs':[{'name':'FrozenLake','n_envs':len(rows),'data_source':'frozenlake','seed_list':[r['seed'] for r in rows]+[-1],'max_turns':5,'response_length_per_turn':512,'config':P['environment']}]}
    OmegaConf.save(OmegaConf.create(spec),ROOT/f'src/configs/smoke/value_head_{phase}.yaml')
assert before==library.read_text()
identities=[board(row['state']) for rows in splits.values() for row in rows]
assert len(identities)==48==len(set(identities)) and not set(identities)&old
result={'status':'CPU_PASS','splits':splits,'screening':screenings,'historical_board_count':len(old),'historical_seed_prefix_count':len(seeds),'historical_json_sources':paths,'identity_rule':'desc board only, regardless player position','train_dev_historical_overlap':0,'source_library_unchanged':True,'model_calls':0,'environment_steps':0,'completed_unix':time.time()}
(OUT/'data_manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['splits','screening']}))
