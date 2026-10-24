"""CPU-only frozen conditional selection; no generation, updates or admissions."""
import json,os,time
from pathlib import Path
from collections import Counter
from omegaconf import OmegaConf
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.skills.mechanism_audit import jsonable
from vagen.skills.provider import ViSkillProvider
ROOT=Path(os.environ['TASK_ROOT']);NAME='card_content_controls_v1'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
OUT=ROOT/'artifacts/component_tests'/NAME;OUT.mkdir(parents=True,exist_ok=False)
def key(s):return json.dumps(jsonable([s['desc'],s['player_pos']]),sort_keys=True)
def state_for(seed):
 e=FrozenLake(P['environment']);e.gym_env=e._create_gym_env(seed);e.gym_env.reset(seed=seed)
 s=jsonable(e.get_state_for_skill());e.gym_env=None
 return s
old=set()
def collect(x):
 if isinstance(x,dict):
  if 'desc' in x and 'player_pos' in x:old.add(key(x))
  for v in x.values():collect(v)
 elif isinstance(x,list):
  for v in x:collect(v)
for path in (ROOT/'artifacts/component_tests').rglob('*.json'):
 if OUT in path.parents:continue
 try:collect(json.loads(path.read_text()))
 except (ValueError,TypeError,KeyError):pass
# Includes all480 old bootstrap splits, with256 reserved Test states.
collect(json.loads((ROOT/'artifacts/component_tests/bootstrap_card_pair_v1/data_manifest.json').read_text()))
for seed in list(range(1,21))+list(range(10001,10021))+list(range(20001,20129))+list(range(40001,40033)):
 old.add(key(state_for(seed)))
for seed in range(P['exclude_previous_screened_seed_bounds'][0],P['exclude_previous_screened_seed_bounds'][1]+1):
 old.add(key(state_for(seed)))
cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system,resolve=True)
cfg['library']={'path':str(ROOT/P['source_library']),'max_size':32};cfg['reward']['guided_weight']=0
provider=ViSkillProvider(cfg);provider.init_env_type('FrozenLake');assert len(provider.library._skills)==5
before=(ROOT/P['source_library']/'skill_index.json').read_text()
rows=[];screen=[];seen=set(old)
for seed in range(P['candidate_seed_bounds'][0],P['candidate_seed_bounds'][1]+1):
 assert time.time()<P['preparation_started_unix']+P['max_attempt_seconds']
 state=state_for(seed);identity=key(state);duplicate=identity in seen;seen.add(identity)
 restored=dict(state,player_pos=tuple(state['player_pos']));skill=provider.retrieve(restored)
 applicable=bool(skill and skill['id']==P['source_skill'] and skill['score']>=.8)
 reason='historical_or_pool_duplicate' if duplicate else 'selected' if applicable else 'no_top1_three_step_applicability'
 screen.append({'seed':seed,'reason':reason,'top1_valid_id':skill['id'] if skill else None,'score':skill['score'] if skill else None})
 if reason=='selected':rows.append({'seed':seed,'state':state,'selected_skill_id':skill['id'],'score':skill['score']})
 if len(rows)==P['target_pairs']:break
assert before==(ROOT/P['source_library']/'skill_index.json').read_text()
result={'status':'CPU_PASS' if len(rows)==16 else 'INSUFFICIENT_APPLICABLE_NEW_STATES','model_calls':0,'historical_exact_states':len(old),'screened':len(screen),'reasons':dict(Counter(r['reason'] for r in screen)),'splits':{'dev':rows},'screening':screen,'source_library_unchanged':True,'conditional_only':True}
source_data=json.loads((ROOT/'artifacts/component_tests/bootstrap_card_pair_v1/data_manifest.json').read_text())
source_state=next(x['state'] for x in source_data['splits']['train'] if x['seed']==P['source_state_seed'])
source_skill=next(s for s in provider.library._skills if s['id']==P['source_skill'])
text=source_skill['strategy']+'\nTrajectory example:\nSource grid (F frozen, H hole, S original start, G goal):\n'+'\n'.join(''.join(line) for line in source_state['desc'])+'\nPlayer positions (row,column): '+json.dumps(source_skill['positions'])+'\nActions: '+json.dumps(source_skill['trajectory'])
(OUT/'text_reference.txt').write_text(text)
result['text_reference']=text
(OUT/'data_manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['splits','screening']}))
