"""Freeze exact curriculum/new conditional Dev before any model requests."""
import os,json,time
from pathlib import Path
from collections import Counter
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.skills.provider import ViSkillProvider
from vagen.skills.mechanism_audit import jsonable
from vagen.bootstrap_pilot import PilotDataset
from verl.utils.dataset.rl_dataset import collate_fn
ROOT=Path(os.environ['TASK_ROOT']);NAME='fixed_library_ppo_v1';P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());OUT=ROOT/'artifacts/component_tests'/NAME;OUT.mkdir(parents=True,exist_ok=False)
def key(s):return json.dumps(jsonable([s['desc'],s['player_pos']]),sort_keys=True)
def state_for(seed):
 e=FrozenLake(P['environment']);e.gym_env=e._create_gym_env(seed);e.gym_env.reset(seed=seed);s=jsonable(e.get_state_for_skill());e.gym_env=None;return s
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
for seed in list(range(1,21))+list(range(10001,10021))+list(range(20001,20129))+list(range(40001,40033)):
 old.add(key(state_for(seed)))
for low,high in P['exclude_cpu_prefixes']:
 for seed in range(low,high+1):old.add(key(state_for(seed)))
cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system,resolve=True);cfg['library']={'path':str(ROOT/P['source_library']),'max_size':32};cfg['reward']['guided_weight']=0
provider=ViSkillProvider(cfg);provider.init_env_type('FrozenLake');assert len(provider.library._skills)==5
before=(ROOT/P['source_library']/'skill_index.json').read_text();seen=set(old);screen=[];categories={k:[] for k in P['training_curriculum']['per_batch']}
for seed in range(P['train_candidate_bounds'][0],P['train_candidate_bounds'][1]+1):
 assert time.time()<P['preparation_started_unix']+P['max_seconds_pair']
 s=state_for(seed);identity=key(s);duplicate=identity in seen;seen.add(identity);state=dict(s,player_pos=tuple(s['player_pos']));bfs=provider.bfs_distance(state);skill=provider.retrieve(state);eligible=bool(skill and skill['id']=='skill_4' and skill['score']>=.8)
 category='single_BFS1' if bfs==1 else 'multi_up3_applicable' if eligible else 'medium_BFS2_3_without_multicard' if bfs in [2,3] else 'long_BFS4plus' if bfs and bfs>=4 else None
 reason='duplicate' if duplicate else 'ineligible' if category is None else 'quota_full' if len(categories[category])==32 else 'selected'
 screen.append({'seed':seed,'reason':reason,'category':category,'bfs':bfs,'applicable_skill4':eligible})
 if reason=='selected':categories[category].append({'seed':seed,'state':s,'category':category,'bfs':bfs,'group':'train'})
 if all(len(v)==32 for v in categories.values()):break
assert all(len(v)==32 for v in categories.values()),'frozen train pool exhausted'
train=[]
for batch in range(4):
 for category in categories:train.extend(categories[category][batch*8:(batch+1)*8])
conditional=[];dev_screen=[]
for seed in range(P['conditional_dev_candidate_bounds'][0],P['conditional_dev_candidate_bounds'][1]+1):
 s=state_for(seed);identity=key(s);duplicate=identity in seen;seen.add(identity);state=dict(s,player_pos=tuple(s['player_pos']));skill=provider.retrieve(state);eligible=bool(skill and skill['id']=='skill_4' and skill['score']>=.8)
 reason='duplicate' if duplicate else 'selected' if eligible else 'not_applicable'
 dev_screen.append({'seed':seed,'reason':reason})
 if reason=='selected':conditional.append({'seed':seed,'state':s,'bfs':provider.bfs_distance(state),'group':'conditional'})
 if len(conditional)==32:break
assert len(conditional)==32,'frozen conditional Dev pool exhausted'
prior=json.loads((ROOT/'artifacts/component_tests/bootstrap_card_pair_v1/data_manifest.json').read_text());natural=[dict(r,group='natural') for r in prior['splits']['dev']];assert len(natural)==64
splits={'train':train,'dev':natural+conditional}
for phase,rows in splits.items():
 spec={'envs':[{'name':'FrozenLake','n_envs':len(rows),'data_source':'frozenlake','seed_list':[r['seed'] for r in rows]+[-1],'max_turns':5,'response_length_per_turn':512,'config':P['environment']}]};target=ROOT/f'src/configs/smoke/fixed_library_{phase}.yaml';OmegaConf.save(OmegaConf.create(spec),target)
 dataset=PilotDataset(str(target),OmegaConf.create({}));loader=StatefulDataLoader(dataset,batch_size=32 if phase=='train' else 8,shuffle=False,drop_last=phase=='train',num_workers=0,collate_fn=collate_fn)
 batches=list(loader);assert len(batches)==(4 if phase=='train' else 12);assert [int(s) for b in batches for s in b['seed']]==[r['seed'] for r in rows]
assert before==(ROOT/P['source_library']/'skill_index.json').read_text()
result={'status':'CPU_PASS','historical_excluded':len(old),'model_calls':0,'splits':splits,'train_screening':screen,'conditional_dev_screening':dev_screen,'train_screened':len(screen),'conditional_dev_screened':len(dev_screen),'categories':{k:len(v) for k,v in categories.items()},'actual_train_batches':4,'source_library_unchanged':True,'train_dev_exact_overlap':0}
(OUT/'data_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['splits','train_screening','conditional_dev_screening']}))
