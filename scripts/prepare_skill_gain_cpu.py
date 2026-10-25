"""Freeze unseen boards, original retrieval eligibility and all384 positions before generation."""
import os,json,time,itertools,random,subprocess
from pathlib import Path
import torch
from omegaconf import OmegaConf
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.skills.provider import ViSkillProvider
from vagen.official_train_observer import plain,write
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='skill_gain_384_v1';OUT=ROOT/'artifacts/reproduction'/NAME;P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text());OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'data_manifest.json').exists()
def key(desc):return '/'.join(''.join(str(c) for c in row) for row in desc)
historical=set();files=0

def collect(obj):
    if isinstance(obj,dict):
        if 'desc' in obj and isinstance(obj['desc'],list) and len(obj['desc'])==4:
            try:historical.add(key(obj['desc']))
            except TypeError:pass
        for v in obj.values():collect(v)
    elif isinstance(obj,list):
        for v in obj:collect(v)
# Only saved manifests and episode state records, not model files or request/image payloads.
for p in (ROOT/'artifacts').rglob('*'):
    if OUT in p.parents or not p.is_file():continue
    if p.suffix=='.json' and (p.name in ['data_manifest.json','data.json','episode.json'] or 'episodes' in p.parts):
        collect(json.loads(p.read_text()));files+=1
    elif p.suffix=='.jsonl' and 'episodes' in p.parts:
        for line in p.read_text().splitlines():collect(json.loads(line))
        files+=1
import runpy
cfg=OmegaConf.to_container(runpy.run_path(str(SRC/'scripts/official_train_config.py'))['config']().skill_system,resolve=True)
cfg.update(cold_start={'enable':False});cfg['library']['path']=str(ROOT/P['source_library']);cfg['distill']['mode']='disabled';cfg['reward']['mode']='success';provider=ViSkillProvider(cfg);provider.init_env_type('FrozenLake');assert len(provider.library._skills)==10
before=json.loads((ROOT/P['source_library']/'skill_index.json').read_text());rows=[];screen=[];seen=set(historical)
for stratum,settings in P['maps'].items():
 if not isinstance(settings,dict) or 'bounds' not in settings:continue
 selected=0
 for seed in range(settings['bounds'][0],settings['bounds'][1]+1):
    assert time.time()<P['cooperative_stop_unix']
    env=FrozenLake(P['environment']);env.gym_env=env._create_gym_env(seed);env.gym_env.reset(seed=seed);state=plain(env.get_state_for_skill());env.gym_env=None;identity=key(state['desc']);skill=provider.retrieve(state);eligible=bool(skill and skill['score']>=.8)
    reason='historical_or_panel_duplicate' if identity in seen else 'below_original_threshold' if stratum=='retrievable' and not eligible else 'selected'
    screen.append({'seed':seed,'stratum':stratum,'reason':reason,'top1':skill['id'] if skill else None,'score':skill['score'] if skill else None})
    if reason!='selected':continue
    seen.add(identity);rows.append({'seed':seed,'stratum':stratum,'state':state,'identity':identity,'eligible':eligible,'skill_id':skill['id'] if eligible else None,'score':skill['score'] if skill else None,'action_pattern':skill.get('trajectory',[]) if eligible else []});selected+=1
    if selected==settings['count']:break
 assert selected==32
assert len(rows)==64 and len({r['identity'] for r in rows})==64
orders=list(itertools.permutations(P['conditions']));schedule=[]
for repeat in range(2):
 indices=list(range(64));random.Random(P['order_seed']+repeat).shuffle(indices)
 for position,index in enumerate(indices):
    for condition in orders[(position+3*repeat)%6]:schedule.append({'repeat':repeat,'condition':condition,'seed':rows[index]['seed']})
assert len(schedule)==384 and len({(x['repeat'],x['condition'],x['seed']) for x in schedule})==384
assert json.loads((ROOT/P['source_library']/'skill_index.json').read_text())==before
write(OUT/'data_manifest.json',{'status':'CPU_DATA_READY','completed_unix':time.time(),'historical_board_count':len(historical),'history_files_read':files,'records':rows,'screened':len(screen),'screening':screen,'schedule':schedule,'library_index':before,'model_calls':0,'GPU_calls':0,'selection_without_model_outcomes':True})
print(json.dumps({'status':'CPU_DATA_READY','historical_boards':len(historical),'screened':len(screen),'natural_eligible':sum(r['eligible'] for r in rows if r['stratum']=='natural'),'selected_skill_counts':{sid:sum(r['skill_id']==sid for r in rows) for sid in sorted({r['skill_id'] for r in rows if r['skill_id']})}}))
