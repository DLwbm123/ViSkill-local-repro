"""Check actual later model-turn retrieval opportunities on the full saved40."""
import os,json,time,signal,copy,collections
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);NAME='turn_retrieval_cpu_v1';P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());deadline=P['preparation_started_unix']+P['max_seconds_including_preparation']
assert time.time()<deadline
signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('original CPU screen deadline')));signal.alarm(max(1,int(deadline-time.time())))
from omegaconf import OmegaConf
from vagen.skills import get_skill_provider
from vagen.envs.frozenlake.utils.utils import parse_response
BASE=ROOT/'artifacts/component_tests/support_matched_curriculum_v1';folder=BASE/'collect';out=ROOT/f'artifacts/component_tests/{NAME}';assert not out.exists();out.mkdir()
cfg=OmegaConf.load(folder/'resolved.yaml');provider=get_skill_provider(OmegaConf.to_container(cfg.skill_system,resolve=True));provider.init_env_type('frozenlake');assert provider.min_score==.8 and provider.contract_guard
library=folder/'context_libraries/cards/skill_index.json';original=json.loads(library.read_text());assert len(original['skills'])==5
requests=collections.defaultdict(list)
for path in (folder/'requests').glob('*.json'):
    r=json.loads(path.read_text())
    if r['kind']=='rollout':requests[r['request_id']].append(r)
expected=json.loads((BASE/'data_manifest.json').read_text())['splits']['train'];raw={json.loads(f.read_text())['seed']:json.loads(f.read_text()) for f in (folder/'episodes').glob('*/episode.json')};assert len(raw)==40
records=[]
for case in expected:
    ep=raw[case['seed']];audit=ep['mechanism_audit'];turns=sorted(requests[audit['request_id']],key=lambda r:r['call']);actions=ep['trajectory_actions'];positions=ep['trajectory_positions'];cursor=0;aligned=len(positions)==len(actions)+1 and positions[0]==ep['initial_env_state']['player_pos'];rows=[]
    assert len(turns)==sum(s['kind']=='generated' for s in audit['spans'])
    for index,req in enumerate(turns):
        state=copy.deepcopy(ep['initial_env_state']);position=positions[cursor] if aligned and cursor<len(positions) else None;state['player_pos']=position
        if position is not None:
            skill=provider.retrieve(state);valid=skill if skill and skill.get('score',0)>=.8 else None
            row={'turn':index+1,'request_call':req['call'],'known_state':True,'position':position,'retrieved_id':valid['id'] if valid else None,'score':valid['score'] if valid else None}
        else:row={'turn':index+1,'request_call':req['call'],'known_state':False,'reason':'saved action/position alignment unknown'}
        rows.append(row)
        parsed=parse_response(req['raw'],max_actions=5)['actions'];prefix=[]
        for action in parsed:
            if action not in ['up','down','left','right']:break
            prefix.append(action)
        if index==len(turns)-1:
            remainder=actions[cursor:];aligned=aligned and remainder==prefix[:len(remainder)] and len(remainder)<=len(prefix);cursor=len(actions)
        elif actions[cursor:cursor+len(prefix)]==prefix and cursor+len(prefix)<=len(actions):cursor+=len(prefix)
        else:aligned=False
    # A later inconsistency invalidates the derived intermediate turn states.
    if not aligned or cursor!=len(actions):
        for row in rows[1:]:row.clear();row.update(known_state=False,reason='episode alignment failed')
    first=rows[0];assert first['known_state'] and first['retrieved_id']==ep['retrieved_skill_id']
    opportunities=[r['turn'] for r in rows[1:] if r.get('known_state') and r.get('retrieved_id') and first['retrieved_id'] is None]
    records.append({'seed':case['seed'],'category':case['category'],'success':bool(ep['traj_success']),'initial_retrieved_id':first['retrieved_id'],'alignment_complete':aligned and cursor==len(actions),'turns':rows,'new_valid_card_turns':opportunities,'initial_none_later_valid':bool(opportunities)})
assert json.loads(library.read_text())==original
counts={}
for category in ['all','conditional','medium','single','long']:
    for outcome in ['all','success','failure']:
        use=[r for r in records if (category=='all' or r['category']==category) and (outcome=='all' or r['success']==(outcome=='success'))]
        counts[category+'/'+outcome]={'episodes':len(use),'actual_model_turns':sum(len(r['turns']) for r in use),'unknown_turn_states':sum(not t['known_state'] for r in use for t in r['turns']),'alignment_complete_episodes':sum(r['alignment_complete'] for r in use),'initial_hit_episodes':sum(r['initial_retrieved_id'] is not None for r in use),'initial_none_later_valid_episodes':sum(r['initial_none_later_valid'] for r in use),'opportunity_turns':sum(len(r['new_valid_card_turns']) for r in use)}
long=counts['long/all']['initial_none_later_valid_episodes'];result={'status':'COMPLETE_CPU_ONLY','preparation_started_unix':P['preparation_started_unix'],'deadline_unix':deadline,'completed_unix':time.time(),'counts':counts,'records':records,'discussion_gate_pass':long>=2,'minimum_long_opportunity_episodes':2,'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'environment_steps':0,'library_json_values_unchanged':True,'actual_model_turns':sum(len(r['turns']) for r in records),'scope':P['scientific_scope'],'decision':'Separate preregistration required before any GPU' if long>=2 else 'Close dynamic-retrieval candidate; no GPU, threshold search or resampling.'}
text=json.dumps(result,indent=2);assert len(text.encode())<P['maximum_additional_bytes'];(out/'result.json').write_text(text);signal.alarm(0);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
