"""Prospective paired complete episodes; only dynamics-description opt-in differs."""
import asyncio,json,os,socket,time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoProcessor,set_seed
from omegaconf import OmegaConf
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop
from vagen.skills.mechanism_audit import jsonable
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
ROOT=Path(os.environ['TASK_ROOT']);NAME='dynamics_wording_v1';OUT=ROOT/'artifacts/component_tests'/NAME
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
def dump(path,obj):path.write_text(json.dumps(jsonable(obj),indent=2))
def board_key(state):return json.dumps(jsonable([state['desc'],state['player_pos']]),sort_keys=True)
async def main(engine):
 started=time.time();set_seed(42);processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
 skill=OmegaConf.to_container(OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system,resolve=True)
 skill['library']={'path':str(OUT/'empty_library'),'max_size':32}
 config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'multi_turn':{},'prompt_length':8192,'response_length':4096}},
  'data':{'apply_chat_template_kwargs':{}},'env_registry':{'FrozenLake':'vagen.envs.frozenlake.frozenlake_env.FrozenLake'},'skill_system':skill})
 base=json.loads((ROOT/'src/configs/low_resource/local_eval.json').read_text())['environment']
 old={board_key(json.loads(p.read_text())['initial_env_state']) for p in (ROOT/'artifacts/component_tests/skill_capacity_v1').glob('episode_*/episode.json')}
 class Manager:
  calls=0;seed=None;arm=None
  async def generate(self,request_id,prompt_ids,sampling_params,image_data=None,**kwargs):
   self.calls+=1;assert self.calls<=P['max_model_calls'];assert len(prompt_ids)+sampling_params.get('max_new_tokens',512)<=P['context_length']
   from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
   handle=SimpleNamespace(config=SimpleNamespace(max_model_len=P['context_length'],response_length=512),tokenizer_manager=engine.tokenizer_manager)
   output=await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle,prompt_ids=prompt_ids,sampling_params=sampling_params.copy(),request_id=request_id,image_data=image_data)
   with (OUT/'requests.jsonl').open('a') as f:f.write(json.dumps({'call':self.calls,'seed':self.seed,'arm':self.arm,'request_id':request_id,
    'sampling':sampling_params,'prompt_tokens':len(prompt_ids),'image_count':len(image_data or []),'output_ids':output.token_ids,
    'raw':processor.tokenizer.decode(output.token_ids,skip_special_tokens=True)})+'\n')
   return SimpleNamespace(token_ids=output.token_ids,log_probs=None)
 manager=Manager();records=[];boards=[]
 for i,seed in enumerate(P['seeds']):
  paired_states=[];prompts={}
  for arm in P['arms']:
   e=FrozenLake(dict(base,accurate_dynamics_prompt=arm=='accurate_description'))
   try:
    await e.reset(seed);paired_states.append(jsonable(e.get_state_for_skill()));prompts[arm]=e.get_system_prompt()
   finally:await e.close()
  assert paired_states[0]==paired_states[1] and not paired_states[0]['is_slippery'] and paired_states[0]['slip_prob']==0
  original_rule='Frozen tiles are slippery, you may move perpendicular to your intended direction.'
  accurate_rule='Movement is deterministic. Actions move in the intended direction; a grid boundary can keep you in place.'
  assert prompts['original_description'].replace(original_rule,accurate_rule)==prompts['accurate_description']
  boards.append(board_key(paired_states[0]));folder=OUT/str(seed);folder.mkdir()
  dump(folder/'paired_input_check.json',{'state':paired_states[0],'same_board_and_dynamics':True,'only_rule_changed':True,'system_prompts':prompts})
  for arm in (P['arms'] if i%2==0 else list(reversed(P['arms']))):
   manager.seed=seed;manager.arm=arm;cfg=dict(base,accurate_dynamics_prompt=arm=='accurate_description')
   agent=GymAgentLoop(SimpleNamespace(config=config),manager,processor.tokenizer,processor)
   result=await agent.run(P['sampling'].copy(),env_name='FrozenLake',config=cfg,seed=seed,max_turns=5,response_length_per_turn=512)
   row=result.extra_fields;assert row['retrieved_skill_id'] is None;assert jsonable(row['initial_env_state'])==paired_states[0]
   audit=row['mechanism_audit'];assert audit['skill_bonus']==0 and audit['distillation_separate_window']
   for span in audit['spans']:
    assert all(v==(1 if span['kind']=='generated' else 0) for v in result.response_mask[span['start']:span['end']])
   bfs=agent.skill_provider.bfs_distance(row['initial_env_state']);positions=jsonable(row['trajectory_positions']);last=positions[-1]
   serial={k:v for k,v in row.items() if k not in ['trajectory_frames','validation_skill_image','validation_initial_state_image']}
   dump(folder/f'{arm}.json',{**serial,'reward_score':result.reward_score,'turns':result.num_turns})
   for j,frame in enumerate(row['trajectory_frames']):frame.save(folder/f'{arm}_frame_{j}.png')
   record={'seed':seed,'arm':arm,'success':bool(row['traj_success']),'quality_eligible':bool(row['traj_success'] and bfs and len(row['trajectory_actions'])/bfs<=1),
    'bfs':bfs,'actions':len(row['trajectory_actions']),'turns':result.num_turns,'distillation':row['distillation_audit'],
    'hole':row['initial_env_state']['desc'][last[0]][last[1]]=='H','boundary_noops':sum(x==y for x,y in zip(positions,positions[1:]))}
   records.append(record);print('EPISODE',seed,arm,record['success'],record['quality_eligible'],flush=True)
 assert len(records)==64
 summary={}
 for arm in P['arms']:
  rows=[r for r in records if r['arm']==arm]
  summary[arm]={'episodes':32,**{k:sum(r[k] for r in rows) for k in ['success','quality_eligible','hole','boundary_noops']},
   'distillation_calls':sum((r['distillation'] or {}).get('model_calls',0) for r in rows),
   'valid_distillations':sum(bool((r['distillation'] or {}).get('valid')) for r in rows),
   'turn_limit_failures':sum(not r['success'] and not r['hole'] and r['turns']>=5 for r in rows)}
 pair={}
 a={r['seed']:r for r in records if r['arm']=='accurate_description'};b={r['seed']:r for r in records if r['arm']=='original_description'}
 for metric in ['success','quality_eligible']:
  pair[metric]={'improved':sum(a[s][metric] and not b[s][metric] for s in P['seeds']),
   'worsened':sum(b[s][metric] and not a[s][metric] for s in P['seeds'])}
 lead=pair['success']['improved']>=4 and pair['success']['worsened']<=1 and pair['quality_eligible']['improved']-pair['quality_eligible']['worsened']>=2
 dump(OUT/'result.json',{'summary':summary,'paired':pair,'performance_lead_gate':lead,'description_correctness_separate':True,
  'unique_new_boards':len(set(boards)),'overlap_with_capacity128':len(set(boards)&old),'model_calls':manager.calls,
  'training':False,'admission':False,'records':records,'wall_seconds':time.time()-started})
def launch():
 import sglang
 OUT.mkdir(parents=True,exist_ok=False);os.environ['MECHANISM_AUDIT_DIR']=str(OUT/'audit');os.environ['MECHANISM_POLICY_LABEL']='b0_frozen_no_update'
 loop=asyncio.new_event_loop();asyncio.set_event_loop(loop);engine=None
 try:
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  args=dict(model_path=str(ROOT/'models/b0'),host='127.0.0.1',port=port,dtype='bfloat16',tp_size=1,mem_fraction_static=.25,
   context_length=P['context_length'],max_running_requests=1,disable_cuda_graph=True,attention_backend='triton',
   mm_attention_backend='sdpa',trust_remote_code=False,random_seed=42,log_level='info')
  dump(OUT/'engine_args.json',args);engine=sglang.Engine(**args);asyncio.set_event_loop(loop);loop.run_until_complete(main(engine))
 finally:
  if engine is not None:engine.shutdown()
  loop.close()
if __name__=='__main__':launch()
