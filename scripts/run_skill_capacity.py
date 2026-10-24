"""Frozen capacity diagnostic. All policy episodes precede any library admission."""
import asyncio,json,os,socket,time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor,set_seed
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop
from vagen.skills.distill_callback import DistillCallback
from vagen.skills.mechanism_audit import jsonable
ROOT=Path(os.environ['TASK_ROOT']);NAME='skill_capacity_v1';OUT=ROOT/'artifacts/component_tests'/NAME
PROTOCOL=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
def dump(path,value):path.write_text(json.dumps(jsonable(value),indent=2))
async def main(engine):
 started=time.time();set_seed(PROTOCOL['sampling_seed'])
 processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
 skill=OmegaConf.to_container(OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system,resolve=True)
 skill['library']={'path':str(OUT/'collection_empty_library'),'max_size':32}
 config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'multi_turn':{},'prompt_length':8192,'response_length':4096}},
  'data':{'apply_chat_template_kwargs':{}},'env_registry':{'FrozenLake':'vagen.envs.frozenlake.frozenlake_env.FrozenLake'},'skill_system':skill})
 environment=json.loads((ROOT/'src/configs/low_resource/local_eval.json').read_text())['environment']
 class Manager:
  calls=0;seed=None
  async def generate(self,request_id,prompt_ids,sampling_params,image_data=None,**kwargs):
   self.calls+=1;assert self.calls<=PROTOCOL['max_model_calls']
   limit=int(sampling_params.get('max_new_tokens',512));assert len(prompt_ids)+limit<=PROTOCOL['context_length']
   from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
   handle=SimpleNamespace(config=SimpleNamespace(max_model_len=PROTOCOL['context_length'],response_length=512),tokenizer_manager=engine.tokenizer_manager)
   generated=await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle,prompt_ids=prompt_ids,
    sampling_params=sampling_params.copy(),request_id=request_id,image_data=image_data)
   with (OUT/'requests.jsonl').open('a') as f:
    f.write(json.dumps({'call':self.calls,'seed':self.seed,'request_id':request_id,'prompt_tokens':len(prompt_ids),
     'image_count':len(image_data or []),'sampling':sampling_params,'output_ids':generated.token_ids,
     'raw':processor.tokenizer.decode(generated.token_ids,skip_special_tokens=True)})+'\n')
   return SimpleNamespace(token_ids=generated.token_ids,log_probs=None)
 manager=Manager();summaries=[];boards=[]
 for seed in range(PROTOCOL['seed_range_inclusive'][0],PROTOCOL['seed_range_inclusive'][1]+1):
  manager.seed=seed
  agent=GymAgentLoop(SimpleNamespace(config=config),manager,processor.tokenizer,processor)
  result=await agent.run(PROTOCOL['sampling'].copy(),env_name='FrozenLake',config=environment,seed=seed,
   max_turns=PROTOCOL['max_turns'],response_length_per_turn=512)
  row=result.extra_fields;assert row['retrieved_skill_id'] is None
  folder=OUT/f'episode_{seed}';folder.mkdir()
  for i,frame in enumerate(row['trajectory_frames']):frame.save(folder/f'frame_{i}.png')
  serial={k:v for k,v in row.items() if k not in ['trajectory_frames','validation_skill_image','validation_initial_state_image']}
  serial['reward_score']=result.reward_score;serial['frame_count']=len(row['trajectory_frames'])
  dump(folder/'episode.json',serial)
  audit=row['mechanism_audit']
  for span in audit['spans']:
   values=result.response_mask[span['start']:span['end']];expected=1 if span['kind']=='generated' else 0
   assert all(value==expected for value in values)
  assert audit['distillation_separate_window'] and audit['skill_bonus']==0
  bfs=agent.skill_provider.bfs_distance(row['initial_env_state'])
  summaries.append({'seed':seed,'success':row['traj_success'],'bfs':bfs,'actions':len(row['trajectory_actions']),
   'distillation':row['distillation_audit']})
  state=jsonable(row['initial_env_state']);boards.append(json.dumps([state['desc'],state['player_pos']],sort_keys=True))
  print('EPISODE',seed,row['traj_success'],bfs,len(row['trajectory_actions']),bool(row['distilled_strategy']),flush=True)
 assert len(summaries)==128
 (OUT/'COLLECTION_COMPLETE.json').write_text(json.dumps({'episodes':128,'model_calls':manager.calls,'time':time.time()}))
 # Separate diagnostic library; it was never visible to the collection loop.
 from PIL import Image
 skill['library']['path']=str(OUT/'diagnostic_library');skill['library']['max_size']=128
 cb=DistillCallback(skill);cb.provider.init_env_type('FrozenLake');admitted=0
 for offset in range(0,128,8):
  rows=[]
  for summary in summaries[offset:offset+8]:
   folder=OUT/f"episode_{summary['seed']}";row=json.loads((folder/'episode.json').read_text())
   row['initial_env_state']['player_pos']=tuple(row['initial_env_state']['player_pos'])
   row['trajectory_frames']=[Image.open(folder/f'frame_{i}.png').convert('RGB') for i in range(row['frame_count'])]
   rows.append(row)
  batch=SimpleNamespace(non_tensor_batch={k:[row.get(k) for row in rows] for k in rows[0]})
  metrics=cb.step(batch);admitted+=metrics['skill/admitted'];dump(OUT/f'admission_batch_{offset//8}.json',metrics)
 by_length={}
 for key in ['1','2','3+','unreachable']:
  rows=[s for s in summaries if ('unreachable' if s['bfs'] is None else str(s['bfs']) if s['bfs']<3 else '3+')==key]
  by_length[key]={'episodes':len(rows),'successes':sum(s['success'] for s in rows),
    'quality_eligible_successes':sum(s['success'] and s['bfs'] and s['actions']/s['bfs']<=1 for s in rows)}
 funnel=Counter()
 for p in (OUT/'audit/admissions').glob('*.json'):
  for row in json.loads(p.read_text())['rows']:funnel[row['admission_audit']]+=1
 dump(OUT/'result.json',{'episodes':128,'successes':sum(s['success'] for s in summaries),'unique_exact_boards':len(set(boards)),
  'exact_board_duplicate_episodes':128-len(set(boards)),'strata':by_length,'admission_funnel':dict(funnel),
  'admitted':admitted,'admitted_action_lengths':[len(s['trajectory']) for s in cb.provider.library._skills],
  'model_calls':manager.calls,'distillation_calls':sum((s['distillation'] or {}).get('model_calls',0) for s in summaries),
  'valid_distillations':sum(bool((s['distillation'] or {}).get('valid',False)) for s in summaries),
  'mask_spans_checked':True,'training':False,'diagnostic_library_only':True,'wall_seconds':time.time()-started,'summaries':summaries})
 print('COMPLETE',admitted,manager.calls,flush=True)
def launch():
 import sglang
 OUT.mkdir(parents=True,exist_ok=False)
 os.environ['MECHANISM_AUDIT_DIR']=str(OUT/'audit');os.environ['MECHANISM_POLICY_LABEL']='b0_frozen_no_update'
 loop=asyncio.new_event_loop();asyncio.set_event_loop(loop);engine=None
 try:
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  args=dict(model_path=str(ROOT/'models/b0'),host='127.0.0.1',port=port,dtype='bfloat16',tp_size=1,mem_fraction_static=.25,
   context_length=PROTOCOL['context_length'],max_running_requests=1,disable_cuda_graph=True,attention_backend='triton',
   mm_attention_backend='sdpa',trust_remote_code=False,random_seed=42,log_level='info')
  dump(OUT/'engine_args.json',args);engine=sglang.Engine(**args);asyncio.set_event_loop(loop)
  loop.run_until_complete(main(engine))
 finally:
  if engine is not None:engine.shutdown()
  loop.close()
if __name__=='__main__':launch()
