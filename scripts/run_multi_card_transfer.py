"""Fixed-policy conditional transfer, complete paired episodes, immutable real cards."""
import asyncio,json,os,shutil,socket,time
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from transformers import AutoProcessor,set_seed
from vagen.agent_loop.bootstrap_agent import BootstrapGymAgentLoop
from vagen.skills.mechanism_audit import jsonable
ROOT=Path(os.environ['TASK_ROOT']);NAME='multi_card_transfer_v1';OUT=ROOT/'artifacts/component_tests'/NAME
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
DATA=json.loads((OUT/'data_manifest.json').read_text());assert DATA['status']=='CPU_PASS'
def dump(path,obj):path.write_text(json.dumps(jsonable(obj),indent=2))
async def main(engine):
 start=time.time();set_seed(42);processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
 original=(ROOT/P['source_library']/'skill_index.json').read_text()
 assert len(DATA['splits']['dev'])==16
 class Manager:
  calls=0
  async def generate(self,request_id,prompt_ids,sampling_params,image_data=None,**kwargs):
   self.calls+=1;assert self.calls<=P['max_model_calls']
   assert len(prompt_ids)+sampling_params.get('max_new_tokens',512)<=P['context_length']
   from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
   handle=SimpleNamespace(config=SimpleNamespace(max_model_len=P['context_length'],response_length=512),tokenizer_manager=engine.tokenizer_manager)
   output=await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle,prompt_ids=prompt_ids,sampling_params=sampling_params.copy(),request_id=request_id,image_data=image_data)
   return SimpleNamespace(token_ids=output.token_ids,log_probs=None)
 manager=Manager();agents={};index_text={}
 for arm in P['arms']:
  folder=OUT/arm;folder.mkdir(exist_ok=False);lib=folder/'library'
  if arm=='cards':shutil.copytree(ROOT/P['source_library'],lib);index_text[arm]=(lib/'skill_index.json').read_text()
  skill=OmegaConf.to_container(OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system,resolve=True)
  skill['library']={'path':str(lib),'max_size':32};skill['reward']['guided_weight']=0
  config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'multi_turn':{},'prompt_length':8192,'response_length':4096}},'data':{'apply_chat_template_kwargs':{}},'env_registry':{'FrozenLake':'vagen.envs.frozenlake.frozenlake_env.FrozenLake'},'skill_system':skill})
  cls=type('Transfer'+arm,(BootstrapGymAgentLoop,),{'_class_initialized':False,'_skill_initialized':False})
  agents[arm]=cls(SimpleNamespace(config=config),manager,processor.tokenizer,processor)
 rows=[]
 for i,item in enumerate(DATA['splits']['dev']):
  for arm in (P['arms'] if i%2==0 else list(reversed(P['arms']))):
   os.environ['PILOT_OUT']=str(OUT/arm);os.environ['TASK_PHASE']='A' if arm=='empty' else 'B'
   os.environ['MECHANISM_POLICY_LABEL']='b0_frozen_transfer_no_update'
   result=await agents[arm].run(P['sampling'].copy(),env_name='FrozenLake',config=P['environment'],seed=item['seed'],max_turns=5,response_length_per_turn=512,pilot_phase='dev',pilot_policy_version=0,validate=True)
   d=json.loads(Path(result.extra_fields['pilot_episode_path']).read_text());sid=d['retrieved_skill_id']
   assert sid==(P['source_skill'] if arm=='cards' else None),(arm,sid)
   assert not (d.get('distillation_audit') or {}).get('model_calls',0)
   assert d['reward_support']['applied_skill_bonus']==0
   if arm=='cards':assert d['reward_support']['reference_length']==3
   row={'seed':item['seed'],'arm':arm,'success':bool(d['traj_success']),'quality':bool(d['traj_success'] and d['bfs'] and len(d['trajectory_actions'])/d['bfs']<=1),'injected_skill':sid,'bfs':d['bfs'],'actions':d['trajectory_actions'],'turns':result.num_turns,'terminal_reason':d['terminal_reason'],'reward_support':d['reward_support'],'episode_path':result.extra_fields['pilot_episode_path']}
   rows.append(row);dump(OUT/'partial_results.json',{'records':rows,'model_calls':manager.calls});print('EPISODE',item['seed'],arm,row['success'],row['quality'],flush=True)
 assert len(rows)==32 and manager.calls<=160
 assert original==(ROOT/P['source_library']/'skill_index.json').read_text()
 assert index_text['cards']==(OUT/'cards/library/skill_index.json').read_text()
 summary={arm:{'episodes':16,**{metric:sum(r[metric] for r in rows if r['arm']==arm) for metric in ['success','quality']},'three_step_injections':sum(r['injected_skill']==P['source_skill'] for r in rows if r['arm']==arm)} for arm in P['arms']}
 pairs={}
 a={r['seed']:r for r in rows if r['arm']=='cards'};b={r['seed']:r for r in rows if r['arm']=='empty'}
 for metric in ['success','quality']:pairs[metric]={'improved':sum(a[s][metric] and not b[s][metric] for s in a),'worsened':sum(b[s][metric] and not a[s][metric] for s in a)}
 gate=summary['cards']['quality']>=3 and summary['cards']['three_step_injections']==16
 dump(OUT/'result.json',{'summary':summary,'paired':pairs,'engineering_reuse_gate':gate,'learning_benefit_established':False,'conditional_only':True,'training':False,'library_unchanged':True,'model_calls':manager.calls,'records':rows,'worker_seconds':time.time()-start,'preparation_seconds':time.time()-P['preparation_started_unix']})
def launch():
 import sglang
 assert not (OUT/'result.json').exists() and not (OUT/'empty').exists() and not (OUT/'cards').exists()
 os.environ['MECHANISM_AUDIT_DIR']=str(OUT/'audit');os.environ['PILOT_DATA_MANIFEST']=str(OUT/'data_manifest.json')
 loop=asyncio.new_event_loop();asyncio.set_event_loop(loop);engine=None
 try:
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  args=dict(model_path=str(ROOT/'models/b0'),host='127.0.0.1',port=port,dtype='bfloat16',tp_size=1,mem_fraction_static=.25,context_length=P['context_length'],max_running_requests=1,disable_cuda_graph=True,attention_backend='triton',mm_attention_backend='sdpa',trust_remote_code=False,random_seed=42,log_level='info')
  dump(OUT/'engine_args.json',args);engine=sglang.Engine(**args);asyncio.set_event_loop(loop);loop.run_until_complete(main(engine))
 finally:
  if engine is not None:engine.shutdown()
  loop.close()
if __name__=='__main__':launch()
