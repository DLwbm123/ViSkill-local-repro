"""One offline VLM observation/action through SGLang; no PPO or skill-gate claim."""
import asyncio,json,os,socket,time
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT'])

def main():
 import sglang as sgl
 from transformers import AutoProcessor
 from vagen.envs.frozenlake.frozenlake_env import FrozenLake
 out=ROOT/'artifacts'/os.environ.get('TASK_ATTEMPT','runtime_engine_v1');out.mkdir(exist_ok=False)
 loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
 cfg=json.loads((ROOT/'src/configs/low_resource/local_eval.json').read_text())
 is_fixture=os.environ.get('TASK_ATTEMPT','').startswith('sglang_fixture_')
 if is_fixture:cfg['environment']['desc']=['SFFF','FHHF','FFFF','FFFG']
 env=FrozenLake(cfg['environment']);engine=None;start=time.time()
 try:
  obs,_=loop.run_until_complete(env.reset(seed=1));system=loop.run_until_complete(env.system_prompt());images=[]
  def blocks(o):
   ims=list((o.get('multi_modal_input') or {}).get('<image>',[]));parts=o['obs_str'].split('<image>');assert len(parts)-1==len(ims);b=[]
   for i,s in enumerate(parts):
    if s:b.append({'type':'text','text':s})
    if i<len(ims):b.append({'type':'image'})
   images.extend(ims);return b
  messages=[{'role':'system','content':blocks(system)},{'role':'user','content':blocks(obs)}]
  model_id='r0' if os.environ.get('TASK_ATTEMPT')=='sglang_fixture_released_contrast_v1' else 'b0'
  processor=AutoProcessor.from_pretrained(ROOT/'models'/model_id,local_files_only=True)
  prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  paths=[]
  for i,im in enumerate(images):
   path=out/f'observation_{i}.png';im.save(path);paths.append(str(path))
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  args=dict(model_path=str(ROOT/'models'/model_id),host='127.0.0.1',port=port,dtype='bfloat16',tp_size=1,mem_fraction_static=.25,context_length=9216,max_running_requests=1,disable_cuda_graph=True,attention_backend='triton',mm_attention_backend='sdpa',trust_remote_code=False,random_seed=42,log_level='info')
  (out/'args.json').write_text(json.dumps(args,indent=2))
  engine=sgl.Engine(**args)
  # Engine installs uvloop policy; re-register the diagnostic loop after init.
  asyncio.set_event_loop(loop)
  if is_fixture:
   import importlib.util
   spec=importlib.util.spec_from_file_location("local_fixture",ROOT/"src/scripts/check_sglang_fixture.py")
   module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   module.run(engine,env,loop,processor,out,ROOT)
   return
  result=engine.generate(prompt,sampling_params={'temperature':0.,'max_new_tokens':128},image_data=paths)
  response=result['text'];obs,reward,done,info=loop.run_until_complete(env.step(response))
  report={'response':response,'actions':info['actions'],'reward':reward,'done':done,'success':info['success'],'generation_metadata':result.get('meta_info',{}),'wall_seconds':time.time()-start,'training':False,'scope':'one local FrozenLake image/action; not a full episode or PPO','backend_deviation':'text Triton and vision SDPA instead of trainer default FA3','image_count':len(images)}
  (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 finally:
  if engine is not None:engine.shutdown()
  loop.run_until_complete(env.close());loop.close()
if __name__=='__main__':main()
