"""One predeclared local base-policy check of the action-grounded self-distilled test card."""
import asyncio,json,os,time
from pathlib import Path
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.skills.provider import ViSkillProvider

async def main():
 root=Path(os.environ['TASK_ROOT']);out=root/'artifacts/component_tests'/os.environ.get('TASK_ATTEMPT','grounded_fixture_v1')
 out.mkdir(exist_ok=False)
 cfg=json.loads((root/'src/configs/low_resource/local_eval.json').read_text())
 envcfg=dict(cfg['environment'],desc=['SFFF','FHHF','FFFF','FFFG'])
 provider=ViSkillProvider({'library':{'path':str(out/'library')},'distill':{'mode':'self'},'render':{'strategy':{'enable':True,'type':'vision'},'trajectory':{'enable':True,'draw_path':True,'thumbnail_size':120}}})
 provider.init_env_type('FrozenLake')
 env=FrozenLake(envcfg);start=time.time();set_seed(42)
 try:
  obs,_=await env.reset(1)
  def convert(o):
   blocks=[];ims=(o.get('multi_modal_input') or {}).get('<image>',[]);parts=o['obs_str'].split('<image>');assert len(parts)-1==len(ims)
   for i,p in enumerate(parts):
    if p:blocks.append({'type':'text','text':p})
    if i<len(ims):blocks.append({'type':'image'})
   return blocks,list(ims)
  processor=AutoProcessor.from_pretrained(root/'models/b0',local_files_only=True)
  model=Qwen2_5_VLForConditionalGeneration.from_pretrained(root/'models/b0',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
  from vagen.skills.solver import FrozenLakeSolver
  from vagen.skills.distill_prompt import build_distill_prompt
  from vagen.skills.api_distiller import parse_strategy
  import re
  state=env.get_state_for_skill();traj=FrozenLakeSolver().replay(state)
  assert traj and traj['actions']==['down','down','down','right','right','right']
  trajectory_card=provider.render_trajectory(traj['frames'],traj['actions'],positions=traj['positions'])
  prompt=build_distill_prompt(provider.task_description,traj['actions'],'frozenlake',True,ground_actions=True)
  (out/'prompt.txt').write_text(prompt)
  b,im=convert({'obs_str':prompt,'multi_modal_input':{'<image>':[traj['frames'][0],trajectory_card]}})
  text=processor.apply_chat_template([{'role':'user','content':b}],tokenize=False,add_generation_prompt=True)+'<strategy>'
  batch=processor(text=[text],images=im,return_tensors='pt').to('cuda');n=batch.input_ids.shape[1]
  with torch.inference_mode():g=model.generate(**batch,max_new_tokens=256,do_sample=False)
  raw='<strategy>'+processor.tokenizer.decode(g[0,n:],skip_special_tokens=True)
  strategy=parse_strategy(raw)
  assert re.search(r'<strategy>.+?</strategy>',raw,re.S)
  card=provider.render_trajectory(traj['frames'],traj['actions'],positions=traj['positions'],strategy=strategy)
  card.save(out/'skill_card.png')
  provider.library.admit({'features':provider.library.feature_extractor.extract(state),'strategy':strategy,'trajectory':traj['actions'],'positions':traj['positions'],'image':card,'seed':'TEST_FIXTURE_ONLY'})
  provider.library.save(out/'library')
  (out/'distillation.json').write_text(json.dumps({'raw':raw,'strategy':strategy,'prompt_tokens':n,'fixture_only':True,'source':'solver fixture; never training library','generation_count':1},indent=2))
  skill=provider.retrieve(state);assert skill['score']>=.8
  sb,images=convert(await env.system_prompt());kb,ki=convert(provider.format_for_prompt(skill));assert ki
  sb+=kb;images+=ki;messages=[{'role':'system','content':sb}]
  rows=[];success=False
  for turn in range(5):
   b,im=convert(obs);images+=im;messages.append({'role':'user','content':b})
   obs['multi_modal_input']['<image>'][0].save(out/f'obs_{turn}.png')
   text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   batch=processor(text=[text],images=images,return_tensors='pt').to('cuda');length=batch.input_ids.shape[1]
   assert length+512<=9216
   with torch.inference_mode():gen=model.generate(**batch,max_new_tokens=512,do_sample=False)
   response=processor.tokenizer.decode(gen[0,length:],skip_special_tokens=True)
   messages.append({'role':'assistant','content':[{'type':'text','text':response}]})
   obs,reward,done,info=await env.step(response);success=bool(info['success'])
   rows.append({'turn':turn,'response':response,'executed_actions':info['actions'],'reward':reward,'success':success,'prompt_tokens':length})
   print('TURN',json.dumps(rows[-1]),flush=True)
   if done:break
  report={'fixture_only':True,'test':'one predeclared action-grounded fixture distillation and reuse, no resampling','success':success,'turns':rows,'wall_seconds':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'skill_id':skill['id'],'retrieval_score':skill['score']}
  (out/'result.json').write_text(json.dumps(report,indent=2))
 finally:await env.close()
if __name__=='__main__':asyncio.run(main())
