"""One predeclared local base-policy check of the previously distilled test card."""
import asyncio,json,os,time
from pathlib import Path
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.skills.provider import ViSkillProvider

async def main():
 root=Path(os.environ['TASK_ROOT']);out=root/'artifacts/component_tests/card_transfer_v1'
 out.mkdir(exist_ok=False)
 cfg=json.loads((root/'src/configs/low_resource/local_eval.json').read_text())
 envcfg=dict(cfg['environment'],desc=['SFFF','FHHF','FFFF','FFFG'])
 provider=ViSkillProvider({'library':{'path':str(root/'artifacts/component_tests/original_v1/roundtrip_library')},'distill':{'mode':'self'},'render':{'strategy':{'type':'vision'}}})
 provider.init_env_type('FrozenLake')
 env=FrozenLake(envcfg);start=time.time();set_seed(42)
 try:
  obs,_=await env.reset(1);skill=provider.retrieve(env.get_state_for_skill());assert skill['score']>=.8
  def convert(o):
   blocks=[];ims=(o.get('multi_modal_input') or {}).get('<image>',[]);parts=o['obs_str'].split('<image>');assert len(parts)-1==len(ims)
   for i,p in enumerate(parts):
    if p:blocks.append({'type':'text','text':p})
    if i<len(ims):blocks.append({'type':'image'})
   return blocks,list(ims)
  sb,images=convert(await env.system_prompt());kb,ki=convert(provider.format_for_prompt(skill));assert ki
  sb+=kb;images+=ki;messages=[{'role':'system','content':sb}]
  processor=AutoProcessor.from_pretrained(root/'models/b0',local_files_only=True)
  model=Qwen2_5_VLForConditionalGeneration.from_pretrained(root/'models/b0',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
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
  report={'fixture_only':True,'test':'one fixed local base-policy card transfer, no resampling','success':success,'turns':rows,'wall_seconds':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'skill_id':skill['id'],'retrieval_score':skill['score']}
  (out/'result.json').write_text(json.dumps(report,indent=2))
 finally:await env.close()
if __name__=='__main__':asyncio.run(main())
