"""Three local-policy calls plus fixed card reuse; explicitly non-paper-exact."""
import asyncio,json,os,re,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.skills.provider import ViSkillProvider

async def main():
 r=Path(os.environ['TASK_ROOT']);out=r/'artifacts/component_tests/applicability_adherence_v1';out.mkdir(exist_ok=False)
 fixture=json.loads((r/'artifacts/component_tests/original_v1/fixture.json').read_text())
 frames=[Image.open(r/f'artifacts/component_tests/original_v1/original_{i}.png').convert('RGB') for i in range(7)]
 cfg=json.loads((r/'src/configs/low_resource/local_eval.json').read_text());env=FrozenLake(dict(cfg['environment'],desc=['SFFF','FHHF','FFFF','FFFG']))
 p=ViSkillProvider({'library':{'path':str(out/'library')},'cold_start':{'enable':False},'distill':{'mode':'self'},'render':{'strategy':{'enable':True,'type':'vision'},'trajectory':{'enable':True,'draw_path':True,'thumbnail_size':120}}});p.init_env_type('FrozenLake')
 set_seed(42);processor=AutoProcessor.from_pretrained(r/'models/b0',local_files_only=True)
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(r/'models/b0',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
 start=time.time();records=[]
 def blocks(text,ims):
  parts=text.split('<image>');assert len(parts)-1==len(ims);b=[]
  for i,t in enumerate(parts):
   if t:b.append({'type':'text','text':t})
   if i<len(ims):b.append({'type':'image'})
  return b
 def generate(name,messages,ims,tokens=256):
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  (out/f'{name}_prompt.txt').write_text(text)
  batch=processor(text=[text],images=ims or None,return_tensors='pt').to('cuda');n=batch.input_ids.shape[1];assert n+tokens<=9216
  with torch.inference_mode():g=model.generate(**batch,max_new_tokens=tokens,do_sample=False,repetition_penalty=1.05)
  raw=processor.tokenizer.decode(g[0,n:],skip_special_tokens=True)
  records.append({'name':name,'raw':raw,'prompt_tokens':n,'image_count':len(ims)})
  (out/'generations.json').write_text(json.dumps(records,indent=2));print(name,raw,flush=True);return raw
 def query(name,text,ims):return generate(name,[{'role':'user','content':blocks(text,ims)}],ims)
 try:
  obs,_=await env.reset(1)
  raw=next(x['response'] for x in json.loads((r/'artifacts/component_tests/applicability_contract_v1/results.json').read_text()) if x['condition']=='explicit_route_clause')
  match=re.search(r'<strategy>(.*?)</strategy>',raw,re.S)
  assert match and match.group(1).strip(),'strict strategy tags required'
  strategy=match.group(1).strip();card=p.render_trajectory(frames,fixture['actions'],positions=fixture['positions'],strategy=strategy);card.save(out/'skill_card.png')
  p.library.admit({'features':p.library.feature_extractor.extract(fixture['state']),'strategy':strategy,'trajectory':fixture['actions'],'positions':fixture['positions'],'image':card,'seed':'TEST_FIXTURE_ONLY'});p.save()
  q=ViSkillProvider(p.config);q.init_env_type('FrozenLake');all_results=[]
  maps=json.loads((r/'src/configs/smoke/strategy_applicability_v1.json').read_text())
  def content(o):
   ims=list((o.get('multi_modal_input') or {}).get('<image>',[]));return blocks(o['obs_str'],ims),ims
  for map_name in ['original_map','counterfactual_map']:
   await env.close();env=FrozenLake(dict(cfg['environment'],desc=maps[map_name]));obs,_=await env.reset(1)
   skill=q.retrieve(env.get_state_for_skill());assert skill
   b,images=content(await env.system_prompt());sb,si=content(q.format_for_prompt(skill));b+=sb;images+=si;messages=[{'role':'system','content':b}];turns=[];success=False
   for turn in range(5):
    b,im=content(obs);messages.append({'role':'user','content':b});images+=im
    response=generate(f'{map_name}_{turn}',messages,images,512);messages.append({'role':'assistant','content':[{'type':'text','text':response}]})
    obs,reward,done,info=await env.step(response);success=bool(info['success']);turns.append({'turn':turn,'response':response,'actions':info['actions'],'step_positions':info.get('step_positions'),'reward':reward,'success':success})
    if done:break
   all_results.append({'map':map_name,'strategy':strategy,'success':success,'turns':turns,'retrieval_score':skill['score'],'policy_rollout':True})
   (out/'result.json').write_text(json.dumps(all_results,indent=2))
  (out/'summary.json').write_text(json.dumps({'wall_seconds':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'fixture_only':True,'training':False,'paper_exact':False},indent=2))
 finally:await env.close()
if __name__=='__main__':asyncio.run(main())
