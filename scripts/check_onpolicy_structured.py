"""Fixed training-seed E diagnostic: real policy successes, no solver trajectories."""
import asyncio,json,os,re,time
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.skills.distill_callback import DistillCallback
from vagen.skills.distill_prompt import build_distill_prompt
from vagen.skills.api_distiller import parse_strategy
from vagen.skills.reward import compute_reward

ROOT=Path(os.environ['TASK_ROOT']);ATTEMPT=os.environ.get('TASK_ATTEMPT','policy_cycle_v1');SAMPLE=True
OUT=ROOT/'artifacts/component_tests'/ATTEMPT
def dump(path,value):path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist()))
def content(obs):
 ims=list((obs.get('multi_modal_input') or {}).get('<image>',[]));parts=obs['obs_str'].split('<image>');assert len(parts)-1==len(ims)
 blocks=[]
 for i,s in enumerate(parts):
  if s:blocks.append({'type':'text','text':s})
  if i<len(ims):blocks.append({'type':'image'})
 return blocks,ims
async def main():
 OUT.mkdir(exist_ok=False);start=time.time();set_seed(42)
 cfg=json.loads((ROOT/'src/configs/low_resource/local_eval.json').read_text())
 cb=DistillCallback({'library':{'path':str(OUT/'library'),'max_size':10000},'cold_start':{'enable':False},'distill':{'mode':'self','quality_ratio':1.,'min_score':.9,'recheck_batch_novelty':True},'retrieval':{'similarity_weight':.9,'min_score':.8},'render':{'strategy':{'enable':True,'type':'vision'},'trajectory':{'enable':True,'draw_path':True,'thumbnail_size':120}}})
 provider=cb.provider;provider.init_env_type('FrozenLake');assert len(provider.library)==0
 processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT/'models/b0',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
 def generate(messages,images,max_tokens=512,prefix='',sample=SAMPLE):
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+prefix
  batch=processor(text=[text],images=images,return_tensors='pt').to('cuda');n=batch.input_ids.shape[1];assert n+max_tokens<=9216
  with torch.inference_mode():g=model.generate(**batch,max_new_tokens=max_tokens,do_sample=sample,**({'temperature':1.0,'top_p':1.0,'top_k':0} if sample else {}))
  return prefix+processor.tokenizer.decode(g[0,n:],skip_special_tokens=True),n
 async def episode(seed,reuse):
  env=FrozenLake(cfg['environment']);ep=OUT/f'{"reuse" if reuse else "collect"}_{seed}';ep.mkdir()
  try:
   obs,_=await env.reset(seed);initial=env.get_state_for_skill();frames=[obs['multi_modal_input']['<image>'][0]];positions=[initial['player_pos']];actions=[];rewards=[];turns=[];success=False
   b,ims=content(await env.system_prompt());messages=[{'role':'system','content':b}];skill=provider.retrieve(initial) if reuse else None
   if skill and skill['score']>=.8:
    sb,si=content(provider.format_for_prompt(skill));assert si;messages[0]['content']+=sb;ims+=si;si[0].save(ep/'retrieved_card.png')
   else:skill=None
   for turn in range(5):
    b,im=content(obs);ims+=im;messages.append({'role':'user','content':b})
    response,n=generate(messages,ims);messages.append({'role':'assistant','content':[{'type':'text','text':response}]})
    obs,reward,done,info=await env.step(response);actions+=info['actions'];frames+=info['step_frames'];positions+=info['step_positions'];rewards.append(reward);success=bool(info['success'])
    turns.append({'response':response,'actions':info['actions'],'reward':reward,'success':success,'prompt_tokens':n});print('TURN',seed,reuse,turn,info['actions'],success,flush=True)
    if done:break
   assert len(frames)==len(positions)==len(actions)+1
   row={'seed':seed,'traj_success':success,'initial_env_state':initial,'trajectory_actions':actions,'trajectory_frames':frames,'trajectory_positions':positions,'retrieved_skill_id':skill['id'] if skill else None,'skill_score':skill['score'] if skill else None,'distilled_strategy':''}
   for i,frame in enumerate(frames):frame.save(ep/f'frame_{i}.png')
   outcome=compute_reward(rewards,'state_progress_bounded' if skill else 'success',success,skill['trajectory'] if skill else None,actions,.8,reference_positions=skill.get('positions') if skill else None,model_positions=positions)
   dump(ep/'episode.json',{**{k:v for k,v in row.items() if k!='trajectory_frames'},'turns':turns,'reward_total':outcome,'environment_rewards':rewards})
   return row
  finally:await env.close()
 rows=[await episode(seed,False) for seed in [1,2,3,4,5,6,7,8]]
 collection_successes=sum(r['traj_success'] for r in rows)
 eligible=cb._filter_distill_candidates(rows)
 for row in eligible:
  card=provider.render_trajectory(row['trajectory_frames'],row['trajectory_actions'],positions=row['trajectory_positions'])
  prompt=build_distill_prompt(provider.task_description,row['trajectory_actions'],'frozenlake',True)
  b,im=content({'obs_str':prompt,'multi_modal_input':{'<image>':[row['trajectory_frames'][0],card]}})
  raw,n=generate([{'role':'user','content':b}],im,256,'<strategy>',sample=False)
  original=raw
  def query(text,images):
   b,ims=content({'obs_str':text,'multi_modal_input':{'<image>':images}})
   return generate([{'role':'user','content':b}],ims or None,256,sample=False)[0]
  initial_image=row['trajectory_frames'][0]
  facts=query('Describe only the visible starting location of the elf and location of the gift goal in this FrozenLake board, using corner or edge words rather than numeric coordinates. The elf is the player, gift is the goal, blue pools are holes. Do not propose actions.\n<image>',[initial_image])
  runs=query('Compress this recorded successful action list into consecutive runs. Return only JSON objects with direction and count, preserving all actions and order; do not plan a route.\n'+json.dumps(row['trajectory_actions']),[])
  prompt=('Write exactly one conditional procedural strategy in <strategy>...</strategy>. The sentence must begin with WHEN and explicitly state the starting/goal conditions before WHAT to do. Use the two local-policy observations below. Preserve action directions and counts exactly, avoid coordinates, do not invent obstacles or extra moves. State that the described route must be clear of holes for reuse. This is trajectory summarization, not new route planning.\nVisual observation by the same local policy:\n'+facts+'\nCompression of recorded actions by the same local policy:\n'+runs+'\nOriginal initial image:\n<image>')
  raw=query(prompt,[initial_image])
  strict=bool(re.search(r'<strategy>.+?</strategy>',raw,re.S))
  row['distilled_strategy']=parse_strategy(raw) if strict else ''
  dump(OUT/f'strategy_{row["seed"]}.json',{'original_control':original,'visual_facts':facts,'action_compression':runs,'raw':raw,'strategy':row['distilled_strategy'],'strict_tags':strict,'source':'current base policy actual successful rollout','paper_exact':False})
 # Reject missing tags instead of silently storing an empty strategy in this diagnostic.
 rejected_ids={id(row) for row in eligible if not row['distilled_strategy']}
 admission_rows=[row for row in rows if id(row) not in rejected_ids]
 def batch(rows):return SimpleNamespace(non_tensor_batch={**{k:[r.get(k) for r in rows] for k in rows[0]},'env_name':['FrozenLake']*len(rows)})
 metrics=cb.step(batch(admission_rows)) if admission_rows else {};dump(OUT/'admission_metrics.json',metrics)
 provider.library.load(OUT/'library') if len(provider.library) else None
 if len(provider.library):
  reuse=[await episode(seed,True) for seed in [1,2,3,4,5,6,7,8]]
  for row in reuse:
   if row['retrieved_skill_id']:provider.update_utility(row['retrieved_skill_id'],float(row['traj_success']))
  provider.save()
 else:reuse=[]
 dump(OUT/'result.json',{'seeds':[1,2,3,4,5,6,7,8],'training_sampling':SAMPLE,'collection_successes':collection_successes,'eligible':len(eligible),'admitted':metrics.get('skill/admitted',0),'reuse_episodes':len(reuse),'reuse_successes':sum(r['traj_success'] for r in reuse),'reused_skills':[r['retrieved_skill_id'] for r in reuse],'no_solver_trajectory':True,'no_training':True,'no_cold_start':True,'original_distill_prompt':False,'paper_exact':False,'paired_original_control_saved':True,'semantic_review_required':True,'wall_seconds':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})
if __name__=='__main__':asyncio.run(main())
