"""Same fixed E fixture through the native rollout backend, no prompt tuning."""
import json,re,time
from PIL import Image

def run(engine,env,loop,processor,out,root):
 from vagen.skills.provider import ViSkillProvider
 from vagen.skills.distill_prompt import build_distill_prompt
 from vagen.skills.api_distiller import parse_strategy
 include_examples=out.name!='sglang_fixture_v2_noexamples'
 start=time.time();source=root/'artifacts/component_tests/original_v1';fixture=json.loads((source/'fixture.json').read_text())
 frames=[Image.open(source/f'original_{i}.png').convert('RGB') for i in range(len(fixture['actions'])+1)]
 provider=ViSkillProvider({'library':{'path':str(out/'library')},'cold_start':{'enable':False},'distill':{'mode':'self'},'render':{'strategy':{'enable':True,'type':'vision'},'trajectory':{'enable':True,'draw_path':True,'thumbnail_size':120}}});provider.init_env_type('FrozenLake');assert len(provider.library)==0
 def blocks(obs):
  ims=list((obs.get('multi_modal_input') or {}).get('<image>',[]));parts=obs['obs_str'].split('<image>');assert len(parts)-1==len(ims);b=[]
  for i,t in enumerate(parts):
   if t:b.append({'type':'text','text':t})
   if i<len(ims):b.append({'type':'image'})
  return b,ims
 card=provider.render_trajectory(frames,fixture['actions'],positions=fixture['positions']);card.save(out/'trajectory_only.png')
 text=build_distill_prompt(provider.task_description,fixture['actions'],'frozenlake',True,include_examples=include_examples)
 b,ims=blocks({'obs_str':text,'multi_modal_input':{'<image>':[frames[0],card]}})
 prompt=processor.apply_chat_template([{'role':'user','content':b}],tokenize=False,add_generation_prompt=True)+'<strategy>'
 result=engine.generate(prompt,sampling_params={'temperature':0.,'max_new_tokens':256},image_data=[str(source/'original_0.png'),str(out/'trajectory_only.png')]);raw='<strategy>'+result['text'];assert re.search(r'<strategy>.+?</strategy>',raw,re.S);strategy=parse_strategy(raw)
 (out/'strategy.json').write_text(json.dumps({'raw':raw,'strategy':strategy,'generation_meta':result.get('meta_info'), 'fixture_only':True,'prompt_changed':not include_examples},indent=2));print('STRATEGY',raw,flush=True)
 final=provider.render_trajectory(frames,fixture['actions'],positions=fixture['positions'],strategy=strategy);final.save(out/'skill_card.png')
 provider.library.admit({'features':provider.library.feature_extractor.extract(fixture['state']),'strategy':strategy,'trajectory':fixture['actions'],'positions':fixture['positions'],'image':final,'seed':'TEST_FIXTURE_ONLY'});provider.save()
 obs,_=loop.run_until_complete(env.reset(seed=1));skill=provider.retrieve(env.get_state_for_skill());assert skill and skill['score']>=.8
 b,images=blocks(loop.run_until_complete(env.system_prompt()));sb,si=blocks(provider.format_for_prompt(skill));b+=sb;images+=si;messages=[{'role':'system','content':b}];turns=[];success=False
 for turn in range(5):
  b,im=blocks(obs);images+=im;messages.append({'role':'user','content':b});paths=[]
  for i,image in enumerate(images):
   path=out/f'request_{i}.png'
   if not path.exists():image.save(path)
   paths.append(str(path))
  prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  result=engine.generate(prompt,sampling_params={'temperature':0.,'max_new_tokens':512},image_data=paths);response=result['text'];messages.append({'role':'assistant','content':[{'type':'text','text':response}]})
  obs,reward,done,info=loop.run_until_complete(env.step(response));success=bool(info['success']);row={'response':response,'actions':info['actions'],'reward':reward,'success':success};turns.append(row);print('TURN',json.dumps(row),flush=True)
  if done:break
 report={'fixture_only':True,'solver_source':'previous component fixture only, not training library','strategy':strategy,'success':success,'turns':turns,'score':skill['score'],'wall_seconds':time.time()-start,'backend':'SGLang0.5.2 Triton/vision SDPA','same_original_prompt':include_examples,'generation_count':1,'training':False}
 (out/'result.json').write_text(json.dumps(report,indent=2))
