"""Sixteen fixed replay requests; all eight quality successes, no answer rewriting."""
import asyncio,json,os,socket,time
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from transformers import AutoProcessor,set_seed
from vagen.skills.directional_contract import build_action_grounded_prompt,build_directional_prompt,validate_directional_observation,parse_directional_contract,displacement_condition
from vagen.skills.structured_contract import MOVES
ROOT=Path(os.environ['TASK_ROOT']);NAME='schema_decoding_v1';OUT=ROOT/'artifacts/component_tests'/NAME
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());SOURCE=ROOT/'artifacts/component_tests/skill_capacity_v1'
def dump(path,obj):path.write_text(json.dumps(obj,indent=2))
async def main(engine):
 started=time.time();set_seed(42);processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
 analysis=json.loads((SOURCE/'analysis.json').read_text())
 assert [d['seed'] for d in analysis['details'] if d['quality_eligible']]==P['seeds']
 rows=[]
 for seed in P['seeds']:
  row=json.loads((SOURCE/f'episode_{seed}/episode.json').read_text())
  positions=row['trajectory_positions'];actions=row['trajectory_actions']
  assert len(positions)==len(actions)+1
  for before,after,action in zip(positions,positions[1:],actions):
   assert tuple(b-a for a,b in zip(before,after))==MOVES[action]
  rows.append(row)
 records=[]
 for i,row in enumerate(rows):
  seed=row['seed'];folder=OUT/str(seed);folder.mkdir()
  image_folder=SOURCE/f'episode_{seed}'
  frames=[Image.open(image_folder/'frame_0.png').convert('RGB'),Image.open(image_folder/f"frame_{row['frame_count']-1}.png").convert('RGB')]
  for arm in (P['arms'] if i%2==0 else list(reversed(P['arms']))):
   images=frames
   prompt=(build_directional_prompt(row['trajectory_actions']) if arm=='original_prompt_schema' else build_action_grounded_prompt(row['trajectory_actions']))
   (folder/f'{arm}_prompt.txt').write_text(prompt)
   control=(row['distillation_audit'] if arm=='original_prompt_schema' else json.loads((ROOT/'artifacts/component_tests/action_grounding_v1'/str(seed)/'action_grounded_with_images.json').read_text()))
   params=P['sampling'].copy();params['json_schema']=(ROOT/'src/configs/smoke/directional_output_schema.json').read_text()
   content=[{'type':'text','text':prompt}]+([{'type':'image'},{'type':'image'}] if images else [])
   text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
   inputs=processor(text=[text],images=images,return_tensors='pt');ids=inputs['input_ids'].squeeze(0).tolist()
   assert len(ids)+256<=P['context_length']
   from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
   handle=SimpleNamespace(config=SimpleNamespace(max_model_len=P['context_length'],response_length=256),tokenizer_manager=engine.tokenizer_manager)
   output=await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle,prompt_ids=ids,
    sampling_params=params,request_id=f'{i}_{arm}',image_data=images)
   raw=processor.tokenizer.decode(output.token_ids,skip_special_tokens=True)
   record={'seed':seed,'arm':arm,'actions':row['trajectory_actions'],'multi_step':len(row['trajectory_actions'])>=2,
    'image_count':len(images or []),'prompt_ids':ids,'output_ids':output.token_ids,'raw':raw,
    'archived_control_valid':control['valid'],'original_raw':control['raw']}
   try:
    validate_directional_observation(raw,row,allow_assistant_envelope=True);record.update(valid=True,reason=None)
   except (ValueError,TypeError,KeyError,IndexError) as e:record.update(valid=False,reason=str(e))
   try:
    contract,actions=parse_directional_contract(raw,allow_assistant_envelope=True)
    dy,dx=[b-a for a,b in zip(row['trajectory_positions'][0],row['trajectory_positions'][-1])];direction,distance=displacement_condition(dy,dx)
    record['checks']={'parse_valid':True,'direction_correct':contract['when']['goal_direction']==direction,
     'distance_correct':contract['when']['goal_manhattan_distance']==distance,'actions_exact':actions==row['trajectory_actions']}
   except (ValueError,TypeError,KeyError) as e:record['checks']={'parse_valid':False,'error':str(e)}
   dump(folder/f'{arm}.json',record);records.append(record)
   print('CASE',seed,arm,record['valid'],record['reason'],flush=True)
 assert len(records)==P['model_calls']
 summaries={}
 for arm in P['arms']:
  rr=[r for r in records if r['arm']==arm]
  summaries[arm]={'valid':sum(r['valid'] for r in rr),'total':len(rr),'multi_step_valid':sum(r['valid'] and r['multi_step'] for r in rr),
   'multi_step_total':sum(r['multi_step'] for r in rr),'single_step_regressions':sum(r['archived_control_valid'] and not r['valid'] for r in rr),
   'improved':sum(not r['archived_control_valid'] and r['valid'] for r in rr),'worsened':sum(r['archived_control_valid'] and not r['valid'] for r in rr)}
 chosen=next((arm for arm in P['arms'] if summaries[arm]['valid']==8),None)
 dump(OUT/'result.json',{'summaries':summaries,'qualified_arm':chosen,'model_calls':16,'new_episodes':0,'training':False,
  'held_out':False,'archived_control_valid':{'original_prompt_schema':5,'action_grounded_prompt_schema':0},'archived_control_multi_step_valid':0,'wall_seconds':time.time()-started})
def launch():
 import sglang
 OUT.mkdir(parents=True,exist_ok=False)
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
