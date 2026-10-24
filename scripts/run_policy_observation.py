"""Fixed 2x2 navigation diagnosis on failure-conditioned saved maps, not skill generation."""
import asyncio,json,os,socket,time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoProcessor,set_seed
from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.agent_loop.gym_agent_loop import convert_obs_to_content
from vagen.skills.mechanism_audit import jsonable
ROOT=Path(os.environ['TASK_ROOT']);NAME='policy_observation_v1';OUT=ROOT/'artifacts/component_tests'/NAME
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());SOURCE=ROOT/'artifacts/component_tests/skill_capacity_v1'
def dump(path,obj):path.write_text(json.dumps(jsonable(obj),indent=2))
def distance(board,position):
 position=tuple(position)
 if board[position[0]][position[1]]=='H':return None
 q=deque([(position,0)]);seen={position}
 while q:
  (r,c),d=q.popleft()
  if board[r][c]=='G':return d
  for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
   rr,cc=r+dr,c+dc
   if 0<=rr<len(board) and 0<=cc<len(board[0]) and board[rr][cc]!='H' and (rr,cc) not in seen:
    seen.add((rr,cc));q.append(((rr,cc),d+1))
 return None
async def main(engine):
 started=time.time();set_seed(42);processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
 analysis=json.loads((SOURCE/'analysis.json').read_text());assert [d['seed'] for d in analysis['details'] if not d['success']][:16]==P['seeds']
 environment=json.loads((ROOT/'src/configs/low_resource/local_eval.json').read_text())['environment']
 records=[]
 for i,seed in enumerate(P['seeds']):
  saved=json.loads((SOURCE/f'episode_{seed}/episode.json').read_text());folder=OUT/str(seed);folder.mkdir()
  offset=i%4;order=P['arms'][offset:]+P['arms'][:offset]
  for arm in order:
   cfg=dict(environment,render_mode='text' if arm.startswith('text') else 'vision',use_example_in_sys_prompt='no_examples' not in arm)
   env=FrozenLake(env_config=cfg)
   try:
    obs,_=await env.reset(seed=seed);system=await env.system_prompt();initial=jsonable(env.get_state_for_skill())
    assert initial==saved['initial_env_state']
    messages=[{'role':'system','content':convert_obs_to_content(system)},{'role':'user','content':convert_obs_to_content(obs)}]
    images=obs.get('multi_modal_input',{}).get('<image>')
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    (folder/f'{arm}_input.txt').write_text(text)
    inputs=processor(text=[text],images=images,return_tensors='pt');ids=inputs['input_ids'].squeeze(0).tolist()
    assert len(ids)+512<=P['context_length']
    if images:images[0].save(folder/f'{arm}.png')
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    handle=SimpleNamespace(config=SimpleNamespace(max_model_len=P['context_length'],response_length=512),tokenizer_manager=engine.tokenizer_manager)
    output=await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle,prompt_ids=ids,sampling_params=P['sampling'].copy(),request_id=f'{i}_{arm}',image_data=images)
    raw=processor.tokenizer.decode(output.token_ids,skip_special_tokens=True)
    _,reward,done,info=await env.step(raw);final=jsonable(env.get_state_for_skill())
    actions=info.get('actions',[]);positions=jsonable(info.get('step_positions',[]));board=initial['desc'];start=initial['player_pos']
    initial_d=distance(board,start);next_d=distance(board,positions[0]) if positions else None
    progress=bool(next_d is not None and next_d==initial_d-1)
    all_positions=[start]+positions
    record={'seed':seed,'arm':arm,'raw':raw,'prompt_ids':ids,'output_ids':output.token_ids,'initial':initial,'final':final,
     'actions':actions,'positions':positions,'format_correct':bool(info.get('format_correct')),
     'first_action_progress':progress,'initial_bfs':initial_d,'first_bfs':next_d,
     'hole':board[final['player_pos'][0]][final['player_pos'][1]]=='H',
     'success':bool(info.get('success')),'reward':float(reward),'done':bool(done),
     'boundary_noops':sum(x==y for x,y in zip(all_positions,all_positions[1:]))}
    dump(folder/f'{arm}.json',record);records.append(record)
    print('CASE',seed,arm,actions,progress,record['hole'],record['success'],flush=True)
   finally:await env.close()
 assert len(records)==64
 summary={}
 for arm in P['arms']:
  rows=[r for r in records if r['arm']==arm]
  summary[arm]={key:sum(r[key] for r in rows) for key in ['format_correct','first_action_progress','hole','success','boundary_noops']}
  summary[arm]['nonempty_actions']=sum(bool(r['actions']) for r in rows);summary[arm]['cases']=16
 comparisons={}
 for test,control in [('vision_no_examples','vision_examples'),('text_examples','vision_examples'),('text_no_examples','vision_no_examples'),('text_no_examples','text_examples')]:
  a={r['seed']:r['first_action_progress'] for r in records if r['arm']==test};b={r['seed']:r['first_action_progress'] for r in records if r['arm']==control}
  improved=sum(a[s] and not b[s] for s in P['seeds']);worsened=sum(b[s] and not a[s] for s in P['seeds'])
  comparisons[f'{test}_vs_{control}']={'improved':improved,'worsened':worsened,'lead_gate':improved>=4 and worsened<=1}
 dump(OUT/'result.json',{'summary':summary,'comparisons':comparisons,'model_calls':64,'full_episode_evaluation':False,
  'failure_conditioned_development_set':True,'training':False,'wall_seconds':time.time()-started})
def launch():
 import sglang
 OUT.mkdir(parents=True,exist_ok=False);loop=asyncio.new_event_loop();asyncio.set_event_loop(loop);engine=None
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
