"""Fixed saved-eight-image test of absolute corner prior versus relative geometry."""
import json,os,re,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
r=Path(os.environ['TASK_ROOT'])
def main():
 out=r/'artifacts/component_tests/relative_geometry_v1';out.mkdir(exist_ok=False);source=r/'artifacts/component_tests/onpolicy_structured_v1'
 set_seed(42);processor=AutoProcessor.from_pretrained(r/'models/b0',local_files_only=True)
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(r/'models/b0',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval();rows=[];start=time.time()
 def generate(name,text,image):
  content=[{'type':'text','text':text},{'type':'image'}];prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True);(out/f'{name}_prompt.txt').write_text(prompt)
  batch=processor(text=[prompt],images=[image],return_tensors='pt').to('cuda');n=batch.input_ids.shape[1]
  with torch.inference_mode():g=model.generate(**batch,max_new_tokens=160,do_sample=False,repetition_penalty=1.05)
  return processor.tokenizer.decode(g[0,n:],skip_special_tokens=True)
 control='Describe only the visible starting location of the elf and location of the gift goal in this FrozenLake board, using corner or edge words rather than numeric coordinates. The elf is the player, gift is the goal, blue pools are holes. Do not propose actions.'
 treatment='In this FrozenLake image, the elf is the player and wrapped gift is the goal. Both may be placed anywhere; do not assume standard corners. Describe the goal RELATIVE TO THE PLAYER, not a route. Return ONLY JSON with direction (one of above, below, left, right, above_left, above_right, below_left, below_right) and adjacent (true only if goal is one horizontal or vertical tile away). Blue pools are holes, not the goal.'
 for seed in range(1,9):
  ep=source/f'collect_{seed}';record=json.loads((ep/'episode.json').read_text());im=Image.open(ep/'frame_0.png').convert('RGB')
  a=generate(f'corner_{seed}',control,im);b=generate(f'relative_{seed}',treatment,im)
  # Truth is read only after generation and never passed to the policy.
  state=record['initial_env_state'];pr,pc=state['player_pos'];gr,gc=next((i,j) for i,row in enumerate(state['desc']) for j,c in enumerate(row) if c=='G')
  dy,dx=gr-pr,gc-pc;vertical='above' if dy<0 else 'below' if dy>0 else '';horizontal='left' if dx<0 else 'right' if dx>0 else '';expected='_'.join(x for x in [vertical,horizontal] if x);adjacent=abs(dy)+abs(dx)==1
  try:parsed=json.loads(re.search(r'\{.*?\}',b,re.S).group(0))
  except (ValueError,AttributeError):parsed={}
  row={'seed':seed,'corner_response':a,'relative_response':b,'parsed':parsed,'truth':{'player':[pr,pc],'goal':[gr,gc],'direction':expected,'adjacent':adjacent},'direction_correct':parsed.get('direction')==expected,'adjacent_correct':parsed.get('adjacent')==adjacent}
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 # Eligibility was frozen from saved policy outcomes, not selected by these answers.
 prior=json.loads((source/'strategy_6.json').read_text());relative=next(x['relative_response'] for x in rows if x['seed']==6);im=Image.open(source/'collect_6/frame_0.png').convert('RGB')
 prompt='Generate one reusable strategy in <strategy>...</strategy>, beginning WHEN with the relative spatial condition and then WHAT actions to take. Use only the raw observations below from the same local policy, retain counts and directions, and do not invent absolute corner locations.\nRelative goal observation:\n'+relative+'\nRecorded-action compression:\n'+prior['action_compression']
 strategy=generate('eligible6_strategy',prompt,im)
 (out/'strategy_6.json').write_text(json.dumps({'raw':strategy,'relative_evidence':relative,'action_compression':prior['action_compression'],'eligibility_from_prior_quality_gate':True,'paper_exact':False,'semantic_review_required':True},indent=2))
 (out/'summary.json').write_text(json.dumps({'direction_correct':sum(x['direction_correct'] for x in rows),'adjacent_correct':sum(x['adjacent_correct'] for x in rows),'both_correct':sum(x['direction_correct'] and x['adjacent_correct'] for x in rows),'episodes':8,'model_calls':17,'wall_seconds':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'training':False},indent=2))
if __name__=='__main__':main()
