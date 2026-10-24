"""Change only the base image processor implementation on fixed saved requests."""
import json,os,re,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
r=Path(os.environ['TASK_ROOT'])
def main():
 out=r/'artifacts/component_tests/processor_parity_v1';out.mkdir(exist_ok=False)
 fast=AutoProcessor.from_pretrained(r/'models/b0',local_files_only=True,use_fast=True);slow=AutoProcessor.from_pretrained(r/'models/b0',local_files_only=True,use_fast=False)
 prior=json.loads((r/'artifacts/component_tests/relative_geometry_v1/results.json').read_text())
 set_seed(42);model=Qwen2_5_VLForConditionalGeneration.from_pretrained(r/'models/b0',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval();rows=[];start=time.time()
 cases=[]
 for old in prior:
  seed=old['seed'];prompt=(r/f'artifacts/component_tests/relative_geometry_v1/relative_{seed}_prompt.txt').read_text();im=Image.open(r/f'artifacts/component_tests/onpolicy_structured_v1/collect_{seed}/frame_0.png').convert('RGB');cases.append((f'relative_{seed}',prompt,[im],old))
 source=r/'artifacts/component_tests/distill_causal_v1';prompt=(source/'original_120_prompt.txt').read_text();ims=[Image.open(r/'artifacts/component_tests/original_v1/original_0.png').convert('RGB'),Image.open(source/'trajectory_120.png').convert('RGB')];cases.append(('original_fixture_strategy',prompt,ims,None))
 for name,prompt,ims,old in cases:
  fb=fast(text=[prompt],images=ims,return_tensors='pt');sb=slow(text=[prompt],images=ims,return_tensors='pt')
  same_shape=fb.pixel_values.shape==sb.pixel_values.shape
  metrics={'fast_grid':fb.image_grid_thw.tolist(),'slow_grid':sb.image_grid_thw.tolist(),'same_input_ids':torch.equal(fb.input_ids,sb.input_ids),'relative_pixel_l2':float((fb.pixel_values-sb.pixel_values).norm()/fb.pixel_values.norm()) if same_shape else None,'max_pixel_absdiff':float((fb.pixel_values-sb.pixel_values).abs().max()) if same_shape else None}
  sb=sb.to('cuda');n=sb.input_ids.shape[1]
  with torch.inference_mode():g=model.generate(**sb,max_new_tokens=256 if old is None else 160,do_sample=False,repetition_penalty=1.05)
  raw=slow.tokenizer.decode(g[0,n:],skip_special_tokens=True);row={'case':name,'processor_comparison':metrics,'slow_response':('<strategy>'+raw if old is None else raw)}
  if old is not None:
   try:parsed=json.loads(re.search(r'\{.*?\}',raw,re.S).group(0))
   except (ValueError,AttributeError):parsed={}
   row.update(fast_response=old['relative_response'],truth=old['truth'],direction_correct=parsed.get('direction')==old['truth']['direction'],adjacent_correct=parsed.get('adjacent')==old['truth']['adjacent'])
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 (out/'summary.json').write_text(json.dumps({'slow_direction_correct':sum(x.get('direction_correct',False) for x in rows),'slow_adjacent_correct':sum(x.get('adjacent_correct',False) for x in rows),'spatial_cases':8,'total_calls':9,'fast_processor_class':type(fast.image_processor).__name__,'slow_processor_class':type(slow.image_processor).__name__,'wall_seconds':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2))
if __name__=='__main__':main()
