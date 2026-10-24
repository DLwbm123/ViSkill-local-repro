"""Pinned released-checkpoint contrast on the exact saved request; no admission."""
import json, os, time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

def main():
    root=Path(os.environ['TASK_ROOT']);out=root/'artifacts/component_tests/structured_checkpoint_contrast_v1';out.mkdir(exist_ok=False)
    source=root/'artifacts/component_tests/structured_contract_v1'
    prompt=(source/'distill_prompt.txt').read_text()
    ep=root/'artifacts/component_tests/onpolicy_structured_v1/collect_6'
    record=json.loads((ep/'episode.json').read_text());assert record['trajectory_actions']==['down']
    images=[Image.open(ep/'frame_0.png').convert('RGB'),Image.open(ep/'frame_1.png').convert('RGB')]
    set_seed(42);processor=AutoProcessor.from_pretrained(root/'models/b0',local_files_only=True)
    compat=json.loads((root/'state/components/structured_checkpoint_compatibility.json').read_text())
    assert compat['vocab_equal'] and compat['special_tokens_equal'] and not compat['generation_config_diffs']
    assert all(v['exact_equal'] for v in compat['tensor_parity'].values())
    assert all(v[0]==v[1] for v in compat['effective_config'].values())
    (out/'compatibility.json').write_text(json.dumps(compat,indent=2))
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(root/'models/r0',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
    assert all(p.dtype==torch.bfloat16 for p in model.parameters() if p.is_floating_point())
    batch=processor(text=[prompt],images=images,return_tensors='pt').to('cuda');n=batch.input_ids.shape[1]
    started=time.time()
    with torch.inference_mode():output=model.generate(**batch,max_new_tokens=256,do_sample=False)
    raw=processor.tokenizer.decode(output[0,n:],skip_special_tokens=True)
    result={'raw_response':raw,'control':json.loads((source/'result.json').read_text())['raw_response'],
        'same_saved_prompt':True,'dtype':'bfloat16','attention':'sdpa','weights':'existing pinned r0','processor':'b0; exact input parity verified','calls':1,'training':False,'admission':False,
        'wall_seconds':time.time()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
