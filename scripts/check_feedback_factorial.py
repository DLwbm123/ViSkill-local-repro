"""Fixed 2x2 diagnostic; no strategy from this audit is admitted."""
import json, os, time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

def main():
    root=Path(os.environ['TASK_ROOT']);source=root/'artifacts/component_tests/verified_self_distill_v1'
    out=root/'artifacts/component_tests/feedback_factorial_v1';out.mkdir(exist_ok=False)
    old=json.loads((source/'result.json').read_text())
    original=(source/'prompt.txt').read_text()
    start=original.index('Correct your previous strategy')
    end=original.index('<|vision_start|>',start)
    request=original[start:end]
    text_without_old=request.replace(old['previous_strategy']['strategy'], '[previous answer withheld]')
    assert request != text_without_old
    set_seed(42);processor=AutoProcessor.from_pretrained(root/'models/b0',local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(root/'models/b0',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
    image=Image.open(root/'artifacts/component_tests/onpolicy_structured_v1/collect_6/frame_0.png').convert('RGB')
    assert processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':request},{'type':'image'}]}],tokenize=False,add_generation_prompt=True)==original
    rows=[{'image':True,'old_answer':True,'response':old['raw_response'],'source':'reused prior exact cell'}]
    started=time.time()
    for with_image,with_old in [(False,True),(True,False),(False,False)]:
        content=[{'type':'text','text':request if with_old else text_without_old}]
        if with_image:content.append({'type':'image'})
        prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        if with_image and with_old:assert prompt==original
        (out/f'image{int(with_image)}_old{int(with_old)}_prompt.txt').write_text(prompt)
        batch=processor(text=[prompt],**({'images':[image]} if with_image else {}),return_tensors='pt').to('cuda')
        n=batch.input_ids.shape[1]
        with torch.inference_mode():output=model.generate(**batch,max_new_tokens=256,do_sample=False,repetition_penalty=1.05)
        row={'image':with_image,'old_answer':with_old,'response':processor.tokenizer.decode(output[0,n:],skip_special_tokens=True)}
        rows.append(row);print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(rows,indent=2))
    (out/'summary.json').write_text(json.dumps({'new_calls':3,'training':False,'admission':False,
        'wall_seconds':time.time()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2))

if __name__=='__main__':main()
