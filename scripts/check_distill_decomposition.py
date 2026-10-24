"""Finite local causal diagnostic; fixture facts never become training labels."""
import json, os, re, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
from vagen.skills.provider import ViSkillProvider
from vagen.skills.distill_prompt import build_distill_prompt

ROOT=Path(os.environ['TASK_ROOT'])

def main():
    out=ROOT/'artifacts/component_tests/distill_decomposition_v1'
    out.mkdir(exist_ok=False)
    source=ROOT/'artifacts/component_tests/original_v1'
    fixture=json.loads((source/'fixture.json').read_text())
    frames=[Image.open(source/f'original_{i}.png').convert('RGB') for i in range(7)]
    cards={}
    for size in (120,256):
        p=ViSkillProvider({'library':{'path':str(out/'unused_library')},'cold_start':{'enable':False},'distill':{'mode':'self'},'render':{'strategy':{'enable':True,'type':'vision'},'trajectory':{'enable':True,'draw_path':True,'thumbnail_size':size}}})
        p.init_env_type('FrozenLake')
        cards[size]=p.render_trajectory(frames,fixture['actions'],positions=fixture['positions'])
        cards[size].save(out/f'trajectory_{size}.png')
    task_description=p.task_description
    processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
    set_seed(42)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT/'models/b0',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
    visual=[(n,m) for n,m in model.named_modules() if m.__class__.__name__=='Qwen2_5_VisionTransformerPretrainedModel']
    assert len(visual)==1, [n for n,_ in visual]
    captured=[]
    def capture(module,args,output):
        assert isinstance(output,torch.Tensor)
        captured.append(output.detach().float().cpu())
    handle=visual[0][1].register_forward_hook(capture)
    (out/'runtime.json').write_text(json.dumps({'visual_module':visual[0][0],'generation_config':model.generation_config.to_dict(),'torch':torch.__version__},indent=2))
    def original(size,prefix=True,examples=True,blank=False):
        prompt=build_distill_prompt(task_description,fixture['actions'],'frozenlake',True,include_examples=examples)
        ims=[frames[0],cards[size]]
        if blank:ims=[Image.new('RGB',im.size,(127,127,127)) for im in ims]
        return prompt,ims,'<strategy>' if prefix else ''
    actions=json.dumps(fixture['actions'])
    cases=[
      ('echo_actions',('Copy the following JSON array exactly. Do not explain or change it.\n'+actions,[],'')),
      ('compress_actions',('Compress this action list into consecutive runs, preserving the total length and order. Return ONLY a JSON array of objects with keys direction and count. Do not plan a route or use coordinates.\nRecorded actions: '+actions,[],'')),
      ('perception_with_legend',('This is a 4 by 4 FrozenLake board. The small elf is the player, the wrapped gift is the goal, blue circular pools are holes, pale squares are safe. Return JSON with player_row_col, goal_row_col, and hole_row_cols. Use zero-based row from top and column from left.\n<image>',[frames[0]],'')),
      ('separate_frames',('These are seven consecutive FrozenLake frames from a successful trajectory. The elf is the player, gift the goal, blue pools are holes. Describe the start and end corners of the player and the sequence of six moves between frames. Use ordinary direction words, not numeric coordinates.\n'+''.join('Frame '+str(i)+':\n<image>\n' for i in range(7)),frames,'')),
      ('two_stage_strategy',('',frames,''))]
    expected=json.loads((ROOT/'src/configs/smoke/distill_decomposition_v1.json').read_text())['conditions']
    assert [n for n,_ in cases]==expected
    rows=[];reference=None;start=time.time()
    for name,(prompt,ims,prefix) in cases:
        if name=='two_stage_strategy':
            compression=next(x['response'] for x in rows if x['condition']=='compress_actions')
            prompt=('Describe this observed successful trajectory as one reusable strategy, in one or two sentences inside <strategy>...</strategy>. State WHEN using the visible starting and goal locations, then WHAT using the recorded direction counts. This is summarization, not new route planning. Preserve the counts and order from the following local-policy compression; avoid coordinates and unsupported obstacle claims.\nLocal policy compression of recorded actions:\n'+compression+'\nThe elf is the player, gift the goal, and blue pools are holes. Original consecutive frames:\n'+''.join('Frame '+str(i)+':\n<image>\n' for i in range(7)))
        captured.clear();parts=prompt.split('<image>');assert len(parts)-1==len(ims)
        blocks=[]
        for i,text in enumerate(parts):
            if text:blocks.append({'type':'text','text':text})
            if i<len(ims):blocks.append({'type':'image'})
        serialized=processor.apply_chat_template([{'role':'user','content':blocks}],tokenize=False,add_generation_prompt=True)+prefix
        (out/f'{name}_prompt.txt').write_text(serialized)
        batch=processor(text=[serialized],images=ims or None,return_tensors='pt').to('cuda')
        length=batch.input_ids.shape[1];assert length+256<=9216
        grid=batch.get('image_grid_thw');image_tokens=int((batch.input_ids==model.config.image_token_id).sum())
        expected_tokens=int(grid.prod(dim=-1).sum().item()//4) if grid is not None else 0
        assert image_tokens==expected_tokens
        pixels=batch.get('pixel_values');stats=None if pixels is None else {'shape':list(pixels.shape),'mean':float(pixels.float().mean()),'std':float(pixels.float().std())}
        t=time.time()
        with torch.inference_mode():generated=model.generate(**batch,max_new_tokens=256,do_sample=False,repetition_penalty=1.05)
        raw=prefix+processor.tokenizer.decode(generated[0,length:],skip_special_tokens=True)
        vision={'calls':len(captured),'shape':[list(v.shape) for v in captured],'norm':[float(v.norm()) for v in captured]}
        assert bool(captured)==bool(ims)
        if name=='original_120':reference=captured[0].clone()
        if name=='blank_120':
            vision['relative_l2_vs_real']=float((captured[0]-reference).norm()/reference.norm())
            vision['cosine_vs_real']=float(torch.nn.functional.cosine_similarity(captured[0].flatten(),reference.flatten(),dim=0))
        row={'condition':name,'response':raw,'strict_tags':bool(re.search(r'<strategy>.+?</strategy>',raw,re.S)),'image_sizes':[list(im.size) for im in ims],'grid':grid.tolist() if grid is not None else [],'image_tokens':image_tokens,'prompt_tokens':length,'completion_tokens':generated.shape[1]-length,'pixel_stats':stats,'vision':vision,'generation_seconds':time.time()-t}
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        del batch,generated
    handle.remove()
    (out/'summary.json').write_text(json.dumps({'conditions_finished':len(rows),'wall_seconds':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'training':False,'fixture_only':True,'semantic_review_required':True},indent=2))
if __name__=='__main__':main()
