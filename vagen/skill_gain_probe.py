"""Frozen native skill loop with only no-card and same-canvas blank controls."""
import json,os,time
from pathlib import Path
from PIL import Image
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop
from vagen.official_train_observer import plain,write
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='skill_gain_384_v1';OUT=ROOT/'artifacts/reproduction'/NAME;P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text())

def prepare_output_folders():
    for name in ['request_images','episodes','trajectory_images']:(OUT/name).mkdir(exist_ok=True)

class GainLoop(GymAgentLoop):
    def _inject_skill(self,data,skill,**kwargs):
        if self.condition=='V':return super()._inject_skill(data,skill,**kwargs)
        assert self.condition=='Blank';original=self.skill_provider.format_for_prompt;obs=original(skill);images=obs.get('multi_modal_input',{}).get('<image>',[]);assert len(images)==1
        controlled={'obs_str':obs['obs_str'],'multi_modal_input':{'<image>':[Image.new('RGB',images[0].size,'white')]}}
        self.skill_provider.format_for_prompt=lambda _:controlled
        try:return super()._inject_skill(data,skill,**kwargs)
        finally:self.skill_provider.format_for_prompt=original
    async def _self_distill(self,*args,**kwargs):raise AssertionError('Distillation is disabled for this experiment')

class Manager:
    def __init__(self,engine,processor,repeat):self.engine=engine;self.processor=processor;self.repeat=repeat
    async def generate(self,request_id,prompt_ids,sampling_params,image_data=None,**kwargs):
        assert time.time()<P['deadline_unix'];counter=OUT/'dispatch_count.json';n=json.loads(counter.read_text())['count']+1 if counter.exists() else 1;assert n<=P['limits']['model_generation_dispatches']
        images=image_data or [];paths=[]
        for i,img in enumerate(images):
            path=OUT/'request_images'/f'{n:05d}_{i}.png';img.save(path);paths.append(str(path.relative_to(OUT)))
        first=self.turn==0
        if first:
            expected=2 if self.item['eligible'] and self.condition!='N' else 1;assert len(images)==expected
            if self.condition=='Blank' and expected==2:assert all(v==(255,255) for v in images[0].getextrema())
        record={'event':'intent','number':n,'repeat':self.repeat,'condition':self.condition,'seed':self.item['seed'],'stratum':self.item['stratum'],'episode_id':self.episode_id,'turn':self.turn,'sampling':sampling_params.copy(),'prompt_ids':prompt_ids,'images':paths,'image_sizes':[list(x.size) for x in images],'first_request':first,'dispatched_unix':time.time()}
        if first:record['image_grid_thw']=self.processor.image_processor(images=images,return_tensors='pt')['image_grid_thw'].tolist()
        write(counter,{'count':n})
        with (OUT/'requests.jsonl').open('a') as f:f.write(json.dumps(plain(record))+'\n')
        from types import SimpleNamespace
        from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
        handle=SimpleNamespace(config=SimpleNamespace(max_model_len=9000,response_length=512),tokenizer_manager=self.engine.tokenizer_manager)
        try:output=await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle,prompt_ids=prompt_ids,sampling_params=sampling_params.copy(),request_id=request_id,image_data=images or None)
        except BaseException as exc:
            with (OUT/'requests.jsonl').open('a') as f:f.write(json.dumps(dict(record,event='error',error=repr(exc),returned_unix=time.time()))+'\n')
            raise
        record.update(event='result',output_ids=output.token_ids,raw=self.processor.tokenizer.decode(output.token_ids,skip_special_tokens=True),returned_unix=time.time())
        with (OUT/'requests.jsonl').open('a') as f:f.write(json.dumps(plain(record))+'\n')
        self.turn+=1;return output
