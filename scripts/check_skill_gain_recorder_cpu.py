"""Exercise the real recorder IO and native transport boundary with a CPU stub."""
import os,json,asyncio,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from transformers import AutoProcessor
import torch,ray
import vagen.skill_gain_probe as probe
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
OUT=probe.OUT;native=SGLangHttpServer.__ray_metadata__.modified_class;original=native.generate;calls=[]
async def fake(self,**kw):
    calls.append(kw);return SimpleNamespace(token_ids=[1,2],log_probs=None)
async def main():
    proc=AutoProcessor.from_pretrained(probe.ROOT/'models/b0',local_files_only=True);item=next(x for x in json.loads((OUT/'data_manifest.json').read_text())['records'] if x['eligible'])
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        probe.OUT=Path(folder);probe.prepare_output_folders();manager=probe.Manager(SimpleNamespace(tokenizer_manager=None),proc,0)
        for condition in ['N','V','Blank']:
            manager.turn=0;manager.condition=condition;manager.item=item;manager.episode_id='cpu_'+condition
            images=[Image.new('RGB',(256,256),'white')]
            if condition!='N':images.insert(0,Image.new('RGB',(308,232),'white'))
            answer=await manager.generate(request_id=condition,prompt_ids=[1,2,3],sampling_params=probe.P['sampling'].copy(),image_data=images);assert answer.token_ids==[1,2]
        events=[json.loads(x) for x in (probe.OUT/'requests.jsonl').read_text().splitlines()];assert len(events)==6 and len(calls)==3 and json.loads((probe.OUT/'dispatch_count.json').read_text())['count']==3;assert len(list((probe.OUT/'request_images').glob('*.png')))==5
try:
    native.generate=fake;asyncio.run(main());assert not torch.cuda.is_initialized() and not ray.is_initialized()
finally:native.generate=original;probe.OUT=OUT
p=OUT/'cpu_check.json';d=json.loads(p.read_text());d.update(real_recorder_IO_CPU_stub_PASS=True,scripted_recorder_fixture_calls=3,recorder_actual_model_calls=0,recorder_fixture_checked_unix=time.time());p.write_text(json.dumps(d,indent=2));print(json.dumps({'actual_recorder_IO':'PASS','scripted_transport_calls':3,'actual_model_calls':0,'GPU_calls':0}))
