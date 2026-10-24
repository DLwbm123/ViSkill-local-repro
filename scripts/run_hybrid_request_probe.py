"""Run one phase of the frozen hybrid request diagnostic."""
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
ROOT = Path(os.environ['TASK_ROOT'])
OUT = ROOT/'artifacts/component_tests/hybrid_request_probe_v1'
PHASE = os.environ['TASK_PHASE']
OUT.mkdir(parents=True, exist_ok=True)
started = time.time()


def hybrid():
    import ray
    from vagen.main_ppo_skill import run_ppo
    from vagen.hybrid_probe_runner import ProbeTaskRunner
    with initialize_config_dir(config_dir=str(ROOT/'src/vagen/configs'), version_base=None):
        cfg = compose(config_name='vagen_multiturn_skill', overrides=[
            'hydra.searchpath=[file://'+str(ROOT/'src/verl/verl/trainer/config')+']'])
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(ROOT/'src/configs/smoke/frozenlake_tiny.yaml'))
    cfg.skill_system.enable = False
    cfg.trainer.resume_mode = 'disable'
    cfg.trainer.val_before_train = True
    cfg.trainer.val_only = True
    cfg.trainer.save_freq = -1
    cfg.trainer.default_local_dir = str(OUT/'forbidden_checkpoint')
    cfg.trainer.validation_data_dir = str(OUT/'validation_log')
    cfg.trainer.rollout_data_dir = None
    cfg.data.val_files = str(ROOT/'src/configs/smoke/hybrid_probe_inputs.yaml')
    cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(ROOT/'src/configs/smoke/hybrid_probe_agent.yaml')
    cfg.ray_kwargs.ray_init._temp_dir = '/data/local/tmp/z0/r2'
    OmegaConf.save(cfg, OUT/'hybrid_resolved.yaml', resolve=True)
    try:
        run_ppo(cfg, task_runner_class=ray.remote(num_cpus=1)(ProbeTaskRunner))
        assert not (OUT/'forbidden_checkpoint').exists()
    finally:
        ray.shutdown()


def standalone():
    import socket
    import sglang
    from PIL import Image
    from transformers import AutoTokenizer
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    records=sorted((OUT/'hybrid').glob('*/result.json'))
    assert len(records)==16
    target=OUT/'standalone';target.mkdir(exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/b0',local_files_only=True)
    loop=asyncio.new_event_loop();asyncio.set_event_loop(loop);engine=None
    async def replay():
        summary=[]
        for file in records:
            row=json.loads(file.read_text())
            handle=SimpleNamespace(config=SimpleNamespace(max_model_len=12288,response_length=4096),tokenizer_manager=engine.tokenizer_manager)
            output=await SGLangHttpServer.__ray_metadata__.modified_class.generate(handle,
                prompt_ids=row['prompt_ids'],sampling_params=dict(row['sampling_params']),
                request_id=file.parent.name,image_data=[Image.open(file.parent/'input.png').copy()])
            ids=list(output.token_ids);prefix=0
            for a,b in zip(ids,row['output_ids']):
                if a!=b:break
                prefix+=1
            result={'seed':row['seed'],'temperature':row['temperature'],'output_ids':ids,
                    'output_text':tokenizer.decode(ids,skip_special_tokens=True),
                    'hybrid_exact_ids':ids==row['output_ids'],'common_prefix_tokens':prefix,
                    'hybrid_length':len(row['output_ids']),'standalone_length':len(ids)}
            (target/(file.parent.name+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False))
            summary.append({k:v for k,v in result.items() if k not in {'output_ids','output_text'}})
        (OUT/'comparison.json').write_text(json.dumps(summary,indent=2))
    try:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        args=dict(model_path=str(ROOT/'models/b0'),host='127.0.0.1',port=port,dtype='bfloat16',tp_size=1,
            mem_fraction_static=.22,context_length=12288,max_running_requests=1,disable_cuda_graph=True,
            attention_backend='triton',mm_attention_backend='sdpa',trust_remote_code=False,random_seed=42,log_level='error')
        engine=sglang.Engine(**args);asyncio.set_event_loop(loop)
        loop.run_until_complete(replay())
    finally:
        if engine is not None:engine.shutdown()
        loop.close()

if __name__=='__main__':
    assert PHASE in {'hybrid','standalone'}
    hybrid() if PHASE=='hybrid' else standalone()
    (OUT/(PHASE+'_COMPLETE.json')).write_text(json.dumps({'phase':PHASE,'seconds':time.time()-started,'training':False},indent=2))
