"""Official loop and bundled native SGLang generate method; no PPO or Ray scheduling."""
import asyncio, json, os, time
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop
from vagen.skills.distill_callback import DistillCallback

ROOT=Path(os.environ['TASK_ROOT'])
OUT=ROOT/'artifacts/component_tests/directional_native_loop_v1'


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,default=lambda v:v.tolist()))


async def main(engine):
    started=time.time();set_seed(42)
    processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
    skill={'enable':True,'library':{'path':str(OUT/'library'),'max_size':10000},
           'cold_start':{'enable':False},'distill':{'mode':'self','quality_ratio':1.,'min_score':.9,
              'recheck_batch_novelty':True,'representation':'directional_contract_v1'},
           'retrieval':{'similarity_weight':.9,'min_score':.8,'contract_guard':True},
           'reward':{'mode':'state_progress_bounded','guided_weight':.8},
           'render':{'strategy':{'enable':True,'type':'vision'},
                     'trajectory':{'enable':True,'draw_path':True,'thumbnail_size':120}}}
    config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'multi_turn':{},'prompt_length':8192,'response_length':8192}},
                            'data':{'apply_chat_template_kwargs':{}},
                            'env_registry':{'FrozenLake':'vagen.envs.frozenlake.frozenlake_env.FrozenLake'},
                            'skill_system':skill})
    environment=json.loads((ROOT/'src/configs/low_resource/local_eval.json').read_text())['environment']
    sampling={'temperature':1.0,'top_p':1.0,'top_k':-1,'max_new_tokens':512}

    class LocalManager:
        calls=0
        phase='collect'
        seed=None

        async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, **kwargs):
            self.calls+=1;assert self.calls<=96
            limit=int(sampling_params.get('max_new_tokens',512));assert len(prompt_ids)+limit<=9216
            from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
            handle=SimpleNamespace(config=SimpleNamespace(max_model_len=9216,response_length=512),tokenizer_manager=engine.tokenizer_manager)
            generated=await SGLangHttpServer.__ray_metadata__.modified_class.generate(
                handle,prompt_ids=prompt_ids,sampling_params=sampling_params.copy(),request_id=request_id,image_data=image_data)
            tokens=generated.token_ids
            with (OUT/'requests.jsonl').open('a') as f:
                f.write(json.dumps({'call':self.calls,'phase':self.phase,'seed':self.seed,
                   'sampling':sampling_params,'prompt_tokens':len(prompt_ids),'image_count':len(image_data or []),
                   'response':processor.tokenizer.decode(tokens,skip_special_tokens=True)})+'\n')
            return SimpleNamespace(token_ids=tokens,log_probs=None)

    manager=LocalManager()
    cb=DistillCallback(skill);cb.provider.init_env_type('FrozenLake');assert len(cb.provider.library)==0

    async def phase(name):
        rows=[];manager.phase=name
        for seed in range(1,9):
            manager.seed=seed
            agent=GymAgentLoop(SimpleNamespace(config=config),manager,processor.tokenizer,processor)
            result=await agent.run(sampling.copy(),env_name='FrozenLake',config=environment,seed=seed,
                                   max_turns=5,response_length_per_turn=512)
            row=result.extra_fields;rows.append(row)
            folder=OUT/f'{name}_{seed}';folder.mkdir()
            for i,frame in enumerate(row['trajectory_frames']):frame.save(folder/f'frame_{i}.png')
            serial={k:v for k,v in row.items() if k not in ['trajectory_frames','validation_skill_image','validation_initial_state_image']}
            dump(folder/'episode.json',{**serial,'reward_score':result.reward_score,'turns':result.num_turns,
                'response_ids':result.response_ids,'response_mask':result.response_mask,'prompt_ids':result.prompt_ids})
            print('EPISODE',name,seed,row['traj_success'],row['retrieved_skill_id'],bool(row['distilled_strategy']),flush=True)
        return rows

    def batch(rows):
        return SimpleNamespace(non_tensor_batch={k:[r.get(k) for r in rows] for k in rows[0]})

    collection=await phase('collect')
    before=cb.step(batch(collection));dump(OUT/'collection_metrics.json',before)
    cb.provider.library.load(OUT/'library')
    if len(cb.provider.library):
        reuse=await phase('reuse')
        after=cb.step(batch(reuse));dump(OUT/'reuse_metrics.json',after)
    else:
        reuse=[];after={}
    cb.provider.library.load(OUT/'library')
    audits=[r.get('distillation_audit') for r in collection+reuse if r.get('distillation_audit')]
    assert sum(a.get('model_calls',0) for a in audits)<=16
    dump(OUT/'result.json',{'official_agent_loop_used':True,'backend':'native SGLang Engine with bundled SGLangHttpServer.generate; no Ray/FSDP synchronization',
        'collection_successes':sum(r['traj_success'] for r in collection),'collection_admitted':before.get('skill/admitted',0),
        'reuse_episodes':len(reuse),'reuse_successes':sum(r['traj_success'] for r in reuse),
        'reuse_skill_ids':[r['retrieved_skill_id'] for r in reuse],
        'distillation_calls':sum(a.get('model_calls',0) for a in audits),'valid_distillations':sum(a['valid'] for a in audits),
        'library_size':len(cb.provider.library),'skills':[{'id':s['id'],'utility':s['utility'],'use_count':s['use_count']} for s in cb.provider.library._skills],
        'total_model_calls':manager.calls,'paper_exact':False,'training':False,'cold_start':False,
        'wall_seconds':time.time()-started,'driver_torch_peak_bytes':torch.cuda.max_memory_allocated(),'native_engine_memory_is_in_child_process':True})

def launch():
    import socket
    import sglang
    OUT.mkdir(exist_ok=False)
    loop=asyncio.new_event_loop();asyncio.set_event_loop(loop);engine=None
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        args=dict(model_path=str(ROOT/'models/b0'),host='127.0.0.1',port=port,dtype='bfloat16',tp_size=1,
            mem_fraction_static=.25,context_length=9216,max_running_requests=1,disable_cuda_graph=True,
            attention_backend='triton',mm_attention_backend='sdpa',trust_remote_code=False,random_seed=42,log_level='info')
        dump(OUT/'engine_args.json',args)
        engine=sglang.Engine(**args);asyncio.set_event_loop(loop)
        loop.run_until_complete(main(engine))
    finally:
        if engine is not None:engine.shutdown()
        loop.close()

if __name__=='__main__':launch()
