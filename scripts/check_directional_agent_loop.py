"""Real official loop + explicit contract variant, with local inference adapter only."""
import asyncio, json, os, time
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop
from vagen.skills.distill_callback import DistillCallback

ROOT=Path(os.environ['TASK_ROOT'])
OUT=ROOT/'artifacts/component_tests/directional_agent_loop_v1'


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,default=lambda v:v.tolist()))


async def main():
    OUT.mkdir(exist_ok=False);started=time.time();set_seed(42)
    processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT/'models/b0',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
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
    sampling={'temperature':1.0,'top_p':1.0,'top_k':0,'max_new_tokens':512}

    class LocalManager:
        calls=0
        phase='collect'
        seed=None

        async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, **kwargs):
            self.calls+=1;assert self.calls<=96
            limit=int(sampling_params.get('max_new_tokens',512));assert len(prompt_ids)+limit<=9216
            ids=torch.tensor([prompt_ids],device='cuda')
            batch={'input_ids':ids,'attention_mask':torch.ones_like(ids)}
            if image_data:
                visual=processor.image_processor(images=image_data,return_tensors='pt').to('cuda')
                expected=int((visual['image_grid_thw'].prod(-1)//processor.image_processor.merge_size**2).sum())
                assert int((ids==model.config.image_token_id).sum())==expected
                batch.update(visual)
            temperature=float(sampling_params.get('temperature',1.0));sample=temperature>0
            options={'temperature':temperature,'top_p':float(sampling_params.get('top_p',1.0)),
                     'top_k':max(0,int(sampling_params.get('top_k',0)))} if sample else {}
            with torch.inference_mode():
                result=model.generate(**batch,max_new_tokens=limit,do_sample=sample,**options)
            tokens=result[0,len(prompt_ids):].tolist()
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
    dump(OUT/'result.json',{'official_agent_loop_used':True,'backend':'local Transformers adapter, not native rollout workers',
        'collection_successes':sum(r['traj_success'] for r in collection),'collection_admitted':before.get('skill/admitted',0),
        'reuse_episodes':len(reuse),'reuse_successes':sum(r['traj_success'] for r in reuse),
        'reuse_skill_ids':[r['retrieved_skill_id'] for r in reuse],
        'distillation_calls':sum(a.get('model_calls',0) for a in audits),'valid_distillations':sum(a['valid'] for a in audits),
        'library_size':len(cb.provider.library),'skills':[{'id':s['id'],'utility':s['utility'],'use_count':s['use_count']} for s in cb.provider.library._skills],
        'total_model_calls':manager.calls,'paper_exact':False,'training':False,'cold_start':False,
        'wall_seconds':time.time()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})

if __name__=='__main__':asyncio.run(main())
