"""Fixed saved-input actor scores; no rollout, critic or update."""
import json,os,time,runpy
from pathlib import Path
import numpy as np
import torch
import ray
from omegaconf import OmegaConf
from vagen.prospective_value_head_eval import EvalActor
from vagen.bf16_delta import apply_to_model

ROOT=Path(os.environ['TASK_ROOT']);NAME='bf16_delta_reload_v1'
BASE=ROOT/f'artifacts/component_tests/{NAME}'

class LocalActor(EvalActor):
    def prepare_loaded_model(self,model):
        assert all(not p.is_meta and p.device.type=='cpu' for p in model.parameters())
        if os.environ['TASK_PHASE']=='full':return {'private_delta_applied':False}
        folder=ROOT/'artifacts/component_tests/bf16_delta_feasibility_v1'
        schema=json.loads((folder/'PRIVATE_PARAMETER_SCHEMA.json').read_text())
        result=apply_to_model(model,folder/'PRIVATE_LEARNED_BF16_DELTA.npz',schema)
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
        return dict(result,private_delta_applied=True)

def make_config(phase,folder):
    cfg=runpy.run_path(str(ROOT/'src/scripts/run_value_head_single_step.py'),run_name='config_helper')['make_config']('C',folder)
    p=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
    cfg.actor_rollout_ref.model.path=str(ROOT/p['model_paths'][phase])
    cfg.ray_kwargs.ray_init._temp_dir='/data/local/tmp/z0/q'+phase
    assert cfg.actor_rollout_ref.model.use_fused_kernels
    assert cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu==1
    assert not cfg.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
    assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype=='fp32'
    assert cfg.actor_rollout_ref.actor.fsdp_config.get('mixed_precision') is None
    assert cfg.actor_rollout_ref.actor.fsdp_config.dtype=='bfloat16'
    return cfg

def fixed_batch():
    batch=torch.load(ROOT/'artifacts/component_tests/value_head_single_step_v1/C/shared_batch.pt',map_location='cpu',weights_only=False)[:2]
    assert [int(x) for x in batch.non_tensor_batch['seed']]==[130014,130007]
    assert len(batch)==2 and bool((batch.batch['response_mask'].sum(-1)>0).all())
    assert len(batch.non_tensor_batch['multi_modal_inputs'])==2
    return batch

def main():
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    p=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
    assert p['status']=='FROZEN_CPU_READY'
    assert time.time()<min(p['absolute_total_deadline'],p['preparation_started_unix']+p['max_seconds_including_preparation'])
    phase=os.environ['TASK_PHASE'];assert phase in p['policy_order']
    folder=BASE/p['policy_output_folders'][phase];assert not folder.exists(),'never repeat a dispatched policy'
    folder.mkdir();os.environ.update(PILOT_OUT=str(folder),FIXED_RUN_SEED='20261024')
    cfg=make_config(phase,folder);OmegaConf.save(cfg,folder/'resolved.yaml',resolve=True)
    batch=fixed_batch()
    try:
        # Ordinary inherited environment keeps private paths out of worker argv.
        kwargs=OmegaConf.to_container(cfg.ray_kwargs.ray_init,resolve=True);kwargs['runtime_env']={}
        ray.init(**kwargs)
        cls=RayClassWithInitArgs(ray.remote(LocalActor),config=cfg.actor_rollout_ref,role='actor')
        colocated=create_colocated_worker_cls({'actor':cls})
        group=RayWorkerGroup(resource_pool=RayResourcePool([2],use_gpu=True,max_colocate_count=1,name_prefix='local'),ray_cls_with_init=colocated,device_name='cuda')
        actor=group.spawn(prefix_set={'actor'})['actor'];actor.init_model()
        loading=json.loads((folder/'actual_loading_info.json').read_text())
        assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'] and not loading['error_msgs']
        (folder/'score_started.json').write_text(json.dumps({'rows':2,'business_forward_samples':2,'expected_extra_alignment_samples':2,'started_unix':time.time()}))
        result=actor.compute_log_prob(batch).batch['old_log_probs'].float().cpu()
        mask=batch.batch['response_mask'].bool();assert bool(torch.isfinite(result[mask]).all())
        np.savez_compressed(folder/'scores.npz',log_probs=result.numpy(),response_mask=mask.numpy(),seeds=np.array([130014,130007]))
        checks=[json.loads(f.read_text()) for f in (folder/'numerical').glob('actor_first_batch_rank*.json')]
        assert len(checks)==2 and all(len(c['samples'])==1 for c in checks)
        (folder/'COMPLETE.json').write_text(json.dumps({'completed_unix':time.time(),'business_forward_samples':2,'extra_alignment_forward_samples':2,'generation':0,'environment_steps':0,'critic_forwards':0,'optimizer_steps':0,'new_model_snapshots':0}))
    finally:ray.shutdown()

if __name__=='__main__':main()
