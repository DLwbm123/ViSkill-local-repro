"""One actor loss numerator intervention; full GAE and denominator preserved."""
import os,json,time,runpy
from pathlib import Path
import numpy as np
import torch,ray
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import register,Dispatch
from vagen.prospective_value_head_eval import EvalActor,EvalTrainer
from vagen.bf16_delta import apply_to_model,retain_state_delta
from vagen.main_ppo_skill import TaskRunner

ROOT=Path(os.environ['TASK_ROOT']);NAME='answer_region_step_v1';BASE=ROOT/f'artifacts/component_tests/{NAME}'

def protocol():return json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
def training_folder():return BASE/protocol()['training_output_folder']

class LocalActor(EvalActor):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        info=json.loads((Path(os.environ['PILOT_OUT'])/'actual_loading_info.json').read_text())
        assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        if os.environ['TASK_PHASE']=='train' and protocol().get('carry_preupdate_from'):
            self.actor._fixed_first_checked=True
            (Path(os.environ['PILOT_OUT'])/f'carried_alignment_rank{self.rank}.json').write_text(json.dumps({'carried_original_alignment_samples':8,'new_alignment_samples':0,'scope':'Original32 scoring and16 checks retained; new same-b0 worker instance after proven pre-forward metadata failure.'}))

    def prepare_loaded_model(self,model):
        if os.environ['TASK_PHASE'] not in ['reload','A']:return {}
        folder=training_folder()/'private_actor'
        schema=json.loads((folder/'PRIVATE_PARAMETER_SCHEMA.json').read_text())
        assert all(not p.is_meta and p.device.type=='cpu' for p in model.parameters())
        return dict(apply_to_model(model,folder/'PRIVATE_LEARNED_BF16_DELTA.npz',schema),answer_region_private_delta_applied=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def retain_delta(self):
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu,get_fsdp_full_state_dict
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        state=get_fsdp_full_state_dict(self.actor_module_fsdp,offload_to_cpu=True,rank0_only=True)
        if self.rank==0:
            p=protocol();result=retain_state_delta(state,self.actor_module_fsdp._fsdp_wrapped_module,ROOT/'models/b0',training_folder()/'private_actor',p['private_archive_write_stop_bytes'],p['preparation_started_unix']+p['max_seconds_including_preparation'])
            (training_folder()/'inference_snapshot.json').write_text(json.dumps(result,indent=2))
            del state
        torch.distributed.barrier();offload_fsdp_model_to_cpu(self.actor_module_fsdp)

def make_config(phase,folder):
    cfg=runpy.run_path(str(ROOT/'src/scripts/run_value_head_single_step.py'),run_name='config_helper')['make_config']('C',folder)
    cfg.actor_rollout_ref.model.path=str(ROOT/('artifacts/component_tests/value_head_single_step_v1/Z/inference_actor' if phase=='Z' else 'models/b0'))
    cfg.data.val_files=str(ROOT/'src/configs/smoke/answer_region_dev.yaml')
    cfg.data.val_batch_size=1;cfg.actor_rollout_ref.rollout.agent.num_workers=1
    cfg.ray_kwargs.ray_init._temp_dir='/data/local/tmp/z0/u'+phase.lower()
    if phase in ['train','reload']:
        cfg.algorithm.adv_estimator='gae'
        cfg.actor_rollout_ref.actor.optim.total_training_steps=1
    return cfg

def training_batch():
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    batch=torch.load(ROOT/'artifacts/component_tests/value_head_single_step_v1/C/shared_batch.pt',map_location='cpu',weights_only=False)
    with np.load(ROOT/'artifacts/component_tests/value_head_single_step_v1/Z/training_signals.npz',allow_pickle=False) as pack:
        adv,ret=compute_gae_advantage_return(batch.batch['token_level_rewards'].float(),torch.zeros_like(batch.batch['token_level_rewards']).float(),batch.batch['response_mask'],1.,1.)
        assert np.array_equal(adv.numpy(),pack['advantages']) and np.array_equal(ret.numpy(),pack['returns'])
        assert np.array_equal(batch.batch['input_ids'].numpy(),pack['input_ids'])
        batch.batch['old_log_probs']=torch.from_numpy(pack['old_log_probs'].copy())
    with np.load(ROOT/'artifacts/component_tests/actor_loss_support_audit_v1/loss_support.npz',allow_pickle=False) as pack:
        answer=torch.from_numpy(pack['answer_loss_mask'].copy()).bool()
        assert np.array_equal(batch.batch['response_mask'].bool().numpy(),pack['full_generation_mask'].astype(bool))
    full=batch.batch['response_mask'].bool();assert len(batch)==32 and bool((answer.sum(-1)>0).all()) and not bool((answer&~full).any())
    batch.batch['advantages']=adv*answer;batch.batch['returns']=ret
    batch.meta_info.update(global_token_num=batch.batch['attention_mask'].sum(-1).tolist(),multi_turn=True,temperature=1.)
    return batch,adv,answer

def actor_group(cfg):
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    cls=RayClassWithInitArgs(ray.remote(LocalActor),config=cfg.actor_rollout_ref,role='actor')
    group=RayWorkerGroup(resource_pool=RayResourcePool([2],use_gpu=True,max_colocate_count=1,name_prefix='local'),ray_cls_with_init=create_colocated_worker_cls({'actor':cls}),device_name='cuda')
    actor=group.spawn(prefix_set={'actor'})['actor'];actor.init_model();return group,actor

def score_gate(actual,expected,mask):
    rows=[]
    for a,e,m in zip(actual.float(),expected.float(),mask.bool(),strict=True):
        diff=(a-e).abs()[m];assert bool(torch.isfinite(diff).all())
        rows.append({'generated_tokens':int(m.sum()),'max_abs_nats':float(diff.max()),'mean_abs_nats':float(diff.mean()),'operational_gate_pass':float(diff.max())<=.05 and float(diff.mean())<=.005})
    return rows

def run_saved(phase,folder,cfg):
    batch,full_adv,answer=training_batch()
    kwargs=OmegaConf.to_container(cfg.ray_kwargs.ray_init,resolve=True);kwargs['runtime_env']={};ray.init(**kwargs)
    group,actor=actor_group(cfg)
    loading=json.loads((folder/'actual_loading_info.json').read_text());assert not any(loading[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    if phase=='reload':
        batch=batch[:2];actual=actor.compute_log_prob(batch).batch['old_log_probs'].cpu()
        np.savez_compressed(folder/'scores.npz',log_probs=actual.float().numpy(),response_mask=batch.batch['response_mask'].numpy())
        with np.load(training_folder()/'training_signals.npz',allow_pickle=False) as pack:expected=torch.from_numpy(pack['post_log_probs'][:2].copy())
        result={'rows':score_gate(actual,expected,batch.batch['response_mask']),'business_forward_samples':2,'extra_alignment_forward_samples':2}
        result['operational_reload_verified']=all(r['operational_gate_pass'] for r in result['rows'])
        (folder/'reload_gate.json').write_text(json.dumps(result,indent=2))
        assert all(r['operational_gate_pass'] for r in result['rows'])
    else:
        assert phase=='train'
        carry=protocol().get('carry_preupdate_from')
        if carry:
            original=BASE/carry
            assert not list((original/'update_audit').glob('actor_*.jsonl')) and not (original/'training_signals.npz').exists()
            with np.load(original/'preupdate_scores.npz',allow_pickle=False) as pack:
                pre=torch.from_numpy(pack['log_probs'].copy());assert np.array_equal(pack['response_mask'],batch.batch['response_mask'].numpy())
            (folder/'pre_scoring_carried.json').write_text(json.dumps({'source_folder':carry,'original_business_forward_samples':32,'original_extra_alignment_samples':16,'new_prescoring_samples':0,'scope':'Same retained b0/config/seed and empty optimizer, but new worker instance; original update failed before any training forward/backward/optimizer action.'},indent=2))
        else:
            pre=actor.compute_log_prob(batch).batch['old_log_probs'].cpu()
            np.savez_compressed(folder/'preupdate_scores.npz',log_probs=pre.float().numpy(),response_mask=batch.batch['response_mask'].numpy())
        prior=score_gate(pre,batch.batch['old_log_probs'],batch.batch['response_mask'])
        (folder/'preupdate_gate.json').write_text(json.dumps(prior,indent=2))
        assert all(r['operational_gate_pass'] for r in prior)
        assert not (folder/'update_started.json').exists()
        (folder/'update_started.json').write_text(json.dumps({'global_actor_steps':1,'rows':32,'time_unix':time.time(),'only_change':'zero nonanswer already-whitened advantages; keep original response_mask and full-generated per-episode denominator'}))
        output=actor.update_actor(batch)
        audit=[json.loads(line) for path in (folder/'update_audit').glob('actor_*.jsonl') for line in path.read_text().splitlines()]
        assert len(audit)==2 and {r['rank'] for r in audit}=={0,1} and all(r['finite'] and r['sample_delta_l2']>0 for r in audit)
        post=actor.compute_log_prob(batch).batch['old_log_probs'].cpu()
        np.savez_compressed(folder/'training_signals.npz',pre_log_probs=pre.float().numpy(),post_log_probs=post.float().numpy(),advantages_before=full_adv.numpy(),advantages_after=batch.batch['advantages'].numpy(),answer_loss_mask=answer.numpy(),response_mask=batch.batch['response_mask'].numpy())
        from verl.utils.metric import reduce_metrics
        (folder/'actor_metrics.json').write_text(json.dumps(reduce_metrics(output.meta_info['metrics']),indent=2))
        actor.retain_delta()
        result={'business_forward_samples':64 if carry else 96,'extra_alignment_forward_samples':0 if carry else 16,'carried_business_forward_samples':32 if carry else 0,'carried_extra_alignment_samples':16 if carry else 0,'global_actor_optimizer_steps':1,'critic_optimizer_steps':0,'private_inference_delta_retained':True}
    result.update(completed_unix=time.time(),generation=0,environment_steps=0)
    (folder/'COMPLETE.json').write_text(json.dumps(result,indent=2))

class EvalTask(TaskRunner):
    def add_actor_rollout_worker(self,config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(LocalActor)
        return LocalActor,RayWorkerGroup
    def run(self,config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer=EvalTrainer
        return super().run(config)
