"""One original next batch: verify retrieved inputs reach native PPO and real updates."""
import os,json,time,math
from pathlib import Path
import torch
import numpy as np
from verl.single_controller.base.decorator import register,make_nd_compute_dataproto_dispatch_fn
from vagen.ray_trainer_skill import RayPPOTrainerSkill as NativeTrainer
from vagen.main_ppo_skill import TaskRunner
from . import official_train_observer as base
ROOT,SRC,OUT,P=base.ROOT,base.SRC,base.OUT,base.P


def parameters(module):
    return {name:p.detach().reshape(-1)[:256].float().cpu().clone() for name,p in module.named_parameters() if p.requires_grad and p.numel()}

def validate_packet(data,episodes):
    seeds=[int(v) for v in data.non_tensor_batch['seed']];mask=data.batch['response_mask'];responses=data.batch['responses'];records=[]
    assert responses.shape==mask.shape and set(torch.unique(mask).tolist())<=set([0,1])
    for key in ['old_log_probs','advantages','returns']:
        assert data.batch[key].shape==mask.shape and torch.isfinite(data.batch[key][mask.bool()]).all(),key
    for i,seed in enumerate(seeds):
        episode=episodes[seed];n=len(episode['response_ids']);sid=data.non_tensor_batch['retrieved_skill_id'][i]
        assert sid==episode.get('retrieved_skill_id') and n<=responses.shape[1]
        assert responses[i,:n].cpu().tolist()==episode['response_ids']
        assert mask[i,:n].cpu().tolist()==episode['response_mask'] and int(mask[i,n:].sum())==0
        mm=data.non_tensor_batch['multi_modal_inputs'][i];grid=mm['image_grid_thw'];pixels=mm['pixel_values']
        assert grid.ndim==2 and grid.shape[1]==3 and int(grid.prod(-1).sum())==pixels.shape[0]
        if sid:assert len(grid)>=2
        records.append({'seed':seed,'retrieved_skill_id':sid,'image_count':len(grid),'grid_shape':list(grid.shape),'pixel_shape':list(pixels.shape),'pixel_dtype':str(pixels.dtype),'generated_tokens':int(mask[i].sum()),'original_response_tokens_and_mask_match':True})
    return records

def audit_update(worker,role,data,native_call):
    assert len(data)==64
    episodes={r['seed']:r for r in [json.loads(x) for x in (OUT/'episodes/train_0002.jsonl').read_text().splitlines()]};rows=validate_packet(data,episodes)
    module=worker.actor_module_fsdp if role=='actor' else worker.critic_module
    optim=worker.actor_optimizer if role=='actor' else worker.critic_optimizer
    before=parameters(module);calls=[]
    hook=optim.register_step_post_hook(lambda *args,**kwargs:calls.append(1))
    if role=='actor':
        assert data.meta_info['multi_turn'] is True and float(data.meta_info['temperature'])==1.
        bank={'tensors':{k:v.detach().cpu() for k,v in data.batch.items()},'multi_modal_inputs':data.non_tensor_batch['multi_modal_inputs'],'seeds':data.non_tensor_batch['seed'],'retrieved_skill_id':data.non_tensor_batch['retrieved_skill_id'],'uid':data.non_tensor_batch['uid'],'meta_info':data.meta_info}
        tensor_bytes=sum(v.numel()*v.element_size() for v in bank['tensors'].values())+sum(v.numel()*v.element_size() for mm in bank['multi_modal_inputs'] for v in mm.values() if torch.is_tensor(v))
        assert tensor_bytes<P['maximum_private_bank_tensor_bytes_per_rank']
        torch.save(bank,OUT/f'PRIVATE_ACTOR_INPUT_rank{worker.rank}.pt')
    base.write(OUT/f'{role}_rank{worker.rank}_intent.json',{'started_unix':time.time(),'rows':rows,'native_update_about_to_call':True})
    try:result=native_call(data)
    finally:hook.remove()
    after=parameters(module);assert set(before)==set(after);stats=[]
    for name,a in before.items():
        b=after[name];assert a.shape==b.shape and torch.isfinite(b).all();d=b-a
        stats.append({'parameter_shard':name,'sampled_elements':d.numel(),'changed_elements':int(torch.count_nonzero(d)),'L2_delta':float(d.norm()),'max_abs_delta':float(d.abs().max())})
    metrics=base.plain(result.meta_info['metrics'])
    def finite(v):
        if isinstance(v,list):return all(finite(x) for x in v)
        return isinstance(v,(int,float)) and math.isfinite(v)
    selected={k:v for k,v in metrics.items() if any(t in k for t in ['grad_norm','pg_loss','vf_loss','ppo_kl'])};assert selected and all(finite(v) for v in selected.values())
    assert len(calls)==4 and any(x['changed_elements'] for x in stats)
    base.write(OUT/f'{role}_rank{worker.rank}_audit.json',{'status':'PASS_NATIVE_RETRIEVED_INPUT_UPDATE','completed_unix':time.time(),'role':role,'rank':worker.rank,'rows':rows,'row_count':len(rows),'retrieved_rows':sum(bool(r['retrieved_skill_id']) for r in rows),'actual_optimizer_step_calls':len(calls),'finite_native_metrics':selected,'parameter_sample_stats':stats,'sampled_elements':sum(x['sampled_elements'] for x in stats),'changed_sample_elements':sum(x['changed_elements'] for x in stats),'native_return_object_preserved':True})
    return result

class CheckActor(base.RunActor):
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='actor'))
    def update_actor(self,data):return audit_update(self,'actor',data,super().update_actor)
class CheckCritic(base.RunCritic):
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='critic'))
    def update_critic(self,data):return audit_update(self,'critic',data,super().update_critic)

class CheckTracking(base.NativeTracking):
    def log(self,data,step,*args,**kwargs):
        base.write(OUT/'metrics'/f'step_{step:04d}.json',{'step':int(step),'recorded_unix':time.time(),'metrics':data})
        result=super().log(data,step,*args,**kwargs)
        assert step==2 and data['training/global_step']==2
        raise base.ControlledStop('one retrieved-input native batch completed')

class CheckTrainer(base.RunTrainer):
    def _load_checkpoint(self):
        result=NativeTrainer._load_checkpoint(self);assert self.global_steps==1
        base.write(OUT/'LOAD_APPLIED.json',{'applied_unix':time.time(),'global_step':1,'native_full_actor_critic_model_optimizer_extra_dataloader_load_returned':True,'reference_checkpoint':'official_frozenlake_correctness_v1/global_step_1','reference_retained_read_only':True});return result
    def _get_gen_batch(self,batch):
        expected=json.loads((OUT/'next_batch_manifest.json').read_text())['next_seeds']
        assert [int(x) for x in batch.non_tensor_batch['seed']]==expected
        return super()._get_gen_batch(batch)
    def _save_checkpoint(self):
        raise RuntimeError('This terminal input/update diagnostic must not overwrite or duplicate the reference checkpoint')
    def fit(self):
        base.ACTIVE=self;assert self.use_critic and not self.use_reference_policy and not self.config.trainer.val_before_train
        base.write(OUT/'training_started.json',{'started_unix':time.time(),'resume_step':1,'only_global_step_target':2,'new_full_checkpoint_requested':False})
        try:NativeTrainer.fit(self)
        except base.ControlledStop as exc:
            assert str(exc)=='one retrieved-input native batch completed' and self.global_steps==2
            audits=[json.loads((OUT/f'{role}_rank{rank}_audit.json').read_text()) for role in ['actor','critic'] for rank in [0,1]]
            for role in ['actor','critic']:
                part=[a for a in audits if a['role']==role];assert sum(a['row_count'] for a in part)==128 and sum(a['retrieved_rows'] for a in part)>0
            base.write(OUT/'WORKER_COMPLETED.json',{'status':'PASS_ONE_RETRIEVED_INPUT_PPO_BATCH','completed_unix':time.time(),'resumed_global_step':1,'completed_global_step':2,'additional_global_batches':1,'actual_distributed_optimizer_steps_per_role':4,'role_rank_audit_files':4,'reference_full_checkpoint_retained':True,'terminal_step2_model_optimizer_state_saved':False});return
        raise AssertionError('Native run must stop after exactly the next original batch')

class CheckTask(base.RunTask):
    def add_actor_rollout_worker(self,config):
        import ray
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(CheckActor);return CheckActor,RayWorkerGroup
    def add_critic_worker(self,config):
        import ray
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.Critic]=ray.remote(CheckCritic)
    def run(self,config):
        import vagen.main_ppo_skill as entry
        import verl.utils.tracking as tracking
        torch.manual_seed(42);tracking.Tracking=CheckTracking;entry.RayPPOTrainer=CheckTrainer;return TaskRunner.run(self,config)
