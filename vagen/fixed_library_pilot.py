"""Independent fixed-memory learning pilot; existing PPO loss and optimizers unchanged."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from vagen.bootstrap_pilot import PilotPlainTrainer,jsonable
from vagen.ray_trainer import RayPPOTrainer
from vagen.main_ppo_skill import TaskRunner
from verl.workers.fsdp_workers import CriticWorker,AsyncActorRolloutRefWorker

def record(kind,name,value):
    folder=Path(os.environ['PILOT_OUT'])/kind;folder.mkdir(parents=True,exist_ok=True)
    (folder/f'{name}.json').write_text(json.dumps(jsonable(value),indent=2))

class PairedCriticWorker(CriticWorker):
    def _build_critic_model_optimizer(self,config):
        import verl.utils.model as models
        original=models.load_valuehead_model
        seed=int(os.environ['FIXED_RUN_SEED'])
        def load(*args,**kwargs):
            model=original(*args,**kwargs)
            heads={name:{'shape':list(p.shape),'values':p.detach().float().cpu().flatten().tolist()}
                   for name,p in model.named_parameters() if not p.is_meta and
                   (name.startswith('v_head.') or name.startswith('score.') or '.v_head.' in name)}
            if self.rank==0:
                assert heads and sum(len(x['values']) for x in heads.values())<1000000
                if os.environ['TASK_PHASE']=='B':
                    prior=Path(os.environ['FIXED_A_INITIAL_HEAD'])
                    assert prior.exists(),'A paired-head evidence absent'
                    assert heads==json.loads(prior.read_text())['head_values'],'paired critic initial head mismatch'
                record('initialization',f'critic_rank{self.rank}',{'run_seed':seed,'head_values':heads,
                    'paired_to_A':os.environ['TASK_PHASE']=='B','scope':'Full small pre-FSDP value head; backbone loaded from same existing b0, not a whole-model identity proof.'})
            return model
        models.load_valuehead_model=load
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                return super()._build_critic_model_optimizer(config)
        finally:models.load_valuehead_model=original


def install_actor_first_batch_check():
    from verl.workers.actor import DataParallelPPOActor
    if getattr(DataParallelPPOActor,'_fixed_check_installed',False):return
    original=DataParallelPPOActor.compute_log_prob
    def checked(self,data,calculate_entropy=False):
        output,entropys=original(self,data,calculate_entropy)
        if calculate_entropy and not getattr(self,'_fixed_first_checked',False):
            self._fixed_first_checked=True
            assert 'response_mask' in data.batch
            from verl.utils.device import get_device_id
            count=min(8,len(data));rows=[];started=time.time();previous=self.actor_module.training
            try:
                self.actor_module.train()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
                    for i,micro in enumerate(data[:count].split(1)):
                        micro=micro.to(get_device_id());inputs={**micro.batch,**micro.non_tensor_batch}
                        _,actual=self._forward_micro_batch(inputs,temperature=data.meta_info['temperature'],calculate_entropy=False)
                        mask=inputs['response_mask'].bool();expected=output[i:i+1]
                        difference=(actual-expected).abs()[mask]
                        assert bool(torch.isfinite(actual[mask]).all())
                        mmi=inputs.get('multi_modal_inputs',[{}])[0] or {};grid=mmi.get('image_grid_thw')
                        rows.append({'seed':int(inputs['seed'][0]),'generated_tokens':int(mask.sum()),
                            'processed_image_count':int(grid.shape[0]) if grid is not None else 0,
                            'max_abs_logprob_error':float(difference.max()),'mean_abs_logprob_error':float(difference.mean())})
            finally:self.actor_module.train(previous)
            max_error=max(r['max_abs_logprob_error'] for r in rows)
            record('numerical',f'actor_first_batch_rank{torch.distributed.get_rank()}',{'samples':rows,
                'extra_forward_seconds':time.time()-started,'max_abs_logprob_error':max_error,
                'rng_preserved':True,'same_preupdate_weights':True,'SGLang_logprob_alignment_verified':False})
            assert max_error<=.05 and max(r['mean_abs_logprob_error'] for r in rows)<=.005,'same-weight actor logprob alignment gate failed'
        return output,entropys
    DataParallelPPOActor.compute_log_prob=checked;DataParallelPPOActor._fixed_check_installed=True

class AuditedActorWorker(AsyncActorRolloutRefWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from transformers import set_seed
        set_seed(int(os.environ['FIXED_RUN_SEED']))
        install_actor_first_batch_check()


def install_return_check():
    import vagen.ray_trainer as trainer
    original=trainer.compute_advantage;done=False
    def checked(batch,*args,**kwargs):
        nonlocal done
        batch=original(batch,*args,**kwargs)
        if not done:
            done=True;assert kwargs['gamma']==kwargs['lam']==1.0
            mask=batch.batch['response_mask'].bool();rewards=batch.batch['token_level_rewards']
            assert bool(((rewards==0)|mask).all())
            assert bool(((rewards!=0).sum(-1)<=1).all()),'expected one terminal aggregate reward per episode'
            total=rewards.sum(-1,keepdim=True).expand_as(rewards)
            error=(batch.batch['returns']-total).abs()[mask]
            samples=[]
            for i in range(min(16,len(batch))):
                ep=json.loads(Path(batch.non_tensor_batch['pilot_episode_path'][i]).read_text())
                env_total=sum(ep['mechanism_audit']['env_rewards']);success=float(ep['traj_success'])
                assert abs(env_total-ep['reward_score'])<1e-6
                samples.append({'seed':int(batch.non_tensor_batch['seed'][i]),'success_reward':success,
                    'format_reward':env_total-success,'skill_reward':0,'episode_reward':ep['reward_score'],
                    'terminal_batch_reward':float(rewards[i].sum()),'generated_tokens':int(mask[i].sum())})
                assert abs(float(rewards[i].sum())-ep['reward_score'])<1e-5
            finite=all(bool(torch.isfinite(batch.batch[k]).all()) for k in ['values','returns','advantages','old_log_probs'])
            maximum=float(error.max())
            record('numerical','first_batch_returns',{'samples':samples,'batch_rows':len(batch),'finite':finite,
                'max_abs_terminal_return_error':maximum,'reward_on_generated_only':True,
                'loss_mask_observation_padding_zero':True,'scope':'Mask used by unchanged actor/critic loss; advantages outside mask need not be zero.'})
            assert finite and maximum<=.001,'first real batch numerical gate failed'
            assert len(list((Path(os.environ['PILOT_OUT'])/'numerical').glob('actor_first_batch_rank*.json')))==2
        return batch
    trainer.compute_advantage=checked

class MidpointStop(RuntimeError):pass

class FixedTrainer(PilotPlainTrainer):
    def _get_gen_batch(self,batch):
        output=super()._get_gen_batch(batch)
        phase=next(iter(set(output.non_tensor_batch['pilot_phase'])))
        context=('empty' if os.environ['TASK_PHASE']=='A' else 'cards') if phase=='train' else self._eval_context
        output.non_tensor_batch['fixed_context']=np.full(len(output),context,dtype=object)
        return output

    def _validate(self):
        version=min(4,self.global_steps);assert version in [0,2,4]
        root=Path(os.environ['PILOT_OUT']);lib=Path(os.environ['FIXED_CARD_LIBRARY'])/'skill_index.json'
        before=lib.read_text();metrics={}
        for context in (['empty','cards'] if version==0 else ['cards','empty']):
            self._eval_context=context
            self.config.trainer.validation_data_dir=str(root/f'dev_{version}_{context}')
            values=RayPPOTrainer._validate(self)
            metrics.update({context+'/'+k:v for k,v in values.items()})
        assert before==lib.read_text(),'evaluation changed frozen library'
        rows=[json.loads(p.read_text()) for p in (root/'episodes').glob(f'dev_{version}_*/episode.json')]
        assert len(rows)==192 and all(sum(r['fixed_context']==c for r in rows)==96 for c in ['empty','cards'])
        natural={}
        for context in ['empty','cards']:
            selected=[r for r in rows if r['fixed_context']==context and r['fixed_group']=='natural' and r['bfs']>1]
            assert len(selected)==53
            natural[context]={'success':sum(r['traj_success'] for r in selected),
                'quality':sum(r['traj_success'] and len(r['trajectory_actions'])<=r['bfs'] for r in selected)}
        self._pilot_record('cross_evaluation',str(version),{'version':version,'episodes':192,'contexts':['empty','cards'],'natural_multi':natural})
        if version==0:self._fixed_baseline=natural
        if version==2:
            context='empty' if os.environ['TASK_PHASE']=='A' else 'cards'
            decline={m:self._fixed_baseline[context][m]-natural[context][m] for m in ['success','quality']}
            if max(decline.values())>=4:
                self._pilot_record('scientific_stop','midpoint',{'decline':decline,'context':context,'version':2,'full_cross_completed':True})
                raise MidpointStop('frozen natural multi-task decline gate')
        return metrics

    def _log_rollout_data(self,batch,*args,**kwargs):
        if os.environ['TASK_PHASE']=='B':
            assert sum(x=='skill_4' for x in batch.non_tensor_batch['retrieved_skill_id'])==8,'eight planned multi-card training exposures absent'
        super()._log_rollout_data(batch,*args,**kwargs)

    def fit(self):
        stopped=False
        try:RayPPOTrainer.fit(self)
        except MidpointStop:stopped=True
        root=Path(os.environ['PILOT_OUT']);updates=len(list((root/'updated_batches').glob('*.json')))
        assert updates==(2 if stopped else 4)
        versions=sorted(int(p.stem) for p in (root/'cross_evaluation').glob('*.json'))
        assert versions==([0,2] if stopped else [0,2,4])
        self._pilot_record('completion','trainer',{'training_batches':updates,'scientific_midpoint_stop':stopped,
            'cross_versions':versions,'checkpoint_saved':False,'final_model_retained':False,
            'learning_benefit_established':False})

class FixedTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self,config):
        import ray
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(AuditedActorWorker)
        return AuditedActorWorker,RayWorkerGroup

    def add_critic_worker(self,config):
        import ray
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.Critic]=ray.remote(PairedCriticWorker)

    def run(self,config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer=FixedTrainer
        install_return_check()
        return super().run(config)
