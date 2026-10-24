"""One shared native-b0 batch; one actor step with two initial value heads."""
import json,os,time,uuid
from pathlib import Path
import numpy as np
import torch
import ray
from verl import DataProto
from verl.single_controller.base.decorator import register,Dispatch
from verl.workers.fsdp_workers import CriticWorker
from vagen.fixed_library_pilot import AuditedActorWorker
from vagen.main_ppo_skill import TaskRunner
from vagen.ray_trainer import RayPPOTrainer,compute_advantage
from vagen.recorded_backend_audit import CheckTrainer

ROOT=Path(os.environ['TASK_ROOT']);NAME='value_head_single_step_v1'
BASE=ROOT/f'artifacts/component_tests/{NAME}'
def out():return Path(os.environ['PILOT_OUT'])
def save(name,value):
    p=out()/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))

class SingleActor(AuditedActorWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        model=self.actor_module_fsdp._fsdp_wrapped_module
        tied=model.get_input_embeddings().weight is model.get_output_embeddings().weight
        assert model.config.tie_word_embeddings and tied,'inference snapshot dedup requires actual tied parameters'
        assert len(self.actor_optimizer.state)==0,'optimizer did not start empty'
        save(f'initial_actor_rank{self.rank}.json',{'optimizer_state_entries':0,'actual_embedding_parameter_tied':tied,'rank':self.rank})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def retain_inference(self):
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu,get_fsdp_full_state_dict
        from safetensors.torch import save_file
        from safetensors import safe_open
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        state=get_fsdp_full_state_dict(self.actor_module_fsdp,offload_to_cpu=True,rank0_only=True)
        result=None
        if self.rank==0:
            model=self.actor_module_fsdp._fsdp_wrapped_module
            assert model.config.tie_word_embeddings
            embedding=[k for k in state if k.endswith('embed_tokens.weight')]
            heads=[k for k in state if k.endswith('lm_head.weight')]
            assert len(embedding)==len(heads)==1 and state[embedding[0]].shape==state[heads[0]].shape
            # init_model verified these are one actual parameter; HF re-ties the alias.
            state.pop(heads[0]);converted={k:v.to(torch.bfloat16).contiguous() if v.is_floating_point() else v.contiguous() for k,v in state.items()}
            payload=sum(v.numel()*v.element_size() for v in converted.values())
            assert payload<7600000000,'unexpected inference snapshot size; refuse write'
            folder=out()/'inference_actor';folder.mkdir(exist_ok=False)
            import copy
            config=copy.deepcopy(model.config);config.torch_dtype=torch.bfloat16
            config.save_pretrained(folder);self.tokenizer.save_pretrained(folder);self.processor.save_pretrained(folder)
            path=folder/'model.safetensors';save_file(converted,str(path),metadata={'format':'pt'})
            with safe_open(str(path),framework='pt',device='cpu') as f:
                assert embedding[0] in f.keys() and heads[0] not in f.keys()
                shape=list(f.get_slice(embedding[0]).get_shape())
            result={'bytes':path.stat().st_size,'payload_bytes':payload,'stored_dtype':'bfloat16 for floating tensors','dropped_tied_alias':heads[0],
                    'embedding_shape':shape,'optimizer_saved':False,'full_training_resume_verified':False,'standalone_reload_verified':False,
                    'scope':'Inference-only weights, key/shape/header readability checked; in-memory post-update evaluation is separate.'}
            save('inference_snapshot.json',result)
            del converted,state
        torch.distributed.barrier();offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return result

class InitialHeadCritic(CriticWorker):
    def _build_critic_model_optimizer(self,config):
        import verl.utils.model as models
        original=models.load_valuehead_model
        stored=json.loads(Path(os.environ['CHECK_HEAD']).read_text())['head_values']
        def load(*args,**kwargs):
            model=original(*args,**kwargs);found={}
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in stored and not p.is_meta:
                        data=stored[name];p.copy_(torch.tensor(data['values'],dtype=p.dtype).reshape(data['shape']))
                        found[name]={'shape':list(p.shape),'values':p.detach().float().cpu().flatten().tolist()}
            assert found==stored,'initial head copy incomplete'
            if self.rank==0:save('initial_critic_head.json',{'exact_saved_head_loaded':True,'head_mode':os.environ['TASK_PHASE'],'parameter_count':sum(len(v['values']) for v in found.values())})
            return model
        models.load_valuehead_model=load
        try:return super()._build_critic_model_optimizer(config)
        finally:models.load_valuehead_model=original

class SingleTrainer(RayPPOTrainer):
    make_batch=CheckTrainer.make_batch
    def _get_gen_batch(self,batch):
        result=super()._get_gen_batch(batch);phase=next(iter(set(result.non_tensor_batch['pilot_phase'])))
        result.non_tensor_batch['pilot_policy_version']=np.full(len(result),self.global_steps,dtype=object)
        result.non_tensor_batch['fixed_context']=np.full(len(result),'cards',dtype=object)
        return result

    def collect(self):
        from verl.trainer.ppo.reward import compute_reward
        assert len(self.train_dataloader)==1 and not self.use_critic
        data=DataProto.from_single_dict(next(iter(self.train_dataloader)))
        data.non_tensor_batch['uid']=np.array([str(uuid.uuid4()) for _ in range(len(data))],dtype=object)
        gen=self._get_gen_batch(data);gen.meta_info['global_steps']=0
        save('collection_started.json',{'rows':32,'time_unix':time.time()})
        data=data.union(self.async_rollout_manager.generate_sequences(gen))
        rewards,extra=compute_reward(data,self.reward_fn);rewards=rewards.float()
        data.batch['token_level_scores']=rewards;data.batch['token_level_rewards']=rewards
        mask=data.batch['response_mask'];assert bool(((rewards==0)|mask.bool()).all())
        assert int((rewards!=0).sum(-1).max())<=1
        manifest=json.loads((BASE/'data_manifest.json').read_text());expected=manifest['splits']['train']
        assert [int(s) for s in data.non_tensor_batch['seed']]==[r['seed'] for r in expected]
        rows=[]
        for i,path in enumerate(data.non_tensor_batch['pilot_episode_path']):
            ep=json.loads(Path(path).read_text());row=expected[i]
            assert ep['initial_env_state']==row['state'] and abs(float(rewards[i].sum())-ep['reward_score'])<1e-5
            if row['category']=='multi_up3_applicable':assert ep['retrieved_skill_id']=='skill_4'
            rows.append({'seed':row['seed'],'category':row['category'],'success':bool(ep['traj_success']),'reward':float(rewards[i].sum()),'episode':str(path)})
        torch.save(data,out()/'shared_batch.pt')
        targets=[r for r in rows if r['category']=='multi_up3_applicable'];assert len(targets)==16
        success=sum(r['success'] for r in targets)
        save('collection_gate.json',{'rows':rows,'target_successes':success,'target_denominator':16,'gate_pass':success>=1,'resampling_allowed':False})
        save('COMPLETE.json',{'phase':'C','time_unix':time.time(),'environment_episodes':32,'optimizer_steps':0,'target_success_gate_pass':success>=1})

    def probe(self):
        m=json.loads((ROOT/'artifacts/component_tests/recorded_backend_audit_v2/inputs.json').read_text())
        frozen=json.loads((BASE/'stop_probe.json').read_text())
        rows=[m['episodes'][key] for key in frozen['episode_ids']]
        return self.make_batch(rows),frozen

    def signals(self,batch,post,probe_pre,probe_post,frozen):
        np.savez_compressed(out()/'training_signals.npz',**{k:batch.batch[k].float().cpu().numpy() for k in ['values','returns','advantages','old_log_probs','token_level_rewards','response_mask']},post_log_probs=post.float().numpy(),historical_stop_probe_pre=probe_pre.float().numpy(),historical_stop_probe_post=probe_post.float().numpy(),input_ids=batch.batch['input_ids'].cpu().numpy(),position_ids=batch.batch['position_ids'].cpu().numpy(),attention_mask=batch.batch['attention_mask'].cpu().numpy())
        rows=[]
        for i,row in enumerate(frozen['rows']):
            p=row['answer_token_positions'];assert len(p)==1
            a=float(probe_pre[i,p[0]]);b=float(probe_post[i,p[0]])
            rows.append(dict(row,pre_log_prob=a,post_log_prob=b,delta_log_prob=b-a,probability_ratio=float(np.exp(b-a))))
        save('stop_probe_result.json',{'rows':rows,'posthoc_historical_two_token_probe':True,'scope':'Actual teacher-forced probability changes; not unseen generalization or original theta4 recovery.'})

    def fit(self):
        self.global_steps=0
        if os.environ['TASK_PHASE']=='C':return self.collect()
        from verl.utils.metric import reduce_metrics
        gate=json.loads((BASE/'C/collection_gate.json').read_text());assert gate['gate_pass']
        batch=torch.load(BASE/'C/shared_batch.pt',map_location='cpu',weights_only=False)
        assert len(batch)==32 and self.use_critic and not self.use_reference_policy
        carried=os.getenv('CARRY_BASELINE_FROM')
        if carried:
            prior=Path(carried)
            assert os.environ['TASK_PHASE']=='O' and not (prior/'update_started.json').exists()
            episodes=[json.loads(p.read_text()) for p in (prior/'episodes').glob('*/episode.json')]
            expected=json.loads((BASE/'data_manifest.json').read_text())['splits']['dev']
            assert len(episodes)==16 and {r['seed'] for r in episodes}=={r['seed'] for r in expected}
            assert all(r['policy_version']==0 and r['phase']=='dev' for r in episodes)
            save('baseline_carried.json',{'source':str(prior),'episodes':16,'new_baseline_calls':0,
                'scope':'Complete original O baseline, same retained b0 source and empty optimizer; new worker instance after pre-update serialization failure.'})
        else:
            save('baseline_eval_started.json',{'time_unix':time.time(),'episodes':16})
            self._validate()
        old=self.actor_rollout_wg.compute_log_prob(batch);old.batch.pop('entropys');batch=batch.union(old)
        batch=batch.union(self.critic_wg.compute_values(batch))
        batch=compute_advantage(batch,adv_estimator=self.config.algorithm.adv_estimator,gamma=1.,lam=1.,num_repeat=1,config=self.config.algorithm)
        mask=batch.batch['response_mask'].bool();assert all(torch.isfinite(batch.batch[k][mask]).all() for k in ['values','returns','advantages','old_log_probs'])
        err=(batch.batch['returns']-batch.batch['token_level_rewards'].sum(-1,keepdim=True)).abs()[mask].max()
        assert err<=.001
        if os.environ['TASK_PHASE']=='Z':assert float(batch.batch['values'].abs()[mask].max())==0
        # DataProto may pack different dtype views into one byte storage.
        torch.save({k:batch.batch[k].detach().cpu().clone() for k in ['values','returns','advantages','old_log_probs','response_mask','token_level_rewards']},out()/'preupdate_signals.pt')
        probe,frozen=self.probe();before=self.actor_rollout_wg.compute_log_prob(probe).batch['old_log_probs'].cpu()
        batch.meta_info.update(global_token_num=batch.batch['attention_mask'].sum(-1).tolist(),multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable)
        assert not (out()/'update_started.json').exists(),'never replay a dispatched actor update'
        save('update_started.json',{'logical_global_optimizer_steps':1,'global_minibatch':32,'microbatch_per_gpu':1,'rows':32,'time_unix':time.time()})
        updated=self.actor_rollout_wg.update_actor(batch)
        save('actor_metrics.json',reduce_metrics(updated.meta_info['metrics']))
        audit=[json.loads(line) for path in (out()/'update_audit').glob('actor_*.jsonl') for line in path.read_text().splitlines()]
        assert len(audit)==2 and {r['rank'] for r in audit}=={0,1} and all(r['finite'] and r['sample_delta_l2']>0 for r in audit)
        after=self.actor_rollout_wg.compute_log_prob(batch).batch['old_log_probs'].cpu()
        probe_after=self.actor_rollout_wg.compute_log_prob(probe).batch['old_log_probs'].cpu()
        self.signals(batch,after,before,probe_after,frozen)
        self.global_steps=1
        save('post_eval_started.json',{'time_unix':time.time(),'episodes':16})
        self._validate()
        self.actor_rollout_wg.retain_inference()
        episodes=[json.loads(p.read_text()) for p in (out()/'episodes').glob('*/episode.json')]
        assert len(episodes)==(16 if carried else 32) and all(r['phase']=='dev' for r in episodes)
        save('COMPLETE.json',{'phase':os.environ['TASK_PHASE'],'time_unix':time.time(),'global_actor_optimizer_steps':1,'critic_optimizer_steps':0,'own_baseline_episodes':16,'baseline_carried_from_original_attempt':bool(carried),'new_environment_episodes':16 if carried else 32,'post_update_episodes':16,'inference_snapshot_retained':True,'learning_benefit_established':False})

class SingleTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self,config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(SingleActor)
        return SingleActor,RayWorkerGroup
    def add_critic_worker(self,config):
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.Critic]=ray.remote(InitialHeadCritic)
    def run(self,config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer=SingleTrainer
        return super().run(config)
