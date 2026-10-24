"""A shared bank and one fixed data-support substitution, not a mask search."""
import os,json,time,runpy,uuid,re
from pathlib import Path
import numpy as np
import torch,ray
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base.decorator import register,Dispatch
from vagen.prospective_value_head_eval import EvalActor,EvalTrainer
from vagen.value_head_single_step import SingleTrainer
from vagen.answer_region_step import score_gate
from vagen.batch_gzip import save_batch,load_batch
from vagen.bf16_delta import apply_to_model,retain_state_delta
from vagen.main_ppo_skill import TaskRunner
from vagen.ray_trainer import RayPPOTrainer

ROOT=Path(os.environ['TASK_ROOT']);NAME='support_matched_curriculum_v1';BASE=ROOT/f'artifacts/component_tests/{NAME}'
def protocol():return json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
def save(name,value):
    folder=Path(os.environ['PILOT_OUT']);(folder/name).write_text(json.dumps(value,indent=2))

def answer_mask(audit,tokenizer,length):
    result=np.zeros(length,dtype=np.uint8);raw=audit['response_ids']
    for span in audit['spans']:
        if span['kind']!='generated':continue
        begin,end=span['start'],span['end'];assert end<=length
        tokens=raw[begin:end];text=tokenizer.decode(tokens,skip_special_tokens=False)
        match=re.search(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>',text,re.S)
        if match:
            sizes=[len(tokenizer.decode(tokens[:k],skip_special_tokens=False)) for k in range(len(tokens)+1)]
            for k in range(len(tokens)):
                if tokens[k] not in [151643,151645] and sizes[k+1]>match.start(2) and sizes[k]<match.end(2):result[begin+k]=1
    return result

class DataActor(EvalActor):
    def prepare_loaded_model(self,model):
        phase=os.environ['TASK_PHASE'];label=phase.split('_')[-1]
        if not phase.startswith(('reload_','eval_')) or label not in ['Q','R']:return {}
        assert all(not v.is_meta and v.device.type=='cpu' for v in model.parameters())
        folder=BASE/label/'private_actor';schema=json.loads((folder/'PRIVATE_PARAMETER_SCHEMA.json').read_text())
        return dict(apply_to_model(model,folder/'PRIVATE_LEARNED_BF16_DELTA.npz',schema),curriculum_private_delta_applied=label)
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model();info=json.loads((Path(os.environ['PILOT_OUT'])/'actual_loading_info.json').read_text())
        assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def retain_delta(self):
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu,get_fsdp_full_state_dict
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        state=get_fsdp_full_state_dict(self.actor_module_fsdp,offload_to_cpu=True,rank0_only=True)
        if self.rank==0:
            p=protocol();result=retain_state_delta(state,self.actor_module_fsdp._fsdp_wrapped_module,ROOT/'models/b0',Path(os.environ['PILOT_OUT'])/'private_actor',p['private_archive_write_stop_bytes'],p['preparation_started_unix']+p['max_seconds_including_preparation'])
            save('inference_snapshot.json',result);del state
        torch.distributed.barrier();offload_fsdp_model_to_cpu(self.actor_module_fsdp)

def make_config(phase,folder):
    cfg=runpy.run_path(str(ROOT/'src/scripts/run_value_head_single_step.py'),run_name='config_helper')['make_config']('C',folder)
    cfg.data.train_files=str(ROOT/'src/configs/smoke/support_curriculum_train.yaml');cfg.data.val_files=str(ROOT/'src/configs/smoke/support_curriculum_dev.yaml')
    cfg.data.train_batch_size=cfg.data.gen_batch_size=40 if phase=='collect' else 32
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size=40 if phase=='collect' else 32
    cfg.actor_rollout_ref.actor.optim.total_training_steps=1
    cfg.data.val_batch_size=1;cfg.actor_rollout_ref.rollout.agent.num_workers=4 if phase=='collect' else 1
    cfg.algorithm.adv_estimator='gae' if phase in ['Q','R','reload_Q','reload_R'] else 'grpo'
    cfg.ray_kwargs.ray_init._temp_dir='/data/tmp/z0/c'+str(protocol()['phase_order'].index(phase))
    return cfg

class BankTrainer(SingleTrainer):
    def fit(self):
        from verl.trainer.ppo.reward import compute_reward
        self.global_steps=0;assert not self.use_critic and len(self.train_dataset)==40 and len(self.train_dataloader)==1
        library=Path(os.environ['PILOT_OUT'])/'context_libraries/cards/skill_index.json';before=library.read_text()
        data=DataProto.from_single_dict(next(iter(self.train_dataloader)));assert len(data)==40
        data.non_tensor_batch['uid']=np.array([str(uuid.uuid4()) for _ in range(40)],dtype=object)
        gen=self._get_gen_batch(data);gen.meta_info['global_steps']=0
        save('collection_started.json',{'rows':40,'started_unix':time.time()})
        data=data.union(self.async_rollout_manager.generate_sequences(gen));rewards,_=compute_reward(data,self.reward_fn)
        data.batch['token_level_rewards']=data.batch['token_level_scores']=rewards.float()
        expected=json.loads((BASE/'data_manifest.json').read_text())['splits']['train']
        assert [int(x) for x in data.non_tensor_batch['seed']]==[r['seed'] for r in expected]
        masks=[];rows=[]
        for i,record in enumerate(expected):
            ep=json.loads(Path(data.non_tensor_batch['pilot_episode_path'][i]).read_text());audit=ep['mechanism_audit']
            raw=audit['response_ids'];assert np.array_equal(data.batch['responses'][i,:len(raw)].numpy(),raw)
            assert np.array_equal(data.batch['response_mask'][i,:len(audit['response_mask'])].numpy(),audit['response_mask'])
            assert ep['initial_env_state']==record['state'] and abs(float(rewards[i].sum())-ep['reward_score'])<1e-5
            if record['category']=='conditional':assert ep['retrieved_skill_id']=='skill_4'
            assert not (ep.get('distillation_audit') or {}).get('model_calls',0)
            masks.append(answer_mask(audit,self.tokenizer,data.batch['responses'].shape[-1]))
            rows.append({'seed':record['seed'],'category':record['category'],'success':bool(ep['traj_success']),'reward':float(rewards[i].sum())})
        mask=np.stack(masks);full=data.batch['response_mask'].numpy().astype(bool)
        assert not (mask.astype(bool)&~full).any()
        counts={k:sum(r['success'] for r in rows if r['category']==k) for k in ['conditional','medium','single','long']}
        gate=all(n>=1 for n in counts.values()) and bool((mask.sum(-1)>0).all())
        save_batch(data,Path(os.environ['PILOT_OUT'])/'PRIVATE_BANK.pt.gz',protocol()['bank_write_stop_bytes'],protocol()['preparation_started_unix']+protocol()['max_seconds_including_preparation'])
        check=load_batch(Path(os.environ['PILOT_OUT'])/'PRIVATE_BANK.pt.gz');assert len(check)==40 and check.batch['response_mask'].sum(-1).tolist()==data.batch['response_mask'].sum(-1).tolist()
        assert all(check.batch[k].shape==v.shape and check.batch[k].dtype==v.dtype for k,v in data.batch.items())
        assert len(check.non_tensor_batch['multi_modal_inputs'])==40
        for original,restored in zip(data.non_tensor_batch['multi_modal_inputs'],check.non_tensor_batch['multi_modal_inputs'],strict=True):
            assert original.keys()==restored.keys()
            assert all(original[k].shape==restored[k].shape and original[k].dtype==restored[k].dtype for k in original)
        np.savez_compressed(Path(os.environ['PILOT_OUT'])/'bank_numeric.npz',answer_loss_mask=mask,**{k:data.batch[k].cpu().numpy() for k in ['input_ids','attention_mask','position_ids','responses','response_mask','token_level_rewards']})
        calls=int((Path(os.environ['PILOT_OUT'])/'model_calls.txt').read_text());assert calls<=200
        requests=[json.loads(f.read_text()) for f in (Path(os.environ['PILOT_OUT'])/'requests').glob('*.json')]
        assert len(requests)==calls and all(r['kind']=='rollout' for r in requests) and library.read_text()==before
        save('bank_ready.json',{'all_rows':rows,'success_counts':counts,'every_episode_nonempty_answer_mask':bool((mask.sum(-1)>0).all()),'gate_pass':gate,'resampling_allowed':False,'private_bank_bytes':(Path(os.environ['PILOT_OUT'])/'PRIVATE_BANK.pt.gz').stat().st_size,'basic_reload_passed':True})
        save('COMPLETE.json',{'completed_unix':time.time(),'episodes':40,'native_calls':calls,'model_optimizer_steps':0,'bank_gate_pass':gate})

class Eval40(EvalTrainer):
    def fit(self):
        assert not self.use_critic and not self.use_reference_policy and len(self.val_dataset)==len(self.val_dataloader)==40
        assert self.config.data.val_batch_size==self.config.actor_rollout_ref.rollout.agent.num_workers==1
        self.global_steps=0;folder=Path(os.environ['PILOT_OUT']);before=(folder/'context_libraries/cards/skill_index.json').read_text()
        save('evaluation_started.json',{'cases':40,'started_unix':time.time()});RayPPOTrainer._validate(self)
        rows=[json.loads(f.read_text()) for f in (folder/'episodes').glob('*/episode.json')]
        expected=json.loads((BASE/'data_manifest.json').read_text())['splits']['dev']
        assert len(rows)==40 and {r['seed'] for r in rows}=={r['seed'] for r in expected}
        assert before==(folder/'context_libraries/cards/skill_index.json').read_text()
        calls=int((folder/'model_calls.txt').read_text());requests=[json.loads(f.read_text()) for f in (folder/'requests').glob('*.json')]
        assert calls==len(requests)<=200 and all(r['kind']=='rollout' for r in requests)
        assert not list((folder/'update_audit').glob('actor_*.jsonl'))
        save('COMPLETE.json',{'completed_unix':time.time(),'episodes':40,'native_calls':calls,'model_optimizer_steps':0})

class LocalTask(TaskRunner):
    def add_actor_rollout_worker(self,config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(DataActor)
        return DataActor,RayWorkerGroup
    def run(self,config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer=BankTrainer if os.environ['TASK_PHASE']=='collect' else Eval40
        return super().run(config)

def training_batch(label):
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    manifest=json.loads((BASE/'data_manifest.json').read_text());indices=manifest['branch_indices'][label]
    bank=load_batch(BASE/'collect/PRIVATE_BANK.pt.gz');batch=bank.select_idxs(indices)
    with np.load(BASE/'collect/bank_numeric.npz',allow_pickle=False) as pack:answer=torch.from_numpy(pack['answer_loss_mask'][indices].copy()).bool()
    assert len(batch)==32 and bool((answer.sum(-1)>0).all())
    adv,returns=compute_gae_advantage_return(batch.batch['token_level_rewards'].float(),torch.zeros_like(batch.batch['token_level_rewards']).float(),batch.batch['response_mask'],1.,1.)
    batch.batch['advantages']=adv*answer;batch.batch['returns']=returns
    batch.meta_info.update(temperature=1.,global_token_num=batch.batch['attention_mask'].sum(-1).tolist(),multi_turn=True)
    return batch,adv,answer

def actor_group(cfg):
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    cls=RayClassWithInitArgs(ray.remote(DataActor),config=cfg.actor_rollout_ref,role='actor')
    group=RayWorkerGroup(resource_pool=RayResourcePool([2],use_gpu=True,max_colocate_count=1,name_prefix='local'),ray_cls_with_init=create_colocated_worker_cls({'actor':cls}),device_name='cuda')
    actor=group.spawn(prefix_set={'actor'})['actor'];actor.init_model();return group,actor

def saved_phase(phase,cfg):
    label=phase[-1] if phase.startswith('reload_') else phase
    batch,original_adv,answer=training_batch(label)
    kwargs=OmegaConf.to_container(cfg.ray_kwargs.ray_init,resolve=True);kwargs['runtime_env']={};ray.init(**kwargs)
    group,actor=actor_group(cfg);folder=Path(os.environ['PILOT_OUT'])
    if phase.startswith('reload_'):
        batch=batch[:2];actual=actor.compute_log_prob(batch).batch['old_log_probs'].float().cpu()
        np.savez_compressed(folder/'scores.npz',log_probs=actual.numpy(),response_mask=batch.batch['response_mask'].numpy())
        with np.load(BASE/label/'training_signals.npz',allow_pickle=False) as pack:expected=torch.from_numpy(pack['post_log_probs'][:2].copy())
        gate=score_gate(actual,expected,batch.batch['response_mask']);save('reload_gate.json',gate);assert all(r['operational_gate_pass'] for r in gate)
        save('COMPLETE.json',{'completed_unix':time.time(),'business_forward_samples':2,'extra_alignment_samples':2,'operational_reload_verified':True});return
    pre=actor.compute_log_prob(batch);pre.batch.pop('entropys');batch=batch.union(pre)
    before=batch.batch['old_log_probs'].float().cpu()
    np.savez_compressed(folder/'pre_scores.npz',log_probs=before.numpy(),response_mask=batch.batch['response_mask'].numpy())
    if label=='R':
        common=json.loads((BASE/'data_manifest.json').read_text())['common_branch_positions']
        with np.load(BASE/'Q/pre_scores.npz',allow_pickle=False) as pack:
            assert np.array_equal(pack['response_mask'][common],batch.batch['response_mask'][common].numpy())
            gate=score_gate(before[common],torch.from_numpy(pack['log_probs'][common].copy()),batch.batch['response_mask'][common])
        save('common24_pre_gate.json',gate);assert len(gate)==24 and all(r['operational_gate_pass'] for r in gate)
    assert not (folder/'update_started.json').exists();save('update_started.json',{'global_actor_steps':1,'started_unix':time.time(),'rows':32})
    output=actor.update_actor(batch)
    audit=[json.loads(line) for f in (folder/'update_audit').glob('actor_*.jsonl') for line in f.read_text().splitlines()]
    assert len(audit)==2 and {r['rank'] for r in audit}=={0,1} and all(r['finite'] and r['sample_delta_l2']>0 for r in audit)
    post=actor.compute_log_prob(batch).batch['old_log_probs'].float().cpu()
    np.savez_compressed(folder/'training_signals.npz',pre_log_probs=before.numpy(),post_log_probs=post.numpy(),advantages_before=original_adv.numpy(),advantages_after=batch.batch['advantages'].numpy(),answer_loss_mask=answer.numpy(),response_mask=batch.batch['response_mask'].numpy())
    assert bool(torch.isfinite(post[batch.batch['response_mask'].bool()]).all())
    from verl.utils.metric import reduce_metrics
    save('actor_metrics.json',reduce_metrics(output.meta_info['metrics']));actor.retain_delta()
    save('COMPLETE.json',{'completed_unix':time.time(),'business_forward_samples':96,'extra_alignment_samples':16,'global_actor_steps':1,'critic_steps':0,'private_inference_artifact_saved':True})
