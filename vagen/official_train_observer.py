"""Observe native released PPO without changing its actions, rewards or updates."""
import os,json,time,math,uuid,fcntl,shutil,subprocess
from pathlib import Path
import numpy as np
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop
from vagen.ray_trainer_skill import RayPPOTrainerSkill
from vagen.main_ppo_skill import TaskRunner
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker,CriticWorker
from verl.utils.tracking import Tracking as NativeTracking
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='official_frozenlake_ppo_v1';OUT=ROOT/f'artifacts/reproduction/{NAME}';P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text());ACTIVE=None

def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    return v

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(plain(value)));temp.replace(path)

def append(path,value):
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(plain(value))+'\n')

def usage(path):return int(subprocess.check_output(['du','-s','-B1','.'],cwd=path,text=True).split()[0])
def total_usage():return sum(usage(x) for x in [ROOT,Path('/data/envs/z0'),Path('/data/tools/z0'),Path('/data/tmp/z0')])
def stop_requested():return (OUT/'STOP_REQUESTED.json').exists() or time.time()>=P['cooperative_stop_unix']
class ControlledStop(Exception):pass

class RecordedManager:
    def __init__(self,agent,original):self.agent,self.original=agent,original
    async def generate(self,**kwargs):
        a=self.agent
        with (OUT/'dispatch_count.json').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);counts=json.loads(f.read() or '{"all":0,"rollout":0,"distill":0}');counts['all']+=1;counts[a.kind]+=1
            assert counts['all']<=P['cost_bounds']['maximum_total_agent_dispatches']
            assert counts['rollout']<=P['cost_bounds']['maximum_rollout_dispatches'] and counts['distill']<=P['cost_bounds']['maximum_post_episode_self_distill_dispatches']
            f.seek(0);f.truncate();f.write(json.dumps(counts));f.flush()
        number=counts['all'];images=kwargs.get('image_data') or [];saved=[]
        if a.fixed_folder is not None:
            for i,img in enumerate(images):
                path=a.fixed_folder/f'{number:06d}_{i}.png';img.save(path);saved.append(str(path.relative_to(OUT)))
        r={'event':'intent','number':number,'mode':a.phase['mode'],'step':a.phase['step'],'episode_id':a.episode_id,'seed':a.seed,'kind':a.kind,'sampling':kwargs['sampling_params'],'request_id':kwargs['request_id'],'prompt_ids':kwargs['prompt_ids'],'image_count':len(images),'fixed_input_images':saved,'dispatched_unix':time.time()}
        path=OUT/'requests'/f"{a.phase['mode']}_{a.phase['step']:04d}.jsonl";append(path,r)
        try:output=await self.original.generate(**kwargs)
        except BaseException as exc:
            append(path,dict(r,event='error',error=repr(exc),returned_unix=time.time()));raise
        r.update(event='result',output_ids=output.token_ids,raw=a.tokenizer.decode(output.token_ids,skip_special_tokens=True),returned_unix=time.time());append(path,r);return output

class RunLoop(GymAgentLoop):
    async def run(self,sampling_params,**kwargs):
        self.phase=json.loads((OUT/'phase.json').read_text());self.seed=int(kwargs['seed']);self.kind='rollout';self.episode_id=uuid.uuid4().hex;self.fixed_folder=None
        assert bool(kwargs.get('validate',False))==(self.phase['mode']=='val')
        assert (10001<=self.seed<=10256) if self.phase['mode']=='val' else (1<=self.seed<=10000)
        if (self.phase['mode'],self.seed) in [('train',1),('val',10001)]:
            folder=OUT/'fixed_inputs'/f"{self.phase['mode']}_{self.seed}"
            try:folder.mkdir();self.fixed_folder=folder
            except FileExistsError:pass
        original=self.server_manager;self.server_manager=RecordedManager(self,original)
        try:output=await super().run(sampling_params,**kwargs)
        finally:self.server_manager=original
        extra=output.extra_fields;r={k:extra.get(k) for k in ['reward_extra_info','traj_success','retrieved_skill_id','skill_score','env_name','initial_env_state','trajectory_actions','trajectory_positions','distilled_strategy']};r.update(mode=self.phase['mode'],step=self.phase['step'],seed=self.seed,episode_id=self.episode_id,num_turns=output.num_turns,response_ids=output.response_ids,response_mask=output.response_mask,completed_unix=time.time());append(OUT/'episodes'/f"{self.phase['mode']}_{self.phase['step']:04d}.jsonl",r);return output
    async def _self_distill(self,*args,**kwargs):
        self.kind='distill'
        try:return await super()._self_distill(*args,**kwargs)
        finally:self.kind='rollout'

def loading(worker,method,args,kwargs,role):
    from transformers import AutoModelForVision2Seq
    cls=AutoModelForVision2Seq;descriptor=cls.__dict__.get('from_pretrained');original=cls.from_pretrained;calls=[]
    def load(*a,**kw):
        kw['output_loading_info']=True;model,info=original(*a,**kw)
        assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
        info.update(actual_embedding_parameter_tied=model.get_input_embeddings().weight is model.get_output_embeddings().weight,actual_first_parameter_dtype=str(next(model.parameters()).dtype),role=role);calls.append(True)
        if worker.rank==0:write(OUT/f'{role}_actual_loading_info.json',info)
        return model
    cls.from_pretrained=staticmethod(load)
    try:result=method(*args,**kwargs);assert len(calls)==1;return result
    finally:
        if descriptor is None:delattr(cls,'from_pretrained')
        else:setattr(cls,'from_pretrained',descriptor)
class RunActor(AsyncActorRolloutRefWorker):
    def _build_model_optimizer(self,*args,**kwargs):return loading(self,super()._build_model_optimizer,args,kwargs,'actor')
class RunCritic(CriticWorker):
    def _build_critic_model_optimizer(self,*args,**kwargs):return loading(self,super()._build_critic_model_optimizer,args,kwargs,'critic')

class RunTracking(NativeTracking):
    def log(self,data,step,*args,**kwargs):
        write(OUT/'metrics'/f'step_{step:04d}.json',{'step':int(step),'recorded_unix':time.time(),'metrics':data})
        result=super().log(data,step,*args,**kwargs)
        if step==1:ACTIVE._save_checkpoint()
        bad=[k for k,v in data.items() if any(word in k for word in ['grad_norm','pg_loss','vf_loss']) and isinstance(v,(float,np.floating)) and not math.isfinite(v)]
        if bad:write(OUT/'STOP_REQUESTED.json',{'reason':'nonfinite training metrics','keys':bad,'at_step':int(step)})
        if stop_requested() and step<301:raise ControlledStop('cooperative resource/finite stop after complete native batch')
        return result

class RunTrainer(RayPPOTrainerSkill):
    def _get_gen_batch(self,batch):
        if not getattr(self,'in_validation',False):
            assert len(batch)==128;write(OUT/'phase.json',{'mode':'train','step':int(self.global_steps),'started_unix':time.time()})
        return super()._get_gen_batch(batch)
    def _validate(self):
        self.in_validation=True;step=int(self.global_steps);write(OUT/'phase.json',{'mode':'val','step':step,'started_unix':time.time()})
        try:metrics=super()._validate()
        finally:self.in_validation=False
        rows=[json.loads(x) for x in (OUT/'episodes'/f'val_{step:04d}.jsonl').read_text().splitlines()];assert len(rows)==256 and {r['seed'] for r in rows}==set(range(10001,10257))
        write(OUT/'validation_metrics'/f'step_{step:04d}.json',{'step':step,'completed_unix':time.time(),'episodes':256,'successes':sum(bool(r['traj_success']) for r in rows),'metrics':metrics});return metrics
    def _save_checkpoint(self):
        if getattr(self,'last_saved_step',None)==self.global_steps:return
        current=total_usage();checkpoint_bytes=usage(OUT/'checkpoints') if (OUT/'checkpoints').exists() else 0;non_checkpoint_new=max(0,current-P['storage_baseline_four_root_bytes']-checkpoint_bytes);forecast=P['storage_baseline_four_root_bytes']+non_checkpoint_new+P['pre_checkpoint_expected_complete_payload_reserve_bytes'];limit=min(P['four_root_stop_bytes'],P['storage_baseline_four_root_bytes']+P['additional_write_stop_bytes'])
        assert forecast<limit and time.time()<P['deadline_unix']-120,'Insufficient complete-checkpoint reserve or clock'
        write(OUT/'checkpoint_intent.json',{'step':int(self.global_steps),'started_unix':time.time(),'current_four_root_bytes':current,'conservative_forecast_bytes':forecast,'write_stop_bytes':limit})
        saving_started=time.time();result=super()._save_checkpoint();folder=OUT/'checkpoints'/f'global_step_{self.global_steps}';source=OUT/'library'
        if source.exists():shutil.copytree(source,folder/'online_library',dirs_exist_ok=False)
        paths=[{'path':str(p.relative_to(folder)),'logical_bytes':p.stat().st_size} for p in folder.rglob('*') if p.is_file()]
        for role in ['actor','critic']:
            for kind in ['model','optim','extra_state']:assert len(list((folder/role).glob(kind+'_world_size_2_rank_*.pt')))==2,(role,kind)
        assert (folder/'data.pt').is_file() and (folder/'actor/huggingface/config.json').is_file()
        write(OUT/'checkpoint_receipts'/f'step_{self.global_steps:04d}.json',{'step':int(self.global_steps),'completed_unix':time.time(),'native_save_returned':True,'save_elapsed_seconds':time.time()-saving_started,'allocated_checkpoint_bytes':usage(folder),'four_root_bytes_after_save':total_usage(),'files':paths,'full_restore_not_executed':True,'online_library_snapshot':source.exists(),'single_generation_native_rotation':True});self.last_saved_step=self.global_steps;return result
    def fit(self):
        global ACTIVE
        ACTIVE=self;assert self.use_critic and not self.use_reference_policy and not self.config.trainer.val_only
        write(OUT/'training_started.json',{'started_unix':time.time(),'steps_target':301,'initial_library_empty':not (OUT/'library/skill_index.json').exists()})
        try:
            super().fit();assert (OUT/'metrics/step_0301.json').exists() and (OUT/'validation_metrics/step_0301.json').exists() and self.last_saved_step==301
            write(OUT/'WORKER_COMPLETED.json',{'status':'COMPLETE_NATIVE301','completed_unix':time.time(),'native_global_batches':301,'final_checkpoint_step':301,'selection':'final step301, not validation-best'})
        except ControlledStop as exc:
            self._save_checkpoint();write(OUT/'WORKER_COMPLETED.json',{'status':'CLOSED_AT_COOPERATIVE_BOUNDARY','completed_unix':time.time(),'last_native_global_batch':int(self.global_steps),'reason':str(exc),'checkpoint_step':self.last_saved_step})
        except BaseException as exc:
            write(OUT/'FIT_FAILURE.json',{'failed_unix':time.time(),'global_step':int(getattr(self,'global_steps',0)),'error':repr(exc),'last_saved_step':getattr(self,'last_saved_step',None)});raise
class RunTask(TaskRunner):
    def add_actor_rollout_worker(self,config):
        import ray
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(RunActor);return RunActor,RayWorkerGroup
    def add_critic_worker(self,config):
        import ray
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.Critic]=ray.remote(RunCritic)
    def run(self,config):
        import torch
        import verl.utils.tracking as tracking
        import vagen.main_ppo_skill as entry
        torch.manual_seed(42);tracking.Tracking=RunTracking;entry.RayPPOTrainer=RunTrainer;return super().run(config)
