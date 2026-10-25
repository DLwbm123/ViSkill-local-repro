"""Observability around the released loop; no retrieval, prompt, reward or action edits."""
import os,json,time,fcntl,uuid
from pathlib import Path
import numpy as np
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop
from vagen.ray_trainer_skill import RayPPOTrainerSkill
from vagen.main_ppo_skill import TaskRunner
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
ROOT=Path(os.environ['TASK_ROOT']);OUT=ROOT/'artifacts/reproduction/released_frozenlake_eval_v1'
def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value
class RecordedManager:
    def __init__(self,agent,original):self.agent,self.original=agent,original
    async def generate(self,**kwargs):
        agent=self.agent;folder=agent.observed_out;counter=folder/'dispatch_count.txt'
        with counter.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);number=int(f.read() or '0')+1;assert number<=1536
            f.seek(0);f.truncate();f.write(str(number));f.flush()
        name=f'{number:04d}';images=kwargs.get('image_data') or [];paths=[]
        for i,img in enumerate(images):
            path=folder/'request_images'/f'{name}_{i}.png';img.save(path);paths.append(str(path.relative_to(folder)))
        intent={'number':number,'seed':agent.observed_seed,'kind':agent.observed_kind,'request_id':kwargs['request_id'],'sampling':kwargs['sampling_params'],'prompt_ids':kwargs['prompt_ids'],'images':paths,'dispatched_unix':time.time()}
        (folder/'requests'/f'{name}.intent.json').write_text(json.dumps(plain(intent)))
        output=await self.original.generate(**kwargs)
        intent.update(output_ids=output.token_ids,raw=agent.tokenizer.decode(output.token_ids,skip_special_tokens=True),returned_unix=time.time())
        (folder/'requests'/f'{name}.json').write_text(json.dumps(plain(intent)))
        return output
class ObservedGymAgentLoop(GymAgentLoop):
    async def run(self,sampling_params,**kwargs):
        assert kwargs.get('validate') and 10001<=int(kwargs['seed'])<=10256
        self.observed_out=OUT;self.observed_seed=int(kwargs['seed']);self.observed_kind='rollout'
        original=self.server_manager;self.server_manager=RecordedManager(self,original)
        try:output=await super().run(sampling_params,**kwargs)
        finally:self.server_manager=original
        extra=output.extra_fields;record={k:extra.get(k) for k in ['reward_extra_info','traj_success','retrieved_skill_id','skill_score','env_name','initial_env_state','trajectory_actions','trajectory_positions','distilled_strategy']}
        record.update(seed=self.observed_seed,num_turns=output.num_turns,response_ids=output.response_ids,response_mask=output.response_mask,completed_unix=time.time())
        (OUT/'episodes'/f'{self.observed_seed}.json').write_text(json.dumps(plain(record)))
        return output
    async def _self_distill(self,*args,**kwargs):
        self.observed_kind='distill'
        try:return await super()._self_distill(*args,**kwargs)
        finally:self.observed_kind='rollout'
class ObserverActor(AsyncActorRolloutRefWorker):
    def _build_model_optimizer(self,*args,**kwargs):
        from transformers import AutoModelForVision2Seq
        cls=AutoModelForVision2Seq;descriptor=cls.__dict__.get('from_pretrained');original=cls.from_pretrained;calls=[]
        def load(*a,**kw):
            model,info=original(*a,**kw,output_loading_info=True)
            assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
            info['actual_embedding_parameter_tied']=model.get_input_embeddings().weight is model.get_output_embeddings().weight;calls.append(True)
            if self.rank==0:(OUT/'actual_loading_info.json').write_text(json.dumps(info,indent=2))
            return model
        cls.from_pretrained=staticmethod(load)
        try:
            result=super()._build_model_optimizer(*args,**kwargs);assert len(calls)==1;return result
        finally:
            if descriptor is None:delattr(cls,'from_pretrained')
            else:setattr(cls,'from_pretrained',descriptor)
class ObserverTrainer(RayPPOTrainerSkill):
    def _validate(self):
        assert len(self.val_dataset)==256 and len(self.val_dataloader)==1 and self.config.data.val_batch_size is None
        (OUT/'evaluation_started.json').write_text(json.dumps({'started_unix':time.time(),'episodes':256,'val_batch_size':256,'official_seed_set':list(range(10001,10257))}))
        metrics=super()._validate();(OUT/'metrics.json').write_text(json.dumps(plain(metrics),indent=2));return metrics
    def fit(self):
        assert self.config.trainer.val_only and not self.use_critic and not self.use_reference_policy
        library=OUT/'library/skill_index.json';before=json.loads(library.read_text());super().fit();assert json.loads(library.read_text())==before
        rows=[json.loads(p.read_text()) for p in (OUT/'episodes').glob('*.json')];assert len(rows)==256 and {r['seed'] for r in rows}==set(range(10001,10257))
        requests=[json.loads(p.read_text()) for p in (OUT/'requests').glob('*.json') if '.intent.' not in p.name];assert len(requests)==int((OUT/'dispatch_count.txt').read_text())<=1536
        assert sum(r['kind']=='rollout' for r in requests)<=1280 and sum(r['kind']=='distill' for r in requests)<=256
        assert not (OUT/'FORBIDDEN_CHECKPOINTS').exists()
        result={'completed_unix':time.time(),'episodes':256,'successes':sum(bool(r['traj_success']) for r in rows),'success_rate':sum(bool(r['traj_success']) for r in rows)/256,'rollout_dispatches':sum(r['kind']=='rollout' for r in requests),'post_episode_distill_dispatches':sum(r['kind']=='distill' for r in requests),'optimizer_steps':0,'admissions':0,'library_json_values_unchanged':True,'paper_reference_success':.84,'release_cold_start_variant_unspecified_by_model_card':True,'scope':'Official released model/library benchmark verification, not from-scratch training reproduction or statistical equivalence proof.'}
        (OUT/'COMPLETE.json').write_text(json.dumps(result,indent=2))
class ObserverTask(TaskRunner):
    def add_actor_rollout_worker(self,config):
        import ray
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(ObserverActor)
        return ObserverActor,RayWorkerGroup
    def run(self,config):
        import torch
        import vagen.main_ppo_skill as entry
        torch.manual_seed(42);entry.RayPPOTrainer=ObserverTrainer
        return super().run(config)
