"""Prospective, serialized evaluation of three already fixed policies."""
import os,json,time
from pathlib import Path
import numpy as np
import ray
from vagen.value_head_single_step import SingleActor
from vagen.main_ppo_skill import TaskRunner
from vagen.ray_trainer import RayPPOTrainer

def validate_loading(info,tied):
    assert not info.get('unexpected_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),info
    missing=info.get('missing_keys',[])
    assert all(k=='lm_head.weight' for k in missing) and tied,info
    return {'missing_keys':missing,'unexpected_keys':[],'mismatched_keys':[],'error_msgs':[],'actual_embedding_parameter_tied':True}

class EvalActor(SingleActor):
    def _build_model_optimizer(self,*args,**kwargs):
        from transformers import AutoConfig,AutoModelForVision2Seq,AutoModelForCausalLM,AutoModelForImageTextToText
        config=AutoConfig.from_pretrained(kwargs['model_path'],trust_remote_code=True)
        selected=next(cls for cls in [AutoModelForVision2Seq,AutoModelForCausalLM,AutoModelForImageTextToText] if type(config) in cls._model_mapping.keys())
        assert selected.__name__=='AutoModelForVision2Seq','unexpected actual auto-model selection'
        descriptor=selected.__dict__.get('from_pretrained');inherited=descriptor is None;original=selected.from_pretrained
        called=[]
        def capture(*a,**kw):
            assert not kw.get('output_loading_info',False)
            model,info=original(*a,**kw,output_loading_info=True)
            tied=model.get_input_embeddings().weight is model.get_output_embeddings().weight
            check=validate_loading(info,tied);called.append(check)
            if self.rank==0:
                check.update(auto_class=selected.__name__,policy=os.environ['TASK_PHASE'],scope='Actual model initialization; no extra load, hash or full-state equality claim.')
                (Path(os.environ['PILOT_OUT'])/'actual_loading_info.json').write_text(json.dumps(check,indent=2))
            return model
        selected.from_pretrained=staticmethod(capture)
        try:
            result=super()._build_model_optimizer(*args,**kwargs)
            assert len(called)==1,'actual loader was not captured exactly once'
            return result
        finally:
            if inherited:delattr(selected,'from_pretrained')
            else:setattr(selected,'from_pretrained',descriptor)

class EvalTrainer(RayPPOTrainer):
    def _get_gen_batch(self,batch):
        result=super()._get_gen_batch(batch)
        assert set(result.non_tensor_batch['pilot_phase'])=={'dev'}
        result.non_tensor_batch['pilot_policy_version']=np.full(len(result),self.global_steps,dtype=object)
        result.non_tensor_batch['fixed_context']=np.full(len(result),'cards',dtype=object)
        return result
    def fit(self):
        assert not self.use_critic and not self.use_reference_policy
        assert self.config.data.val_batch_size==self.config.actor_rollout_ref.rollout.agent.num_workers==1
        assert len(self.val_dataset)==32 and len(self.val_dataloader)==32
        self.global_steps=0;folder=Path(os.environ['PILOT_OUT'])
        before=(folder/'context_libraries/cards/skill_index.json').read_text()
        assert not (folder/'evaluation_started.json').exists(),'never replay a dispatched policy evaluation'
        (folder/'evaluation_started.json').write_text(json.dumps({'started_unix':time.time(),'cases':32,'serial_batch_size':1}))
        RayPPOTrainer._validate(self)
        rows=[json.loads(p.read_text()) for p in (folder/'episodes').glob('*/episode.json')]
        manifest=json.loads(Path(os.environ['PILOT_DATA_MANIFEST']).read_text())
        assert len(rows)==32 and {r['seed'] for r in rows}=={r['seed'] for r in manifest['splits']['dev']}
        assert all(r['phase']=='dev' and r['policy_version']==0 for r in rows)
        assert before==(folder/'context_libraries/cards/skill_index.json').read_text()
        assert not list((folder/'update_audit').glob('*.jsonl'))
        assert all(not (r.get('distillation_audit') or {}).get('model_calls',0) for r in rows)
        requests=[json.loads(p.read_text()) for p in (folder/'requests').glob('*.json')]
        assert all(r['kind']=='rollout' for r in requests)
        assert len(requests)==int((folder/'model_calls.txt').read_text())<=160
        (folder/'COMPLETE.json').write_text(json.dumps({'completed_unix':time.time(),'policy':os.environ['TASK_PHASE'],'episodes':32,'optimizer_steps':0,'distill_calls':0,'new_snapshots':0,'actual_HF_loading_info_checked':True},indent=2))

class EvalTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self,config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(EvalActor)
        return EvalActor,RayWorkerGroup
    def run(self,config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer=EvalTrainer
        return super().run(config)
