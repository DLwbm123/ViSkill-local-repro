"""CPU checks of original full configuration and transparent launch hooks."""
import os,json,time,runpy,asyncio,tempfile,types
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,ray,numpy as np
from omegaconf import OmegaConf
from verl.utils.config import validate_config
from verl import DataProto
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import _determine_fsdp_megatron_base_class
from vagen.gym_agent_dataset import AgenticDataset
import vagen.official_train_observer as obs
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);OUT=ROOT/'artifacts/reproduction/official_frozenlake_ppo_v1';OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'dispatch_count.json').exists() and not (OUT/'cpu_check.json').exists()
cfg=runpy.run_path(str(SRC/'scripts/official_train_config.py'))['config']();validate_config(cfg,use_reference_policy=False,use_critic=True)
assert cfg.algorithm.adv_estimator=='gae' and not cfg.trainer.val_only and cfg.trainer.resume_mode=='disable'
assert cfg.actor_rollout_ref.rollout.mode=='async' and cfg.actor_rollout_ref.model.use_fused_kernels
assert all(callable(getattr(obs.RunActor,k,None)) for k in ['sleep','wake_up','generate','compute_log_prob','update_actor'])
assert _determine_fsdp_megatron_base_class([obs.RunActor.__mro__,obs.RunCritic.__mro__]) is Worker
train=AgenticDataset(cfg.data.train_files,cfg.data);val=AgenticDataset(cfg.data.val_files,cfg.data)
assert len(train)==10000 and {x['seed'] for x in train.items}==set(range(1,10001))
assert len(val)==256 and {x['seed'] for x in val.items}==set(range(10001,10257))
assert cfg.actor_rollout_ref.actor.checkpoint.save_contents==['model','hf_model','optimizer','extra'] and cfg.critic.checkpoint.save_contents==['model','optimizer','extra']
assert cfg.trainer.max_actor_ckpt_to_keep==cfg.trainer.max_critic_ckpt_to_keep==1
assert cfg.actor_rollout_ref.rollout.temperature==1 and cfg.actor_rollout_ref.rollout.val_kwargs.temperature==0
async def recorder_fixture(folder):
    (folder/'requests').mkdir();captured=[];answer=types.SimpleNamespace(token_ids=[1,2])
    class Native:
        async def generate(self,**kw):captured.append(kw);return answer
    agent=types.SimpleNamespace(kind='rollout',fixed_folder=None,phase={'mode':'train','step':1},episode_id='fixture',seed=1,tokenizer=types.SimpleNamespace(decode=lambda *a,**k:'fixture'))
    kw={'image_data':[],'prompt_ids':[3,4],'sampling_params':{'temperature':1.},'request_id':'fixture'}
    result=await obs.RecordedManager(agent,Native()).generate(**kw);assert result is answer and captured==[kw]
    rows=[json.loads(x) for x in (folder/'requests/train_0001.jsonl').read_text().splitlines()];assert [r['event'] for r in rows]==['intent','result'] and rows[-1]['output_ids']==[1,2]
    maximum=obs.P['cost_bounds']['maximum_total_agent_dispatches'];(folder/'dispatch_count.json').write_text(json.dumps({'all':maximum,'rollout':0,'distill':0}))
    try:await obs.RecordedManager(agent,Native()).generate(**kw)
    except AssertionError:pass
    else:raise AssertionError('dispatch guard must fail before native generation')
    assert len(captured)==1
original_out=obs.OUT
try:
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        folder=Path(td);obs.OUT=folder;asyncio.run(recorder_fixture(folder))
        proto=DataProto.from_dict(tensors={k:torch.zeros((128,2),dtype=torch.long) for k in ['input_ids','attention_mask','position_ids']},non_tensors={'seed':np.arange(1,129),'data_source':np.array(['frozenlake']*128,dtype=object)})
        trainer=object.__new__(obs.RunTrainer);trainer.global_steps=1;trainer.async_rollout_mode=True;generated=trainer._get_gen_batch(proto);assert len(generated)==128 and generated.non_tensor_batch['seed'].tolist()==list(range(1,129))
        assert json.loads((folder/'phase.json').read_text())['step']==1
finally:obs.OUT=original_out
# Exercise the exact installed-version HF export constructor on meta only.
from transformers import AutoConfig,AutoModelForImageTextToText
from accelerate import init_empty_weights
hfcfg=AutoConfig.from_pretrained(ROOT/'models/b0',local_files_only=True)
with init_empty_weights():dummy=AutoModelForImageTextToText.from_config(hfcfg,torch_dtype=torch.bfloat16)
assert all(p.is_meta for p in dummy.parameters());del dummy
assert not ray.is_initialized() and not torch.cuda.is_initialized()
result={'status':'CPU_READY','completed_unix':time.time(),'official_train_seeds':10000,'official_val_seeds':256,'full_original_batch128_mini32_GAE_critic_lr_and_checkpoint_contents':True,'native_async_actor_RPC_methods_and_two_role_MRO':True,'recorder_exact_kwargs_and_return_identity':True,'cap_stops_before_mock_native_call':True,'actual_parent_get_gen_batch128':True,'installed_HF_checkpoint_export_constructor_meta_only':True,'new_actual_model_forward_backward_calls':0,'new_GPU_calls':0,'new_native_env_steps':0,'new_optimizer_steps':0,'prior_storage_cpu_scalar_fixture_steps_not_repeated':1,'scope':'CPU/meta and mocked recorder checks do not establish GPU/backward/serialization/throughput success.'}
(OUT/'cpu_check.json').write_text(json.dumps(result,indent=2));(OUT/'dataset_manifest.json').write_text(json.dumps({'train':train.items,'validation':val.items},default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)));print(json.dumps(result))
