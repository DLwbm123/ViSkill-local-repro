"""Freeze the actual next sampler batch and test new input/update audit hooks on CPU."""
import os,json,runpy,tempfile,types,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np,ray
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.config import validate_config
from verl.single_controller.ray.base import _determine_fsdp_megatron_base_class
from verl.single_controller.base.worker import Worker
from verl import DataProto
from vagen.gym_agent_dataset import AgenticDataset
from vagen.main_ppo_skill import create_rl_sampler
import vagen.retrieved_update_check as check
ROOT,SRC,OUT=check.ROOT,check.SRC,check.OUT;OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'cpu_check.json').exists()
cfg=runpy.run_path(str(SRC/'scripts/run_retrieved_update_check.py'))['config']();validate_config(cfg,use_reference_policy=False,use_critic=True)
assert cfg.data.train_batch_size==128 and cfg.actor_rollout_ref.actor.ppo_mini_batch_size==32 and cfg.algorithm.adv_estimator=='gae'
assert cfg.trainer.resume_mode=='resume_path' and not cfg.trainer.val_before_train and not cfg.trainer.del_local_ckpt_after_load
assert _determine_fsdp_megatron_base_class([check.CheckActor.__mro__,check.CheckCritic.__mro__]) is Worker
train=AgenticDataset(cfg.data.train_files,cfg.data);sampler=create_rl_sampler(cfg.data,train)
loader=StatefulDataLoader(dataset=train,batch_size=cfg.data.get('gen_batch_size',cfg.data.train_batch_size),num_workers=cfg.data.dataloader_num_workers,drop_last=True,collate_fn=collate_fn,sampler=sampler)
state=torch.load(Path(cfg.trainer.resume_from_path)/'data.pt',weights_only=False);loader.load_state_dict(state);iterator=iter(loader);batch=next(iterator);next_seeds=[int(x) for x in batch['seed']];del iterator,loader
previous=[json.loads(x)['seed'] for x in (ROOT/'artifacts/reproduction/official_frozenlake_correctness_v1/episodes/train_0001.jsonl').read_text().splitlines()]
assert len(next_seeds)==len(set(next_seeds))==128 and not set(next_seeds)&set(previous)
manifest={'source_checkpoint_global_step':1,'next_global_step':2,'next_seeds':next_seeds,'previous128_seeds':previous,'disjoint_previous_batch':True,'selection':'Actual original StatefulDataLoader restored from read-only data.pt; no model queries or outcome-based selection'}
(OUT/'next_batch_manifest.json').write_text(json.dumps(manifest,indent=2))
original_out=check.OUT;original_base=check.base.OUT
try:
 with tempfile.TemporaryDirectory(dir=OUT) as td:
  folder=Path(td);check.OUT=check.base.OUT=folder;(folder/'episodes').mkdir()
  seeds=np.arange(1,65);ids=np.array(['skill_0' if i%2==0 else None for i in range(64)],dtype=object);episodes={int(s):{'seed':int(s),'retrieved_skill_id':ids[i],'response_ids':[7,8],'response_mask':[1,1]} for i,s in enumerate(seeds)}
  (folder/'episodes/train_0002.jsonl').write_text('\n'.join(json.dumps(x) for x in episodes.values()))
  mm=np.empty(64,dtype=object)
  for i in range(64):
   n=2 if ids[i] else 1;mm[i]={'image_grid_thw':torch.tensor([[1,2,2]]*n),'pixel_values':torch.zeros((4*n,1176))}
  tensors={k:torch.ones((64,4)) for k in ['old_log_probs','advantages','returns']};tensors.update(responses=torch.tensor([[7,8,0,0]]*64),response_mask=torch.tensor([[1,1,0,0]]*64))
  data=DataProto.from_dict(tensors=tensors,non_tensors={'seed':seeds,'retrieved_skill_id':ids,'multi_modal_inputs':mm,'uid':np.array([str(x) for x in seeds],dtype=object)},meta_info={'multi_turn':True,'temperature':1.})
  rows=check.validate_packet(data,episodes);assert sum(bool(x['retrieved_skill_id']) for x in rows)==32
  model=torch.nn.Linear(2,2,bias=False);optim=torch.optim.AdamW(model.parameters(),lr=1e-3);worker=types.SimpleNamespace(actor_module_fsdp=model,actor_optimizer=optim,rank=0);answer=types.SimpleNamespace(meta_info={'metrics':{'actor/pg_loss':torch.tensor(.1),'actor/grad_norm':[torch.tensor(1.)]}})
  def native(value):
   assert value is data
   for _ in range(4):
    for p in model.parameters():p.grad=torch.ones_like(p)
    optim.step()
   return answer
  assert check.audit_update(worker,'actor',data,native) is answer
  assert json.loads((folder/'actor_rank0_audit.json').read_text())['actual_optimizer_step_calls']==4
  data.batch['responses'][0,0]=99
  try:check.validate_packet(data,episodes)
  except AssertionError:pass
  else:raise AssertionError('Mismatched generated token must be rejected')
finally:check.OUT=original_out;check.base.OUT=original_base
assert not ray.is_initialized() and not torch.cuda.is_initialized()
result={'status':'CPU_READY','completed_unix':time.time(),'actual_next_original128_sampler_seeds_frozen':True,'source_checkpoint_not_modified':True,'disjoint_first_training_batch':True,'native_two_role_async_MRO':True,'packet_token_mask_grid_checks_and_mismatch_rejection':True,'optimizer_post_hook_counts_exactly4_and_native_result_identity':True,'tiny_CPU_optimizer_fixture_calls':4,'actual_model_forward_backward_calls':0,'actual_GPU_calls':0,'native_env_steps':0,'scope':'Fixture uses a2x2 CPU linear parameter with assigned gradients, not the real model. No GPU update correctness claim.'}
(OUT/'cpu_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
