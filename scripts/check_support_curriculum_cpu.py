"""New40-row interfaces and exact unchanged answer-mask operator; no Ray startup."""
import os,json,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np,ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from verl.utils.config import validate_config
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import _determine_fsdp_megatron_base_class
from verl.trainer.ppo.core_algos import compute_gae_advantage_return
from vagen.support_curriculum import ROOT,BASE,protocol,make_config,answer_mask,BankTrainer,Eval40,DataActor
from vagen.bootstrap_pilot import PilotDataset
from verl.utils.dataset.rl_dataset import collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader

p=protocol();manifest=json.loads((BASE/'data_manifest.json').read_text());cfgs={phase:make_config(phase,BASE/phase) for phase in p['phase_order']}
for phase,cfg in cfgs.items():
    validate_config(cfg,use_reference_policy=False,use_critic=False)
    assert cfg.actor_rollout_ref.model.use_fused_kernels and cfg.actor_rollout_ref.actor.entropy_coeff==0
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size==(40 if phase=='collect' else 32)
    assert cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu==1
    assert not cfg.actor_rollout_ref.actor.use_dynamic_bsz and not cfg.actor_rollout_ref.actor.use_kl_loss
    assert cfg.ray_kwargs.ray_init.runtime_env=={}
collect=cfgs['collect'];data=PilotDataset(str(ROOT/'src/configs/smoke/support_curriculum_train.yaml'),collect.data)
loader=StatefulDataLoader(data,batch_size=40,shuffle=False,num_workers=0,collate_fn=collate_fn)
actual=next(iter(loader));assert [int(v) for v in actual['seed']]==[r['seed'] for r in manifest['splits']['train']]
fake=BankTrainer.__new__(BankTrainer);fake.global_steps=0;fake.async_rollout_mode=True
gen=fake._get_gen_batch(DataProto.from_single_dict(actual));assert len(gen)==40 and set(gen.non_tensor_batch['fixed_context'])=={'cards'}
val=PilotDataset(str(ROOT/'src/configs/smoke/support_curriculum_dev.yaml'),collect.data)
valloader=StatefulDataLoader(val,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_fn)
assert [int(v['seed'][0]) for v in valloader]==[r['seed'] for r in manifest['splits']['dev']]
fake=Eval40.__new__(Eval40);fake.global_steps=0;fake.async_rollout_mode=True
gen=fake._get_gen_batch(DataProto.from_single_dict(collate_fn([val[0]])));assert len(gen)==1 and set(gen.non_tensor_batch['pilot_phase'])=={'dev'}
assert _determine_fsdp_megatron_base_class([DataActor.__mro__]) is Worker
toy=DataProto.from_dict(tensors={'input_ids':torch.arange(160).reshape(40,4),'response_mask':torch.ones(40,4),'token_level_rewards':torch.zeros(40,4)},non_tensors={'seed':np.array([r['seed'] for r in manifest['splits']['train']])})
chosen={label:toy.select_idxs(indices) for label,indices in manifest['branch_indices'].items()};common=manifest['common_branch_positions']
assert len(chosen['Q'])==len(chosen['R'])==32 and len(common)==24
assert torch.equal(chosen['Q'].batch['input_ids'][common],chosen['R'].batch['input_ids'][common])
for b in chosen.values():
    b.batch['token_level_rewards'][0,-1]=1.;adv,ret=compute_gae_advantage_return(b.batch['token_level_rewards'],torch.zeros_like(b.batch['token_level_rewards']),b.batch['response_mask'],1.,1.)
    assert torch.isfinite(adv).all() and torch.equal(ret,b.batch['token_level_rewards'].sum(-1,keepdim=True).expand_as(ret))
old=ROOT/'artifacts/component_tests/value_head_single_step_v1';tok=AutoTokenizer.from_pretrained(ROOT/'models/b0',trust_remote_code=True,local_files_only=True)
prior={json.loads(f.read_text())['seed']:json.loads(f.read_text()) for f in (old/'C/episodes').glob('*/episode.json')}
original=json.loads((old/'data_manifest.json').read_text())['splits']['train']
with np.load(ROOT/'artifacts/component_tests/actor_loss_support_audit_v1/loss_support.npz',allow_pickle=False) as pack:
    expected=pack['answer_loss_mask'];result=np.stack([answer_mask(prior[r['seed']]['mechanism_audit'],tok,expected.shape[1]) for r in original]);assert np.array_equal(result,expected) and int(result.sum())==233
assert not ray.is_initialized()
result={'status':'CPU_READY','completed_unix':time.time(),'bank_rows':40,'dev_rows':40,'branch_rows':32,'common24_same_positions':True,'actual_collect_and_eval_get_gen_batch_passed':True,'unchanged_old_answer_mask_exact32_rows_233_tokens':True,'GAE_zero_baseline_toy_and_subset_mapping_passed':True,'all_actual_phase_configs_valid':True,'MRO_base_correct':True,'model_forwards':0,'model_optimizer_steps':0,'environment_steps':0}
(BASE/'cpu_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
