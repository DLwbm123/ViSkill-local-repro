"""Check the intervention and stream-retention path before any model call."""
import os,json,time,tempfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np,ray
from types import SimpleNamespace
from omegaconf import OmegaConf
from safetensors.torch import save_file
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import _determine_fsdp_megatron_base_class
from vagen.answer_region_step import ROOT,BASE,protocol,make_config,training_batch,LocalActor
from vagen.bf16_delta import retain_state_delta,apply_to_model
from vagen.bootstrap_pilot import PilotDataset
from verl.utils.dataset.rl_dataset import collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader

p=protocol();batch,adv,answer=training_batch();cfg=make_config('train',BASE/'train')
old=OmegaConf.load(ROOT/'artifacts/component_tests/value_head_single_step_v1/Z/resolved.yaml')
old.actor_rollout_ref.actor.optim.total_training_steps=1  # production trainer sets this before workers start
assert OmegaConf.to_container(old.actor_rollout_ref.actor,resolve=True)==OmegaConf.to_container(cfg.actor_rollout_ref.actor,resolve=True)
assert OmegaConf.to_container(old.actor_rollout_ref.model,resolve=True)==OmegaConf.to_container(cfg.actor_rollout_ref.model,resolve=True)
actor=cfg.actor_rollout_ref.actor
assert actor.ppo_mini_batch_size==32 and actor.ppo_micro_batch_size_per_gpu==1 and actor.ppo_epochs==1
assert actor.loss_agg_mode=='token-mean' and actor.entropy_coeff==0 and not actor.use_kl_loss and not actor.get('use_rollout_log_probs',False)
assert actor.policy_loss.loss_mode=='vanilla' and actor.optim.lr==1e-6
assert torch.equal(batch.batch['advantages'][answer],adv[answer]) and bool((batch.batch['advantages'][~answer]==0).all())
assert _determine_fsdp_megatron_base_class([LocalActor.__mro__]) is Worker
logp=torch.tensor([[-.2,-.3,-.4]],requires_grad=True);a=torch.tensor([[1.,-2.,4.]]);m=torch.ones_like(a);selected=torch.tensor([[0.,1.,0.]])
loss,_=compute_policy_loss_vanilla(logp.detach(),logp,a*selected,m,'token-mean',actor)
(loss/32).backward();assert torch.allclose(logp.grad,-a*selected/(3*32),rtol=0,atol=1e-7)
class Toy(torch.nn.Module):
    _checkpoint_conversion_mapping={}
    def __init__(self,weight):
        super().__init__();self.embed=torch.nn.Embedding(2,4);self.lm_head=torch.nn.Linear(4,2,bias=False)
        self.embed.weight=torch.nn.Parameter(weight.float().clone());self.lm_head.weight=self.embed.weight
        self.config=SimpleNamespace(tie_word_embeddings=True)
    def get_input_embeddings(self):return self.embed
    def get_output_embeddings(self):return self.lm_head
with tempfile.TemporaryDirectory(dir=BASE) as tmp:
    tmp=Path(tmp);base=tmp/'base';base.mkdir()
    original=torch.arange(8).reshape(2,4).to(torch.bfloat16);target=original+torch.tensor([[.5,0,0,0],[0,0,0,-.5]],dtype=torch.bfloat16)
    save_file({'embed.weight':original.clone(),'lm_head.weight':original.clone()},str(base/'model.safetensors'))
    (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'embed.weight':'model.safetensors','lm_head.weight':'model.safetensors'}}))
    trained=Toy(target);restored=Toy(original)
    encoded=retain_state_delta(trained.state_dict(),trained,base,tmp/'private',4096,p['preparation_started_unix']+p['max_seconds_including_preparation'])
    schema=json.loads((tmp/'private/PRIVATE_PARAMETER_SCHEMA.json').read_text())
    apply_to_model(restored,tmp/'private/PRIVATE_LEARNED_BF16_DELTA.npz',schema)
    assert torch.equal(restored.embed.weight.detach().to(torch.bfloat16),target) and restored.embed.weight is restored.lm_head.weight
data=PilotDataset(str(ROOT/'src/configs/smoke/answer_region_dev.yaml'),cfg.data)
loader=StatefulDataLoader(data,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_fn)
manifest=json.loads((BASE/'data_manifest.json').read_text())
assert [int(row['seed'][0]) for row in loader]==[r['seed'] for r in manifest['splits']['dev']]
assert len(data)==32 and not ray.is_initialized()
result={'status':'CPU_READY','completed_unix':time.time(),'all32_GAE_and_returns_exactly_match_savedZ':True,'actor_model_and_optimizer_config_match_oldZ_runtime':True,'full_generated_denominator_preserved_scalar_autograd':True,'nonanswer_advantages_zero_after_full_whitening':True,'tiny_tied_stream_encode_apply_passed':True,'actual_MRO_base_checked':True,'fresh_serial_dev_rows':32,'answer_overlap_tokens':int(answer.sum()),'original_generated_tokens':int(batch.batch['response_mask'].sum()),'model_forwards':0,'model_backwards':0,'scalar_autograd_fixtures':1,'environment_steps':0,'optimizer_steps':0}
(BASE/'cpu_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
