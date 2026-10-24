"""Complete saved tensor analysis on CPU, with no repeated model forwards."""
import os,json
from pathlib import Path
import torch
from verl.utils import hf_tokenizer,hf_processor
from verl.trainer.ppo.core_algos import compute_gae_advantage_return
from verl.utils.torch_functional import masked_whiten
from vagen.recorded_backend_audit import CheckTrainer

ROOT=Path(os.environ['TASK_ROOT']);OUT=ROOT/'artifacts/component_tests/recorded_backend_audit_v2'
assert not (OUT/'COMPLETE.json').exists()
manifest=json.loads((OUT/'inputs.json').read_text())
rows=[manifest['episodes'][k] for k in manifest['training_episode_ids']]
old=torch.load(OUT/'audit_actor.pt',map_location='cpu',weights_only=True)
values=torch.load(OUT/'audit_critic.pt',map_location='cpu',weights_only=True)
assert len(rows)==len(old)==len(values)==32
mask=torch.tensor([r['response_mask']+[0]*(4096-len(r['response_mask'])) for r in rows])
# Retain the reconstruction-only BF16 mistake; the original PPO used FP32 rewards.
bad_reward=torch.zeros_like(values)
for i,row in enumerate(rows):bad_reward[i,row['turns'][-1]['end']-1]=row['reward']
bad_adv,bad_return=compute_gae_advantage_return(bad_reward,values,mask,1.,1.)
constant=float((values*mask).sum()/mask.sum())
bad_shift=masked_whiten(bad_return-(values-constant),mask)
active=mask.bool()
incident={'scope':'New audit CPU reconstruction only; not an original PPO defect',
    'wrong_reward_dtype':str(bad_reward.dtype),'production_reward_dtype':'torch.float32',
    'original_reconstruction_return_error':float((bad_return-bad_reward.sum(-1,keepdim=True)).abs()[active].max()),
    'original_reconstruction_shift_error':float((bad_shift-bad_adv).abs()[active].max()),
    'additional_model_forwards':0,'thresholds_unchanged':True}
p=OUT/'incidents/repair3/cpu_analysis_dtype.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(incident,indent=2))
worker=CheckTrainer.__new__(CheckTrainer)
worker.tokenizer=hf_tokenizer(str(ROOT/'models/b0'),trust_remote_code=False)
worker.processor=hf_processor(str(ROOT/'models/b0'),trust_remote_code=False,use_fast=True)
batch=worker.make_batch(rows)
assert torch.equal(batch.batch['response_mask'],mask)
generated=json.loads((OUT/'score_results.json').read_text())['native_returned_generated_tokens']
worker.finish_signal_audit(rows,batch,old,values,generated)
print(json.dumps({'completed_from_saved_tensors':True,'additional_model_forwards':0,'incident':incident}))
