"""Reproduce missing metadata and reach the first training-forward boundary on CPU."""
import os,json,inspect,importlib,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from types import SimpleNamespace
from omegaconf import OmegaConf
from vagen.answer_region_step import BASE,training_batch,make_config
from verl.workers.actor import DataParallelPPOActor
batch,adv,answer=training_batch();fn=inspect.unwrap(DataParallelPPOActor.update_policy)
module=importlib.import_module('verl.workers.actor.dp_actor')
class ForwardBoundary(Exception):pass
counts={'forward_boundary':0,'optimizer_steps':0}
def boundary(*args,**kwargs):
    assert kwargs['temperature']==1.
    counts['forward_boundary']+=1
    raise ForwardBoundary('CPU dispatch reached; actual model forward deliberately absent')
old_device=module.get_device_id;module.get_device_id=lambda:'cpu'
try:
    for shard in batch.chunk(2):
        shard.meta_info.pop('temperature')
        try:fn(SimpleNamespace(actor_module=SimpleNamespace(train=lambda:None)),shard)
        except KeyError as exc:assert exc.args==('temperature',)
        else:raise AssertionError('old entry error was not reproduced')
        shard.meta_info['temperature']=1.
        cfg=OmegaConf.create(OmegaConf.to_container(make_config('train',BASE/'train_repair1').actor_rollout_ref.actor,resolve=True));cfg.ppo_mini_batch_size=16
        fake=SimpleNamespace(actor_module=SimpleNamespace(train=lambda:None),actor_optimizer=SimpleNamespace(zero_grad=lambda:None),config=cfg,_forward_micro_batch=boundary)
        try:fn(fake,shard)
        except ForwardBoundary:pass
        else:raise AssertionError('fixed metadata did not reach the expected forward boundary')
finally:module.get_device_id=old_device
assert counts['forward_boundary']==2
original=BASE/'train'
assert not list((original/'update_audit').glob('actor_*.jsonl')) and not (original/'training_signals.npz').exists()
prior=json.loads((original/'preupdate_gate.json').read_text());assert len(prior)==32 and all(r['operational_gate_pass'] for r in prior)
result={'status':'CPU_PREUPDATE_REPAIR_READY','completed_unix':time.time(),'both_DP_shards_old_temperature_error_reproduced':True,'both_fixed_shards_reach_first_training_forward_boundary':True,'original_business_scores_to_carry':32,'original_extra_alignment_samples_to_carry':16,'original_actual_training_forwards_before_failure':0,'original_actual_optimizer_steps':0,'new_model_forwards':0,'new_model_backwards':0,'new_optimizer_steps':0,'scope':'Restore required temperature=1 metadata normally supplied by union(compute_log_prob output). Carry existing32 scores/16 alignment checks without replay; new same-source worker instance is disclosed.'}
(BASE/'cpu_repair_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
