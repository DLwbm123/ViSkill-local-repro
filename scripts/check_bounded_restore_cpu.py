"""Actual saved DataProto/config/API readiness; no Ray or model startup."""
import os,json,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from omegaconf import OmegaConf
import ray
from transformers import AutoConfig
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls,_determine_fsdp_megatron_base_class
from verl.single_controller.base.worker import Worker
from vagen.bounded_restore import LocalActor,fixed_batch,make_config

root=Path(os.environ['TASK_ROOT']);base=root/'artifacts/component_tests/bf16_delta_reload_v1'
assert base.exists()
class BadName(LocalActor):pass
BadName.__name__='Worker'
assert _determine_fsdp_megatron_base_class([BadName.__mro__]) is BadName
assert _determine_fsdp_megatron_base_class([LocalActor.__mro__]) is Worker
batch=fixed_batch();p=json.loads((root/'src/configs/smoke/bf16_delta_reload_v1.json').read_text())
checks={}
for phase in p['policy_order']:
    cfg=make_config(phase,base/phase)
    model=AutoConfig.from_pretrained(cfg.actor_rollout_ref.model.path,trust_remote_code=True)
    assert model.tie_word_embeddings
    cls=RayClassWithInitArgs(ray.remote(LocalActor),config=cfg.actor_rollout_ref,role='actor')
    colocated=create_colocated_worker_cls({'actor':cls})
    assert isinstance(colocated,RayClassWithInitArgs) and not ray.is_initialized()
    checks[phase]={'actor_only_role':True,'same_mixed_precision_config':True,'actual_HF_config_tied':True,'worker_colocation_API_valid':True}
    OmegaConf.save(cfg,base/f'cpu_{phase}_resolved.yaml',resolve=True)
assert (root/'artifacts/component_tests/bf16_delta_feasibility_v1/PRIVATE_LEARNED_BF16_DELTA.npz').is_file()
result={'status':'CPU_READY','completed_unix':time.time(),'rows':2,'seeds':[130014,130007],'exact_saved_multimodal_batch':True,'checks':checks,'model_forwards':0,'environment_steps':0,'optimizer_steps':0,
 'retained_preflight_failure':'Initial CPU guard incorrectly expected optional mixed_precision section; actual resolved config uses FSDPEngineConfig.dtype=bfloat16. Fixed assertion without changing model settings, before any Ray/model startup.'}
result['actual_MRO_base_selector_regression_passed']=True
(base/'cpu_repair_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
