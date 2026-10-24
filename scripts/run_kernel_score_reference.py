"""Frozen actor-only diagnostic; no native server, rollout, critic, or update."""
import os,json,time
from pathlib import Path
import ray
from omegaconf import OmegaConf
ROOT=Path(os.environ['TASK_ROOT']);NAME='kernel_score_reference_v1'
OUT=ROOT/f'artifacts/component_tests/{NAME}'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
assert P['status']=='FROZEN' and time.time()<P['preparation_started_unix']+P['max_seconds_including_preparation']
manifest=json.loads((ROOT/'artifacts/component_tests/recorded_backend_audit_v2/inputs.json').read_text())
os.environ.update(PILOT_OUT=str(OUT),PILOT_DATA_MANIFEST=str(ROOT/'artifacts/component_tests/fixed_library_ppo_v1/data_manifest.json'))
cfg=OmegaConf.load(manifest['original_resolved_config'])
cfg.skill_system.enable=False
cfg.algorithm.adv_estimator='grpo'  # no critic is initialized; no advantages or updates are computed
cfg.actor_rollout_ref.rollout.mode='sync'
cfg.actor_rollout_ref.rollout.temperature=1.
cfg.actor_rollout_ref.model.use_fused_kernels=False
cfg.actor_rollout_ref.actor.use_fused_kernels=False
cfg.trainer.resume_mode='disable';cfg.trainer.save_freq=-1;cfg.trainer.test_freq=-1
cfg.trainer.default_local_dir=str(OUT/'FORBIDDEN_CHECKPOINTS')
cfg.trainer.rollout_data_dir=None;cfg.trainer.validation_data_dir=None
cfg.ray_kwargs.ray_init._temp_dir='/data/local/tmp/z0/s3'
cfg.ray_kwargs.ray_init.runtime_env={}
OmegaConf.save(cfg,OUT/'resolved.yaml',resolve=True)
try:
    from vagen.main_ppo_skill import run_ppo
    from vagen.kernel_score_reference import ReferenceTaskRunner
    run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(ReferenceTaskRunner))
    assert (OUT/'COMPLETE.json').exists() and not (OUT/'FORBIDDEN_CHECKPOINTS').exists()
finally:ray.shutdown()
