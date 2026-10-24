"""Initialize production workers solely for the frozen saved-input audit."""
import os,json,time
from pathlib import Path
from omegaconf import OmegaConf
import ray
ROOT=Path(os.environ['TASK_ROOT']);NAME='recorded_backend_audit_v2'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
OUT=ROOT/f'artifacts/component_tests/{NAME}'
assert P['status']=='FROZEN_CPU_PREFLIGHT_PASS'
assert time.time()<P['preparation_started_unix']+P['max_seconds_including_preparation']
manifest=json.loads((OUT/'inputs.json').read_text())
head=json.loads(Path(manifest['training_initial_head']).read_text())
os.environ.update(CHECK_HEAD=manifest['training_initial_head'],CHECK_RUN_SEED=str(head['run_seed']),
    PILOT_OUT=str(OUT),PILOT_DATA_MANIFEST=str(ROOT/'artifacts/component_tests/fixed_library_ppo_v1/data_manifest.json'))
cfg=OmegaConf.load(manifest['original_resolved_config'])
cfg.skill_system.enable=False
cfg.trainer.resume_mode='disable';cfg.trainer.save_freq=-1;cfg.trainer.test_freq=-1
cfg.trainer.default_local_dir=str(OUT/'FORBIDDEN_CHECKPOINTS')
cfg.trainer.rollout_data_dir=None;cfg.trainer.validation_data_dir=None
cfg.ray_kwargs.ray_init._temp_dir='/data/local/tmp/z0/s2'
# All required variables are set before Ray starts and inherited by its workers.
# Explicit runtime_env serializes private paths into transient worker argv.
cfg.ray_kwargs.ray_init.runtime_env={}
cfg.actor_rollout_ref.rollout.temperature=1.
OmegaConf.save(cfg,OUT/'resolved.yaml',resolve=True)
try:
    from vagen.main_ppo_skill import run_ppo
    from vagen.recorded_backend_audit import CheckTaskRunner
    run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(CheckTaskRunner))
    assert (OUT/'COMPLETE.json').exists() and not (OUT/'FORBIDDEN_CHECKPOINTS').exists()
finally:ray.shutdown()
