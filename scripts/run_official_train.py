"""Start one released full-training attempt; existing dispatch evidence is never replayed."""
import os,json,time,runpy
from pathlib import Path
import ray
from omegaconf import OmegaConf
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME=os.environ.get('TASK_EXPERIMENT_NAME','official_frozenlake_ppo_v1');OUT=ROOT/f'artifacts/reproduction/{NAME}';P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text())
assert P['status']=='FROZEN_CPU_READY' and time.time()<P['cooperative_stop_unix']
stage=os.environ.get('TASK_RUN_STAGE','train')
if stage=='train':
    assert not (OUT/'dispatch_count.json').exists() and not (OUT/'training_started.json').exists() and not (OUT/'library').exists()
    for folder in ['requests','episodes','fixed_inputs','metrics','validation_metrics','checkpoint_receipts']:(OUT/folder).mkdir(exist_ok=False)
else:
    assert stage=='reload' and not (OUT/'WORKER_RELOAD_COMPLETED.json').exists() and not (OUT/'RELOAD_APPLIED.json').exists()
    assert json.loads((OUT/'WORKER_COMPLETED.json').read_text())['reason']=='planned correctness stop after global batch1'
cfg=runpy.run_path(str(SRC/'scripts/official_train_config.py'))['config']()
if stage=='reload':
    cfg.trainer.val_only=True;cfg.trainer.resume_mode='resume_path';cfg.trainer.resume_from_path=str(OUT/'checkpoints/global_step_1');cfg.trainer.del_local_ckpt_after_load=False
OmegaConf.save(cfg,OUT/('resolved_reload.yaml' if stage=='reload' else 'resolved.yaml'),resolve=True)

try:
    from vagen.main_ppo_skill import run_ppo
    from vagen.official_train_observer import RunTask
    run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(RunTask));assert (OUT/('WORKER_RELOAD_COMPLETED.json' if stage=='reload' else 'WORKER_COMPLETED.json')).exists()
finally:ray.shutdown()
