"""Start one released full-training attempt; existing dispatch evidence is never replayed."""
import os,json,time,runpy
from pathlib import Path
import ray
from omegaconf import OmegaConf
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='official_frozenlake_ppo_v1';OUT=ROOT/f'artifacts/reproduction/{NAME}';P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text())
assert P['status']=='FROZEN_CPU_READY' and time.time()<P['cooperative_stop_unix']
assert not (OUT/'dispatch_count.json').exists() and not (OUT/'training_started.json').exists() and not (OUT/'library').exists()
for folder in ['requests','episodes','fixed_inputs','metrics','validation_metrics','checkpoint_receipts']:(OUT/folder).mkdir(exist_ok=False)
cfg=runpy.run_path(str(SRC/'scripts/official_train_config.py'))['config']();OmegaConf.save(cfg,OUT/'resolved.yaml',resolve=True)
try:
    from vagen.main_ppo_skill import run_ppo
    from vagen.official_train_observer import RunTask
    run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(RunTask));assert (OUT/'WORKER_COMPLETED.json').exists()
finally:ray.shutdown()
