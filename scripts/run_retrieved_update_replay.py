"""Run the native update path with the CPU-validated existing generation batch."""
import os,runpy,time
from pathlib import Path
import ray
from omegaconf import OmegaConf
from vagen.retrieved_replay import ROOT,SRC,OUT,P,ReplayTask
assert P['status']=='FROZEN_CPU_READY' and time.time()<P['cooperative_stop_unix'] and not (OUT/'training_started.json').exists()
cfg=runpy.run_path(str(SRC/'scripts/run_retrieved_update_check.py'))['config']();cfg.ray_kwargs.ray_init._temp_dir='/data/tmp/z0/p4';OmegaConf.save(cfg,OUT/'resolved.yaml',resolve=True)
try:
 from vagen.main_ppo_skill import run_ppo
 run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(ReplayTask));assert (OUT/'WORKER_COMPLETED.json').exists()
finally:ray.shutdown()
