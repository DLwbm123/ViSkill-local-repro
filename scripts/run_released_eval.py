"""One official256-map validation pass; refuse replay of any dispatched evaluation."""
import os,json,time,shutil,runpy
from pathlib import Path
import ray
from omegaconf import OmegaConf
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);OUT=ROOT/'artifacts/reproduction/released_frozenlake_eval_v1'
p=json.loads((SRC/'configs/reproduction/released_frozenlake_eval_v1.json').read_text());assert time.time()<p['deadline_unix'] and not (OUT/'evaluation_started.json').exists()
assert not (OUT/'library').exists();shutil.copytree(ROOT/'models/r0/skill_library',OUT/'library')
for sub in ['requests','request_images','episodes']:(OUT/sub).mkdir(exist_ok=False)
cfg,official=runpy.run_path(str(SRC/'scripts/release_eval_config.py'),run_name='configuration')['config']()
OmegaConf.save(OmegaConf.create(official),OUT/'official_script_resolved.yaml');OmegaConf.save(cfg,OUT/'resolved.yaml',resolve=True)
try:
    from vagen.main_ppo_skill import run_ppo
    from vagen.released_eval_observer import ObserverTask
    run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(ObserverTask));assert (OUT/'COMPLETE.json').is_file()
finally:ray.shutdown()
