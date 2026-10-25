"""Resume the verified full state and run only the next native training batch."""
import os,json,time,runpy,shutil
from pathlib import Path
import ray
from omegaconf import OmegaConf
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME=os.environ['TASK_EXPERIMENT_NAME'];OUT=ROOT/f'artifacts/reproduction/{NAME}';P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text())
def config():
    cfg=runpy.run_path(str(SRC/'scripts/official_train_config.py'))['config']();cfg.trainer.resume_mode='resume_path';cfg.trainer.resume_from_path=str(ROOT/'artifacts/reproduction/official_frozenlake_correctness_v1/checkpoints/global_step_1');cfg.trainer.del_local_ckpt_after_load=False;cfg.trainer.val_before_train=False
    cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path=str(SRC/'configs/reproduction/observed_train_agent.yaml');cfg.ray_kwargs.ray_init._temp_dir='/data/tmp/z0/p3'
    return cfg
if __name__=='__main__':
    assert P['status']=='FROZEN_CPU_READY' and time.time()<P['cooperative_stop_unix'] and not (OUT/'dispatch_count.json').exists() and not (OUT/'training_started.json').exists()
    assert not (OUT/'library').exists();shutil.copytree(ROOT/'artifacts/reproduction/official_frozenlake_correctness_v1/checkpoints/global_step_1/online_library',OUT/'library')
    for folder in ['requests','episodes','fixed_inputs','metrics','validation_metrics','checkpoint_receipts']:(OUT/folder).mkdir(exist_ok=False)
    cfg=config();OmegaConf.save(cfg,OUT/'resolved.yaml',resolve=True)
    try:
        from vagen.main_ppo_skill import run_ppo
        from vagen.retrieved_update_check import CheckTask
        run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(CheckTask));assert (OUT/'WORKER_COMPLETED.json').is_file()
    finally:ray.shutdown()
