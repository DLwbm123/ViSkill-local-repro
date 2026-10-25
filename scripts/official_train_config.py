"""Restore released training overrides; apply listed runtime/observability settings only."""
import os,runpy
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME=os.environ.get('TASK_EXPERIMENT_NAME','official_frozenlake_ppo_v1');OUT=ROOT/f'artifacts/reproduction/{NAME}'
def config():
    _,original=runpy.run_path(str(SRC/'scripts/release_eval_config.py'))['config']()
    def paths(value):
        if isinstance(value,dict):return {k:paths(v) for k,v in value.items()}
        if isinstance(value,list):return [paths(v) for v in value]
        if isinstance(value,str):return value.replace(str(ROOT/'models/r0'),str(ROOT/'models/b0')).replace(str(ROOT/'artifacts/reproduction/released_frozenlake_eval_v1'),str(OUT)).replace('FORBIDDEN_CHECKPOINTS','checkpoints')
        return value
    cfg=OmegaConf.create(paths(original));OmegaConf.set_struct(cfg,False)
    cfg.trainer.logger=['console'];cfg.trainer.resume_mode='disable';cfg.trainer.val_only=False
    cfg.trainer.default_local_dir=str(OUT/'checkpoints');cfg.skill_system.library.path=str(OUT/'library')
    cfg.actor_rollout_ref.rollout.engine_kwargs.sglang.update(attention_backend='triton',mm_attention_backend='sdpa',random_seed=42,context_length=9000)
    cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path=str(SRC/'configs/reproduction/observed_train_agent.yaml')
    cfg.ray_kwargs.ray_init.update(num_cpus=16,num_gpus=2,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir='/data/tmp/z0/p1',object_store_memory=4294967296,runtime_env={})
    assert cfg.algorithm.adv_estimator=='gae' and cfg.trainer.total_training_steps==301
    assert cfg.data.train_batch_size==128 and cfg.actor_rollout_ref.actor.ppo_mini_batch_size==32
    assert cfg.actor_rollout_ref.actor.optim.lr==1e-6 and cfg.critic.optim.lr==1e-5
    assert cfg.skill_system.distill.mode=='self' and not cfg.skill_system.cold_start.enable
    assert cfg.actor_rollout_ref.model.path==cfg.critic.model.path==str(ROOT/'models/b0')
    return cfg
