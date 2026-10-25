"""Resolve released script overrides first, then list execution-only eval adaptations."""
import os,re
from pathlib import Path
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);OUT=ROOT/'artifacts/reproduction/released_frozenlake_eval_v1'
def config():
    script=SRC/'examples/train/frozenlake/train_ppo_qwen25vl3b_skill.sh';body=script.read_text().split('python3 -m vagen.main_ppo_skill',1)[1]
    values={'BASEDIR':str(SRC),'SCRIPTDIR':str(script.parent),'EXPERIMENT_DIR':str(OUT),'SAVE_CHECKPOINT_DIR':str(OUT/'FORBIDDEN_CHECKPOINTS'),'DATASET_TRAIN':str(script.parent/'train_frozenlake_vision_free_think.yaml'),'DATASET_VAL':str(script.parent/'val_frozenlake_vision_free_think.yaml'),'REF_MODEL_PATH':str(ROOT/'models/r0'),'PROJECT_NAME':'local','EXPERIMENT_NAME':'evaluation','agent_loop_config_path':str(SRC/'vagen/configs/agent_skill.yaml')};overrides=[]
    for line in body.splitlines():
        line=line.strip().removesuffix('\\').strip().split('2>&1',1)[0].strip()
        if not re.match(r'^\w+(?:\.\w+)*=',line):continue
        if line.startswith('trainer.n_gpus_per_node='):line='trainer.n_gpus_per_node=2'
        else:
            for key,value in values.items():line=line.replace('${'+key+'}',value).replace('$'+key,value)
        assert '$' not in line,line;overrides.append(line)
    with initialize_config_dir(config_dir=str(SRC/'vagen/configs'),version_base=None):
        cfg=compose(config_name='vagen_multiturn_skill',overrides=['hydra.searchpath=[file://'+str(SRC/'verl/verl/trainer/config')+']',*overrides])
    OmegaConf.set_struct(cfg,False);official=OmegaConf.to_container(cfg,resolve=True)
    cfg.trainer.logger=['console'];cfg.trainer.val_only=True;cfg.trainer.resume_mode='disable';cfg.trainer.save_freq=-1
    cfg.trainer.validation_data_dir=str(OUT/'validation');cfg.skill_system.library.path=str(OUT/'library')
    cfg.actor_rollout_ref.rollout.engine_kwargs.sglang.update(attention_backend='triton',mm_attention_backend='sdpa',random_seed=42)
    cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path=str(SRC/'configs/reproduction/observed_agent.yaml')
    cfg.algorithm.adv_estimator='grpo'  # No advantages or updates in val_only; omit the unused critic.
    cfg.ray_kwargs.ray_init.update(num_cpus=16,num_gpus=2,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir='/data/tmp/z0/p0',object_store_memory=4294967296,runtime_env={})
    assert cfg.data.train_batch_size==128 and cfg.actor_rollout_ref.actor.ppo_mini_batch_size==32
    assert cfg.data.max_prompt_length==5000 and cfg.data.max_response_length==4000
    assert cfg.skill_system.reward.mode=='skill_guided' and cfg.skill_system.reward.guided_weight==.8
    assert cfg.skill_system.distill.mode=='self' and not cfg.skill_system.cold_start.enable
    return cfg,official
