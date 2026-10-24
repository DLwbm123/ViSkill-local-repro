"""One fixed arm of the paired bootstrap pilot; no checkpoint or external API."""
import json
import os
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import ray

ROOT = Path(os.environ['TASK_ROOT'])
ARM = os.environ['TASK_PHASE']
assert ARM in ['A', 'B']
OUT = ROOT / 'results/ppo_smoke/bootstrap_card_pair_v1' / ARM


def make_config():
    with initialize_config_dir(config_dir=str(ROOT / 'src/vagen/configs'), version_base=None):
        cfg = compose(config_name='vagen_multiturn_skill', overrides=[
            'hydra.searchpath=[file://' + str(ROOT / 'src/verl/verl/trainer/config') + ']'])
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(ROOT / 'src/configs/low_resource/directional_skill_system.yaml'),
                         OmegaConf.load(ROOT / 'src/configs/smoke/frozenlake_tiny.yaml'))
    cfg.data.train_files = str(ROOT / 'src/configs/smoke/bootstrap_train.yaml')
    cfg.data.val_files = str(ROOT / 'src/configs/smoke/bootstrap_dev.yaml')
    cfg.data.train_batch_size = cfg.data.gen_batch_size = 32
    cfg.data.val_batch_size = 8
    cfg.data.shuffle = cfg.data.validation_shuffle = False
    cfg.data.custom_cls = {'path': 'pkg://vagen.bootstrap_pilot', 'name': 'PilotDataset'}
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 8
    cfg.critic.ppo_mini_batch_size = 8
    cfg.actor_rollout_ref.rollout.temperature = 1.0
    cfg.actor_rollout_ref.rollout.top_p = 1.0
    cfg.actor_rollout_ref.rollout.val_kwargs.update(temperature=0.0, top_p=1.0, do_sample=False, n=1)
    cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(ROOT / 'src/configs/smoke/bootstrap_agent.yaml')
    cfg.trainer.total_training_steps = 4
    cfg.trainer.total_epochs = 1
    cfg.trainer.val_before_train = True
    cfg.trainer.test_freq = 4
    cfg.trainer.save_freq = -1
    cfg.trainer.default_local_dir = str(OUT / 'FORBIDDEN_CHECKPOINTS')
    cfg.trainer.rollout_data_dir = str(OUT / 'rollout')
    cfg.trainer.validation_data_dir = str(OUT / 'dev_validation')
    cfg.trainer.experiment_name = f'pilot_{ARM}'
    cfg.trainer.log_val_generations = 0
    cfg.skill_system.library.path = str(OUT / 'library')
    cfg.skill_system.checkpoint_pairing.enable = False
    cfg.skill_system.reward.mode = 'state_progress_bounded'
    cfg.skill_system.reward.guided_weight = 0.0
    cfg.huggingface_hub.hf_save_freq = None
    cfg.huggingface_hub.repo_id = None
    cfg.ray_kwargs.ray_init._temp_dir = f'/data/local/tmp/z0/p{ARM.lower()}'
    cfg.bootstrap_pilot = {'arm': ARM, 'probe_file': str(ROOT / 'src/configs/smoke/bootstrap_probe.yaml')}
    assert cfg.skill_system.distill.mode == 'self' and not cfg.skill_system.cold_start.enable
    assert cfg.actor_rollout_ref.model.lora_rank == 0 and not cfg.actor_rollout_ref.actor.freeze_vision_tower
    assert cfg.actor_rollout_ref.rollout.n == 1
    return cfg


def main():
    protocol = json.loads((ROOT / 'src/configs/smoke/bootstrap_card_pair_v1.json').read_text())
    assert protocol['status'] == 'FROZEN_CPU_PREFLIGHT_PASS'
    assert not OUT.exists(), 'refuse duplicate arm'
    OUT.mkdir(parents=True)
    os.environ.update(PILOT_OUT=str(OUT),
        PILOT_DATA_MANIFEST=str(ROOT / 'artifacts/component_tests/bootstrap_card_pair_v1/data_manifest.json'),
        MECHANISM_AUDIT_DIR=str(OUT / 'audit'), MECHANISM_POLICY_LABEL='pilot_dynamic_versioned',
        LOCAL_UPDATE_AUDIT_DIR=str(OUT / 'update_audit'), LOCAL_SYNC_AUDIT_DIR=str(OUT / 'sync_audit'))
    cfg = make_config()
    # Ray actors inherit evidence paths explicitly through the configured runtime.
    cfg.ray_kwargs.ray_init.runtime_env = {'env_vars': {k: os.environ[k] for k in [
        'PILOT_OUT', 'PILOT_DATA_MANIFEST', 'MECHANISM_AUDIT_DIR', 'MECHANISM_POLICY_LABEL',
        'LOCAL_UPDATE_AUDIT_DIR', 'LOCAL_SYNC_AUDIT_DIR', 'TASK_PHASE', 'TASK_ROOT', 'TASK_RUN_TOKEN']}}
    OmegaConf.save(cfg, OUT / 'resolved.yaml', resolve=True)
    started = time.time()
    try:
        from vagen.main_ppo_skill import run_ppo
        from vagen.bootstrap_pilot import PilotTaskRunner
        run_ppo(cfg, task_runner_class=ray.remote(num_cpus=1)(PilotTaskRunner))
        assert (OUT / 'completion/trainer.json').exists()
        assert not (OUT / 'FORBIDDEN_CHECKPOINTS').exists()
        (OUT / 'COMPLETE.json').write_text(json.dumps({'arm': ARM, 'seconds': time.time() - started,
            'actual_run_ppo_returned': True, 'checkpoint_saved': False, 'final_model_retained': False}, indent=2))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
