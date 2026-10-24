"""One real PPO batch or independent checkpoint-load validation, never extra training."""
import json, os, time
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import ray
ROOT = Path(os.environ['TASK_ROOT'])
PHASE = os.environ['TASK_PHASE']
OUT = ROOT / 'results/ppo_smoke/directional_ppo_v1'
OUT.mkdir(parents=True, exist_ok=True)
with initialize_config_dir(config_dir=str(ROOT / 'src/vagen/configs'), version_base=None):
    cfg = compose(config_name='vagen_multiturn_skill', overrides=[
        'hydra.searchpath=[file://' + str(ROOT / 'src/verl/verl/trainer/config') + ']'])
OmegaConf.set_struct(cfg, False)
cfg = OmegaConf.merge(cfg, OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml'),
                     OmegaConf.load(ROOT/'src/configs/smoke/frozenlake_tiny.yaml'))
assert not cfg.skill_system.cold_start.enable and cfg.skill_system.distill.mode == 'self'
assert cfg.trainer.total_training_steps == 1
if PHASE == 'reload':
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = cfg.trainer.default_local_dir + '/global_step_1'
    cfg.trainer.val_before_train = True
    cfg.trainer.val_only = True
    cfg.skill_system.library.path = str(ROOT/'artifacts/ppo_smoke/directional_ppo_v1/restored_library')
    cfg.trainer.validation_data_dir = str(OUT/'reload_validation')
    cfg.ray_kwargs.ray_init._temp_dir = '/data/local/tmp/z0/r1'
else:
    assert PHASE == 'train'
    assert not Path(cfg.skill_system.library.path).exists(), 'cold start must be empty'
    assert not Path(cfg.trainer.default_local_dir).exists(), 'refuse duplicate training'
OmegaConf.save(cfg, OUT/f'{PHASE}_resolved.yaml', resolve=True)
started = time.time()
try:
    from vagen.main_ppo_skill import run_ppo
    run_ppo(cfg)
    (OUT/f'{PHASE}_COMPLETE.json').write_text(json.dumps({'phase':PHASE,'seconds':time.time()-started,
        'actual_run_ppo_returned':True,'optimizer_continuity_claim':False},indent=2))
finally:
    ray.shutdown()
