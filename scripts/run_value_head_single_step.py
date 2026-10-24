"""One phase of a frozen shared-data value-head intervention."""
import os,json,time,shutil
from pathlib import Path
from omegaconf import OmegaConf
import ray
ROOT=Path(os.environ['TASK_ROOT']);NAME='value_head_single_step_v1'
BASE=ROOT/f'artifacts/component_tests/{NAME}'

def make_config(phase,folder):
    manifest=json.loads((ROOT/'artifacts/component_tests/recorded_backend_audit_v2/inputs.json').read_text())
    cfg=OmegaConf.load(manifest['original_resolved_config']);OmegaConf.set_struct(cfg,False)
    cfg.data.train_files=str(ROOT/'src/configs/smoke/value_head_train.yaml')
    cfg.data.val_files=str(ROOT/'src/configs/smoke/value_head_dev.yaml')
    cfg.data.train_batch_size=cfg.data.gen_batch_size=32;cfg.data.val_batch_size=8
    cfg.data.shuffle=cfg.data.validation_shuffle=False
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size=32
    cfg.critic.ppo_mini_batch_size=32
    cfg.actor_rollout_ref.rollout.temperature=1.
    cfg.actor_rollout_ref.rollout.val_kwargs.update(temperature=0.,top_p=1.,do_sample=False,n=1)
    cfg.trainer.total_training_steps=1;cfg.trainer.total_epochs=1
    cfg.trainer.resume_mode='disable';cfg.trainer.save_freq=-1;cfg.trainer.test_freq=-1
    cfg.trainer.default_local_dir=str(folder/'FORBIDDEN_FULL_CHECKPOINTS')
    cfg.trainer.rollout_data_dir=None;cfg.trainer.validation_data_dir=str(folder/'validation')
    cfg.skill_system.library.path=str(folder/'context_libraries/cards')
    cfg.skill_system.checkpoint_pairing.enable=False
    cfg.skill_system.reward.guided_weight=0.
    cfg.ray_kwargs.ray_init._temp_dir='/data/local/tmp/z0/h'+phase.lower()
    cfg.ray_kwargs.ray_init.runtime_env={}
    if phase=='C':cfg.algorithm.adv_estimator='grpo'  # skip unused critic during collection
    assert cfg.algorithm.gamma==cfg.algorithm.lam==1.
    assert cfg.actor_rollout_ref.model.use_fused_kernels
    assert cfg.actor_rollout_ref.actor.ppo_epochs==1
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==1
    assert not cfg.actor_rollout_ref.actor.use_kl_loss and not cfg.algorithm.use_kl_in_reward
    assert cfg.actor_rollout_ref.rollout.n==1
    return cfg

def main():
    p=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
    assert p['status']=='FROZEN_MODEL_READY_REPAIR1'
    assert time.time()<min(p['absolute_total_deadline'],p['preparation_started_unix']+p['max_seconds_including_preparation'])
    phase=os.environ['TASK_PHASE'];assert phase in ['C','O','Z']
    repair=phase=='O' and os.getenv('TASK_ATTEMPT')=='repair1'
    folder=BASE/('O_repair1' if repair else phase)
    assert not folder.exists(),'refuse restarting a dispatched phase'
    folder.mkdir();(folder/'context_libraries/empty').mkdir(parents=True)
    shutil.copytree(ROOT/p['source_library'],folder/'context_libraries/cards')
    head=json.loads((ROOT/'artifacts/component_tests/recorded_backend_audit_v2/inputs.json').read_text())['training_initial_head']
    os.environ.update(PILOT_OUT=str(folder),PILOT_DATA_MANIFEST=str(BASE/'data_manifest.json'),
        MECHANISM_AUDIT_DIR=str(folder/'audit'),MECHANISM_POLICY_LABEL='single_controlled_update',
        LOCAL_UPDATE_AUDIT_DIR=str(folder/'update_audit'),LOCAL_SYNC_AUDIT_DIR=str(folder/'sync_audit'),
        FIXED_EMPTY_LIBRARY=str(folder/'context_libraries/empty'),FIXED_CARD_LIBRARY=str(folder/'context_libraries/cards'),
        FIXED_RUN_SEED=str(p['run_seed']),PILOT_MAX_MODEL_CALLS=str(160-int((BASE/'O/model_calls.txt').read_text()) if repair else 192 if phase=='C' else 160),
        CHECK_HEAD=str(BASE/'zero_head_private.json') if phase=='Z' else head)
    if repair:os.environ['CARRY_BASELINE_FROM']=str(BASE/'O')
    else:os.environ.pop('CARRY_BASELINE_FROM',None)
    cfg=make_config(phase,folder);OmegaConf.save(cfg,folder/'resolved.yaml',resolve=True)
    before=(folder/'context_libraries/cards/skill_index.json').read_text()
    try:
        from vagen.main_ppo_skill import run_ppo
        from vagen.value_head_single_step import SingleTaskRunner
        run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(SingleTaskRunner))
        assert (folder/'COMPLETE.json').exists() and not (folder/'FORBIDDEN_FULL_CHECKPOINTS').exists()
        assert before==(folder/'context_libraries/cards/skill_index.json').read_text()
    finally:ray.shutdown()

if __name__=='__main__':main()
