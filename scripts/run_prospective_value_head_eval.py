"""One predeclared policy on one newly frozen prospective panel."""
import os,json,time,runpy,shutil
from pathlib import Path
from omegaconf import OmegaConf
import ray
ROOT=Path(os.environ['TASK_ROOT']);NAME='prospective_value_head_eval_v1'
BASE=ROOT/f'artifacts/component_tests/{NAME}'

def make_config(policy,folder):
    cfg=runpy.run_path(str(ROOT/'src/scripts/run_value_head_single_step.py'),run_name='config_helper')['make_config']('C',folder)
    p=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
    cfg.data.val_files=str(ROOT/'src/configs/smoke/prospective_value_head_dev.yaml')
    cfg.data.val_batch_size=1;cfg.actor_rollout_ref.rollout.agent.num_workers=1
    cfg.actor_rollout_ref.model.path=str(ROOT/p['policies'][policy])
    cfg.trainer.validation_data_dir=str(folder/'validation')
    cfg.ray_kwargs.ray_init._temp_dir='/data/local/tmp/z0/p'+policy.lower()
    assert cfg.algorithm.adv_estimator=='grpo'
    return cfg

def main():
    p=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());assert p['status']=='FROZEN_MODEL_READY_REPAIR1'
    assert time.time()<min(p['absolute_total_deadline'],p['preparation_started_unix']+p['max_seconds_including_preparation'])
    policy=os.environ['TASK_PHASE'];assert policy in p['policy_order']
    folder=BASE/p['policy_output_folders'][policy];assert not folder.exists(),'refuse repeating a dispatched policy'
    folder.mkdir();(folder/'context_libraries/empty').mkdir(parents=True)
    shutil.copytree(ROOT/p['source_library'],folder/'context_libraries/cards')
    os.environ.update(PILOT_OUT=str(folder),PILOT_DATA_MANIFEST=str(BASE/'data_manifest.json'),
        MECHANISM_AUDIT_DIR=str(folder/'audit'),MECHANISM_POLICY_LABEL='prospective_fixed_policy',
        LOCAL_UPDATE_AUDIT_DIR=str(folder/'update_audit'),LOCAL_SYNC_AUDIT_DIR=str(folder/'sync_audit'),
        FIXED_EMPTY_LIBRARY=str(folder/'context_libraries/empty'),FIXED_CARD_LIBRARY=str(folder/'context_libraries/cards'),
        FIXED_RUN_SEED='20261024',PILOT_MAX_MODEL_CALLS='160')
    cfg=make_config(policy,folder);OmegaConf.save(cfg,folder/'resolved.yaml',resolve=True)
    try:
        from vagen.main_ppo_skill import run_ppo
        from vagen.prospective_value_head_eval import EvalTaskRunner
        run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(EvalTaskRunner))
        assert (folder/'COMPLETE.json').exists() and not (folder/'FORBIDDEN_FULL_CHECKPOINTS').exists()
    finally:ray.shutdown()

if __name__=='__main__':main()
