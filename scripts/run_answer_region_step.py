"""Dispatch exactly one declared train/reload/evaluation phase."""
import os,json,time,shutil
from pathlib import Path
import ray
from omegaconf import OmegaConf
from vagen.answer_region_step import ROOT,BASE,protocol,make_config,run_saved,EvalTask

def main():
    p=protocol();assert p['status']=='FROZEN_CPU_READY'
    assert time.time()<p['preparation_started_unix']+p['max_seconds_including_preparation']
    phase=os.environ['TASK_PHASE'];assert phase in p['policy_order']
    if phase in ['b0','Z','A']:
        reload=json.loads((BASE/'reload/COMPLETE.json').read_text());assert reload['operational_reload_verified']
    folder=BASE/(p['training_output_folder'] if phase=='train' else phase);assert not folder.exists(),'never replay a dispatched phase'
    folder.mkdir();(folder/'context_libraries/empty').mkdir(parents=True)
    shutil.copytree(ROOT/p['source_library'],folder/'context_libraries/cards')
    os.environ.update(PILOT_OUT=str(folder),FIXED_RUN_SEED='20261024',PILOT_DATA_MANIFEST=str(BASE/'data_manifest.json'),
        MECHANISM_AUDIT_DIR=str(folder/'audit'),MECHANISM_POLICY_LABEL='answer_region_fixed_policy',LOCAL_UPDATE_AUDIT_DIR=str(folder/'update_audit'),LOCAL_SYNC_AUDIT_DIR=str(folder/'sync_audit'),
        FIXED_EMPTY_LIBRARY=str(folder/'context_libraries/empty'),FIXED_CARD_LIBRARY=str(folder/'context_libraries/cards'),PILOT_MAX_MODEL_CALLS='160')
    cfg=make_config(phase,folder);OmegaConf.save(cfg,folder/'resolved.yaml',resolve=True)
    try:
        if phase in ['train','reload']:run_saved(phase,folder,cfg)
        else:
            from vagen.main_ppo_skill import run_ppo
            run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(EvalTask))
        assert (folder/'COMPLETE.json').exists() and not (folder/'FORBIDDEN_FULL_CHECKPOINTS').exists()
    finally:ray.shutdown()

if __name__=='__main__':main()
