"""One phase; completed phases and updates cannot be replayed."""
import os,time,json,shutil
from pathlib import Path
import ray
from omegaconf import OmegaConf
from vagen.support_curriculum import ROOT,BASE,protocol,make_config,saved_phase,LocalTask

def main():
    p=protocol();phase=os.environ['TASK_PHASE'];assert p['status']=='FROZEN_CPU_READY' and phase in p['phase_order']
    assert time.time()<p['preparation_started_unix']+p['max_seconds_including_preparation']
    if phase!='collect':assert json.loads((BASE/'collect/bank_ready.json').read_text())['gate_pass']
    if phase.startswith('eval_'):
        assert all(json.loads((BASE/f'reload_{label}/COMPLETE.json').read_text())['operational_reload_verified'] for label in ['Q','R'])
    folder=BASE/phase;assert not folder.exists();folder.mkdir();(folder/'context_libraries/empty').mkdir(parents=True)
    shutil.copytree(ROOT/p['source_library'],folder/'context_libraries/cards')
    os.environ.update(PILOT_OUT=str(folder),PILOT_DATA_MANIFEST=str(BASE/'data_manifest.json'),FIXED_RUN_SEED='20261024',
        FIXED_EMPTY_LIBRARY=str(folder/'context_libraries/empty'),FIXED_CARD_LIBRARY=str(folder/'context_libraries/cards'),
        MECHANISM_AUDIT_DIR=str(folder/'audit'),MECHANISM_POLICY_LABEL='fixed_data_support_policy',LOCAL_UPDATE_AUDIT_DIR=str(folder/'update_audit'),LOCAL_SYNC_AUDIT_DIR=str(folder/'sync_audit'),PILOT_MAX_MODEL_CALLS='200')
    cfg=make_config(phase,folder);OmegaConf.save(cfg,folder/'resolved.yaml',resolve=True)
    try:
        if phase in ['Q','R','reload_Q','reload_R']:saved_phase(phase,cfg)
        else:
            from vagen.main_ppo_skill import run_ppo
            run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(LocalTask))
        assert (folder/'COMPLETE.json').exists() and not (folder/'FORBIDDEN_FULL_CHECKPOINTS').exists()
    finally:ray.shutdown()

if __name__=='__main__':main()
