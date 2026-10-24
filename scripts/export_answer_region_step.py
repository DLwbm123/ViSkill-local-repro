"""Allowlist synthetic outcomes and numeric signals; never private weights."""
import os,json,shutil
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);name='answer_region_step_v1';base=root/f'artifacts/component_tests/{name}'
p=json.loads((root/f'src/configs/smoke/{name}.json').read_text())
assert json.loads((base/'result.json').read_text())['status'].startswith('COMPLETE')
dest=base/'public_bundle';dest.mkdir(exist_ok=False)
def copy(source,relative):
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for relative in ['result.json','closure.json','cpu_check.json','cpu_repair_check.json','data_manifest.json','posthoc_signal_plan.json','posthoc_signal_audit.json']:
    copy(base/relative,relative)
for phase in ['train',p['training_output_folder'],'reload','b0','Z','A']:
    folder=base/phase
    if phase!='train':assert (folder/'COMPLETE.json').exists()
    for name_only in ['COMPLETE.json','actual_loading_info.json','initial_actor_rank0.json','initial_actor_rank1.json','inference_snapshot.json','actor_metrics.json','preupdate_gate.json','pre_scoring_carried.json','carried_alignment_rank0.json','carried_alignment_rank1.json','reload_gate.json','score_started.json','evaluation_started.json','update_started.json','training_signals.npz','preupdate_scores.npz','scores.npz']:
        source=folder/name_only
        if source.exists():copy(source,source.relative_to(base))
    for sub,pattern in [('numerical','*.json'),('update_audit','*.jsonl'),('sync_audit','*.jsonl'),('requests','*.json'),('episodes','*/*')]:
        for source in (folder/sub).glob(pattern):
            if source.is_file() and source.suffix in ['.json','.jsonl','.png']:copy(source,source.relative_to(base))
for suffix in ['', '_repair1']:
    receipt=json.loads((root/f'state/components/{name}{suffix}_receipt.json').read_text());receipt.pop('run_token',None)
    (dest/f'receipt{suffix}.json').write_text(json.dumps(receipt,indent=2));copy(root/f'state/components/{name}{suffix}_EXIT.json',f'EXIT{suffix}.json')
assert not any('PRIVATE' in path.name or path.suffix in ['.pt','.bin','.safetensors','.ckpt'] for path in dest.rglob('*'))
files=[path for path in dest.rglob('*') if path.is_file()];print(json.dumps({'files':len(files),'bytes':sum(path.stat().st_size for path in files),'model_parameters_exported':0}))
