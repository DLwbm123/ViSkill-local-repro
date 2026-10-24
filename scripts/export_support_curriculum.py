"""Saved synthetic outcomes and numerical evidence only; no private containers."""
import os,json,shutil
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);name='support_matched_curriculum_v1';base=root/f'artifacts/component_tests/{name}'
result=json.loads((base/'result.json').read_text());dest=base/'public_bundle';dest.mkdir(exist_ok=False)
def copy(source,relative):
    assert source.is_file() and not source.name.startswith('PRIVATE_')
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for name_only in ['result.json','closure.json','failure_audit.json','cpu_distill_guard_check.json','cpu_check.json','data_manifest.json']:
    if (base/name_only).exists():copy(base/name_only,name_only)
for phase in ['collect','Q','reload_Q','R','reload_R','eval_b0','eval_Q','eval_R']:
    folder=base/phase
    for name_only in ['COMPLETE.json','actual_loading_info.json','initial_actor_rank0.json','initial_actor_rank1.json','inference_snapshot.json','actor_metrics.json','common24_pre_gate.json','reload_gate.json','collection_started.json','evaluation_started.json','update_started.json','training_signals.npz','pre_scores.npz','scores.npz','bank_numeric.npz','bank_ready.json']:
        source=folder/name_only
        if source.exists():copy(source,source.relative_to(base))
    for sub,pattern in [('numerical','*.json'),('update_audit','*.jsonl'),('sync_audit','*.jsonl'),('requests','*.json'),('episodes','*/*')]:
        for source in (folder/sub).glob(pattern):
            if source.is_file() and source.suffix in ['.json','.jsonl','.png']:copy(source,source.relative_to(base))
receipt=json.loads((root/f'state/components/{name}_receipt.json').read_text());receipt.pop('run_token',None)
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));copy(root/f'state/components/{name}_EXIT.json','EXIT.json')
assert not any('PRIVATE' in p.name or p.suffix in ['.pt','.gz','.safetensors','.ckpt'] for p in dest.rglob('*'))
files=[p for p in dest.rglob('*') if p.is_file()];print(json.dumps({'files':len(files),'bytes':sum(p.stat().st_size for p in files),'private_containers_or_weights_exported':False}))
