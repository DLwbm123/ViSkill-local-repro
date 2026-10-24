"""Export the completed prospective panel without weights or run tokens."""
import os,json,shutil
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);NAME='prospective_value_head_eval_v1';BASE=ROOT/f'artifacts/component_tests/{NAME}'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
assert (BASE/'result.json').exists()
for folder in P['policy_output_folders'].values():assert (BASE/folder/'COMPLETE.json').exists()
OUT=BASE/'public_bundle';OUT.mkdir(exist_ok=False)
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for name in ['result.json','data_manifest.json','cpu_runner_check.json','cpu_runner_check_before_gen_method_regression.json','pre_model_incidents.json','before_generation_failure_audit.json','snapshot_generation_metadata.json','closure.json']:
    if (BASE/name).exists():copy(BASE/name,OUT/name)
for folder in ['b0']+list(P['policy_output_folders'].values()):
    base=BASE/folder
    for name in ['COMPLETE.json','evaluation_started.json','actual_loading_info.json','initial_actor_rank0.json','initial_actor_rank1.json']:
        if (base/name).exists():copy(base/name,OUT/folder/name)
    for sub,pattern in [('episodes','*/*'),('requests','*.json'),('sync_audit','*.jsonl')]:
        for path in (base/sub).glob(pattern):
            if path.is_file() and path.suffix in ['.json','.jsonl','.png']:
                copy(path,OUT/folder/sub/path.relative_to(base/sub))
for suffix in ['', '_repair1']:
    state=ROOT/f'state/components/{NAME}{suffix}_receipt.json'
    data=json.loads(state.read_text());data.pop('run_token',None)
    (OUT/f'receipt{suffix}.json').write_text(json.dumps(data,indent=2))
    copy(ROOT/f'state/components/{NAME}{suffix}_EXIT.json',OUT/f'EXIT{suffix}.json')
assert not any(path.suffix in ['.pt','.bin','.safetensors','.ckpt'] for path in OUT.rglob('*'))
files=[path for path in OUT.rglob('*') if path.is_file()]
print(json.dumps({'files':len(files),'bytes':sum(path.stat().st_size for path in files),'weights_included':False,'scope':'All prospective synthetic episodes and requests, loading proof, gates and retained zero-query failure.'}))
