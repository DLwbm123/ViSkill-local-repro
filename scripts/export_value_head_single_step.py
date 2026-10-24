"""Copy only the declared synthetic/numeric public evidence, never weights."""
import os,json,shutil
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);BASE=ROOT/'artifacts/component_tests/value_head_single_step_v1'
assert (BASE/'result.json').exists() and (BASE/'O_repair1/COMPLETE.json').exists() and (BASE/'Z/COMPLETE.json').exists()
OUT=BASE/'public_bundle';OUT.mkdir(exist_ok=False)
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for name in ['result.json','data_manifest.json','cpu_runner_check.json','pre_model_incidents.json','serialization_failure_audit.json','serialization_fix_cpu_check.json','stop_probe.json']:
    copy(BASE/name,OUT/name)
for phase in ['C','O','O_repair1','Z']:
    folder=BASE/phase
    for name in ['COMPLETE.json','collection_gate.json','collection_started.json','baseline_eval_started.json','baseline_carried.json','update_started.json','post_eval_started.json','actor_metrics.json','stop_probe_result.json','inference_snapshot.json','initial_actor_rank0.json','initial_actor_rank1.json','initial_critic_head.json','actual_inputs.npz','training_signals.npz','fresh_answer_signals.json']:
        if (folder/name).exists():copy(folder/name,OUT/phase/name)
    for sub,pattern in [('requests','*.json'),('episodes','*/*'),('numerical','*.json'),('update_audit','*.jsonl'),('sync_audit','*.jsonl')]:
        for path in (folder/sub).glob(pattern):
            if path.is_file() and path.suffix in ['.json','.jsonl','.png']:
                copy(path,OUT/phase/sub/path.relative_to(folder/sub))
for suffix in ['', '_repair1']:
    source=ROOT/f'state/components/value_head_single_step_v1{suffix}_receipt.json'
    receipt=json.loads(source.read_text());receipt.pop('run_token',None)
    (OUT/f'receipt{suffix}.json').write_text(json.dumps(receipt,indent=2))
    copy(ROOT/f'state/components/value_head_single_step_v1{suffix}_EXIT.json',OUT/f'EXIT{suffix}.json')
library=BASE/'C/context_libraries/cards'
for path in library.rglob('*'):
    if path.is_file() and path.suffix in ['.json','.png']:copy(path,OUT/'frozen_library'/path.relative_to(library))
assert not any(p.suffix in ['.pt','.bin','.safetensors','.ckpt'] for p in OUT.rglob('*'))
files=[p for p in OUT.rglob('*') if p.is_file()]
print(json.dumps({'files':len(files),'bytes':sum(p.stat().st_size for p in files),'scope':'Synthetic episode inputs/outputs, numeric training signals, status and source library; no weights, full runtime logs, private head or run token.'}))
