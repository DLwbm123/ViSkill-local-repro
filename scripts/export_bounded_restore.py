"""Export scores and metadata only; private parameter artifacts are excluded."""
import json,os,shutil
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);name='bf16_delta_reload_v1';base=root/f'artifacts/component_tests/{name}'
result=json.loads((base/'result.json').read_text());assert result['status'].startswith('COMPLETE')
dest=base/'public_bundle';dest.mkdir(exist_ok=False)
files=['result.json','closure.json','cpu_check.json','cpu_repair_check.json']
for phase in ['full_repair1','delta']:
    files.extend(str(p.relative_to(base)) for p in (base/phase).rglob('*') if p.is_file() and (p.suffix in ['.json','.npz']))
for relative in files:
    source=base/relative;target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for suffix in ['_receipt.json','_EXIT.json','_repair1_receipt.json','_repair1_EXIT.json']:
    s=json.loads((root/f'state/components/{name}{suffix}').read_text());s.pop('run_token',None)
    (dest/('supervisor'+suffix)).write_text(json.dumps(s,indent=2))
assert not any(p.suffix in ['.pt','.bin','.safetensors','.ckpt'] or 'PRIVATE' in p.name for p in dest.rglob('*'))
print(json.dumps({'public_files':sum(p.is_file() for p in dest.rglob('*')),'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'learned_parameter_exports':0}))
