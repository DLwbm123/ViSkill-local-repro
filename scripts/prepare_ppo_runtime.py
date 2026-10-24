"""Install the official inference runtime in the independent environment, no GPU."""
import importlib.metadata as md
import json,os,pathlib,subprocess,time
import torch
root=pathlib.Path(os.environ['TASK_ROOT'])
assert torch.__version__=='2.8.0+cu128' and torch.compiled_with_cxx11_abi()
constraints=root/'environment/runtime_constraints.txt'
critical=['torch','torchvision','transformers','numpy','ray','tensordict','peft','datasets','trl','gymnasium','pillow','torchdata','accelerate']
constraints.write_text(''.join(f'{name}=={md.version(name)}\n' for name in critical))
os.environ.update(PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
wheel='https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
variant=os.environ.get('TASK_RUNTIME_VARIANT','blackwell')
assert variant in ['blackwell','srt']
record_path=root/'state'/f'ppo_runtime_install_{variant}.json'
args=['install','--constraint',str(constraints),f'sglang[{variant}]==0.5.2','torch_memory_saver==0.0.8',wheel]
env=dict(os.environ,TASK_ARGS=json.dumps(args));start=time.time()
p=subprocess.run(['/data/local/envs/z0/bin/python','-u','-c','import os,json;from pip._internal.cli.main import main;raise SystemExit(main(json.loads(os.environ["TASK_ARGS"])))'],env=env)
record={'args':args,'exit_code':p.returncode,'start_unix':start,'end_unix':time.time(),'training':False}
record_path.write_text(json.dumps(record,indent=2))
if p.returncode:raise SystemExit(p.returncode)
p=subprocess.run(['/data/local/envs/z0/bin/python','-m','pip','check']);record['pip_check_exit']=p.returncode
with (root/'environment/pip_freeze_runtime.txt').open('w') as h:subprocess.run(['/data/local/envs/z0/bin/python','-m','pip','freeze'],stdout=h,check=True)
record_path.write_text(json.dumps(record,indent=2))
raise SystemExit(p.returncode)
