"""Resolve observed FA cute/CUTLASS API conflict without altering core model stack."""
import json,os,pathlib,subprocess,time
root=pathlib.Path(os.environ['TASK_ROOT']);start=time.time()
args=['install','--constraint',str(root/'environment/runtime_constraints.txt'),'nvidia-cudnn-frontend==1.13.0','nvidia-cutlass-dsl==4.2.0']
env=dict(os.environ,TASK_ARGS=json.dumps(args),PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
p=subprocess.run(['/data/local/envs/z0/bin/python','-u','-c','import os,json;from pip._internal.cli.main import main;raise SystemExit(main(json.loads(os.environ["TASK_ARGS"])))'],env=env)
record={'args':args,'start_unix':start,'end_unix':time.time(),'exit_code':p.returncode,'training':False}
(root/'state/ppo_runtime_pins_install.json').write_text(json.dumps(record,indent=2))
if p.returncode:raise SystemExit(p.returncode)
p=subprocess.run(['/data/local/envs/z0/bin/python','-m','pip','check']);record['pip_check_exit']=p.returncode
with (root/'environment/pip_freeze_runtime_pins.txt').open('w') as f:subprocess.run(['/data/local/envs/z0/bin/python','-m','pip','freeze'],stdout=f,check=True)
(root/'state/ppo_runtime_pins_install.json').write_text(json.dumps(record,indent=2));raise SystemExit(p.returncode)
