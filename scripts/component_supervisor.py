"""Bounded, isolated original-code component audit; no training library writes."""
import json,os,pathlib,signal,subprocess,time
root=pathlib.Path(os.environ['TASK_ROOT']);state=root/'state/components'
state.mkdir(parents=True,exist_ok=True)
receipt_path=state/'original_v1_receipt.json'
if receipt_path.exists():raise RuntimeError('original component attempt already exists')
paths=[str(root),'/data/local/envs/z0','/data/local/tools/z0','/data/local/tmp/z0']
def usage():
 p=subprocess.run(['du','-s','-B1',*paths],capture_output=True,text=True,check=True)
 return sum(int(s.split()[0]) for s in p.stdout.splitlines())
if usage()>=85000000000:raise RuntimeError('storage guard')
free=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<20000:raise RuntimeError('insufficient GPU4 memory')
start=time.time();deadline=start+1800
args=['-v','-s',str(root/'src/tests'),'--basetemp',str(root/'artifacts/component_tests/pytest_tmp'),'--junitxml',str(root/'artifacts/component_tests/original_v1/junit.xml')]
env=dict(os.environ,TASK_ARGS=json.dumps(args),CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',RAY_USAGE_STATS_ENABLED='0',HF_HOME=str(root/'cache/hf'),TMPDIR='/data/local/tmp/z0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
p=subprocess.Popen(['/data/local/envs/z0/bin/python','-u','-c','import os,json,pytest;raise SystemExit(pytest.main(json.loads(os.environ["TASK_ARGS"])))'],env=env,start_new_session=True)
receipt_path.write_text(json.dumps({'start_unix':start,'deadline_unix':deadline,'supervisor_pid':os.getpid(),'worker_pid':p.pid,'physical_gpu':4,'fixture_only':True,'training':False},indent=2))
reason=None;peak=0
try:
 while p.poll() is None:
  size=usage();peak=max(peak,size)
  (state/'original_v1_resource.json').write_text(json.dumps({'time_unix':time.time(),'allocated_bytes':size,'peak_bytes':peak}))
  if size>=85000000000 or time.time()>=deadline:reason='storage/time guard';break
  time.sleep(5)
except BaseException as e:reason=repr(e)
if reason:
 os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
(state/'original_v1_EXIT.json').write_text(json.dumps({'exit_code':p.returncode,'reason':reason,'original_start_unix':start,'end_unix':time.time(),'peak_allocated_bytes':peak},indent=2))
