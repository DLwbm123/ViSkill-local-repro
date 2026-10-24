"""Own-process setup supervisor with independent time and storage cutoffs."""
import json,os,pathlib,signal,subprocess,time
ROOT=pathlib.Path(os.environ['TASK_ROOT']);state=ROOT/'state';state.mkdir(parents=True,exist_ok=True)
paths=[str(ROOT),'/data/local/envs/z0','/data/local/tools/z0','/data/local/tmp/z0']
def usage():
 p=subprocess.run(['du','-s','-B1',*[x for x in paths if os.path.exists(x)]],capture_output=True,text=True)
 if p.returncode:raise RuntimeError('cannot verify storage usage: '+p.stderr)
 return sum(int(line.split()[0]) for line in p.stdout.splitlines())
started=time.time();budget=100000000000;cutoff=85000000000;timeout=6*3600
receipt={'started_unix':started,'pid':os.getpid(),'budget_bytes':budget,'stop_writes_at_bytes':cutoff,'setup_timeout_seconds':timeout,'paths':paths,'baseline_allocated_bytes':usage(),'gpu_indices_for_future_work':[3,4,5],'training':False}
(state/'setup_receipt.json').write_text(json.dumps(receipt,indent=2))
env=dict(os.environ,ENTRY=str(ROOT/'src/scripts/prepare_environment_worker.py'),CUDA_VISIBLE_DEVICES='')
child=subprocess.Popen(['python3','-u','-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=env,start_new_session=True)
receipt['worker_pid']=child.pid;(state/'setup_receipt.json').write_text(json.dumps(receipt,indent=2))
reason=None;peak=0
try:
 while child.poll() is None:
  n=usage();peak=max(n,peak)
  (state/'setup_resource.json').write_text(json.dumps({'time_unix':time.time(),'allocated_bytes':n,'peak_allocated_bytes':peak}))
  if n>=cutoff:reason='storage guard cutoff';break
  if time.time()-started>=timeout:reason='setup hard timeout';break
  time.sleep(5)
except BaseException as e:reason='supervisor error: '+repr(e)
if reason:
 os.killpg(child.pid,signal.SIGTERM)
 try:child.wait(timeout=10)
 except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
(state/'SETUP_EXIT.json').write_text(json.dumps({'exit_code':child.returncode,'reason':reason,'start_unix':started,'end_unix':time.time(),'peak_allocated_bytes':peak},indent=2))
