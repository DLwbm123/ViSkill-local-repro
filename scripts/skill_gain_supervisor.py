"""Bounded independent neutral-process guard; cleanup is restricted to this run token."""
import os,json,time,subprocess,signal,uuid,getpass
from pathlib import Path
import psutil
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='skill_gain_384_v1';OUT=ROOT/'artifacts/reproduction'/NAME;STATE=ROOT/'state/reproduction';P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text());assert P['status']=='FROZEN_CPU_READY' and json.loads((OUT/'cpu_check.json').read_text())['status']=='CPU_READY'
roots=[ROOT,Path('/data/envs/z0'),Path('/data/tools/z0'),Path('/data/tmp/z0')]
def usage():return sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in roots)
start=time.time();assert start+600<P['cooperative_stop_unix'];receipt=STATE/f'{NAME}_receipt.json';assert not receipt.exists();baseline=P['four_root_baseline_bytes'];initial=usage();hard=baseline+P['maximum_incremental_bytes'];limit=baseline+P['write_stop_incremental_bytes'];assert initial<limit
assert psutil.disk_usage(ROOT).free>10000000000 and psutil.virtual_memory().available>64000000000
probe=STATE/'.skill_gain_probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink();free=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=60000
token=uuid.uuid4().hex;r={'supervisor_pid':os.getpid(),'supervisor_ctime':psutil.Process().create_time(),'start_unix':start,'run_token':token,'initial_four_root_bytes':initial,'baseline_bytes':baseline,'write_stop_bytes':limit,'strict_hard_bytes':hard,'deadline_unix':P['deadline_unix'],'GPU_free_MiB':free};receipt.write_text(json.dumps(r,indent=2));known={};violations=[];peak=initial;reason=None;exitcode=None

def census():
 for p in psutil.process_iter(['pid','create_time','uids']):
  try:
   if p.info['uids'].real==os.getuid() and p.info['create_time']>=start-1 and p.environ().get('TASK_RUN_TOKEN')==token:
    known[p.pid]=p.info['create_time']
    if any(x in ' '.join(p.cmdline()).lower() for x in [getpass.getuser().lower(),ROOT.name.lower(),'skill_gain']):violations.append(p.pid)
  except(psutil.NoSuchProcess,psutil.AccessDenied):pass

def cleanup():
 census()
 for sig,delay in [(signal.SIGTERM,2),(signal.SIGKILL,1)]:
  for pid,ct in known.items():
   try:
    p=psutil.Process(pid)
    if p.create_time()==ct:p.send_signal(sig)
   except(psutil.NoSuchProcess,psutil.AccessDenied):pass
  time.sleep(delay)
env=dict(os.environ,TASK_RUN_TOKEN=token,ENTRY=str(SRC/'scripts/run_skill_gain_384.py'),CUDA_VISIBLE_DEVICES='3',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(SRC)+':'+str(SRC/'verl'),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',RAY_USAGE_STATS_ENABLED='0',TOKENIZERS_PARALLELISM='false',HF_HOME=str(ROOT/'cache/hf'),TMPDIR='/data/tmp/z0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PATH='/data/envs/z0/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda',XDG_CACHE_HOME='/data/tools/z0/cache/xdg',TORCH_EXTENSIONS_DIR='/data/tools/z0/cache/torch_extensions',TORCHINDUCTOR_CACHE_DIR='/data/tools/z0/cache/inductor',TRITON_CACHE_DIR='/data/tools/z0/cache/triton',FLASHINFER_WORKSPACE_BASE='/data/tools/z0/cache/flashinfer',CUDA_CACHE_PATH='/data/tools/z0/cache/cuda',OMP_NUM_THREADS='4')
try:
 with (OUT/'worker.log').open('w') as log:
  child=subprocess.Popen(['/data/envs/z0/bin/python','-u','-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=env,cwd=SRC,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while child.poll() is None:
   census();peak=max(peak,usage());now=time.time();(STATE/f'{NAME}_status.json').write_text(json.dumps({'status':'RUNNING','updated_unix':now,'peak_four_root_bytes':peak,'incremental_bytes':peak-baseline,'owned_seen':len(known),'argv_violations':len(set(violations))}))
   if now>=P['cooperative_stop_unix']:(OUT/'STOP_REQUESTED.json').write_text(json.dumps({'reason':'cooperative deadline'}))
   if now>=P['deadline_unix']-15 or peak>=limit or peak>=hard or violations:raise RuntimeError('hard clock/storage/argv stop')
   time.sleep(5)
  exitcode=child.returncode
  if exitcode:raise RuntimeError('worker failed; retain all records and no blind replay')
  assert (OUT/'WORKER_COMPLETED.json').exists()
except BaseException as exc:reason=repr(exc)
finally:
 cleanup();(STATE/f'{NAME}_EXIT.json').write_text(json.dumps({'exit_code':exitcode,'reason':reason,'end_unix':time.time(),'peak_four_root_bytes':peak,'new_peak_bytes':peak-baseline,'argv_violations':sorted(set(violations)),'deadline_unix':P['deadline_unix']},indent=2))
