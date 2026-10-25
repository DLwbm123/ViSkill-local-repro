"""Independent original-clock/space/own-process guard for one official evaluation."""
import os,json,time,subprocess,signal,uuid,getpass
from pathlib import Path
import psutil
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='released_frozenlake_eval_v1';P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text());OUT=ROOT/f'artifacts/reproduction/{NAME}';STATE=ROOT/'state/reproduction';STATE.mkdir(exist_ok=True)
assert P['status']=='FROZEN_CPU_READY' and json.loads((OUT/'cpu_check.json').read_text())['status']=='CPU_READY'
receipt=STATE/f'{NAME}_receipt.json';assert not receipt.exists();started=time.time();deadline=P['deadline_unix'];assert started+120<deadline
assert json.loads((ROOT/'state/autonomous_20261024T153827.json').read_text())['status']=='CLOSED'
roots=[ROOT,Path('/data/envs/z0'),Path('/data/tools/z0'),Path('/data/tmp/z0')]
def usage():return sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in roots)
initial=usage();limit=min(P['project_write_stop_bytes'],P['storage_baseline_bytes']+P['write_stop_additional_bytes']);assert initial+100000000<limit
v=os.statvfs(ROOT);assert v.f_bavail*v.f_frsize>P['maximum_additional_bytes'];probe=STATE/'.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
free={g:int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for g in [3,5]};assert min(free.values())>=60000
run_token=uuid.uuid4().hex;info={'supervisor_pid':os.getpid(),'start_unix':started,'preparation_started_unix':P['preparation_started_unix'],'deadline_unix':deadline,'run_token':run_token,'initial_four_root_bytes':initial,'write_limit_bytes':limit,'initial_GPU_free_MiB':free,'source_base_commit':P['upstream_commit']};receipt.write_text(json.dumps(info,indent=2))
env=dict(os.environ,ENTRY=str(SRC/'scripts/run_released_eval.py'),TASK_RUN_TOKEN=run_token,CUDA_VISIBLE_DEVICES='3,5',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(SRC)+':'+str(SRC/'verl'),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',RAY_USAGE_STATS_ENABLED='0',TOKENIZERS_PARALLELISM='false',HF_HOME=str(ROOT/'cache/hf'),TMPDIR='/data/tmp/z0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PATH='/data/envs/z0/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda',XDG_CACHE_HOME='/data/tools/z0/cache/xdg',TORCH_EXTENSIONS_DIR='/data/tools/z0/cache/torch_extensions',TORCHINDUCTOR_CACHE_DIR='/data/tools/z0/cache/inductor',TRITON_CACHE_DIR='/data/tools/z0/cache/triton',FLASHINFER_WORKSPACE_BASE='/data/tools/z0/cache/flashinfer',CUDA_CACHE_PATH='/data/tools/z0/cache/cuda',SGLANG_ENABLE_JIT_DEEPGEMM='0',OMP_NUM_THREADS='4')
env.pop('TASK_CAMPAIGN_TOKEN',None);known={};audited=set();violations=[];blocked=[getpass.getuser().lower(),ROOT.name.lower(),'released_frozenlake','paper_repro'];peak=initial;reason=None;exit_code=None

def discover():
 for p in psutil.process_iter(['pid','create_time','uids']):
  try:
   if p.info['uids'].real==os.getuid() and p.info['create_time']>=started-1 and p.environ().get('TASK_RUN_TOKEN')==run_token:
    known[p.pid]=p.info['create_time']
    if p.pid not in audited:
     audited.add(p.pid);argv=p.cmdline()
     if any(word in ' '.join(argv).lower() for word in blocked):violations.append({'pid':p.pid,'argv':argv})
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass

def cleanup():
 discover()
 for sig,delay in [(signal.SIGTERM,2),(signal.SIGKILL,1)]:
  for pid,created in known.items():
   try:
    p=psutil.Process(pid)
    if p.create_time()==created:p.send_signal(sig)
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  time.sleep(delay)
try:
 with (OUT/'worker.log').open('w') as log:
  child=subprocess.Popen(['/data/envs/z0/bin/python','-u','-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=env,cwd=SRC,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while child.poll() is None:
   discover();peak=max(peak,usage())
   if peak>=limit or time.time()>=deadline-15:raise RuntimeError('original clock/storage hard stop')
   if violations:raise RuntimeError('non-neutral command line')
   time.sleep(5)
  exit_code=child.returncode
  if exit_code:raise RuntimeError(f'worker exit{exit_code}; preserve actual calls, no replay')
except BaseException as exc:reason=repr(exc)
finally:
 cleanup();end={'exit_code':exit_code,'reason':reason,'end_unix':time.time(),'deadline_unix':deadline,'peak_four_root_bytes':peak,'command_line_violations':violations};(STATE/f'{NAME}_EXIT.json').write_text(json.dumps(end,indent=2))
