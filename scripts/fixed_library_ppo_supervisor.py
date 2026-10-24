"""Own-tree independent deadline and write guard for the fixed-memory pair."""
import getpass,json,os,signal,subprocess,time,uuid
from pathlib import Path
import psutil
ROOT=Path(os.environ['TASK_ROOT']);NAME='fixed_library_ppo_v1';STATE=ROOT/'state/components'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());assert P['status']=='FROZEN_CPU_PREFLIGHT_PASS'
original=json.loads((ROOT/'state/setup_receipt.json').read_text());assert original['started_unix']==P['original_total_clock_start']
receipt=STATE/f'{NAME}_receipt.json';assert not receipt.exists(),'refuse duplicate pair'
started=time.time();deadline=min(P['absolute_total_deadline'],P['preparation_started_unix']+P['max_seconds_pair']);assert started+300<deadline
def usage():return sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in original['paths'])
baseline=usage();limit=min(P['project_write_stop_bytes'],baseline+P['maximum_additional_bytes_pair']-200000000);assert baseline+300000000<limit
assert P['maximum_additional_bytes_pair']<50000000000
probe=ROOT/'state/.fixed_probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
token=uuid.uuid4().hex
info={'start_unix':started,'preparation_start_unix':P['preparation_started_unix'],'deadline_unix':deadline,'original_total_deadline':P['absolute_total_deadline'],'supervisor_pid':os.getpid(),'run_token':token,'baseline_bytes':baseline,'write_limit_bytes':limit,'maximum_additional_bytes':P['maximum_additional_bytes_pair'],'physical_gpus':[3,5],'checkpoint_save':False,'arms':{}}
receipt.write_text(json.dumps(info,indent=2))
env=dict(os.environ,ENTRY=str(ROOT/'src/scripts/run_fixed_library_ppo.py'),TASK_RUN_TOKEN=token,
 CUDA_VISIBLE_DEVICES='3,5',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',RAY_USAGE_STATS_ENABLED='0',
 HF_HOME=str(ROOT/'cache/hf'),TMPDIR='/data/local/tmp/z0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
 XDG_CACHE_HOME='/data/local/tools/z0/cache/xdg',TORCH_EXTENSIONS_DIR='/data/local/tools/z0/cache/torch_extensions',TORCHINDUCTOR_CACHE_DIR='/data/local/tools/z0/cache/inductor',TRITON_CACHE_DIR='/data/local/tools/z0/cache/triton',FLASHINFER_WORKSPACE_BASE='/data/local/tools/z0/cache/flashinfer',CUDA_CACHE_PATH='/data/local/tools/z0/cache/cuda',SGLANG_ENABLE_JIT_DEEPGEMM='0',PATH='/data/local/envs/z0/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'src/verl'),OMP_NUM_THREADS='4')
known={};audited=set();violations=[];blocked=[getpass.getuser().lower(),ROOT.name.lower(),'directional']
def discover():
 for p in psutil.process_iter(['pid','create_time','uids']):
  try:
   if p.info['uids'].real==os.getuid() and p.info['create_time']>=started-1 and p.environ().get('TASK_RUN_TOKEN')==token:
    known[p.pid]=p.info['create_time']
    if p.pid not in audited:
     audited.add(p.pid);argv=p.cmdline()
     with (STATE/f'{NAME}_processes.jsonl').open('a') as f:f.write(json.dumps({'pid':p.pid,'argv':argv})+'\n')
     if any(w in ' '.join(argv).lower() for w in blocked):violations.append({'pid':p.pid,'argv':argv})
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass

def cleanup():
 discover()
 for sig in [signal.SIGTERM,signal.SIGKILL]:
  for pid,created in known.items():
   try:
    p=psutil.Process(pid)
    if p.create_time()==created:p.send_signal(sig)
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  time.sleep(2)
reason=None;exits={};peak=baseline
try:
 for arm in ['A','B']:
  arm_start=time.time();arm_deadline=min(deadline,arm_start+P['max_seconds_per_arm'])
  info['arms'][arm]={'start_unix':arm_start,'deadline_unix':arm_deadline};receipt.write_text(json.dumps(info,indent=2))
  for gpu in [3,5]:
   free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   assert free>=44000,f'insufficient free memory GPU{gpu}: {free}'
  logfile=ROOT/f'logs/ppo_smoke/{NAME}_{arm}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
  with logfile.open('w') as log:
   process=subprocess.Popen(['/data/local/envs/z0/bin/python','-u','-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=dict(env,TASK_PHASE=arm),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT/'src')
   while process.poll() is None:
    discover();peak=max(peak,usage())
    if peak>=limit or time.time()>=arm_deadline:raise RuntimeError('storage/time guard')
    if violations:raise RuntimeError('non-neutral process command line')
    time.sleep(5)
   exits[arm]=process.returncode
  cleanup();info['arms'][arm]['exit_code']=process.returncode;receipt.write_text(json.dumps(info,indent=2))
  if process.returncode:raise RuntimeError(f'{arm} exited {process.returncode}; preserve completed calls/updates, no replay')
 summary_env=dict(env,ENTRY=str(ROOT/'src/scripts/summarize_fixed_library_ppo.py'))
 with (ROOT/f'logs/ppo_smoke/{NAME}_summary.log').open('w') as log:
  subprocess.run(['/data/local/envs/z0/bin/python','-u','-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=summary_env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,min(60,int(deadline-time.time()))),cwd=ROOT/'src')
except BaseException as exc:reason=repr(exc)
finally:
 cleanup();(STATE/f'{NAME}_EXIT.json').write_text(json.dumps({'exits':exits,'reason':reason,'end_unix':time.time(),'peak_allocated_bytes':peak,'deadline_unix':deadline,'command_line_violations':violations},indent=2))
