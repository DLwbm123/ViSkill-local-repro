"""Own-tree hard timeout, disk guard and two sequential isolated Ray phases."""
import json, os, pathlib, signal, subprocess, time, uuid
import psutil
ROOT=pathlib.Path(os.environ['TASK_ROOT']); STATE=ROOT/'state/components'; NAME='skill_capacity_v1'
protocol=json.loads((ROOT/'src/configs/smoke/skill_capacity_v1.json').read_text())
original=json.loads((ROOT/'state/setup_receipt.json').read_text())
assert original['started_unix']==protocol['original_total_clock_start']
receipt=STATE/f'{NAME}_receipt.json'
if receipt.exists():raise RuntimeError('receipt exists; refuse duplicate')
start=time.time();deadline=min(protocol['absolute_total_deadline'],start+protocol['max_attempt_seconds'])
def usage():
 return sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=folder,text=True).split()[0]) for folder in original['paths'])
baseline=usage();limit=min(protocol['project_write_stop_bytes'],baseline+protocol['maximum_additional_bytes'])
assert start<deadline and baseline+500000000<limit
for gpu in [4]:
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 assert free>=28000, (gpu,free)
probe=ROOT/'state/.ppo_probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
token=uuid.uuid4().hex
ENV=dict(os.environ,ENTRY=str(ROOT/'src/scripts/run_skill_capacity.py'),TASK_RUN_TOKEN=token,
 CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',RAY_USAGE_STATS_ENABLED='0',
 HF_HOME=str(ROOT/'cache/hf'),TMPDIR='/data/local/tmp/z0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
 XDG_CACHE_HOME='/data/local/tools/z0/cache/xdg',TORCH_EXTENSIONS_DIR='/data/local/tools/z0/cache/torch_extensions',TRITON_CACHE_DIR='/data/local/tools/z0/cache/triton',
 FLASHINFER_WORKSPACE_BASE='/data/local/tools/z0/cache/flashinfer',CUDA_CACHE_PATH='/data/local/tools/z0/cache/cuda',SGLANG_ENABLE_JIT_DEEPGEMM='0',
 PATH='/data/local/envs/z0/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda',
 PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'src/verl'),OMP_NUM_THREADS='4')
receipt.write_text(json.dumps({'start_unix':start,'original_start_unix':original['started_unix'],'deadline_unix':deadline,'supervisor_pid':os.getpid(),'run_token':token,'baseline_bytes':baseline,'write_limit_bytes':limit,'physical_gpus':[4]},indent=2))
known={};audited=set();peak=baseline

def discover():
 for process in psutil.process_iter(['pid','create_time','uids']):
  try:
   if process.info['uids'].real==os.getuid() and process.info['create_time']>=start-1 and process.environ().get('TASK_RUN_TOKEN')==token:
    known[process.pid]=process.info['create_time']
    if process.pid not in audited:
     argv=process.cmdline(); audited.add(process.pid)
     with (STATE/f'{NAME}_processes.jsonl').open('a') as f:f.write(json.dumps({'pid':process.pid,'argv':argv})+'\n')
     assert not any(word in ' '.join(argv).lower() for word in ['local-user','viskill','directional']), 'non-neutral child argv'
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

reason=None;exits={}
try:
 for phase in ['capacity']:
  env=dict(ENV,TASK_PHASE=phase)
  logfile=ROOT/f'logs/ppo_smoke/{NAME}_{phase}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
  with logfile.open('w') as log:
   p=subprocess.Popen(['/data/local/envs/z0/bin/python','-u','-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT/'src')
   while p.poll() is None:
    discover();size=usage();peak=max(peak,size)
    if size>=limit or time.time()>=deadline:raise RuntimeError('storage/time guard')
    time.sleep(5)
   exits[phase]=p.returncode
  cleanup()
  if p.returncode:raise RuntimeError(f'{phase} exited {p.returncode}')
except BaseException as exc:reason=repr(exc)
finally:
 try:cleanup()
 except BaseException as exc:reason=str(reason)+'; cleanup '+repr(exc)
 (STATE/f'{NAME}_EXIT.json').write_text(json.dumps({'exits':exits,'reason':reason,'end_unix':time.time(),'peak_allocated_bytes':peak,'deadline_unix':deadline},indent=2))
