"""Independent original-clock/storage guard for a single forward-only audit."""
import os,json,time,subprocess,signal,getpass,uuid
from pathlib import Path
import psutil
ROOT=Path(os.environ['TASK_ROOT']);NAME='recorded_backend_audit_v2'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
assert P['status']=='FROZEN_CPU_PREFLIGHT_PASS'
state=ROOT/'state/components';attempt=os.environ.get('TASK_ATTEMPT','initial')
suffix='' if attempt=='initial' else '_'+attempt
receipt=state/f'{NAME}{suffix}_receipt.json'
assert not receipt.exists(),'refuse duplicate audit'
started=time.time();deadline=min(P['absolute_total_deadline'],P['preparation_started_unix']+P['max_seconds_including_preparation'])
assert started+120<deadline
original=json.loads((ROOT/'state/setup_receipt.json').read_text())
assert original['started_unix']==1792772381.9046373
def usage():
    return sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in original['paths'])
baseline=usage();limit=min(85000000000,P['storage_baseline_reference_bytes']+450000000)
assert baseline+50000000<limit
v=os.statvfs(ROOT);assert v.f_bavail*v.f_frsize>500000000
probe=ROOT/'state/.score_probe';probe.write_text('ready');assert probe.read_text()=='ready';probe.unlink()
for gpu in [3,5]:
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=44000,(gpu,free)
token=uuid.uuid4().hex
info={'started_unix':started,'preparation_started_unix':P['preparation_started_unix'],'deadline_unix':deadline,
    'original_total_deadline':P['absolute_total_deadline'],'supervisor_pid':os.getpid(),'token':token,
    'baseline_current_bytes':baseline,'baseline_before_preparation_reference':P['storage_baseline_reference_bytes'],
    'write_limit_bytes':limit,'physical_gpus':[3,5],'new_generated_tokens':0,'optimizer_steps':0}
receipt.write_text(json.dumps(info,indent=2))
env=dict(os.environ,TASK_RUN_TOKEN=token,ENTRY=str(ROOT/'src/scripts/run_recorded_backend_audit.py'),
    CUDA_VISIBLE_DEVICES='3,5',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',WANDB_DISABLED='true',RAY_USAGE_STATS_ENABLED='0',
    HF_HOME=str(ROOT/'cache/hf'),TMPDIR='/data/local/tmp/z0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',
    PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',PYGAME_HIDE_SUPPORT_PROMPT='1',
    XDG_CACHE_HOME='/data/local/tools/z0/cache/xdg',TORCH_EXTENSIONS_DIR='/data/local/tools/z0/cache/torch_extensions',
    TORCHINDUCTOR_CACHE_DIR='/data/local/tools/z0/cache/inductor',TRITON_CACHE_DIR='/data/local/tools/z0/cache/triton',
    FLASHINFER_WORKSPACE_BASE='/data/local/tools/z0/cache/flashinfer',CUDA_CACHE_PATH='/data/local/tools/z0/cache/cuda',
    SGLANG_ENABLE_JIT_DEEPGEMM='0',PATH='/data/local/envs/z0/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''),
    CUDA_HOME='/usr/local/cuda',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'src/verl'),OMP_NUM_THREADS='4')
known={};audited=set();violations=[];blocked=[getpass.getuser().lower(),ROOT.name.lower(),'directional','recorded_backend']
def discover():
    for p in psutil.process_iter(['pid','create_time','uids']):
        try:
            if p.info['uids'].real!=os.getuid() or p.info['create_time']<started-1:continue
            if p.environ().get('TASK_RUN_TOKEN')!=token:continue
            known[p.pid]=p.info['create_time']
            if p.pid not in audited:
                audited.add(p.pid);argv=p.cmdline()
                with (state/f'{NAME}{suffix}_processes.jsonl').open('a') as f:f.write(json.dumps({'pid':p.pid,'argv':argv})+'\n')
                if any(s in ' '.join(argv).lower() for s in blocked):violations.append({'pid':p.pid,'argv':argv})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
def cleanup():
    discover()
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        for pid,created in known.items():
            try:
                p=psutil.Process(pid)
                if p.create_time()==created:p.send_signal(sig)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        time.sleep(1)
reason=None;code=None;peak=baseline
try:
    with (ROOT/f'logs/ppo_smoke/{NAME}{suffix}.log').open('w') as log:
        p=subprocess.Popen(['/data/local/envs/z0/bin/python','-u','-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT/'src',start_new_session=True)
        info['worker_pid']=p.pid;receipt.write_text(json.dumps(info,indent=2))
        while p.poll() is None:
            discover();peak=max(peak,usage())
            if time.time()>=deadline:raise RuntimeError('original preparation-inclusive deadline reached')
            if peak>=limit:raise RuntimeError('additional-storage write stop reached')
            if violations:raise RuntimeError('non-neutral process argv')
            time.sleep(5)
        code=p.returncode
except BaseException as exc:reason=repr(exc)
finally:
    cleanup()
    (state/f'{NAME}{suffix}_EXIT.json').write_text(json.dumps({'exit_code':code,'reason':reason,'end_unix':time.time(),
        'deadline_unix':deadline,'peak_allocated_bytes':peak,'command_line_violations':violations},indent=2))
