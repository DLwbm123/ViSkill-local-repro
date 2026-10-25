"""Independent same-clock, storage and own-process guard for released PPO training."""
import os,json,time,subprocess,signal,uuid,getpass
from pathlib import Path
import psutil
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='official_frozenlake_ppo_v1';P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text());OUT=ROOT/f'artifacts/reproduction/{NAME}';STATE=ROOT/'state/reproduction'
assert P['status']=='FROZEN_CPU_READY' and json.loads((OUT/'cpu_check.json').read_text())['status']=='CPU_READY'
assert P['storage_exception_directly_authorized_by_user'] and P['strict_maximum_additional_bytes']==150000000000 and P['four_root_stop_bytes']==210000000000
receipt=STATE/f'{NAME}_receipt.json';assert not receipt.exists();started=time.time();deadline=P['deadline_unix'];assert started+600<P['cooperative_stop_unix']
roots=[ROOT,Path('/data/envs/z0'),Path('/data/tools/z0'),Path('/data/tmp/z0')]
def usage():return sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in roots)
initial=usage();baseline=P['storage_baseline_four_root_bytes'];limit=min(P['four_root_stop_bytes'],baseline+P['additional_write_stop_bytes']);hard=min(P['four_root_stop_bytes'],baseline+P['strict_maximum_additional_bytes']);assert initial+P['pre_checkpoint_expected_complete_payload_reserve_bytes']<limit
free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize;assert free_bytes>P['strict_maximum_additional_bytes'];probe=STATE/'.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
free={g:int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for g in [3,5]};assert min(free.values())>=60000
run_token=uuid.uuid4().hex;info={'supervisor_pid':os.getpid(),'supervisor_ctime':psutil.Process().create_time(),'start_unix':started,'preparation_started_unix':P['preparation_started_unix'],'deadline_unix':deadline,'cooperative_stop_unix':P['cooperative_stop_unix'],'run_token':run_token,'initial_four_root_bytes':initial,'baseline_four_root_bytes':baseline,'write_limit_bytes':limit,'hard_limit_bytes_strict':hard,'initial_GPU_free_MiB':free,'filesystem':subprocess.check_output(['findmnt','-T',str(ROOT),'-n','-o','FSTYPE,TARGET'],text=True).strip(),'free_filesystem_bytes':free_bytes};receipt.write_text(json.dumps(info,indent=2))
env=dict(os.environ,ENTRY=str(SRC/'scripts/run_official_train.py'),TASK_RUN_TOKEN=run_token,CUDA_VISIBLE_DEVICES='3,5',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(SRC)+':'+str(SRC/'verl'),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',RAY_USAGE_STATS_ENABLED='0',TOKENIZERS_PARALLELISM='false',HF_HOME=str(ROOT/'cache/hf'),TMPDIR='/data/tmp/z0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PATH='/data/envs/z0/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda',XDG_CACHE_HOME='/data/tools/z0/cache/xdg',TORCH_EXTENSIONS_DIR='/data/tools/z0/cache/torch_extensions',TORCHINDUCTOR_CACHE_DIR='/data/tools/z0/cache/inductor',TRITON_CACHE_DIR='/data/tools/z0/cache/triton',FLASHINFER_WORKSPACE_BASE='/data/tools/z0/cache/flashinfer',CUDA_CACHE_PATH='/data/tools/z0/cache/cuda',SGLANG_ENABLE_JIT_DEEPGEMM='0',OMP_NUM_THREADS='4')
env.pop('TASK_CAMPAIGN_TOKEN',None);known={};audited=set();violations=[];blocked=[getpass.getuser().lower(),ROOT.name.lower(),'official_frozenlake','paper_repro'];peak=initial;reason=None;exit_code=None

def discover():
    for p in psutil.process_iter(['pid','create_time','uids']):
        try:
            if p.info['uids'].real==os.getuid() and p.info['create_time']>=started-1 and p.environ().get('TASK_RUN_TOKEN')==run_token:
                known[p.pid]=p.info['create_time']
                if p.pid not in audited:
                    audited.add(p.pid)
                    if any(word in ' '.join(p.cmdline()).lower() for word in blocked):violations.append(p.pid)
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
            discover();peak=max(peak,usage());now=time.time()
            status={'updated_unix':now,'status':'RUNNING','four_root_bytes':peak,'deadline_unix':deadline,'owned_processes_seen':len(known),'argv_violation_count':len(violations)};(STATE/f'{NAME}_status.json').write_text(json.dumps(status))
            if now>=P['cooperative_stop_unix'] and not (OUT/'STOP_REQUESTED.json').exists():(OUT/'STOP_REQUESTED.json').write_text(json.dumps({'reason':'24h original clock cooperative prestop','requested_unix':now,'deadline_unix':deadline}))
            if peak>=limit or peak>=hard:raise RuntimeError('storage write stop; preserve latest completed native checkpoint')
            if now>=deadline-15:raise RuntimeError('original24h hard clock')
            if violations:raise RuntimeError('non-neutral command line')
            time.sleep(5)
        exit_code=child.returncode
        if exit_code:raise RuntimeError(f'worker exit{exit_code}; retain dispatches and latest checkpoint, no blind replay')
except BaseException as exc:reason=repr(exc)
finally:
    cleanup();end={'exit_code':exit_code,'reason':reason,'end_unix':time.time(),'deadline_unix':deadline,'peak_four_root_bytes':peak,'new_peak_bytes_from_original_baseline':peak-baseline,'command_line_violations':violations};(STATE/f'{NAME}_EXIT.json').write_text(json.dumps(end,indent=2))
