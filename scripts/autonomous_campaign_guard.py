"""Enforce a user-authorized campaign clock and cumulative storage independently."""
import os,json,time,signal,subprocess
from pathlib import Path
import psutil
state=Path(os.environ['CAMPAIGN_STATE']);p=json.loads(state.read_text());started=p['start_unix']
known={};reason=None;peak=p['baseline_allocated_bytes'];own=os.getpid()
def usage():return sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=r,text=True).split()[0]) for r in p['roots'])
def discover():
    for proc in psutil.process_iter(['pid','uids','create_time']):
        try:
            if proc.pid==own or proc.info['uids'].real!=os.getuid() or proc.info['create_time']<started-1:continue
            if proc.environ().get('TASK_CAMPAIGN_TOKEN')==p['token']:known[proc.pid]=proc.info['create_time']
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
while True:
    discover();current=usage();peak=max(peak,current)
    p.update(last_check_unix=time.time(),current_allocated_bytes=current,peak_allocated_bytes=peak,
             additional_bytes=max(0,current-p['baseline_allocated_bytes']),guard_pid=own)
    if time.time()>=p['deadline_unix']:reason='12_hour_campaign_deadline'
    if current-p['baseline_allocated_bytes']>=p['additional_write_stop_bytes']:reason='cumulative_additional_storage_stop'
    if current>=p['project_write_stop_bytes']:reason='project_conservative_storage_stop'
    p['status']='STOPPING' if reason else 'ACTIVE';p['stop_reason']=reason
    tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(p,indent=2)+'\n');tmp.replace(state)
    if reason:break
    time.sleep(20)
for sig,delay in [(signal.SIGTERM,30),(signal.SIGKILL,1)]:
    discover()
    for pid,created in known.items():
        try:
            proc=psutil.Process(pid)
            if proc.create_time()==created:proc.send_signal(sig)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    time.sleep(delay)
p.update(status='CLOSED',end_unix=time.time(),stop_reason=reason,registered_owned_processes=len(known))
state.write_text(json.dumps(p,indent=2)+'\n')
