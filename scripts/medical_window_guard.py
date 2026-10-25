"""Enforce one fixed autonomous window across all its experiments and descendants."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

ROOT = Path(os.environ['TASK_ROOT'])
STATE = ROOT / 'state/medical_window_12h_20261025'
P = json.loads((STATE / 'protocol.json').read_text())
TOKEN = os.environ['TASK_WINDOW_TOKEN']
roots = [Path(p) for p in json.loads(os.environ['TASK_BUDGET_ROOTS'])]
known = {}
assert not (STATE / 'guard_receipt.json').exists()
(STATE / 'guard_receipt.json').write_text(json.dumps({'pid': os.getpid(), 'ctime': psutil.Process().create_time(), 'start_unix': time.time(), 'deadline_unix': P['deadline_unix'], 'window_token': TOKEN}, indent=2))


def census():
    alive = []
    for proc in psutil.process_iter(['pid', 'status', 'uids', 'create_time']):
        try:
            if proc.pid != os.getpid() and proc.info['uids'].real == os.getuid() and proc.environ().get('TASK_WINDOW_TOKEN') == TOKEN:
                known[proc.pid] = proc.info['create_time']
                if proc.info['status'] != psutil.STATUS_ZOMBIE:
                    alive.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return alive


def stop_owned():
    census()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, created in known.items():
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == created and proc.status() != psutil.STATUS_ZOMBIE:
                    proc.send_signal(sig)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(3)


reason = None
try:
    while True:
        now = time.time()
        used = sum(int(subprocess.check_output(['du', '-s', '-B1', '.'], cwd=p, text=True, timeout=30).split()[0]) for p in roots)
        alive = census()
        if now >= P['cooperative_stop_unix']:
            (STATE / 'STOP_NEW_WORK.json').write_text(json.dumps({'reason': 'window closing', 'time_unix': now}))
        if now >= P['deadline_unix'] - 30 or used - P['four_root_baseline_bytes'] >= P['write_stop_incremental_bytes']:
            reason = 'deadline' if now >= P['deadline_unix'] - 30 else 'storage_pre_stop'
            break
        if (STATE / 'CLOSE_WINDOW.json').exists():
            reason = 'explicit_window_close'
            break
        (STATE / 'guard_status.json').write_text(json.dumps({'status': 'GUARDING', 'time_unix': now, 'incremental_bytes': used - P['four_root_baseline_bytes'], 'experiment_processes_alive': len(alive), 'note': 'The guardian is not an experiment; idle time requires agent action when viable work remains.'}))
        time.sleep(20)
finally:
    stop_owned()
    (STATE / 'guard_EXIT.json').write_text(json.dumps({'reason': reason, 'end_unix': time.time(), 'remaining_owned': len(census())}, indent=2))
