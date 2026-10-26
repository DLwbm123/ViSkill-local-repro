"""Bounded preparation using the project's existing setup-supervisor pattern."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import psutil

ROOT = Path(os.environ['TASK_ROOT'])
P = json.loads((ROOT / 'plan.json').read_text())
TOKEN = uuid.uuid4().hex
STATE = ROOT / 'state'
STATE.mkdir(exist_ok=True)
ROOTS = [Path(p) for p in P['budget_roots']]


def usage():
    return sum(int(subprocess.check_output(['du', '-s', '-B1', '.'], cwd=p, text=True).split()[0]) for p in ROOTS)


start = time.time()
receipt = dict(start_unix=start, supervisor_pid=os.getpid(), token=TOKEN,
               deadline_unix=P['deadline_unix'], limit_bytes=50_000_000_000,
               write_stop_bytes=45_000_000_000, baseline_bytes=P['baseline_bytes'])
(STATE / 'receipt.json').write_text(json.dumps(receipt, indent=2))
env = dict(os.environ, TASK_RUN_TOKEN=TOKEN, ENTRY=str(ROOT / 'src/prepare_medcta.py'))
peak = usage()
reason = None
violations = []
with (STATE / 'worker.log').open('w') as log:
    child = subprocess.Popen(['python3', '-u', '-c', 'import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],
                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt['worker_pid'] = child.pid
    (STATE / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    try:
        while child.poll() is None:
            n = usage()
            peak = max(peak, n)
            for proc in psutil.process_iter(['pid', 'uids', 'cmdline']):
                try:
                    if proc.info['uids'].real != os.getuid() or proc.environ().get('TASK_RUN_TOKEN') != TOKEN:
                        continue
                    command = ' '.join(proc.info['cmdline'] or []).lower()
                    if any(word in command for word in tuple(term for term in (os.environ.get('PRIVATE_ARGV_TERM'), 'viskill', 'medcta', 'llava') if term)):
                        violations.append(proc.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            (STATE / 'resource.json').write_text(json.dumps(dict(updated_unix=time.time(), allocated_bytes=n,
                incremental_bytes=n-P['baseline_bytes'], peak_bytes=peak, argv_violations=sorted(set(violations)))))
            if n-P['baseline_bytes'] >= 45_000_000_000:
                raise RuntimeError('storage pre-stop')
            if time.time() >= P['deadline_unix'] or violations:
                raise RuntimeError('clock or visible-command guard')
            time.sleep(2)
    except BaseException as exc:
        reason = repr(exc)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(child.pid, sig)
                child.wait(timeout=5)
                break
            except ProcessLookupError:
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        child.wait()
        peak = max(peak, usage())
        result = dict(exit_code=child.returncode, reason=reason, end_unix=time.time(),
            allocated_peak_bytes=peak, incremental_peak_bytes=peak-P['baseline_bytes'],
            argv_violations=sorted(set(violations)))
        if not reason and child.returncode == 0:
            result['ready_receipt_exists'] = (ROOT / 'READY.json').is_file()
        (STATE / 'EXIT.json').write_text(json.dumps(result, indent=2))
