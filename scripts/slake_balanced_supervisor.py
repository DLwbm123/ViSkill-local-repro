"""Independent limits and token-scoped cleanup for one final paired-reference round."""
import getpass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'slake_balanced_final_v1'
OUT = ROOT / 'artifacts/medical' / NAME
STATE = ROOT / 'state/reproduction'
P = json.loads((SRC / 'configs/reproduction' / (NAME + '.json')).read_text())
assert P['status'] == 'FROZEN_CPU_READY'
assert json.loads((OUT / 'source_cpu_check.json').read_text())['status'] == 'CPU_READY'
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + str(Path(os.environ['CUDA_HOME']) / 'bin') + os.pathsep + os.environ.get('PATH', '')
assert shutil.which('ninja') and shutil.which('nvcc')
roots = [Path(p) for p in json.loads(os.environ['TASK_BUDGET_ROOTS'])]


def usage():
    return sum(int(subprocess.check_output(['du', '-s', '-B1', '.'], cwd=p, text=True).split()[0]) for p in roots)


start = time.time()
assert start + 300 < P['cooperative_stop_unix']
token = uuid.uuid4().hex
known, violations = {}, set()
baseline = P['four_root_baseline_bytes']
peak = usage()
assert peak < baseline + P['write_stop_incremental_bytes']
receipt = STATE / (NAME + '_receipt.json')
assert not receipt.exists()
receipt.write_text(json.dumps({'supervisor_pid': os.getpid(), 'supervisor_ctime': psutil.Process().create_time(), 'start_unix': start, 'run_token': token, 'first_preparation_unix': P['first_preparation_unix'], 'deadline_unix': P['deadline_unix'], 'baseline_bytes': baseline, 'initial_bytes': peak}, indent=2))


def census():
    for proc in psutil.process_iter(['pid', 'uids', 'create_time']):
        try:
            if proc.info['uids'].real == os.getuid() and proc.info['create_time'] >= start - 1 and proc.environ().get('TASK_RUN_TOKEN') == token:
                known[proc.pid] = proc.info['create_time']
                if any(s in ' '.join(proc.cmdline()).lower() for s in (getpass.getuser().lower(), ROOT.name.lower(), 'slake', 'balanced_probe')):
                    violations.add(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def cleanup():
    census()
    for sig, delay in ((signal.SIGTERM, 2), (signal.SIGKILL, 1)):
        for pid, created in known.items():
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == created and proc.status() != psutil.STATUS_ZOMBIE:
                    proc.send_signal(sig)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(delay)


reason, exit_code, outcome = None, None, None
stage_costs = []
try:
    for stage in ('source', 'freeze_sources', 'features', 'freeze_confirmation', 'repeat_0', 'repeat_1'):
        gpu = stage in ('source', 'features', 'repeat_0', 'repeat_1')
        if gpu:
            free = int(subprocess.check_output(['nvidia-smi', '-i', os.environ['TASK_GPU'], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            assert free >= 30000
        assert psutil.disk_usage(ROOT).free > 10000000000
        env = dict(os.environ, TASK_RUN_TOKEN=token, ENTRY=str(SRC / 'scripts/slake_balanced_probe.py'))
        if stage == 'source' or stage.startswith('repeat_'):
            env.update(TASK_STAGE='infer', TASK_PHASE='source' if stage == 'source' else 'confirmation', TASK_REPEAT=stage[-1] if stage.startswith('repeat_') else '0')
        else:
            env['TASK_STAGE'] = stage
        if not gpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        stage_start = time.time()
        with (OUT / (stage + '.log')).open('w') as log:
            child = subprocess.Popen([sys.executable, '-u', '-c', 'import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'], cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            while child.poll() is None:
                census()
                peak = max(peak, usage())
                now = time.time()
                (STATE / (NAME + '_status.json')).write_text(json.dumps({'status': 'RUNNING', 'stage': stage, 'updated_unix': now, 'peak_bytes': peak, 'incremental_bytes': peak - baseline, 'owned_seen': len(known), 'argv_violations': len(violations)}))
                if now >= P['cooperative_stop_unix']:
                    (OUT / 'STOP_REQUESTED.json').write_text('{"reason":"cooperative deadline"}')
                if now >= P['deadline_unix'] - 15 or peak >= baseline + P['write_stop_incremental_bytes'] or violations:
                    raise RuntimeError('hard clock/storage/argv stop')
                time.sleep(5)
            exit_code = child.returncode
            stage_costs.append({'stage': stage, 'start_unix': stage_start, 'end_unix': time.time(), 'exit_code': exit_code})
            if exit_code:
                raise RuntimeError('worker failed; preserve results without blind replay')
        cleanup()
        if stage == 'freeze_sources' and json.loads((OUT / 'SOURCE_FROZEN.json').read_text())['status'] != 'SOURCE_GATE_PASS':
            outcome = 'SOURCE_GATE_STOP'
            break
        if stage == 'freeze_confirmation' and json.loads((OUT / 'CONFIRMATION_FROZEN.json').read_text())['status'] != 'CONFIRMATION_GATE_PASS':
            outcome = 'CONFIRMATION_GATE_STOP'
            break
    outcome = outcome or 'COMPLETE_CONFIRMATION'
    (OUT / 'WORKER_COMPLETED.json').write_text(json.dumps({'status': outcome, 'completed_unix': time.time(), 'optimizer_steps': 0, 'test_answers_accessed_by_executor': False, 'stage_costs': stage_costs}, indent=2))
except BaseException as exc:
    reason = repr(exc)
    exit_code = exit_code or 1
finally:
    cleanup()
    (STATE / (NAME + '_EXIT.json')).write_text(json.dumps({'exit_code': exit_code, 'reason': reason, 'outcome': outcome, 'end_unix': time.time(), 'peak_bytes': peak, 'incremental_bytes': peak - baseline, 'argv_violations': sorted(violations), 'stage_costs': stage_costs}, indent=2))
