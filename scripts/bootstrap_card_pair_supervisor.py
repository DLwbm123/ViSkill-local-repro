"""Own-process time/storage guard for two sequential, separately isolated arms."""
import getpass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import psutil

ROOT = Path(os.environ['TASK_ROOT'])
NAME = 'bootstrap_card_pair_v1'
STATE = ROOT / 'state/components'
P = json.loads((ROOT / f'src/configs/smoke/{NAME}.json').read_text())
assert P['status'] == 'FROZEN_CPU_PREFLIGHT_PASS'
original = json.loads((ROOT / 'state/setup_receipt.json').read_text())
assert original['started_unix'] == P['original_total_clock_start']
recover_b = os.getenv('TASK_RESUME_UNSTARTED_B') == '1'
state_name = NAME + ('_recovery' if recover_b else '')
receipt = STATE / f'{state_name}_receipt.json'
assert not receipt.exists(), 'refuse duplicate supervisor'
prior = None
prior_exit = None
if recover_b:
    prior = json.loads((STATE / f'{NAME}_receipt.json').read_text())
    prior_exit = json.loads((STATE / f'{NAME}_EXIT.json').read_text())
    assert prior_exit['exits'] == {'A': 0}
    assert 'insufficient free memory' in prior_exit['reason']
    assert 'B' not in prior['arms']
    a_out = ROOT / f'results/ppo_smoke/{NAME}/A'
    assert (a_out / 'COMPLETE.json').exists() and (a_out / 'completion/trainer.json').exists()
    assert len(list((a_out / 'updated_batches').glob('*.json'))) == 4
    assert not (ROOT / f'results/ppo_smoke/{NAME}/B').exists(), 'B already attempted'
else:
    assert not (STATE / f'{NAME}_receipt.json').exists(), 'refuse duplicate pair'
started = time.time()
deadline = prior['deadline_unix'] if prior else min(P['absolute_total_deadline'], P['preparation_started_unix'] + P['max_seconds_pair'])
assert started + 300 < deadline


def usage():
    return sum(int(subprocess.check_output(['du', '-s', '-B1', '.'], cwd=p, text=True).split()[0])
               for p in original['paths'])


baseline = prior['baseline_bytes'] if prior else usage()
write_limit = prior['write_limit_bytes'] if prior else min(P['project_write_stop_bytes'], baseline + P['maximum_additional_bytes_pair'] - 200_000_000)
assert usage() + 300_000_000 < write_limit
assert P['maximum_additional_bytes_pair'] < 50_000_000_000
token = uuid.uuid4().hex
info = {'start_unix': prior['start_unix'] if prior else started, 'recovery_started_unix': started if prior else None,
        'preparation_start_unix': P['preparation_started_unix'],
        'deadline_unix': deadline, 'original_total_deadline': P['absolute_total_deadline'],
        'supervisor_pid': os.getpid(), 'run_token': token, 'baseline_bytes': baseline,
        'write_limit_bytes': write_limit, 'maximum_additional_bytes': P['maximum_additional_bytes_pair'],
        'physical_gpus': [3, 5], 'checkpoint_save': False,
        'arms': dict(prior['arms']) if prior else {},
        'resume_only_unstarted_B': recover_b, 'original_run_token': prior['run_token'] if prior else None}
receipt.write_text(json.dumps(info, indent=2))
env = dict(os.environ, ENTRY=str(ROOT / 'src/scripts/run_bootstrap_card_pair.py'), TASK_RUN_TOKEN=token,
    CUDA_VISIBLE_DEVICES='3,5', CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled', RAY_USAGE_STATS_ENABLED='0',
    HF_HOME=str(ROOT / 'cache/hf'), TMPDIR='/data/local/tmp/z0', SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
    PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
    XDG_CACHE_HOME='/data/local/tools/z0/cache/xdg', TORCH_EXTENSIONS_DIR='/data/local/tools/z0/cache/torch_extensions',
    TORCHINDUCTOR_CACHE_DIR='/data/local/tools/z0/cache/inductor', TRITON_CACHE_DIR='/data/local/tools/z0/cache/triton',
    FLASHINFER_WORKSPACE_BASE='/data/local/tools/z0/cache/flashinfer', CUDA_CACHE_PATH='/data/local/tools/z0/cache/cuda',
    SGLANG_ENABLE_JIT_DEEPGEMM='0', PATH='/data/local/envs/z0/bin:/usr/local/cuda/bin:' + os.environ.get('PATH', ''),
    CUDA_HOME='/usr/local/cuda', PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'src/verl'), OMP_NUM_THREADS='4')
known = {}
audited = set()
violations = []
blocked = [getpass.getuser().lower(), ROOT.name.lower(), 'directional']


def discover():
    for process in psutil.process_iter(['pid', 'create_time', 'uids']):
        try:
            if (process.info['uids'].real == os.getuid() and process.info['create_time'] >= started - 1
                    and process.environ().get('TASK_RUN_TOKEN') == token):
                known[process.pid] = process.info['create_time']
                if process.pid not in audited:
                    audited.add(process.pid)
                    argv = process.cmdline()
                    with (STATE / f'{NAME}_processes.jsonl').open('a') as stream:
                        stream.write(json.dumps({'pid': process.pid, 'argv': argv}) + '\n')
                    if any(word in ' '.join(argv).lower() for word in blocked):
                        violations.append({'pid': process.pid, 'argv': argv})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def cleanup():
    discover()
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        for pid, created in known.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == created:
                    process.send_signal(sig)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(2)


reason = None
exits = {}
peak = baseline
try:
    for arm in (['B'] if recover_b else ['A', 'B']):
        for gpu in [3, 5]:
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            assert free >= 44000, f'insufficient free memory on allowed GPU {gpu}: {free}'
        arm_start = time.time()
        arm_deadline = min(deadline, arm_start + P['max_seconds_per_arm'])
        if recover_b:
            # Count the resource-wait period against B's original allowance too.
            arm_deadline = min(arm_deadline, prior_exit['end_unix'] + P['max_seconds_per_arm'])
            assert arm_start + 300 < arm_deadline
        info['arms'][arm] = {'start_unix': arm_start, 'deadline_unix': arm_deadline}
        receipt.write_text(json.dumps(info, indent=2))
        logfile = ROOT / f'logs/ppo_smoke/{NAME}_{arm}.log'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open('w') as log:
            process = subprocess.Popen(['/data/local/envs/z0/bin/python', '-u', '-c',
                'import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],
                env=dict(env, TASK_PHASE=arm), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=ROOT / 'src')
            while process.poll() is None:
                discover()
                peak = max(peak, usage())
                if peak >= write_limit or time.time() >= arm_deadline:
                    raise RuntimeError('storage/time guard')
                if violations:
                    raise RuntimeError('non-neutral process command line')
                time.sleep(5)
            exits[arm] = process.returncode
        cleanup()
        info['arms'][arm]['exit_code'] = process.returncode
        receipt.write_text(json.dumps(info, indent=2))
        if process.returncode:
            raise RuntimeError(f'arm {arm} exited {process.returncode}; do not rerun completed episodes or updates')
    summary_env = dict(env, ENTRY=str(ROOT / 'src/scripts/summarize_bootstrap_pair.py'))
    with (ROOT / f'logs/ppo_smoke/{NAME}_summary.log').open('w') as log:
        subprocess.run(['/data/local/envs/z0/bin/python', '-u', '-c',
            'import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],
            env=summary_env, stdout=log, stderr=subprocess.STDOUT, check=True,
            timeout=max(1, min(60, int(deadline - time.time()))), cwd=ROOT / 'src')
except BaseException as error:
    reason = repr(error)
finally:
    cleanup()
    (STATE / f'{state_name}_EXIT.json').write_text(json.dumps({'exits': exits, 'reason': reason,
        'end_unix': time.time(), 'peak_allocated_bytes': peak, 'deadline_unix': deadline,
        'command_line_violations': violations}, indent=2))
