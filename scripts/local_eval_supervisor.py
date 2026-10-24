"""Bounded evaluation worker; all three arms/retries share one 6-hour clock."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path(os.environ["TASK_ROOT"])
arm = os.environ["TASK_ARM"]
assert arm in ["A", "B", "C"]
state = root / "state/local_eval"
state.mkdir(parents=True, exist_ok=True)
clock = state / "clock.json"
if not clock.exists():
    clock.write_text(json.dumps({"started_unix": time.time(), "max_seconds": 21600}))
original = json.loads(clock.read_text())
deadline = original["started_unix"] + original["max_seconds"]
paths = [str(root), "/data/local/envs/z0", "/data/local/tools/z0", "/data/local/tmp/z0"]

def usage():
    result = subprocess.run(["du", "-s", "-B1", *paths], capture_output=True, text=True, check=True)
    return sum(int(line.split()[0]) for line in result.stdout.splitlines())

if (state / f"{arm}_receipt.json").exists():
    raise RuntimeError("arm already dispatched; inspect outcome and preserve original attempt before repair")
if time.time() >= deadline or usage() >= 85000000000:
    raise RuntimeError("evaluation time/storage budget exhausted")
free = int(subprocess.check_output(["nvidia-smi", "-i", "4", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
if free < 25000:
    raise RuntimeError("GPU4 has less than 25GiB free; reassess allocation")
env = dict(os.environ, ENTRY=str(root / "src/scripts/local_eval.py"), CUDA_VISIBLE_DEVICES="4",
           CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
           HF_HOME=str(root / "cache/hf"), WANDB_MODE="disabled", WANDB_DISABLED="true",
           RAY_USAGE_STATS_ENABLED="0", TMPDIR="/data/local/tmp/z0", PYTHONDONTWRITEBYTECODE="1",
           TOKENIZERS_PARALLELISM="false", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
child = subprocess.Popen(["/data/local/envs/z0/bin/python", "-u", "-c",
    'import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'], env=env, start_new_session=True)
receipt = dict(original, arm=arm, supervisor_pid=os.getpid(), worker_pid=child.pid,
               deadline_unix=deadline, physical_gpu=4, storage_stop_bytes=85000000000, training=False)
(state / f"{arm}_receipt.json").write_text(json.dumps(receipt, indent=2))
reason = None
peak = 0
try:
    while child.poll() is None:
        size = usage()
        peak = max(size, peak)
        (state / f"{arm}_resource.json").write_text(json.dumps({"time_unix": time.time(), "allocated_bytes": size, "peak_bytes": peak}))
        if size >= 85000000000 or time.time() >= deadline:
            reason = "storage/time guard"
            break
        time.sleep(5)
except BaseException as error:
    reason = repr(error)
if reason:
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
(state / f"{arm}_EXIT.json").write_text(json.dumps({
    "exit_code": child.returncode, "reason": reason, "end_unix": time.time(),
    "original_start_unix": original["started_unix"], "peak_allocated_bytes": peak}, indent=2))
