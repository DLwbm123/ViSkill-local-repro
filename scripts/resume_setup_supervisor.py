"""Download repair guard: preserve original setup deadline and failure artifacts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path(os.environ["TASK_ROOT"])
state = root / "state"
prefix = os.environ.get("TASK_STATE_PREFIX", "setup_repair")
assert prefix in ["setup_repair", "setup_repair_2", "ppo_runtime_setup", "ppo_runtime_srt", "ppo_runtime_pins"]
if (state / f"{prefix}_receipt.json").exists():
    raise RuntimeError("attempt receipt already exists; inspect instead of duplicate launch")
original = json.loads((state / "setup_receipt.json").read_text())
deadline = original["started_unix"] + original["setup_timeout_seconds"]
cutoff = original["stop_writes_at_bytes"]
paths = original["paths"]

def usage():
    result = subprocess.run(["du", "-s", "-B1", *paths], capture_output=True, text=True, check=True)
    return sum(int(line.split()[0]) for line in result.stdout.splitlines())

if time.time() >= deadline or usage() >= cutoff:
    raise RuntimeError("original setup time/storage budget exhausted")
worker = os.environ.get("TASK_WORKER_SCRIPT", "resume_downloads.py")
assert worker in ["resume_downloads.py", "prepare_ppo_runtime.py", "repair_runtime_pins.py"]
env = dict(os.environ, ENTRY=str(root / "src/scripts" / worker),
           CUDA_VISIBLE_DEVICES="", HF_HOME=str(root / "cache/hf"),
           HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1",
           HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DOWNLOAD_TIMEOUT="120",
           WANDB_MODE="disabled", RAY_USAGE_STATS_ENABLED="0",
           TMPDIR="/data/local/tmp/z0", PYTHONDONTWRITEBYTECODE="1",
           HTTPS_PROXY="http://127.0.0.1:28791", HTTP_PROXY="http://127.0.0.1:28791",
           ALL_PROXY="http://127.0.0.1:28791", NO_PROXY="", no_proxy="")
child = subprocess.Popen(["/data/local/envs/z0/bin/python", "-u", "-c",
    'import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],
    env=env, start_new_session=True)
receipt = dict(original, supervisor_pid=os.getpid(), worker_pid=child.pid,
               repair_started_unix=time.time(), original_deadline_unix=deadline,
               attempt=prefix, proxy="loopback SSH reverse forward")
(state / f"{prefix}_receipt.json").write_text(json.dumps(receipt, indent=2))
peak = max(json.loads(p.read_text())["peak_allocated_bytes"] for p in state.glob("setup*resource.json"))
reason = None
try:
    while child.poll() is None:
        allocated = usage()
        peak = max(peak, allocated)
        (state / f"{prefix}_resource.json").write_text(json.dumps({
            "time_unix": time.time(), "allocated_bytes": allocated, "peak_allocated_bytes": peak}))
        if allocated >= cutoff:
            reason = "storage guard cutoff"
            break
        if time.time() >= deadline:
            reason = "original setup hard timeout"
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
(state / f"{prefix.upper()}_EXIT.json").write_text(json.dumps({
    "exit_code": child.returncode, "reason": reason,
    "original_start_unix": original["started_unix"], "end_unix": time.time(),
    "peak_allocated_bytes": peak}, indent=2))
