"""Resume only public fixed-revision assets after setup network failure."""
import json
import os
from pathlib import Path
import time

from huggingface_hub import snapshot_download

root = Path(os.environ["TASK_ROOT"])
specs = [
    ("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3", "b0"),
    ("hongxingli/ViSkill-FrozenLake", "edafb2b71f73afb1c35b82d264346c2ae2b5981e", "r0"),
]
deadline = json.loads((root / "state/setup_receipt.json").read_text())
deadline = deadline["started_unix"] + deadline["setup_timeout_seconds"]
attempt_name = os.environ.get("TASK_STATE_PREFIX", "setup_repair")
for repo, revision, directory in specs:
    if os.environ.get("TASK_DOWNLOAD_ONLY") and directory != os.environ["TASK_DOWNLOAD_ONLY"]:
        continue
    print("DOWNLOAD", directory, revision, flush=True)
    for retry in range(4):
        try:
            snapshot_download(
                repo_id=repo, revision=revision, local_dir=root / "models" / directory,
                token=False, max_workers=2,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "skill_library/**"],
            )
            break
        except Exception as error:
            event = {"model": directory, "retry": retry, "error": repr(error), "time_unix": time.time()}
            with (root / "state" / f"{attempt_name}_errors.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print("DOWNLOAD_ERROR", json.dumps(event), flush=True)
            if retry == 3 or time.time() + 20 >= deadline:
                raise
            time.sleep(20)
    print("COMPLETE", directory, flush=True)
(root / "state/SETUP_COMPLETE.json").write_text(json.dumps({
    "status": "setup_and_downloads_done_NOT_pipeline_pass",
    "finished_at_unix": time.time(), "download_attempt": attempt_name,
    "next": "C runtime/import checks, then D/E; SGLang/FA pending",
}, indent=2))
