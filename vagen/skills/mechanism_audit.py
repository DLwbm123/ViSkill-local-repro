"""Opt-in local mechanism evidence. No model calls or gate changes."""
import json
import os
from pathlib import Path


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


def write_record(kind, identity, value):
    root = os.getenv("MECHANISM_AUDIT_DIR")
    if root:
        folder = Path(root) / kind
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{identity}.json").write_text(json.dumps(jsonable(value), indent=2))


def library_snapshot(provider):
    if provider is None or provider.library is None:
        return None
    return {"next_id": provider.library._next_id,
            "skills": [{k: s.get(k) for k in ("id", "seed", "utility", "use_count", "trajectory")}
                       for s in provider.library._skills]}
