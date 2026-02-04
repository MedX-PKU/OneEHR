from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    data: dict[str, Any]


def read_run_manifest(run_root: Path) -> RunManifest | None:
    path = run_root / "run_manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    v = int(data.get("schema_version", 0) or 0)
    return RunManifest(schema_version=v, data=data)

