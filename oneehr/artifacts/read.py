from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    data: dict[str, Any]

    def dynamic_feature_columns(self) -> list[str]:
        cols = (((self.data.get("features") or {}).get("dynamic") or {}).get("feature_columns")) or []
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("Invalid run_manifest: features.dynamic.feature_columns must be list[str]")
        return list(cols)

    def static_feature_columns(self) -> list[str]:
        cols = (((self.data.get("features") or {}).get("static") or {}).get("feature_columns")) or []
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("Invalid run_manifest: features.static.feature_columns must be list[str]")
        return list(cols)

    def static_matrix_path(self) -> Path | None:
        p = (((self.data.get("features") or {}).get("static") or {}).get("matrix_parquet_path")) or None
        if p is None:
            return None
        if not isinstance(p, str) or not p:
            raise ValueError("Invalid run_manifest: features.static.matrix_parquet_path must be str")
        return Path(p)


def read_run_manifest(run_root: Path) -> RunManifest | None:
    path = run_root / "run_manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    v = int(data.get("schema_version", 0) or 0)
    return RunManifest(schema_version=v, data=data)
