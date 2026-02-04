from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oneehr.artifacts.read import read_run_manifest


@dataclass(frozen=True)
class RunIO:
    run_root: Path

    @classmethod
    def from_cfg(cls, *, root: Path, run_name: str) -> "RunIO":
        return cls(run_root=Path(root) / run_name)

    def require_manifest(self):
        manifest = read_run_manifest(self.run_root)
        if manifest is None:
            raise SystemExit(
                f"Missing run_manifest.json under {self.run_root}. "
                "Run `oneehr preprocess` first."
            )
        return manifest

    def load_binned(self, manifest) -> pd.DataFrame:
        p = (manifest.data.get("artifacts") or {}).get("binned_parquet")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing binned_parquet in run_manifest.json. Re-run `oneehr preprocess`.")
        return pd.read_parquet(self.run_root / p)

    def load_patient_view(self, manifest) -> tuple[pd.DataFrame, pd.Series]:
        p = (manifest.data.get("artifacts") or {}).get("patient_tabular_parquet")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing patient_tabular_parquet in run_manifest.json. Re-run `oneehr preprocess`.")
        df = pd.read_parquet(self.run_root / p)
        if "patient_id" not in df.columns or "label" not in df.columns:
            raise SystemExit("Invalid patient_tabular.parquet: missing patient_id/label.")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        X = df.drop(columns=["label"]).set_index("patient_id")
        y = df["label"]
        return X, y

    def load_time_view(self, manifest) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        p = (manifest.data.get("artifacts") or {}).get("time_tabular_parquet")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing time_tabular_parquet in run_manifest.json. Re-run `oneehr preprocess`.")
        df = pd.read_parquet(self.run_root / p)
        required = {"patient_id", "bin_time", "label"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Invalid time_tabular.parquet: missing columns {missing}")
        key = df[["patient_id", "bin_time"]].reset_index(drop=True)
        y = df["label"].reset_index(drop=True)
        X = df.drop(columns=["patient_id", "bin_time", "label"]).reset_index(drop=True)
        return X, y, key

    def load_static_all(self, manifest) -> tuple[pd.DataFrame | None, list[str] | None]:
        if not bool(((manifest.data.get("static_features") or {}).get("enabled"))):
            return None, None
        st_path = manifest.static_matrix_path()
        if st_path is None:
            raise SystemExit(
                "Static features enabled but static matrix not found in run_manifest.json. "
                "Re-run `oneehr preprocess`."
            )
        static_all = pd.read_parquet(self.run_root / st_path)
        cols = manifest.static_feature_columns()
        if list(static_all.columns) != list(cols):
            raise SystemExit(
                "Static feature_columns mismatch with run_manifest.json. "
                "Re-run `oneehr preprocess`."
            )
        return static_all, cols


def read_feature_columns_json(path: Path) -> list[str]:
    """Read a feature_columns.json file.

    Expected format:
      {"feature_columns": [...]}
    """

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    cols = data.get("feature_columns")
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError(f"Invalid feature_columns.json at {path}")
    return list(cols)
