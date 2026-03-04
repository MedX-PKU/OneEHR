from __future__ import annotations

import json
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
        p = (manifest.data.get("artifacts") or {}).get("binned_parquet_path")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing binned_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
        df = pd.read_parquet(self.run_root / p)
        required = {"patient_id", "bin_time"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Invalid binned.parquet: missing columns {missing}")

        feat_cols = manifest.dynamic_feature_columns()
        missing_feat = [c for c in feat_cols if c not in df.columns]
        if missing_feat:
            raise SystemExit(f"Invalid binned.parquet: missing feature columns {missing_feat}")

        # Standardize column order: keys -> (optional label) -> features
        base = ["patient_id", "bin_time"]
        if "label" in df.columns:
            base.append("label")
        df = df[base + feat_cols]
        return df

    def load_patient_view(self, manifest) -> tuple[pd.DataFrame, pd.Series]:
        p = (manifest.data.get("artifacts") or {}).get("patient_tabular_parquet_path")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing patient_tabular_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
        df = pd.read_parquet(self.run_root / p)
        if "patient_id" not in df.columns or "label" not in df.columns:
            raise SystemExit("Invalid patient_tabular.parquet: missing patient_id/label.")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        feat_cols = manifest.dynamic_feature_columns()
        missing = [c for c in ["patient_id", "label", *feat_cols] if c not in df.columns]
        if missing:
            raise SystemExit(f"Invalid patient_tabular.parquet: missing columns {missing}")
        df = df[["patient_id", "label", *feat_cols]]
        X = df[["patient_id", *feat_cols]].set_index("patient_id")
        y = df["label"]
        return X, y

    def load_time_view(self, manifest) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        p = (manifest.data.get("artifacts") or {}).get("time_tabular_parquet_path")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing time_tabular_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
        df = pd.read_parquet(self.run_root / p)
        required = {"patient_id", "bin_time", "label"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Invalid time_tabular.parquet: missing columns {missing}")
        feat_cols = manifest.dynamic_feature_columns()
        missing2 = [c for c in feat_cols if c not in df.columns]
        if missing2:
            raise SystemExit(f"Invalid time_tabular.parquet: missing feature columns {missing2}")
        df = df[["patient_id", "bin_time", "label", *feat_cols]].reset_index(drop=True)
        key = df[["patient_id", "bin_time"]]
        y = df["label"]
        X = df[feat_cols]
        return X, y, key

    def load_static_all(self, manifest) -> tuple[pd.DataFrame | None, list[str] | None]:
        st_path = manifest.static_matrix_path()
        if st_path is None:
            return None, None
        static_all = pd.read_parquet(self.run_root / st_path)
        cols = manifest.static_feature_columns()
        if list(static_all.columns) != list(cols):
            raise SystemExit(
                "Static feature_columns mismatch with run_manifest.json. "
                "Re-run `oneehr preprocess`."
            )
        return static_all, cols

    def load_labels(self, manifest) -> pd.DataFrame | None:
        p = (manifest.data.get("artifacts") or {}).get("labels_parquet_path")
        if p is None:
            return None
        if not isinstance(p, str) or not p:
            raise SystemExit("Invalid labels_parquet_path in run_manifest.json.")
        path = self.run_root / p
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        mode = str((manifest.data.get("task") or {}).get("prediction_mode", "patient"))
        from oneehr.artifacts.labels import validate_patient_labels, validate_time_labels

        if mode == "patient":
            return validate_patient_labels(df)
        if mode == "time":
            return validate_time_labels(df)
        raise ValueError(f"Unsupported prediction_mode in run_manifest.json: {mode!r}")

    def load_fitted_postprocess(self, split_name: str):
        """Load fitted postprocess pipeline for a given split.

        Reads ``preprocess/{split_name}/pipeline.json`` and returns a
        ``FittedPostprocess`` instance, or ``None`` if the file doesn't exist.
        """
        pp_path = self.run_root / "preprocess" / split_name / "pipeline.json"
        if not pp_path.exists():
            return None
        from oneehr.data.postprocess import FittedPostprocess

        data = json.loads(pp_path.read_text(encoding="utf-8"))
        pipeline = data.get("pipeline")
        if not isinstance(pipeline, list):
            return None
        return FittedPostprocess(pipeline=pipeline)
