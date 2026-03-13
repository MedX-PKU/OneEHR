from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_TRAIN_RAW = Path("/home/yhzhu/projects/pyehr/datasets/tjh/raw/time_series_375_prerpocess_en.xlsx")
DEFAULT_EXTERNAL_RAW = Path("/home/yhzhu/projects/pyehr/datasets/tjh/raw/time_series_test_110_preprocess_en.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert real TJH Excel files into OneEHR standardized CSV tables.")
    parser.add_argument("--train-raw", type=Path, default=DEFAULT_TRAIN_RAW, help="Path to the TJH training-cohort Excel file.")
    parser.add_argument(
        "--external-raw",
        type=Path,
        default=DEFAULT_EXTERNAL_RAW,
        help="Path to the TJH external-test Excel file.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".cache/tjh_demo"),
        help="Cache root for standardized outputs and manifests.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_root = (repo_root / args.cache_root).resolve() if not args.cache_root.is_absolute() else args.cache_root.resolve()
    standardized_root = cache_root / "standardized"
    standardized_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "train": _convert_excel(repo_root=repo_root, raw_path=args.train_raw.resolve(), out_dir=standardized_root / "train"),
        "external": _convert_excel(
            repo_root=repo_root,
            raw_path=args.external_raw.resolve(),
            out_dir=standardized_root / "external",
        ),
    }
    manifest["cache_root"] = str(cache_root)
    manifest_path = standardized_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest_path": str(manifest_path), "datasets": manifest}, indent=2))


def _convert_excel(*, repo_root: Path, raw_path: Path, out_dir: Path) -> dict[str, object]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing TJH raw file: {raw_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = repo_root / "examples" / "tjh_pipeline" / "convert_tjh.py"
    subprocess.check_call(
        [sys.executable, str(script_path), "--raw", str(raw_path), "--out-dir", str(out_dir)],
        cwd=repo_root,
    )

    dynamic_csv = out_dir / "dynamic.csv"
    static_csv = out_dir / "static.csv"
    label_csv = out_dir / "label.csv"
    return {
        "raw_path": str(raw_path),
        "out_dir": str(out_dir),
        "dynamic_csv": str(dynamic_csv),
        "static_csv": str(static_csv),
        "label_csv": str(label_csv),
        "n_dynamic_rows": int(len(pd.read_csv(dynamic_csv))),
        "n_static_rows": int(len(pd.read_csv(static_csv))),
        "n_label_rows": int(len(pd.read_csv(label_csv))),
    }


if __name__ == "__main__":
    main()
