from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_csv(index=False), encoding="utf-8")


def _ensure_pid_str(df: pd.DataFrame, col: str = "patient_id") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    out = df.copy()
    out[col] = out[col].astype(str)
    return out


def _split_ids(
    patient_ids: list[str],
    *,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0,1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1")

    ids = pd.Series(sorted(set(patient_ids)), dtype=str)
    ids = ids.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(ids)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_test = max(1, n_test)
    n_val = max(0, n_val)
    if n_test + n_val >= n:
        n_test = max(1, n - 1)
        n_val = 0

    test_ids = set(ids.iloc[:n_test].tolist())
    val_ids = set(ids.iloc[n_test : n_test + n_val].tolist())
    train_ids = set(ids.iloc[n_test + n_val :].tolist())
    return train_ids, val_ids, test_ids


def main(
    *,
    in_dir: Path,
    out_dir: Path,
    test_size: float,
    val_size: float,
    seed: int,
) -> None:
    dynamic = pd.read_csv(in_dir / "dynamic.csv")
    static = pd.read_csv(in_dir / "static.csv") if (in_dir / "static.csv").exists() else None
    label = pd.read_csv(in_dir / "label.csv") if (in_dir / "label.csv").exists() else None

    dynamic = _ensure_pid_str(dynamic)
    if static is not None:
        static = _ensure_pid_str(static)
    if label is not None:
        label = _ensure_pid_str(label)

    # Split is label-driven to ensure supervised test has labels.
    if label is None or label.empty:
        raise ValueError("label.csv is required for offline evaluation splits.")
    label_pids = label["patient_id"].astype(str).unique().tolist()

    train_ids, val_ids, test_ids = _split_ids(label_pids, test_size=test_size, val_size=val_size, seed=seed)

    def _filt(df: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
        return df.loc[df["patient_id"].astype(str).isin(ids)].copy()

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        base = out_dir / name
        _write_csv(_filt(dynamic, ids), base / "dynamic.csv")
        if static is not None:
            _write_csv(_filt(static, ids), base / "static.csv")
        if label is not None:
            _write_csv(_filt(label, ids), base / "label.csv")

    # Also write split manifest for inspection.
    manifest = out_dir / "split_manifest.json"
    import json

    manifest.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "test_size": float(test_size),
                "val_size": float(val_size),
                "n_train": int(len(train_ids)),
                "n_val": int(len(val_ids)),
                "n_test": int(len(test_ids)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split TJH exported CSVs into train/val/test by patient_id")
    p.add_argument("--in-dir", required=True, help="Directory containing dynamic.csv/static.csv/label.csv")
    p.add_argument("--out-dir", required=True, help="Output directory (will contain train/val/test subdirs)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(
        in_dir=Path(args.in_dir),
        out_dir=Path(args.out_dir),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        seed=int(args.seed),
    )

