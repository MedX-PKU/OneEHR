from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimConfig:
    n_patients: int = 200
    seed: int = 7
    start: str = "2025-01-01"
    n_days: int = 30


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def simulate_tables(cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = _rng(cfg.seed)
    patient_ids = [f"p{i:05d}" for i in range(1, cfg.n_patients + 1)]

    # static.csv
    age = rng.integers(18, 90, size=cfg.n_patients)
    sex = rng.choice(["F", "M"], size=cfg.n_patients, replace=True)
    static = pd.DataFrame({"patient_id": patient_ids, "age": age, "sex": sex})

    # dynamic.csv (irregular events)
    start = pd.Timestamp(cfg.start)
    rows = []
    for pid, a in zip(patient_ids, age, strict=True):
        n_events = int(rng.integers(12, 60))
        for _ in range(n_events):
            t = start + pd.to_timedelta(rng.uniform(0, cfg.n_days * 24), unit="h")
            if rng.random() < 0.6:
                code = rng.choice(["hr", "sbp", "temp"])
                if code == "hr":
                    v = float(np.clip(rng.normal(75 + 0.15 * (a - 50), 12), 40, 160))
                elif code == "sbp":
                    v = float(np.clip(rng.normal(120 + 0.3 * (a - 50), 18), 70, 220))
                else:
                    v = float(np.clip(rng.normal(36.8, 0.6), 34.0, 41.0))
                value = f"{v:.2f}"
            else:
                code = "dx"
                value = rng.choice(["none", "cold", "flu", "copd"])
            rows.append((pid, t, code, value))

    dynamic = pd.DataFrame(rows, columns=["patient_id", "event_time", "code", "value"])
    dynamic = dynamic.sort_values(["patient_id", "event_time"], kind="stable").reset_index(drop=True)

    # label.csv (patient-level outcome at last observation)
    last = dynamic.groupby("patient_id", sort=False)["event_time"].max().rename("label_time").reset_index()

    # Outcome depends on age, sex, and whether "flu" ever appeared.
    flu_flag = (
        dynamic.loc[(dynamic["code"] == "dx") & (dynamic["value"].astype(str) == "flu"), ["patient_id"]]
        .drop_duplicates()
        .assign(has_flu=1)
    )
    base = static.merge(flu_flag, on="patient_id", how="left").fillna({"has_flu": 0})
    logit = -3.0 + 0.035 * (base["age"] - 50) + 0.9 * base["has_flu"] + 0.2 * (base["sex"] == "M").astype(int)
    p = 1 / (1 + np.exp(-logit.to_numpy()))
    y = (rng.random(cfg.n_patients) < p).astype(int)

    label = last.copy()
    label["label_code"] = "outcome"
    label["label_value"] = y
    label["label_time"] = pd.to_datetime(label["label_time"], errors="raise")
    label = label[["patient_id", "label_time", "label_code", "label_value"]]

    return dynamic, static, label


def write_simulated_dataset(out_dir: str | Path, cfg: SimConfig) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dynamic, static, label = simulate_tables(cfg)
    (out / "dynamic.csv").write_text(dynamic.to_csv(index=False), encoding="utf-8")
    (out / "static.csv").write_text(static.to_csv(index=False), encoding="utf-8")
    (out / "label.csv").write_text(label.to_csv(index=False), encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate a simulated OneEHR 3-table dataset")
    p.add_argument("--out-dir", default="examples/simulated", help="Output directory")
    p.add_argument("--n-patients", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--n-days", type=int, default=30)
    args = p.parse_args()

    cfg = SimConfig(n_patients=args.n_patients, seed=args.seed, start=args.start, n_days=args.n_days)
    write_simulated_dataset(args.out_dir, cfg)


if __name__ == "__main__":
    main()

