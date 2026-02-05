from __future__ import annotations

import pandas as pd


def build_labels(
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    cfg,
) -> pd.DataFrame:
    """Example label_fn (new signature).

    Demonstrates N-N (time) label generation by combining inputs:
    - `dynamic`: used to find each patient's last observed time
    - `label`: optional task-agnostic label event table (long format)

    Output columns for N-N:
    - patient_id
    - label_time (OneEHR will floor to bin_time if needed)
    - label
    - mask
    """

    _ = static  # not used in this example

    # Find last observed dynamic time per patient.
    pid = cfg.dataset.dynamic.patient_id_col
    tcol = cfg.dataset.dynamic.time_col
    df = dynamic[[pid, tcol]].copy()
    df[pid] = df[pid].astype(str)
    df[tcol] = pd.to_datetime(df[tcol], errors="raise")
    df = df.sort_values([pid, tcol], kind="stable")
    last_time = df.groupby(pid, sort=False).tail(1).rename(columns={pid: "patient_id", tcol: "label_time"})

    # If label.csv exists, pick one label_code for the task; else synthesize a dummy label.
    if label is not None and not label.empty:
        lpid = cfg.dataset.label.patient_id_col if cfg.dataset.label is not None else "patient_id"
        ltime = cfg.dataset.label.time_col if cfg.dataset.label is not None else "label_time"
        lcode = cfg.dataset.label.code_col if cfg.dataset.label is not None else "label_code"
        lval = cfg.dataset.label.value_col if cfg.dataset.label is not None else "label_value"

        lab = label[[lpid, ltime, lcode, lval]].copy()
        lab[lpid] = lab[lpid].astype(str)
        lab[ltime] = pd.to_datetime(lab[ltime], errors="raise")

        # Example rule: use label_code="outcome" when binary, else "los".
        wanted = "outcome" if cfg.task.kind == "binary" else "los"
        lab = lab.loc[lab[lcode].astype(str) == wanted].copy()
        if lab.empty:
            raise ValueError(f"label.csv has no rows with {lcode}={wanted!r}")

        # For each patient, take the last label at/before the last observed time.
        merged = last_time.merge(lab, left_on="patient_id", right_on=lpid, how="left")
        merged = merged.loc[merged[ltime] <= merged["label_time"]].copy()
        merged = merged.sort_values(["patient_id", ltime], kind="stable").groupby("patient_id", sort=False).tail(1)
        out = merged.rename(columns={lval: "label"})[["patient_id", "label_time", "label"]].copy()
    else:
        out = last_time.copy()
        out["label"] = 1

    out["mask"] = 1
    return out[["patient_id", "label_time", "label", "mask"]]

