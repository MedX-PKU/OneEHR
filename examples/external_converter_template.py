from __future__ import annotations

from pathlib import Path

import pandas as pd


def main(raw_path: str, out_path: str) -> None:
    raw_path = str(Path(raw_path))
    out_path = str(Path(out_path))

    # 1) Read your raw dataset (csv/xlsx/parquet/anything).
    # df_raw = pd.read_csv(raw_path)
    df_raw = pd.read_excel(raw_path)

    # 2) Convert to the OneEHR unified event table schema (long format).
    # Required columns: patient_id, event_time, code, value
    events = pd.DataFrame(
        {
            "patient_id": df_raw["patient_id"].astype(str),
            "event_time": pd.to_datetime(df_raw["event_time"]),
            "code": df_raw["code"].astype(str),
            "value": df_raw["value"],
        }
    )

    # 3) Save as a plain table. OneEHR can read it directly.
    events.to_csv(out_path, index=False)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert a raw dataset to OneEHR unified schema.")
    p.add_argument("--raw", required=True, help="Path to raw dataset file")
    p.add_argument("--out", required=True, help="Path to output events.csv")
    args = p.parse_args()
    main(args.raw, args.out)

