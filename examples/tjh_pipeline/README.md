# TJH dataset example pipeline (outside OneEHR)

This folder demonstrates the intended workflow:

1. **Convert** TJH raw files into OneEHR standardized CSV inputs:
   - `dynamic.csv` (required): `patient_id,event_time,code,value`
   - `static.csv` (optional): must include `patient_id`
   - `label.csv` (optional): `patient_id,label_time,label_code,label_value`
2. Run OneEHR using only those CSVs (no built-in dataset adapters; no `converter_fn`).

## 1) Convert TJH raw -> standard CSVs

Run:

```bash
uv run python examples/tjh_pipeline/convert_tjh.py \
  --raw /path/to/time_series_375_prerpocess_en.xlsx \
  --out-dir /path/to/out
```

Outputs:
- `/path/to/out/dynamic.csv`
- `/path/to/out/static.csv`
- `/path/to/out/label.csv`

## 2) Train with OneEHR

Edit `examples/tjh_pipeline/experiment.toml` to point to the exported CSVs, then:

```bash
oneehr preprocess --config examples/tjh_pipeline/experiment.toml
oneehr train --config examples/tjh_pipeline/experiment.toml
```

