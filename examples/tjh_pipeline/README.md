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

## 3) External test split (train/test) + `oneehr test`

If you want a **true external evaluation** (train on one patient cohort, test on a held-out cohort),
use the helper splitter:

```bash
uv run python examples/tjh_pipeline/split_tjh.py \
  --in-dir /tmp/oneehr_tjh_example \
  --out-dir /tmp/oneehr_tjh_split \
  --test-size 0.2 \
  --seed 42
```

Then run OneEHR with `datasets.train` + `datasets.test`:

```bash
uv run oneehr preprocess --config examples/tjh_pipeline/experiment_external_test.toml
uv run oneehr train --config examples/tjh_pipeline/experiment_external_test.toml --force
uv run oneehr test --config examples/tjh_pipeline/experiment_external_test.toml --force
```

### Fixed train/val split (no CV)

If you prefer a fixed train/val split (instead of k-fold CV on the training cohort),
use:

```bash
uv run oneehr preprocess --config examples/tjh_pipeline/experiment_external_fixed_split.toml
uv run oneehr train --config examples/tjh_pipeline/experiment_external_fixed_split.toml --force
uv run oneehr test --config examples/tjh_pipeline/experiment_external_fixed_split.toml --force
```

Artifacts to inspect:
- Training run: `logs/tjh_external_test/`
- Test run outputs: `logs/tjh_external_test/test_runs/<dataset_stem>/`
```
