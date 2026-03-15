# TJH Tutorial Demo

This folder contains a reproducible, real-data demo flow for the KDD hands-on tutorial. The showcase path is eval-first: it trains baseline models, freezes a shared EHR evaluation set, runs configured framework systems, and saves comparison artifacts alongside the standard analysis outputs.

## Inputs

- Training cohort Excel: `/home/yhzhu/projects/pyehr/datasets/tjh/raw/time_series_375_prerpocess_en.xlsx`
- External test Excel: `/home/yhzhu/projects/pyehr/datasets/tjh/raw/time_series_test_110_preprocess_en.xlsx`

## Fast preset

```bash
uv run python examples/tjh_demo/run_demo.py --preset fast --force
```

## Showcase preset

Requires `ZENMUX_API_KEY`:

```bash
uv run python examples/tjh_demo/run_demo.py --preset showcase --force
```

The showcase preset builds a fast baseline run plus a richer run with held-out testing, structured analysis, unified eval artifacts, and compare-run summaries.

## Screenshot export

```bash
examples/tjh_demo/capture_screenshots.sh --run-name tjh_tutorial_showcase --baseline-run tjh_tutorial_fast
```

The screenshot exporter targets a `2560x1600` viewport by default so the assets match a laptop presentation flow. It captures the runs landing page, run overview, audit dashboards, and comparison view.

## Slides

Open `examples/tjh_demo/slides/index.html` in a browser for the tutorial deck. It references the generated screenshots under `.cache/tjh_demo/screenshots/tjh_tutorial_showcase/`.

## Output locations

- Standardized CSV cache: `.cache/tjh_demo/standardized/`
- Generated configs: `.cache/tjh_demo/configs/`
- Screenshot assets: `.cache/tjh_demo/screenshots/`
- Runs: `logs/tjh_tutorial_fast/`, `logs/tjh_tutorial_showcase/`
