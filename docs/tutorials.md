# Tutorials

The tutorial notebooks walk through common OneEHR workflows with runnable examples.

| # | Tutorial | Covers |
|---|----------|--------|
| 1 | [Quickstart](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/01_quickstart.ipynb) | End-to-end TJH mortality prediction |
| 2 | [Custom Dataset](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/02_custom_dataset.ipynb) | Preparing dynamic, static, and label CSV files |
| 3 | [Model Comparison](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/03_model_comparison.ipynb) | Comparing configured models with confidence intervals and statistical tests |
| 4 | [Fairness And Explainability](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/04_fairness_analysis.ipynb) | Fairness metrics and feature importance outputs |
| 5 | [Survival Analysis](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/05_survival_analysis.ipynb) | DeepSurv, DeepHit, concordance index, and survival curves |

## Run Locally

```bash
pip install oneehr jupyter
jupyter notebook tutorials/
```

When working from a source checkout, install the package in editable mode first:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test]"
jupyter notebook tutorials/
```

## Run In Colab

Upload a notebook to [Google Colab](https://colab.research.google.com/) and install OneEHR at the top:

```python
!pip install oneehr
```

## What You Will Practice

Tutorial 1:

- Define an experiment with a TOML config file.
- Preprocess EHR events into binned features.
- Train XGBoost and GRU models.
- Evaluate on a held-out test split.
- Run comparison analysis.

Tutorial 2:

- Prepare dynamic, static, and label CSV files.
- Convert MIMIC-III, MIMIC-IV, or eICU data.
- Apply ICD, CCS, and ATC code mapping helpers.

Tutorial 3:

- Configure multiple models in one experiment.
- Read `predictions.parquet` and `metrics.json`.
- Use bootstrap confidence intervals and pairwise statistical tests.
- Include configured LLM or agent systems in the same comparison table.

Tutorial 4:

- Compute fairness metrics across available static attributes.
- Review feature importance outputs.
- Connect analysis JSON files to plots and notebooks.

Tutorial 5:

- Configure DeepSurv and DeepHit.
- Train with survival task settings.
- Evaluate with concordance index.
- Plot survival curves from saved results.
