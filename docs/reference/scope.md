# Scope

Use this page to understand what OneEHR covers and where to connect external tools.

## What OneEHR Covers

| Area | Included |
|------|----------|
| Input data | Standard CSV tables for dynamic events, static covariates, and labels |
| Preprocessing | Event binning, feature encoding, train-split-fitted preprocessing steps, and patient-level splits |
| Models | 42 built-in model config names across tabular ML, DL, irregular-time, multimodal, KG-enhanced, and survival families |
| Systems | LLM and agent system definitions that write predictions into the same test artifact |
| Evaluation | Metrics, bootstrap confidence intervals, calibration, fairness, feature importance, statistical tests, and missing-data summaries |
| Outputs | Run manifests, Parquet tables, JSON reports, checkpoints, and figures |
| Dataset conversion | MIMIC-III, MIMIC-IV, and eICU converters for supported tasks |
| Medical codes | ICD-9/10 parsing, CCS grouping, ATC hierarchy, and code mapping helpers |

## Boundaries

OneEHR is an experimentation package. It does not provide clinical decision support, protected health information hosting, credential management, or access to licensed datasets.

Raw clinical datasets must be obtained and stored according to their own data-use agreements. OneEHR converters read local files and write derived CSV tables.

Built-in model baselines train from the run config and saved preprocessing artifacts. Models that use optional external assets, such as external KG files or text embeddings, record those paths through explicit model parameters and checkpoint metadata.

## Public Terms

Use these terms consistently across docs, examples, and tutorials:

- `OneEHR`
- `EHR AI platform`
- `longitudinal EHR experiments`
- `standardized EHR tables`
- `TOML experiment config`
- `run directory`
- `ML/DL models`
- `LLM or agent systems`
- `structured artifacts`
- `dataset converters`
- `medical code utilities`

Prefer concrete descriptions of what a user can run or inspect. Avoid competitive claims and broad statements that do not change how someone uses the package.
