# Positioning

Use this page when you need canonical product language for README copy, docs updates, slides, demos, or release notes.

## Canonical Product Identity

Preferred short label:

- `EHR AI platform`

Preferred one-sentence description:

> OneEHR is a unified Python platform for longitudinal EHR experiments across ML, DL, and LLM agents.

Preferred expanded description:

> OneEHR is a unified Python platform for longitudinal EHR experiments. It provides shared infrastructure for preprocessing, modeling, analysis, and reproducible evaluation across AI agents, LLM systems, and conventional ML/DL models on one shared run contract — the first toolkit bridging classical machine learning, deep learning, and agentic AI for clinical prediction.

## Core Claims

Use these ideas repeatedly and consistently:

- **Unified ML/DL/LLM comparison** — the first platform that evaluates classical models, deep learning, and LLM agents on the same contract.
- **25 model architectures** — tabular ML, recurrent, transformer, Mamba, EHR-specialised, and survival models.
- **Built-in dataset support** — converters for MIMIC-III, MIMIC-IV, and eICU with standard clinical tasks.
- **Medical code ontologies** — ICD-9/10 mapping, CCS grouping, ATC drug classification for dimensionality reduction.
- **Survival analysis** — Cox PH and discrete-time models with concordance index and Kaplan-Meier visualization.
- **Statistical rigor** — bootstrap confidence intervals, DeLong test, McNemar test, BH FDR correction as defaults, not afterthoughts.
- **Fairness-aware** — demographic parity, equalized odds, predictive parity, SMD integrated into the standard workflow.
- **Interpretability** — SHAP, LIME, integrated gradients, permutation importance, attention visualization.
- **Publication-ready outputs** — ROC, PR, calibration, DCA, forest plots, KM curves with Nature/Lancet style presets.
- **Reproducibility by design** — TOML config = complete experiment specification, Parquet + JSON artifacts.

## Preferred Terms

- `platform`
- `AI agents`
- `LLM systems`
- `conventional ML/DL models`
- `shared run contract`
- `cross-system evaluation`
- `dataset converters`
- `medical code ontologies`
- `survival analysis`
- `standardized EHR tables`
- `reproducible artifacts`

## Terms To Avoid

Do not use these as top-level positioning language:

- `toolkit` (use `platform`)
- `library`
- `all-in-one`
- `task-first`
- `artifact-first`
- `single-LLM`
- `multi-agent medical framework`
- `infra platform`
- `ML/DL baselines`

## Competitive Positioning

| Dimension | OneEHR | PyHealth | ehrapy |
|-----------|--------|----------|--------|
| Focus | Unified ML/DL/LLM evaluation | DL model breadth | Statistical EHR analysis |
| Models | 25 (ML + DL + survival) | 33+ (DL-focused) | ~0 DL |
| LLM support | Native (unified contract) | None | None |
| Datasets | MIMIC-III/IV, eICU converters | 10+ built-in loaders | MIMIC via ehrdata |
| Medical codes | ICD-9/10, CCS, ATC | ICD, ATC, NDC, RxNorm, UMLS | FHIR |
| Survival | DeepSurv, DeepHit, KM | None | KM, Cox PH, AFT |
| Statistical tests | DeLong, McNemar, bootstrap CI | None | GLM, ANOVA |
| Fairness | 4 metrics + auto-detect | None | Bias detection + SMD |
| Interpretability | SHAP, LIME, IG, attention | 15+ methods | Feature ranking |
| Causal inference | Not in scope | None | DoWhy integration |
| Config | TOML (complete contract) | Python code | Python code |

## Reusable Copy Blocks

Homepage or hero eyebrow:

- `EHR AI platform`

Short intro:

- `OneEHR is a unified Python platform for longitudinal EHR experiments across ML, DL, and LLM agents.`

Medium intro:

- `OneEHR is a unified Python platform for longitudinal EHR experiments. It provides 25 model architectures, built-in dataset converters for MIMIC and eICU, medical code ontologies, survival analysis, and publication-quality visualization — all on one shared run contract.`

Long intro:

- `OneEHR is a unified Python platform for longitudinal EHR experiments across ML, DL, and LLM agents. It provides shared infrastructure for preprocessing, modeling, analysis, and reproducible evaluation on one shared run contract — the first toolkit bridging classical machine learning, deep learning, and agentic AI for clinical prediction. With 25 model architectures, built-in converters for MIMIC-III/IV and eICU, ICD/CCS/ATC ontologies, survival analysis, fairness analysis, and publication-quality visualization with Nature/Lancet style presets.`

## Scope Boundaries

Prefer:

- `AI agents, LLM systems, and conventional ML/DL models`

The canonical abstraction levels for OneEHR:

- product identity: `platform`
- system scope: `AI agents`, `LLM systems`, `conventional ML/DL models`
- architecture: `shared run contract`
- data: `MIMIC-III/IV`, `eICU`, `ICD-9/10`, `CCS`, `ATC`
