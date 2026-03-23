# Tutorials

Step-by-step Jupyter notebooks covering the full OneEHR workflow.

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | [Quickstart](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/01_quickstart.ipynb) | End-to-end mortality prediction on the TJH COVID-19 dataset |
| 2 | [Custom Dataset](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/02_custom_dataset.ipynb) | Bring your own data into OneEHR's three-table format + medical code mapping |
| 3 | [Model Comparison](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/03_model_comparison.ipynb) | Compare ML vs DL models with bootstrap CI and DeLong/McNemar statistical tests |
| 4 | [Fairness & Explainability](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/04_fairness_analysis.ipynb) | Bias detection across demographic groups + feature importance visualization |
| 5 | [Survival Analysis](https://github.com/MedX-PKU/OneEHR/blob/main/tutorials/05_survival_analysis.ipynb) | DeepSurv, DeepHit, concordance index, and Kaplan-Meier curves |

## Running Tutorials

### Local

```bash
pip install oneehr jupyter
jupyter notebook tutorials/
```

### Google Colab

Upload any notebook to [Google Colab](https://colab.research.google.com/) and add this cell at the top:

```python
!pip install oneehr
```

## What You'll Learn

### Tutorial 1: Quickstart
- Define an experiment with a TOML config file
- Preprocess EHR events into binned features
- Train XGBoost (ML) and GRU (DL) models
- Evaluate on a held-out test set
- Run cross-system comparison

### Tutorial 2: Custom Dataset
- Prepare dynamic, static, and label CSV files
- Use built-in dataset converters (MIMIC-III/IV, eICU)
- Apply medical code ontologies (ICD-9/10, CCS, ATC)
- Aggregate codes by ontology group for dimensionality reduction

### Tutorial 3: Model Comparison
- Configure multiple models in one experiment
- Bootstrap confidence intervals for AUROC, AUPRC, F1
- Pairwise statistical tests (DeLong, McNemar)
- Include LLM agents in the comparison

### Tutorial 4: Fairness & Explainability
- Detect bias across age, sex, ethnicity groups
- Compute demographic parity, equalized odds, predictive parity
- Feature importance: permutation, SHAP, attention weights
- Visualize attributions with waterfall and heatmap plots

### Tutorial 5: Survival Analysis
- Build DeepSurv and DeepHit models
- Train with Cox partial likelihood loss
- Evaluate with concordance index
- Plot stratified Kaplan-Meier survival curves
