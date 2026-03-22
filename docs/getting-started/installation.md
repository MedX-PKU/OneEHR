# Installation

OneEHR requires **Python 3.12+**.

## Quick Install (PyPI)

```bash
pip install oneehr
```

## From Source

```bash
git clone https://github.com/MedXLab/OneEHR.git
cd OneEHR
uv venv .venv --python 3.12
uv pip install -e .
```

## Verify The CLI

```bash
oneehr --help
```

You should see these top-level commands:

- `preprocess` — bin features, split patients
- `train` — train ML/DL models
- `test` — evaluate on test set
- `analyze` — cross-system comparison
- `plot` — publication-quality figures
- `convert` — convert raw datasets to OneEHR format

## Optional Extras

```bash
# Development and testing
pip install oneehr[test]

# Documentation
pip install oneehr[docs]

# Survival analysis (lifelines for fast C-index)
pip install oneehr[survival]
```

## GPU Support

OneEHR uses PyTorch for deep learning models. If you have a CUDA-capable GPU, install a CUDA-enabled PyTorch build. The trainer defaults to `device = "auto"` and will use CUDA when available.

```toml
[trainer]
device = "auto"       # auto | cpu | cuda
precision = "fp16"    # fp32 | fp16 | bf16
```

## LIME Support (Optional)

For LIME-based interpretability, install the `lime` package:

```bash
pip install lime
```
