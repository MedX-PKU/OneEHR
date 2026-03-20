# Installation

OneEHR requires **Python 3.12** and uses **uv** for environment and package management.

## Prerequisites

- Python 3.12, pinned in `.python-version`
- [uv](https://docs.astral.sh/uv/)

## Install OneEHR

```bash
uv venv .venv --python 3.12
uv pip install -e .
```

## Verify The CLI

```bash
uv run oneehr --help
```

You should see these top-level commands:

- `preprocess`
- `train`
- `test`
- `analyze`

## Optional Extras

Docs build dependencies:

```bash
uv pip install -e ".[docs]"
uv run mkdocs serve
uv run mkdocs build
```

## GPU Support

OneEHR uses PyTorch for deep learning models. If you have a CUDA-capable GPU, install a CUDA-enabled PyTorch build. The trainer defaults to `device = "auto"` and will use CUDA when available.

```toml
[trainer]
device = "auto"  # auto | cpu | cuda
```
