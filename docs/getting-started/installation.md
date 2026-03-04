# Installation

OneEHR requires **Python 3.12** and uses **uv** as its dependency manager.

---

## Prerequisites

- Python 3.12 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
# Create a virtual environment
uv venv .venv --python 3.12

# Install OneEHR in editable mode
uv pip install -e .
```

## Verify

```bash
uv run oneehr --help
```

You should see the four subcommands: `preprocess`, `train`, `test`, `analyze`.

## Optional: docs dependencies

To build the documentation site locally:

```bash
uv pip install -e ".[docs]"
uv run mkdocs serve  # MkDocs 2 pre-release, default http://127.0.0.1:5000
```

## GPU support

OneEHR uses PyTorch for deep learning models. If you have a CUDA-capable GPU, ensure your PyTorch installation includes CUDA support. The `trainer.device` config defaults to `"auto"`, which will use CUDA if available.

```toml
[trainer]
device = "auto"  # auto | cpu | cuda
```
