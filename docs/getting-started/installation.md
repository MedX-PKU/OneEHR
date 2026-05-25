# Installation

OneEHR requires Python 3.12 or newer.

## PyPI

```bash
pip install oneehr
oneehr --help
```

## Source Checkout

```bash
git clone https://github.com/MedX-PKU/OneEHR.git
cd OneEHR
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test]"
uv run oneehr --help
```

Use `oneehr ...` when the package is installed in the active environment. Use `uv run oneehr ...` when you are working from a checkout and want `uv` to run the command inside the project environment.

## CLI Commands

```bash
oneehr --help
```

Top-level commands:

| Command | Purpose |
|---------|---------|
| `preprocess` | Bin event data, build features, create labels, and save the split contract |
| `train` | Train configured ML/DL models |
| `test` | Evaluate trained models and configured systems on the test split |
| `analyze` | Write comparison, feature importance, fairness, calibration, statistical test, and missing-data outputs |
| `plot` | Render figures from run artifacts |
| `convert` | Convert supported raw datasets into OneEHR CSV tables |

## Optional Packages

```bash
pip install "oneehr[survival]"  # lifelines support for survival analysis
pip install lime                # optional LIME interpretability support
```

## GPU Setup

OneEHR uses PyTorch for deep learning models. Install a CUDA-enabled PyTorch build if you want GPU training, then keep the trainer device on `auto` or set it explicitly:

```toml
[trainer]
device = "auto"     # auto | cpu | cuda
precision = "fp32"  # fp32 | fp16 | bf16
```

## Documentation Preview

From a source checkout:

```bash
uv run --group docs mkdocs serve
```

MkDocs prints the local preview URL, usually `http://127.0.0.1:5000/`.
