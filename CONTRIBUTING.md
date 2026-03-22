# Contributing to OneEHR

Thank you for your interest in contributing to OneEHR! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/MedXLab/OneEHR.git
cd OneEHR

# Create a virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install in development mode with test dependencies
uv pip install -e ".[test]"

# Run the test suite
pytest tests/ -v
```

## Code Style

- Follow PEP 8 with 120-character line limit
- Use type hints for all function signatures
- Use `from __future__ import annotations` in all modules
- Prefer `dataclass(frozen=True)` for data containers
- Use Google-style docstrings

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check oneehr/ tests/
ruff format oneehr/ tests/
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `test:` — adding or updating tests
- `docs:` — documentation changes
- `refactor:` — code refactoring (no behavior change)
- `ci:` — CI/CD changes
- `chore:` — other maintenance tasks

## Adding a New Model

1. Create `oneehr/models/<model_name>.py` with patient-level and time-level variants
2. Both classes must accept `(x: Tensor, lengths: Tensor, **kwargs) -> Tensor`
3. Register the model name in `oneehr/models/__init__.py`:
   - Add to `DL_MODELS` frozenset
   - Add default hyperparameters to `_DL_DEFAULTS`
   - Add builder case in `build_dl_model()`
4. Add tests in `tests/test_all_models.py`
5. Document in `docs/reference/models.md`

## Adding a New Dataset Converter

1. Create `oneehr/datasets/<dataset>.py` inheriting from `BaseConverter`
2. Implement the `convert()` method returning `ConvertedDataset`
3. Register in `oneehr/datasets/__init__.py`
4. Add CLI support in `oneehr/cli/main.py` and `oneehr/cli/convert.py`
5. Add tests with synthetic data in `tests/test_datasets.py`

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=oneehr --cov-report=term-missing

# Run a specific test file
pytest tests/test_metrics.py -v
```

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass before requesting review
