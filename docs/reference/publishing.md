# Publishing To PyPI

Use this checklist when publishing OneEHR as the `oneehr` package on PyPI.

Official references:

- [Python Packaging User Guide: Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)

## 1. Prepare Accounts

Create accounts on both services:

- [PyPI](https://pypi.org/)
- [TestPyPI](https://test.pypi.org/)

Enable two-factor authentication on both accounts. For manual uploads, create an API token and use it only through your local keyring, `.pypirc`, or the interactive `twine` prompt. Do not commit tokens.

## 2. Check The Package Name

Before the first release, confirm that the package name is available:

```bash
uvx --from pip pip index versions oneehr
```

If PyPI returns an existing project that you do not control, change `[project].name` in `pyproject.toml` before publishing.

## 3. Update Release Metadata

For every release, update both version locations:

- `pyproject.toml`: `[project] version`
- `oneehr/__init__.py`: `__version__`

Then verify the public project URLs in `pyproject.toml`:

```toml
[project.urls]
Homepage = "https://medx-pku.github.io/OneEHR/"
Documentation = "https://medx-pku.github.io/OneEHR/"
Repository = "https://github.com/MedX-PKU/OneEHR"
Issues = "https://github.com/MedX-PKU/OneEHR/issues"
```

## 4. Run Local Checks

Run tests and lint before building distributions:

```bash
uv pip install -e ".[test]"
uv run pytest tests/ -v
uv run ruff check oneehr/ tests/
uv run ruff format --check oneehr/ tests/
```

## 5. Build And Validate Distributions

Build a source distribution and wheel from a clean working tree:

```bash
rm -rf dist/
uv run --with build python -m build
uv run --with twine twine check dist/*
```

The `twine check` command must pass before uploading.

## 6. Test On TestPyPI

Upload to TestPyPI first:

```bash
uv run --with twine twine upload --repository testpypi dist/*
```

Install from TestPyPI in a clean environment. The `--extra-index-url` keeps normal dependencies available from PyPI while the OneEHR package comes from TestPyPI:

```bash
python -m venv /tmp/oneehr-testpypi
source /tmp/oneehr-testpypi/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  oneehr
oneehr --help
python -c "import oneehr; print(oneehr.__version__)"
```

## 7. Upload To PyPI

After TestPyPI passes, upload the same distributions to PyPI:

```bash
uv run --with twine twine upload dist/*
```

Validate the public release:

```bash
python -m venv /tmp/oneehr-pypi
source /tmp/oneehr-pypi/bin/activate
python -m pip install --upgrade pip
python -m pip install oneehr
oneehr --help
python -c "import oneehr; print(oneehr.__version__)"
```

## 8. Tag The Release

Commit the release metadata before tagging:

```bash
git status --short
git tag v0.1.0
git push origin main --tags
```

Use the actual version number for the tag.

## 9. Prefer Trusted Publishing For Future Releases

After the first release, configure PyPI trusted publishing so GitHub Actions can publish without stored API tokens.

Create `.github/workflows/publish.yml`:

```yaml
name: Publish Python Package

on:
  release:
    types: [published]
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install uv
        run: python -m pip install --upgrade uv
      - name: Build
        run: uv run --with build python -m build
      - name: Check distributions
        run: uv run --with twine twine check dist/*
      - name: Publish
        uses: pypa/gh-action-pypi-publish@release/v1
```

In PyPI, add a trusted publisher for:

- Owner: `MedX-PKU`
- Repository: `OneEHR`
- Workflow name: `publish.yml`
- Environment name: `pypi`

For production releases, create a GitHub release for the version tag and let the workflow publish to PyPI.
