# Installation

ATOM is currently used from source.

## Requirements

- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+

## Core Installation

From the repository root:

```bash
pip install -e .
```

## Optional Extras

| Use case | Command |
| --- | --- |
| Core (CPU) | `pip install -e .` |
| ML dependencies | `pip install -e ".[ml]"` |
| Visualization | `pip install -e ".[viz]"` |
| Development and tests | `pip install -e ".[dev]"` |
| Documentation build | `pip install -e ".[docs]"` |
| All optional extras | `pip install -e ".[all]"` |

## Quick Sanity Check

```bash
python -c "from atom import AtomicDFTSolver; print('ATOM import OK')"
```

## Documentation Build

If you want to build the Jupyter Book locally:

```bash
pip install -e ".[docs]"
jupyter-book build docs/ --all
```

The HTML output is written to `docs/_build/html/`.
