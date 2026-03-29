<div align="center">
<img src="logo.png" alt="logo" width="250"></img>
</div>

> **Note:** This project is still in development.

# ATOM — Atomic DFT with finite elements

![CI](https://img.shields.io/badge/CI-private%20repo-lightgrey) [![PyPI](https://img.shields.io/badge/PyPI-not%20published-lightgrey)](https://pypi.org/project/atom-dft/)
<!-- After making the repo public, use the dynamic badge: [![CI](https://github.com/phanish-suryanarayana/atom/actions/workflows/ci.yaml/badge.svg)](https://github.com/phanish-suryanarayana/atom/actions/workflows/ci.yaml) -->

[**Features**](#features)
| [**Quick start**](#quick-start)
| [**Installation**](#installation)
| [**Change log**](ChangeLog)
| [**Documentation**](docs/)


## What is ATOM?

**ATOM** is a Python library for atomic (spherical) density functional theory (DFT) with a finite-element discretization in real space. It is heavily optimized and aims for high numerical accuracy.

ATOM solves the Kohn–Sham equations self-consistently and supports a wide range of exchange–correlation functionals. Calculations can be run in all-electron mode or by reading norm-conserving pseudopotential (PSP) files.

Advanced capabilities include the optimized effective potential (OEP) method, hybrid functionals with exact exchange (e.g. HF), RPA with parallelization, configurable parameters and advanced options, and more.

This is a research code. Please try it out, [report issues](https://github.com/phanish-suryanarayana/atom/issues), and share feedback.

```python
from atom import AtomicDFTSolver

# Single-atom DFT with GGA-PBE
solver = AtomicDFTSolver(atomic_number=13, xc_functional="GGA_PBE")
results = solver.solve()

# Access total energy, density, eigenvalues, etc.
print(results["energy"])
```

### Contents

- [ATOM — Atomic DFT with finite elements](#atom--atomic-dft-with-finite-elements)
  - [What is ATOM?](#what-is-atom)
    - [Contents](#contents)
  - [Features](#features)
  - [Quick start](#quick-start)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Instructions](#instructions)
  - [Project structure](#project-structure)
  - [Optional dependencies](#optional-dependencies)
  - [Citing ATOM](#citing-atom)
  - [Reference documentation](#reference-documentation)


## Features

* **Finite-element discretization** — Real-space mesh and operators in `atom.mesh`.
* **Pseudopotentials** — Norm-conserving pseudopotential support (e.g. psp8) in `atom.pseudo`.
* **SCF driver** — Density, Hamiltonian, eigensolver, Poisson, mixing, and convergence in `atom.scf`.
* **Exchange–correlation** — LDA, GGA-PBE, hybrid (HF), and ML-XC in `atom.xc`.
* **Data and ML** — Dataset generation, loading, and ML-XC training interfaces in `atom.data` and `atom.xc.ml_xc`.


## Quick start

```python
from atom import AtomicDFTSolver

# xc_functional can be any supported functional (e.g. GGA_PBE, LDA_PZ, PBE0, ...)
solver = AtomicDFTSolver(atomic_number=29, xc_functional="GGA_PBE")
results = solver.solve()

# Many options available: domain_size, mesh, grid, SCF settings, verbose, etc.
solver = AtomicDFTSolver(
    atomic_number=6,
    xc_functional="LDA_PZ",
    domain_size=15.0,
    verbose=True,
)
results = solver.solve()
```


## Installation

### Requirements

* Python ≥ 3.8
* NumPy ≥ 1.20
* SciPy ≥ 1.7

### Instructions

| Use case        | Command |
|-----------------|---------|
| Core (CPU)      | `pip install -e .` or `pip install atom` |
| With ML-XC      | `pip install -e ".[ml]"` |
| With viz        | `pip install -e ".[viz]"` |
| Dev + tests     | `pip install -e ".[dev]"` |
| All optional    | `pip install -e ".[all]"` |

From the repository root:

```bash
cd delta/atom
pip install -e .
```


## Project structure

| Directory / module | Description |
|--------------------|-------------|
| `atom/mesh`        | Grid construction and operators |
| `atom/pseudo`      | Pseudopotential reading and evaluation (local / non-local) |
| `atom/scf`         | SCF loop: density, Hamiltonian, eigensolver, Poisson, mixer |
| `atom/xc`          | XC functionals: LDA, GGA, HF, ML-XC, etc. |
| `atom/data`        | Data generation, loading, and processing |
| `atom/utils`       | Occupation states, periodicity helpers |
| `tests`            | Unit and integration tests |
| `docs`             | Tutorial and documentation source |


## Optional dependencies

| Extra   | Purpose |
|---------|---------|
| `ml`    | PyTorch, scikit-learn for ML-XC |
| `viz`   | Matplotlib for plotting |
| `dev`   | pytest, Jupyter for development |
| `threadpool` | threadpoolctl for RPA/thread control |
| `docs`  | Jupyter Book, Sphinx for building docs |


## Citing ATOM

If you use this code in your research, please cite the repository:

```
@software{atom2026,
  author = {Qihao Cheng and Shubhang Trivedi and Phanish Suryanarayana},
  title = {{ATOM}: Atomic density functional theory with finite elements},
  url = {https://github.com/phanish-suryanarayana/atom},
  version = {0.1.0},
  year = {2026},
}
```


## Reference documentation

For API details and tutorials, see the [documentation](docs/) in this repository.

For development and contribution guidelines, see the [repository](https://github.com/phanish-suryanarayana/atom).
