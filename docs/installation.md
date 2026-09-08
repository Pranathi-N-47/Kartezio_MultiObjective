---
title: Installation Guide - Kartezio
description: How to install Kartezio with uv or pip, including dev dependencies.
---

# Installation

This guide walks you through setting up Kartezio and the Multi-Objective (MOO) extension on your system.

---

## System Prerequisites

Kartezio is designed to run efficiently on standard CPU architectures without requiring dedicated GPUs, CUDA drivers, or heavy tensor runtimes.

- **Operating System**: Linux, macOS, or Windows (x86_64 or ARM64)
- **Python Version**: Python 3.11 or higher (`python --version`)
- **C/C++ Build Tools**: Standard C compiler for building Numba / OpenCV wheels if binaries are not pre-cached

---

## Recommended: Installation with `uv`

[`uv`](https://github.com/astral-sh/uv) is an extremely fast, modern Python package installer and resolver.

### 1. Clone the Repository

```bash
git clone https://github.com/Pranathi-N-47/kartezio-multiobjective.git
cd kartezio-multiobjective
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment with Python 3.11+
uv venv .venv

# Activate environment:
# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Install Package in Editable Mode

```bash
# Standard installation
uv pip install -e .

# With optional development & testing dependencies
uv pip install -e ".[dev]"
```

---

## Alternative: Installation with Standard `pip`

If you prefer using standard Python `pip` and `venv`:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 2. Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# 3. Install in editable development mode
pip install -e .

# 4. Install test and documentation tools
pip install pytest pytest-cov ruff mkdocs mkdocs-material
```

---

## Core Dependencies

The package requires the following core computer vision and numerical libraries, which are resolved automatically:

| Library | Purpose |
| :--- | :--- |
| `numpy` | High-performance multi-dimensional array operations |
| `scipy` | Mathematical routines and spatial filters |
| `opencv-python` | High-speed image processing primitives and transformations |
| `scikit-image` | Advanced image segmentation, morphology, and sample datasets |
| `numba` | Just-In-Time (JIT) acceleration for graph execution loops |
| `pandas` | Tabular structuring and CSV export of Pareto front results |
| `matplotlib` | Visualization, Pareto frontier plots, and robustness charts |
| `tabulate` | Clean ASCII and Markdown table formatting for terminal output |

---

## Documentation Website Setup

To build and preview this interactive documentation locally:

```bash
# Install MkDocs and Material theme
pip install mkdocs mkdocs-material pymdown-extensions

# Launch local hot-reloading dev server
mkdocs serve
```

Navigate to `http://127.0.0.1:8000` in your web browser. Any edits made to Markdown files in `docs/` will automatically trigger an instant live reload.

To compile static HTML files for deployment:

```bash
mkdocs build --strict
```

The compiled site is written to the `site/` directory.

---

## Verifying the Installation

Verify that the core Kartezio and MOO extension components can be imported successfully:

```bash
python -c "
import kartezio
import cv2
import numpy as np
print('Kartezio successfully imported!')
print(f'NumPy version: {np.__version__}')
print(f'OpenCV version: {cv2.__version__}')
"
```

If no errors are displayed, your environment is ready to start evolving multi-objective pipelines!
