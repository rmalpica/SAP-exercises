# Sustainable Aircraft Propulsion — Python Exercises

This repository contains **Python scripts (`.py`)** and **Jupyter notebooks (`.ipynb`)** for the course **Sustainable Aircraft Propulsion**.
Exercises are organized by **course chapter** under `chapters/`.

---

## Repository structure

- `chapters/` — chapter-by-chapter exercises (each chapter contains `notebooks/`, `scripts/`, `data/`, `outputs/`)
- `docs/` — source for the MkDocs website (GitHub Pages)

---
## Installation and dependencies

All exercises run inside a dedicated **conda environment** to ensure reproducibility across systems.

### Required software
- Anaconda or Miniconda
- Git

---
### 1) Create and activate the conda environment

From the repository root:
```bash
conda env create -f environment.yml
conda activate saprop
```

### 2) Register the Jupyter kernel (one-time)

```bash
python -m ipykernel install --user --name saprop --display-name "SAProp (saprop)"
```

### 3) Install this repository (recommended) 
```bash
pip install -e .
```

### 4) Install the external dependency: PhlyGreen
Some exercises rely on the PhlyGreen Python package, developed separately and hosted on GitHub:
```bash
pip install git+https://github.com/rmalpica/PhlyGreen.git
```

---

## Running scripts

Run scripts from the repo root so relative paths behave consistently:

```bash
python chapters/300_mechanisms/scripts/EI.py
```

