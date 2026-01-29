# Installation

## Conda environment

From the repository root:

```bash
conda env create -f environment.yaml
conda activate saprop
```

## Jupyter kernel (one-time)

```bash
python -m ipykernel install --user --name saprop --display-name "SAProp (saprop)"
```


## Install PhlyGreen
```bash
pip install "git+https://github.com/rmalpica/PhlyGreen.git@5e5f082#subdirectory=trunk"
```

## Install pyAircraftEngineFramework
```bash
pip install "git+https://github.com/rmalpica/pyAircraftEngineFramework
```


