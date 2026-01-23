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


### Install PhlyGreen
```bash
pip install "git+https://github.com/rmalpica/PhlyGreen.git@3f42cbf907dba655bad3cfe8114d7c9bcda7f02a#subdirectory=trunk"
```


