# Troubleshooting

## I get `ModuleNotFoundError`
- Confirm you activated the env: `conda activate saprop`
- In notebooks, confirm kernel is **SAProp (saprop)**

## Cantera import fails
- Recreate env from scratch:
```bash
conda env remove -n saprop
conda env create -f environment.yaml
```
