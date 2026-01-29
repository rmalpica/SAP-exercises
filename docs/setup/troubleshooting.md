# Troubleshooting

## I get `ModuleNotFoundError`
- Confirm you activated the env: `conda activate saprop`
- In notebooks, confirm kernel is **SAProp (saprop)**

## Package import fails
- Recreate env from scratch:
```bash
conda env remove -n saprop
conda env create -f environment.yaml
```

## Conda command not found (Windows)

- Open **Anaconda Prompt**, not PowerShell or CMD

## Wrong Python version in VS Code

- In VS Code:

    - press `Ctrl+Shift+P`
    - select **Python: Select Interpreter**
    - choose the one from the `saprop` environment

## Jupyter does not see the kernel

- Re-run the kernel installation step (Section 5)