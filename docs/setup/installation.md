# Installation and setup

This page explains how to set up a **working Python environment** for the
*Sustainable Aircraft Propulsion* exercises.

Instructions are provided for:

- **Windows**
- **macOS**

No prior Python experience is assumed.

---

## 1. Install Python and basic tools

### 1.1 Install Miniconda (recommended)

We use **Conda** to manage dependencies in a reproducible way.

#### Windows

1. Go to: <https://docs.conda.io/en/latest/miniconda.html>  
2. Download **Miniconda (Python 3.x, 64-bit, Windows)**  
3. Run the installer:
    - choose *“Just Me”*
    - allow the installer to **initialize Conda**
    - allow Conda to modify your PATH (recommended)

After installation, open **Anaconda Prompt** (from the Start Menu).

#### macOS

1. Go to: <https://docs.conda.io/en/latest/miniconda.html>  
2. Download **Miniconda (Python 3.x, macOS)** for your architecture:
    - Apple Silicon (arm64), or
    - Intel (x86_64)
3. Run the installer (`.pkg`) and follow the default options.

After installation, open **Terminal**.

---

### 1.2 Verify Conda installation

In a terminal (macOS) or Anaconda Prompt (Windows):

```bash
conda --version
```

You should see a Conda version printed.

---

## 2. Install a code editor (Visual Studio Code)

We strongly recommend **Visual Studio Code (VS Code)**.

1. Download from: <https://code.visualstudio.com/>
2. Install using default options.

### Recommended VS Code extensions

After opening VS Code, install:

- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)

These enable:

- Python script execution
- Jupyter notebook support
- integrated debugging

---

## 3. Get the course repository

Clone the repository (recommended):

```bash
git clone https://github.com/rmalpica/SAP-exercises.git
cd SAP-exercises
```

If you do not have Git installed:

- Windows: <https://git-scm.com/download/win>  
- macOS: Git is usually available by default, or via Xcode Command Line Tools

---

## 4. Create the Conda environment

From the **repository root directory**:

```bash
conda env create -f environment.yaml
conda activate saprop
```

This installs all required dependencies (NumPy, SciPy, Cantera, PyTorch, etc.).

---

## 5. Register the Jupyter kernel (one time only)

This makes the environment available inside Jupyter:

```bash
python -m ipykernel install --user --name saprop --display-name "SAProp (saprop)"
```

---

## 6. Install external course packages

### 6.1 Install PhlyGreen

PhlyGreen is used in the **Hybrid-Electric Propulsion** chapter.

```bash
pip install ""git+https://github.com/rmalpica/PhlyGreen.git@5e5f082#subdirectory=trunk"
```

---

### 6.2 Install pyAircraftEngineFramework

This package is used in selected engine-modeling exercises.

```bash
pip install "git+https://github.com/rmalpica/pyAircraftEngineFramework"
```

---

## 7. Running the exercises

### 7.1 Python scripts

Activate the environment and run:

```bash
conda activate saprop
python script.py
```

Example:

```bash
cd chapters/300_mechanisms/scripts/ 
python EI.py
```

---

### 7.2 Jupyter notebooks

From the repository root:

```bash
conda activate saprop
jupyter lab
```

Then:

1. Open the notebook you want to run
2. Select kernel **“SAProp (saprop)”**

---

## 8. Common issues

See [Troubleshooting](./troubleshooting.md).

---

## 9. Final check

If everything is correctly installed:

- Python scripts should run without import errors
- Jupyter notebooks should execute using the **saprop** kernel

You are now ready to work on the course exercises.
