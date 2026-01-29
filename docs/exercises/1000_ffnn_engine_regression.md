# Exercise 1000.1 — FFNN regression for engine prediction  
## A simple neural network surrogate (PyTorch)

### Files
🧪 **Script**  
[`FFNN.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/1000_ffnn_engine_regression/scripts/FFNN.py)

🧪 **Dataset**  
[`b777_engine_inputs.dat`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/1000_ffnn_engine_regression/data/b777_engine_inputs.dat)
[`b777_engine_outputs.dat`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/1000_ffnn_engine_regression/data/b777_engine_outputs.dat)

🧪 **Supporting script**  
[`b777_engine.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/1000_ffnn_engine_regression/scripts/b777_engine.py)

- Supporting script (model/data context): `b777_engine.py`

---

## How to run

From the script folder (chapters/1000_ML/scripts):

```bash
python FFNN.py
```

Expected outputs typically include:

- printed training progress (loss vs epoch)
- saved plots (e.g., loss history, parity plots) in the chapter outputs/ folder

---


## Learning objectives

By completing this exercise, you will learn to:

- Train a feed-forward neural network (FFNN) to approximate an engineering mapping
- Split data into train/validation/test sets and evaluate generalization
- Use appropriate regression diagnostics beyond a single loss value
- Understand why scaling and data leakage matter
- Interpret ML results in an engineering way: *where does the surrogate work, and where does it fail?*

---

## What you are modeling

You are learning a surrogate of the form:

\[
\mathbf{x} \rightarrow \mathbf{y}
\]

where \(\mathbf{x}\) comes from `b777_engine_inputs.dat` and \(\mathbf{y}\) from `b777_engine_outputs.dat`.

You should treat this like a real engineering surrogate task:
- inputs may have different units and scales,
- outputs may have different magnitudes and sensitivities,
- the model may behave poorly near the edges of the data domain.

The role of `b777_engine.py` is to give context on what the variables represent (and/or how the dataset was generated).

---

## Guided questions

### 1) Data and scaling

- Are the input variables on comparable numerical scales?
- What happens to training if **no normalization** is applied?
- If normalization is used, how can **data leakage** be avoided  
  (i.e. using test-set statistics during training)?

*Hint:* scaling parameters should be computed using training data only.

---

### 2) Train / validation / test split

- How sensitive are the results to the random split seed?
- Do all outputs generalize equally well, or do some exhibit larger errors?

---

### 3) Underfitting vs overfitting

- If the network capacity is strongly reduced (few layers / neurons), what failure mode is observed?
- If the network capacity is strongly increased, what failure mode appears?
- How do training and validation loss curves help diagnose these behaviors?

---

### 4) What is a good validation metric?

- Is mean squared error (MSE) sufficient for this problem?
- Would you prefer:

    - coefficient of determination \(R^2\),
    - relative error,
    - mean absolute error (MAE),
    - error normalized by the output scale?

Explain your choice from an **engineering interpretation** perspective.

---

### 5) Where does the surrogate fail?

Using parity plots or residual plots:

- Are errors larger near the boundaries of the input domain?
- Are errors predominantly:

    - systematic (bias), or
    - random (variance)?
- Does the model violate any expected physical trends  
  (e.g. monotonicity with respect to certain inputs)?

---

## Student tasks

### Task 1 — Baseline training and evaluation (core)

Run the script as provided and report:

- final training loss,
- final validation loss,
- test-set performance using **at least one metric** of your choice.

Produce:
- one parity plot (predicted vs true) for at least one output variable.

Write **6–10 lines** discussing whether the surrogate is acceptable for engineering use,
and justify your assessment.

---

### Task 2 — Architecture study (capacity vs generalization)

Train at least **three** different networks by varying **one** architectural choice:

- number of hidden layers, or
- number of neurons per layer, or
- activation function (if supported by the script).

Deliver:
- training and validation loss curves for each case,
- a short interpretation explaining:

    - where underfitting occurs,
    - where overfitting begins,
    - where performance saturates.

---

### Task 3 — Data efficiency experiment

Repeat the training using reduced fractions of the available dataset
(e.g. 20%, 50%, and 100% of the training data).

Deliver:
- a plot of test error versus number of training samples,
- a short paragraph answering:

    - how much data is required to reach acceptable accuracy,
    - which outputs are most data-hungry.

---

### Task 4 — Engineering interpretation (short essay)

In **10–12 lines**, answer the following:

- What makes this neural-network surrogate trustworthy (or not)?
- What additional validation steps would you require before using it
  inside a mission analysis or design loop?
- What risks arise when optimizing designs using a surrogate
  that is inaccurate near the boundaries of the input domain?

---

## Limitations (important)

This exercise demonstrates supervised regression with a feed-forward neural network,
but it does **not** address:

- uncertainty quantification or confidence intervals,
- extrapolation outside the data domain,
- physical constraints (e.g. conservation laws, monotonicity),
- bias or sparsity in the training dataset,
- coupling with optimization or decision-making loops.

The FFNN should be treated as a **conditional surrogate**:
its predictions are meaningful only within the region of the input space
covered by the training data.
