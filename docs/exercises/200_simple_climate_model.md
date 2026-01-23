# Exercise 200 — Simple climate model  
## A minimal climate–carbon system

📓 **Notebook**  
[`simple_climate_model.ipynb`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/200_sustainability/notebooks/simple_climate_model.ipynb)

---

## Learning objectives

By completing this exercise, you will learn to:

- Translate **CO₂ emissions** into changes in atmospheric concentration
- Compute **radiative forcing** from CO₂ using a logarithmic law
- Simulate the **temperature response** using a zero-dimensional energy balance model
- Understand the role of **system inertia**, feedbacks, and time scales
- Interpret numerical results in a **policy- and engineering-relevant** way

This exercise is a *thinking tool*, not a predictive climate model.

---

## Model overview

The notebook implements two related models of increasing physical realism.

---

### Model A — Temperature anomaly + CO₂ concentration

State variables:

- \(T(t)\): global-mean **temperature anomaly** (°C)
- \(C_{\mathrm{atm}}(t)\): atmospheric CO₂ (ppm)

#### Temperature equation

\[
\frac{dT}{dt}
=
\frac{
F_0 + 5.35\ln\!\left(C_{\mathrm{atm}}/280\right)
-\lambda_{\mathrm{cl}}\,T
}{C}
\]

- \(C\): effective heat capacity (response time)
- \(\lambda_{\mathrm{cl}}\): climate feedback parameter
- \(F_0\): background forcing term (optional)

#### Carbon cycle (one-box)

\[
\frac{dC_{\mathrm{atm}}}{dt}
=
E_{\mathrm{ppm}}(t)
-\beta\left(C_{\mathrm{atm}}-280\right)
\]

- Emissions are converted using  
  \(1\,\mathrm{ppm} \approx 2.13\,\mathrm{GtC}\)
- \(\beta\) represents net carbon uptake by sinks

---

### Model B — Blackbody energy balance + carbon mass

This variant tracks absolute temperature \(T\) (K) and carbon mass (GtC):

\[
C\,\frac{dT}{dt}
=
\frac{S(1-\alpha)}{4}
+
5.35\ln\!\left(\frac{C_{\mathrm{atm}}}{C_{\mathrm{atm},0}}\right)
-
\sigma T^4
\]

It highlights:

- the nonlinear nature of outgoing radiation
- the difference between **temperature anomaly** and **absolute temperature**

---

## How to run the notebook

```bash
jupyter lab
```

---

## Guided questions

Use these questions to structure your exploration.

### 1) Emissions vs temperature

- What happens to temperature if emissions become zero instantly?
- Does temperature stop increasing immediately?
- Why or why not?

### 2) Time scales

- Which variable responds fastest: emissions, CO₂ concentration, or temperature?
- Which parameter controls this behavior?

### 3) Carbon sinks

- How does changing the uptake parameter $\beta$ affect long-term CO₂?
- Is temperature recovery symmetric with temperature increase?

### 4) Feedback strength

- How does $\lambda_{cl}$ influence equilibrium temperature?
- Can you interpret it physically?

