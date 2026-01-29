# Exercise 1000.UQ.1 — Bayesian inference of a reduced climate model

## Reference and scope

This exercise is **fully detailed in the lecture notes**, in section 10.14:

> **“An end-to-end uncertainty quantification workflow: a reduced climate model”**

Students are expected to **read this section carefully before starting** the exercise.
All equations, model structure, and physical assumptions are defined there and are **not re-derived here**.

The Python script provided in this repository is a **direct implementation** of the workflow
described in the notes.

---

## Script and data

🧪 **Script**  
[`bayesian_climate.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/1000_bayesian_climate_inference/scripts/bayesian_climate.py)

🧪 **Datasets**  

[Historical temperature data](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/1000_bayesian_climate_inference/data/T_data.csv)

[Historical CO₂ data](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/1000_bayesian_climate_inference/data/CO2_data.csv)

[Historical CH₄ data](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/1000_bayesian_climate_inference/data/CH4_data.csv)

These datasets correspond to those shown in the lecture notes (e.g. figures of historical data and posterior predictive checks).

---

## Learning objectives

By completing this exercise, you will learn to:

- Implement an **end-to-end Bayesian uncertainty quantification workflow**
- Calibrate a **reduced-order dynamical climate model** using historical data
- Distinguish clearly between:

    - prior assumptions,
    - likelihood (data constraints),
    - posterior uncertainty

- Interpret MCMC output beyond point estimates
- Perform **posterior predictive checks**
- Analyze **time-dependent parameter importance** using variance-based sensitivity analysis
- Understand why uncertainty grows in future projections even when historical fit is good

---

## What the script does (high-level workflow)

The script follows exactly the steps described in the lecture notes:

1. **Define the forward model**

    - A coupled system of ODEs describing:

        - CO₂ concentration
        - CH₄ concentration
        - radiative forcing terms
        - planetary albedo
        - global mean temperature anomaly
        - aerosol forcing

2. **Load historical observations**

    - Temperature anomaly
    - CO₂ concentration
    - CH₄ concentration

3. **Specify uncertainty**

    - Bounded uniform priors for all uncertain parameters
    - Gaussian observational error model

4. **Bayesian inference**

    - Compute the log-posterior
    - Find a MAP estimate (optimization)
    - Use MCMC (ensemble sampler) to explore the posterior distribution

5. **Posterior analysis**

    - Marginal and joint posterior distributions
    - Parameter correlations

6. **Posterior predictive uncertainty**

    - Forward propagation of posterior samples
    - Probabilistic future projections under different emission scenarios

7. **Global sensitivity analysis**

    - Time-dependent Sobol-style variance contributions
    - Time-integrated parameter importance

---

## How to run

From the repository root:

```bash
python chapters/1000_UQ/scripts/bayesian_climate.py
```

---

## Practical notes

- This script can be computationally expensive depending on:

    - number of MCMC walkers,
    - number of MCMC steps,
    - ODE solver tolerances,
    - use of multiprocessing.

- For a **quick test run**, you may:

    - reduce the number of walkers,
    - reduce the number of MCMC steps,
    - shorten the historical time window,
    - disable multiprocessing.

- A full run reproduces figures and behaviors discussed in the lecture notes.
  Expect runtimes from several minutes to longer, depending on settings.

- Always verify that chains are well mixed before interpreting results.

---

## Guided questions

### 1) Prior vs data: who constrains what?

- Which parameters are strongly constrained by the historical data?
- Which parameters remain close to their prior bounds?
- How can this be inferred from marginal posterior distributions?

Relate your answer to the discussion of weak and informative priors in the lecture notes.

---

### 2) Parameter correlations and identifiability

- Which parameters exhibit strong posterior correlations?
- What physical or structural reasons explain these correlations?
- Why is joint inference essential in this model?

Support your answer using joint posterior plots.

---

### 3) MAP estimate vs posterior distribution

- Does the MAP parameter set provide a good fit to historical data?
- Are there distinct parameter combinations that fit the data almost equally well?
- What does this imply about relying on a single “best-fit” solution?

---

### 4) Posterior predictive checks

- Does the posterior predictive interval capture the historical observations?
- Are there systematic deviations during specific time periods?
- Should these deviations be interpreted as parameter uncertainty or model limitations?

---

### 5) Uncertainty growth in future projections

- Why does uncertainty grow with time even after calibration?
- Why does uncertainty differ between business-as-usual and mitigation scenarios?

Relate your answer to feedback mechanisms discussed in the notes.

---

### 6) Time-dependent sensitivity

- Which parameters dominate uncertainty in the early historical period?
- Which parameters dominate long-term projections?
- Why does parameter importance change with time?

Use the time-resolved variance contribution results to justify your answer.

---

## Student tasks

### Task 1 — Baseline calibration and posterior summary 

Run the script as provided and deliver:

- one plot comparing historical temperature data with:

    - MAP prediction,
    - posterior predictive uncertainty band;

- one marginal or joint posterior plot (or a subset of a corner plot).

Write **8–12 lines** discussing:

- quality of the fit,
- dominant sources of uncertainty,
- whether the model appears overconfident or underconfident.

---

### Task 2 — Robustness to inference choices

Modify **one** modeling or inference assumption, for example:

- widen or narrow selected parameter priors,
- change the assumed observational noise level,
- change the number of MCMC walkers or steps.

Deliver:

- one figure illustrating the impact on posterior or predictive uncertainty,
- **6–10 lines** explaining what changed and why.

---

### Task 3 — Identifiability analysis

Select two strongly correlated parameters and:

- plot their joint posterior distribution,
- explain the physical or mathematical origin of the correlation,
- discuss what additional data or modeling changes could reduce this degeneracy.

---

### Task 4 — Sensitivity interpretation

Using the sensitivity-analysis outputs:

- identify the most important parameters by **time-integrated contribution**,
- identify at least one parameter whose importance changes with time,
- explain what this implies for long-term projections.

Write **8–12 lines**.


---

## Limitations (important)

This exercise illustrates a complete Bayesian UQ workflow, but it has important limitations:

- The climate model is highly reduced and omits many physical processes.
- Structural model discrepancy is not explicitly represented.
- Observational uncertainty is treated in a simplified manner.
- No rigorous convergence diagnostics are enforced.
- Results are conditional on the chosen model structure and data.

All results should be interpreted as:

> uncertainty conditioned on a specific model class,  
> not as definitive predictions of the real climate system.


---

## Key takeaway

This exercise shows that:

- good agreement with historical data does **not** imply predictive certainty;
- uncertainty redistributes across parameters and time horizons;
- long-term projections are dominated by feedback-driven uncertainty.

Replacing single trajectories with **probability distributions**
is essential for responsible engineering analysis and decision-making.

