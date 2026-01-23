# Exercise 300.1 — Contrail formation models  
## Schmidt–Appleman criterion and contrail diagrams

📓 **Notebook**  
[`Contrail_Formation_Models.ipynb`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/300_mechanisms/notebooks/Contrail_Formation_Models.ipynb)

---

## Learning objectives

By completing this exercise, you will learn to:

- Explain the **Schmidt–Appleman criterion (SAC)** as a thermodynamic condition for contrail formation
- Construct and interpret saturation vapor pressure curves (over **water** and **ice**)
- Use an **isobaric mixing line** to determine whether the exhaust/ambient mixture crosses saturation
- Understand the geometric method for finding the **critical formation temperature**
- Explore how engine and ambient parameters shift contrail formation thresholds

This is a *first-principles* way to understand contrail formation before introducing meteorology and microphysics.

---

## Conceptual model (what the notebook is doing)

The notebook builds the pieces needed for the SAC:

### 1) Saturation curves
It computes saturation vapor pressure over:

- liquid water \(e_w(T)\)
- ice \(e_i(T)\)

and plots the curves vs. temperature.

### 2) Isobaric mixing line
A simplified mixing model links the exhaust plume state to ambient conditions along a line in \((T, e)\) space with slope \(G\).  
The slope depends on quantities such as:

- water emission index \(EI_{H_2O}\)
- specific heat \(c_p\)
- ambient pressure \(p\)
- fuel lower heating value \(Q\)
- an overall efficiency parameter \(\eta\)

### 3) Critical temperature (geometric approach)
The contrail formation threshold can be obtained by finding a **tangency condition** between the mixing line and a saturation curve.
This yields the *critical formation temperature* and allows building a Schmidt–Appleman diagram.

---

## How to run

```bash
jupyter lab
```

Open:
```bash
chapters/300_mechanisms/notebooks/Contrail_Formation_Models.ipynb
```

---

## Guided questions

### 1) What exactly triggers contrail formation in this model?

- **In the Schmidt–Appleman criterion (SAC), what does it mean physically for the mixing line to intersect the saturation curve?**  
  Interpret the intersection as the point at which the exhaust–ambient mixture becomes saturated with respect to water (or ice). Discuss what “saturation” represents in terms of phase change potential in the expanding plume.

- **What role does the choice “over water” vs “over ice” play?**  
  Compare the saturation vapor pressure over liquid water and over ice. Explain why, at typical cruise temperatures, ice saturation is often the relevant threshold and how this affects the predicted contrail formation conditions.

---

### 2) Sensitivity to engine parameters

Change **one parameter at a time**, keeping all others fixed:

- Water emission index \( EI_{H_2O} \)
- Efficiency-like parameter \( \eta \)
- Fuel lower heating value \( Q \)

For each variation:

- **How does the critical formation temperature shift?**
- Which parameter appears to have the strongest influence, and why?

Relate your observations to the physical meaning of each parameter in the mixing-line formulation.

---

### 3) Sensitivity to ambient conditions (altitude proxy)

The model includes ambient pressure \( p \), which can be interpreted as an **altitude proxy**.

- **What happens to the contrail formation threshold when \( p \) decreases (higher altitude)?**
- Does the resulting trend align with the intuition that contrails are more likely at cruise altitude?

Explain your answer in terms of thermodynamics rather than operational experience alone.

---

### 4) Geometry and interpretation

- **Why is tangency used for the “critical” case in the Schmidt–Appleman construction?**  
  Explain why the tangent condition represents the limiting case between contrail formation and no formation.

- **What does being “above” vs “below” the saturation curve imply?**  
  Discuss the physical meaning of these regions in the context of plume evolution and phase change potential.

---

## Student tasks

### Task 1 — Build and explain a Schmidt–Appleman diagram (core)

Produce a clean figure showing:

- saturation curve(s) (water and/or ice)
- the isobaric mixing line
- the critical point (tangent or intersection)

Write **6–10 lines** explaining:

- how the diagram is constructed,
- what the critical point represents,
- how contrail formation is determined from the geometry.

---

### Task 2 — Parameter sweep (engineering intuition)

Choose **one** of the following parameters and vary it over a reasonable range:

- water emission index \( EI_{H_2O} \), or
- efficiency parameter \( \eta \), or
- ambient pressure \( p \)

**Deliver:**

- a plot showing how the **critical formation temperature** changes with the chosen parameter,
- a short physical interpretation answering: *which parameter matters most and why?*

---

### Task 3 — Link back to propulsion choices (course integration)

In a short paragraph, answer the following:

> Which propulsion or operational levers can change contrail propensity in this simplified view  
> (e.g. efficiency, fuel type, water emission index, operating altitude),  
> and which ones cannot?

Explicitly relate your answer to the assumptions built into the Schmidt–Appleman criterion.

---

## Limitations (important)

This notebook isolates the **thermodynamic threshold** for contrail formation. It does **not** model:

- ice microphysics or particle growth
- ambient humidity variability and supersaturation fields
- plume turbulence and detailed mixing processes
- contrail persistence or radiative forcing

Results should therefore be interpreted as **necessary conditions for formation**, not as predictions of contrail lifetime or climate impact.

