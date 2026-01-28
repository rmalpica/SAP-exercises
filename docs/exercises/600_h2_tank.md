# Exercise 600.4 — Hydrogen tank thermodynamics  
## Pressure build-up and boil-off in a sealed LH₂ tank

🧪 **Script**  
[`h2_tank.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/600_hydrogen_combustion/scripts/h2_tank.py)

### Reference
This exercise implements the models described in **Section 6.4.5** of the course notes  
(*Hydrogen aircraft design: the storage challenge*).

Students are expected to **read the reference section carefully** before working on this exercise.
Equations and assumptions are **not re-derived here**.

---

## How to run

From the script folder (chapters/600_hydrogen_combustion/scripts):

```bash
python h2_tank.py
```

---

## Learning objectives

By completing this exercise, you will learn to:

- Interpret the **thermodynamic evolution of a sealed cryogenic hydrogen tank**
- Distinguish between **non-equilibrium (isothermal)** and **two-phase equilibrium** modeling approaches
- Understand how **heat leaks translate into pressure rise and boil-off**
- Identify numerical and physical consistency requirements in two-phase simulations
- Connect time-domain results to the **hydrogen phase diagram**

---

## Physical problem (context)

A rigid, non-vented cryogenic tank contains liquid hydrogen near its normal boiling point.
A small but continuous **heat leak** from the environment causes:

- partial vaporization of the liquid,
- growth of the vapor mass and volume,
- increase in tank pressure.

Two modeling approaches are considered in the reference material:

1. **Approach 1** — Constant liquid temperature, fixed ullage, ideal-gas vapor  
2. **Approach 2** — Two-phase equilibrium at saturation, with evolving temperature and vapor volume  

The script focuses on **illustrating these mechanisms numerically**, not on detailed tank design.

---

## What the script does

Depending on the selected model option, the script:

- applies a prescribed heat leak \( Q \),
- evolves the tank state in time,
- computes and plots quantities such as:

    - pressure \( P(t) \),
    - temperature \( T(t) \),
    - vapor mass \( M_{\mathrm{vap}}(t) \),
    - liquid mass \( M_{\mathrm{liq}}(t) \).

For the two-phase model, saturation properties are enforced consistently with the phase diagram,
as described in the reference notes.

---

## Guided questions

### 1) Physical interpretation of pressure rise

- Why can **very small heat inputs** (order of 1–10 W) lead to **large pressure increases** over a few hours?
- Which physical mechanism dominates the pressure rise:

    - sensible heating,
    - or latent heat associated with vaporization?

Relate your answer explicitly to the energy balance discussed in the reference.

---

### 2) Comparison of modeling approaches

Using the results from the script:

- How does the pressure evolution differ between:

    - the constant-temperature / ideal-gas approach,
    - the two-phase equilibrium approach?
- In which sense can Approach 1 be interpreted as a **bounding or limiting case**?

Discuss under which physical conditions each approach might be more appropriate.

---

### 3) Role of saturation and the phase diagram

- In the two-phase model, why is the condition  
  $P(t) = p_{\mathrm{sat}}(T(t))$
  enforced at all times?
- How does this constraint shape the trajectory of the solution in the \( P\text{–}T \) plane?

Use the hydrogen phase diagram from the reference notes to support your explanation.

---

### 4) Initial-condition consistency (numerical robustness)

The reference emphasizes the importance of choosing a **consistent initial vapor mass**.

- What happens if the initial vapor mass is not consistent with:

    - total mass,
    - tank volume,
    - saturation properties at \( T(0) \)?
- Which unphysical behaviors may appear in the numerical solution?

Explain why these issues are *not* merely numerical bugs, but indicators of an inconsistent physical state.

---

### 5) Time scales and operational relevance

- Over what time scale does pressure reach potentially critical values?
- How does this compare with:

    - typical ground operation times,
    - turnaround times,
    - cruise durations?

What does this imply for **tank venting, pressure regulation, or mission planning**?

---

## Student tasks

### Task 1 — Baseline reproduction

Run the script using the baseline parameters provided.

Produce plots of:

- \( P(t) \),
- \( T(t) \),
- \( M_{\mathrm{vap}}(t) \),
- \( M_{\mathrm{liq}}(t) \).

Briefly describe the qualitative behavior of each variable.

---

### Task 2 — Heat-leak sensitivity

Increase and decrease the heat leak \( Q \) by at least a factor of 5.

For each case:

- compare pressure rise rates,
- identify whether temperature or phase change dominates.

Discuss whether the response is linear or nonlinear.

---

### Task 3 — Initial-condition robustness test

Deliberately modify the initial vapor mass so that it is **inconsistent** with saturation at \( T(0) \).

Observe:

- early-time behavior of pressure and temperature,
- any numerical instabilities or nonphysical values.

Explain why enforcing consistency at \( t = 0 \) is essential in two-phase modeling.

---

### Task 4 — Engineering interpretation 

In ~10–12 lines, answer:

- Why is hydrogen storage primarily a **thermal management problem**, not an energy problem?
- Why does tank pressurization remain a concern even with excellent insulation?
- Which design or operational strategies could mitigate these issues?

---

## Limitations (important)

This exercise uses a **lumped-parameter, rigid-tank model**.
It does **not** account for:

- structural stresses and tank mechanics,
- active venting or pressure regulation,
- sloshing or stratification,
- coupling with aircraft mission phases,
- transient non-equilibrium phase-change kinetics.

The goal is **physical insight**, not certification-level modeling.

---

## Key takeaway

Even in an idealized setting, cryogenic hydrogen storage exhibits
**strong coupling between heat transfer, phase change, and pressure rise**.

This coupling — not combustion — is one of the dominant constraints
on hydrogen-powered aircraft.
