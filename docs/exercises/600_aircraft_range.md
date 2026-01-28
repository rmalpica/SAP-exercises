# Exercise 600.5 — Aircraft range with alternative fuels  
## Breguet analysis

🧪 **Scripts**  
[`breguet_baseline_designpoint.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/600_hydrogen_combustion/scripts/equilibrium_calculation.py)
[`breguet_altern_fuel_designpoint.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/600_hydrogen_combustion/scripts/equilibrium_calculation.py)

---

This chapter explores how **fuel choice affects aircraft range and energy performance**
using **Breguet’s range equation** as a low-order system model.

We focus on **liquid hydrogen (LH₂)** and **conventional kerosene**, following the
methodology of:

> S. S. Jagtap et al., *Energy performance evaluation of alternative energy vectors for subsonic long-range tube-wing aircraft*,  
> Transportation Research Part D, 2023.

This exercise implements the models described in **Section 6.5.2** of the course notes  
(*Retrofitting an existing aircraft*).

Students are expected to **read the reference section carefully** before working on this exercise.
Equations and assumptions are **not re-derived here**.

---

## How to run

From the script folder (chapters/600_hydrogen_combustion/scripts):

```bash
python breguet_baseline_designpoint.py
python breguet_altern_fuel_designpoint.py
```

---

## Physical problem (context)


Rather than treating hydrogen as a “drop-in” fuel, this approach explicitly accounts for:

- gravimetric *and* volumetric energy density,
- aircraft weight breakdown (OEW, fuel, payload),
- aerodynamic penalties from increased fuselage length,
- system-level consequences on range and energy consumption.

The accompanying Python scripts reproduce the **logic of the paper**, not just its final numbers.

---

## Learning objectives

- Apply the Breguet range equation consistently
- Compare aircraft range for kerosene vs hydrogen
- Understand the role of fuel specific energy and storage penalties

---


## Guided questions

### 1) Why Breguet’s equation still matters

Breguet’s range equation provides a first-order estimate of aircraft range:

\[
R = \frac{h}{g} \, \eta_0 \, \frac{L}{D} \, \ln\!\left(\frac{W_\text{initial}}{W_\text{final}}\right)
\]

Despite its simplicity, it remains powerful because it exposes **where fuel choice enters the system**:

- \( h \): fuel lower heating value
- \( \eta_0 \): overall propulsion efficiency
- \( L/D \): aerodynamic efficiency
- \( W_\text{initial}, W_\text{final} \): aircraft mass evolution during cruise

Hydrogen fundamentally modifies **all four terms**, not just \( h \).

---

### 2) What the script actually computes

The Breguet scripts in this module implement the **design-point sizing loop** used in the paper.

At a high level, the algorithm:

1. Fixes:

    - payload mass
    - target range
    - cruise conditions
    - baseline aircraft geometry

2. Iterates on **total fuel mass** until:

    - the target range is achieved,
    - the aircraft stays below the MTOW constraint.

3. Updates, at each iteration:

    - fuel volume and required fuselage extension,
    - operating empty weight (OEW),
    - lift-to-drag ratio penalties,
    - initial and final cruise weights.

This is *not* a fuel substitution problem — it is a **vehicle resizing problem**.

---

### 3) Why hydrogen changes the aircraft, not just the fuel tank

Liquid hydrogen has:

- **very high gravimetric energy density** (~120 MJ/kg),
- **very poor volumetric energy density**.

As a consequence:

- Fuel mass decreases dramatically,
- Fuel volume increases dramatically,
- Fuselage length must increase to house cryogenic tanks,
- Wetted area, drag, and OEW increase,
- \( L/D \) decreases.

The script explicitly models these effects through:

- additional fuselage length,
- OEW scaling laws,
- aerodynamic penalties.

This is the central lesson:  
**hydrogen improves one term in Breguet’s equation while degrading others.**

---

### 4) Energy efficiency vs range: a non-intuitive result

One of the key results reproduced by the script is that:

- LH₂ aircraft have **higher specific energy consumption** than kerosene aircraft,
- but their SEC becomes **less sensitive to range beyond ~10,000 km**.

This happens because:

- LH₂ aircraft have a high OEW/GTOW ratio,
- energy consumption is dominated by carrying the aircraft itself,
- marginal range extension becomes relatively “cheap”.

This contradicts the naive intuition that *lighter fuel always means better efficiency*.

---

### 5) What this model does *not* include

This is intentionally a **low-order system model**.

It does **not** include:

- detailed structural sizing,
- detailed drag build-up or wave drag,
- engine cycle redesign,
- mission-level operational strategies,
- non-CO₂ climate effects (e.g. contrails).

Its strength lies in clarity, not completeness.

---

### 6) What you should learn from this exercise

By working with this script, you should be able to:

- explain why hydrogen aircraft require **vehicle-level redesign**,
- trace fuel properties through **mass → aerodynamics → range**,
- critically interpret published performance claims,
- distinguish *fuel efficiency* from *system efficiency*,
- understand why hydrogen is **not a silver bullet**, even at long range.

---

## Scope and reference aircraft

All results in this chapter are obtained using two **reference tube-and-wing aircraft**:

- an A320-class aircraft representing short-range missions,
- an A350-class aircraft representing long-range missions.

Hydrogen is introduced by modifying fuel properties, tank volume, and associated mass and aerodynamic penalties,
**without changing the underlying aircraft architecture**.

Therefore, conclusions drawn here apply to hydrogen integration into *conventional aircraft layouts* and
should not be extrapolated to radically different configurations.


---

## Student tasks

## Task 1 — Baseline reproduction (paper literacy)

Run the script for:

- kerosene (baseline),
- liquid hydrogen.

Report:

- achieved range,
- total fuel mass,
- operating empty weight (OEW),
- lift-to-drag ratio,
- specific energy consumption (SEC).

**Question:**  
Which Breguet term improves with hydrogen, and which ones degrade?

---

## Task 2 — Payload sensitivity

Reduce the payload in steps (e.g. −10 %, −20 %, −30 %).

For each case:

- recompute the maximum achievable range,
- plot range vs payload for kerosene and LH₂.

**Guided questions:**

- Which fuel benefits more from payload reduction?
- Why does LH₂ respond differently than kerosene?

---

## Task 3 — Range sensitivity and asymptotic behavior

Sweep target range from short-haul to ultra-long-haul.

Plot:

- SEC vs range for both fuels.

**Interpretation:**

- Why does LH₂ SEC flatten at long range?
- What dominates the energy balance in that regime?

---

## Task 4 — Engineering interpretation

In ~10 lines, answer:

- Why does LH₂ reduce GTOW but increase SEC?
- Why is volumetric energy density a *design* problem, not a fuel problem?
- Under what conditions (if any) could LH₂ become competitive?

---

## Task 5 — Critical reflection 

The paper concludes that LH₂ is viable but not energetically superior.

Write a short paragraph addressing:

- What assumptions drive this conclusion?
- Which assumptions might change with future aircraft architectures?
- Which ones are fundamentally hard to escape?
