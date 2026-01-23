# Exercise 300.2 — Emission indices with Cantera  
## Homogeneous constant-pressure reactor + pollutant EI (CO, NO, CO₂, H₂O)

🧪 **Script**  
[`EI.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/300_mechanisms/scripts/EI.py)

Supporting files (same folder):

- Mechanism: [`CRECK_2003_TOT_HT_NOX.yaml`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/300_mechanisms/scripts/CRECK_2003_TOT_HT_NOX.yaml)
- (Optional) Mechanism: [`nDodecane_Reitz.yaml`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/300_mechanisms/scripts/nDodecane_Reitz.yaml)
- Helper functions: [`efficiency_functions.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/300_mechanisms/scripts/efficiency_functions.py)

---

## Learning objectives

By completing this exercise, you will learn to:

- Run a **zero-dimensional homogeneous reactor** simulation in Cantera
- Compute time-dependent **emission indices (EI)** on a mass basis:
  $EI_X = \frac{m_X}{m_{\text{fuel consumed}}} \quad [\mathrm{kg/kg}]$
- Understand how EI depends on:
  - equivalence ratio \(\phi\)
  - temperature history
  - residence time
  - kinetic mechanism (species set and rate models)
- Relate emissions formation qualitatively to combustion completeness (CO vs CO₂) and NO formation

---

## What the script does (high level)

The script:

1) Loads a chemical mechanism that includes **dodecane** fuel chemistry and **NOx**
2) Sets initial conditions (temperature \(T_0\), pressure \(P_0\), equivalence ratio \(\phi\))
3) Runs a **constant-pressure ideal gas reactor**
4) At each time step:

   - extracts mass fractions of CO₂, H₂O, CO, NO
   - estimates fuel consumed from the decrease in fuel mass fraction
   - computes emission indices \(EI\)
   - estimates a combustion efficiency proxy using enthalpy change vs. LHV

It then plots EI(t), temperature(t), and the efficiency proxy.

---

## How to run

From the repository root:

```bash
python chapters/300_mechanisms/scripts/EI.py
```

--- 

## Guided questions

### 1) Definitions and bookkeeping

- **Why is EI defined per kg of fuel consumed rather than per kg of exhaust?**
  Think about what you want EI to represent: a quantity that is comparable across engines, operating points, and dilution levels. Normalizing by exhaust mass would make EI depend strongly on how much air is mixed/diluted, not on how much pollutant is produced *per unit of fuel energy input*.

- **In the script, how is “fuel consumed” estimated? What assumptions does that imply?**
  Identify how the script computes fuel consumption (typically from the decrease of the fuel species amount/mass fraction relative to its initial value, or from carbon balance).
  Then discuss assumptions such as:
  - the selected “fuel species” represents the whole fuel (single surrogate)
  - no other fuel-containing species should be counted as “unburned fuel”
  - the reactor is closed (no mass enters/leaves) and perfectly mixed
  - numerical stiffness/step size does not distort the fuel consumption estimate

---

### 2) Residence time (reactor time)

- **Increase/decrease `t_end` (end time) by 10×.**
  Run the same case with a much longer/shorter integration horizon.

- **Do EI values converge to a steady value?**
  Which EIs approach a plateau, and which keep drifting?

- **Which species converge fast vs slow?**
  Typical expectations (to be checked against your runs):
  - **CO₂ and H₂O** often stabilize relatively quickly once major oxidation is complete.
  - **CO** may continue to decrease as CO oxidizes to CO₂ (if conditions allow).
  - **NO** may continue evolving depending on temperature history and the balance of formation/destruction pathways, and can be sensitive to residence time.

---

### 3) Equivalence ratio effects

- **Change `eqratio`** (lean → stoichiometric → rich).

- **Leaner mixtures: do you expect CO to rise or fall? Why?**
  Consider how oxygen availability and temperature change with mixture strength. In many simplified settings:
  - lean conditions provide more oxygen and can reduce CO *if temperatures stay high enough* for CO oxidation,
  - but very lean conditions can lower temperature and slow kinetics, potentially increasing incomplete combustion indicators depending on the model.

- **How does NO behave with temperature and oxygen availability?**
  Discuss qualitatively:
  - higher temperatures often promote NO formation (strong temperature sensitivity),
  - oxygen availability matters (leaner mixtures can provide more O, but may reduce temperature),
  - rich conditions may suppress NO due to lower O availability even if temperatures are high early.

---

### 4) Temperature sensitivity

- **Change `T0` upward/downward.**

- **How do the trends for CO and NO differ?**
  Compare sensitivities:
  - CO is strongly linked to oxidation completeness and kinetic time scales.
  - NO is often extremely temperature-sensitive and can respond differently than CO.

- **What does that tell you about formation pathways?**
  Use your results to infer whether the dominant behavior is:
  - “oxidation completion” controlled (CO),
  - “high-temperature formation” controlled (NO),
  - or influenced by both through the temperature–chemistry coupling.

---

### 5) Mechanism dependence (optional)

- **Switch between available mechanisms.**

- **What changes: absolute EI values, trends, or both?**
  Note whether mechanisms preserve qualitative trends but shift magnitudes, or whether trends can change too.

- **Why should an engineer be cautious when quoting EI numbers?**
  Because EI can depend on:
  - mechanism completeness (species/rates)
  - NOx sub-model choices
  - surrogate fuel choice
  - reactor assumptions (perfect mixing, no quench, no dilution)
  - residence time representation
  So EI from a 0-D homogeneous reactor is best treated as a *model-based indicator*, not a “ground-truth engine EI”.

---

## Student tasks

### Task 1 — “Baseline” EI and convergence

Run the script as provided and report:

- Final values at the end time:
  - \(EI_{CO_2}\)
  - \(EI_{H_2O}\)
  - \(EI_{CO}\)
  - \(EI_{NO}\)

- For each EI, state whether it has **converged** by the final time
  (**yes/no + brief justification** based on the slope or change over the last portion of the simulation).

**Suggested convergence check:**
Compare EI at the final time with EI at (final time − 10% of simulation window). If the relative change is below a small threshold, treat as “converged”.

---

### Task 2 — Parametric study (choose one axis)

Pick **one** parameter and sweep **5–8 values**:

- Equivalence ratio \( \phi \)
- Initial temperature \( T_0 \)
- Pressure \( P_0 \)
- Residence time \( t_{end} \)

**Deliver:**
- One plot of **final EI** vs the chosen parameter (you can plot multiple species EIs on the same figure if readable).
- One paragraph interpreting the trends physically (oxygen availability, temperature effects, kinetic time scales, etc.).

---

### Task 3 — Engineering interpretation: which EI matter for what?

Write **8–12 lines** connecting:

- **CO₂** → climate impact mainly through **cumulative emissions** (long-lived greenhouse gas)
- **NOx** → atmospheric chemistry (ozone formation, methane lifetime changes) and potential climate relevance
- **CO** → indicator of incomplete combustion and relevance for local air quality and operational issues

Your answer should distinguish between:
- *direct climate forcing* vs *chemistry-mediated effects* vs *air-quality/local impacts*.

---

### Task 4 — Reactor model matters

Switch to a Continuously Stirred Tank Reactor (CSTR) model (available in Cantera). This is an open system (unlike a batch reactor, which is isolated). The mass flow rate at which reactants come in is the same as that of the products which exit, and on average this mass stays in the reactor for a characteristic time $\tau$, called the residence time. 

- Observe how residence time affects the steady solution.

---

## Limitations (important)

This is a homogeneous, perfectly stirred **0-D reactor** model. It does **not** capture:

- mixing-controlled combustion or flame structure
- turbulence–chemistry interaction
- finite-rate mixing, quenching, wall heat transfer
- combustor staging, dilution zones, liners, residence-time distributions

Treat results as **conceptual and comparative**, not as certified engine EI values.
