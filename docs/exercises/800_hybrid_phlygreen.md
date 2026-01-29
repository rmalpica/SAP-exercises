# Exercise 800.1 — Hybrid-electric mission test case (PhlyGreen)

## Reference documentation

This notebook uses the **PhlyGreen** library. The official documentation is here:

- PhlyGreen docs: https://rmalpica.github.io/PhlyGreen/

You are expected to consult the PhlyGreen documentation for:

- class definitions and available models,
- parameter meanings and units,
- plotting utilities and conventions.

This exercise page focuses on *what to explore and how to report results*.

---

## Notebook
🧪 **Jupyter Notebook**  
[`hybrid.ipynb`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/800_hybrid_electric/notebooks/hybrid.ipynb)

This is a simple test case with:

- a prescribed **hybridization strategy** (power split),
- a **battery model (Class II)**,
- a mission-level simulation loop.

---

## Learning objectives

By completing this exercise, you will learn to:

- Run a hybrid-electric design loop using an existing engineering codebase
- Identify key drivers of feasibility:
  - peak power,
  - battery energy capacity,
  - efficiency losses,
  - battery constraints (C-rate, usable SoC window, etc.)
- Explore sensitivity to assumptions in a structured way
- Produce a short engineering narrative from simulation results

---

## How to run (minimum path)

Provided that the PhlyGreen code has been successfully installed (see [Installation Setup →](../setup/installation.md)), launch jupyter:

```bash
jupyter lab
```

and open the notebook hybrid.ipynb.

---

## What to focus on

PhlyGreen exposes many parameters and modeling options.  
For this exercise, you are **not expected** to explore everything.

Focus on a **small set of high-impact levers**:

- **Power split ratio** along the mission  
  (how much power is provided electrically vs thermally, and when). This is controlled by the phi values in the MissionStages dictionary.

- **Battery cell and pack assumptions**  
  (energy density, power capability, C-rate,usable state-of-charge window). These are controlled by the Battery class parameters in the CellInput dictionary.

- **Powertrain efficiencies**  
  (electrical path vs thermal path losses). These are controlled by the Efficiencies in the EnergyInput dictionary.

- **Mission constraints**  
  (mission profile, range, payload, etc.). These are controlled by the MissionInput and MissionStages dictionaries.

Your goal is to understand **which assumptions dominate the results**,  
not to fine-tune a single “optimal” configuration.

---

## Guided questions

### 1) Baseline behavior: where does mass come from?

In the baseline configuration:

- What are the main contributors to aircraft mass?
- How is total mass partitioned between:
    - propulsion system,
    - fuel,
    - battery system,
    - remaining aircraft structure,
    - payload

Identify which subsystem dominates mass **before any modification is applied**.

---

### 2) Snowball effects of hybridization level

Increase the global hybridization level (power split) gradually.

- How does required battery mass evolve?
- How does increased battery mass feed back into:
    - required power,
    - required energy,
    - overall aircraft mass?

Describe the **snowball mechanism** linking hybridization → mass → power → mass.

---

### 3) Power vs energy driven redesign

Modify the power split strategy while keeping all components feasible.

- When hybridization is increased mainly during high-power phases (e.g. climb):
    - which mass component grows most rapidly?
- When hybridization is increased during lower-power phases (e.g. cruise):
    - does mass growth behave differently?

- What happens to the battery pack architecture (number of cells in series/parallel)?

Explain whether the redesign is primarily **power-driven** or **energy-driven**.

---

### 4) Efficiency improvements and indirect mass effects

Improve drivetrain efficiency by a small amount.

- Does total aircraft mass decrease?
- Is the mass reduction proportional to the efficiency improvement?
- Which mass components benefit most indirectly from improved efficiency?

Explain how **small efficiency gains** can propagate nonlinearly through the design loop.

---

### 5) Battery technology assumptions and feasibility illusion

Modify battery technology parameters (e.g. energy density).

- Does improved battery performance always lead to a lighter aircraft?
- Are there regimes where mass continues to increase despite “better” batteries?

Discuss why improving a component does not necessarily improve the **system**.

---

### 6) Design stability and diminishing returns

As hybridization is increased:

- Does aircraft mass grow smoothly or accelerate?
- Can you identify a region of **diminishing returns**, where added hybridization yields little benefit but large mass penalties?

Explain how this behavior emerges from repeated aircraft re-design rather than explicit constraints.


## Student tasks

### Task 1 — Baseline run and constraint diagnosis (core)

Run the notebook as provided and deliver:

- one figure showing the mission power split (thermal vs electric),
- one figure showing battery state evolution (SoC and/or relevant limits),
- a short diagnosis (**6–10 lines**) explaining:
  - what limits the system in the baseline case,
  - during which mission phase the limit is reached.

---

### Task 2 — One structured sensitivity sweep

Choose **one** parameter and sweep **5–8 values**, for example:

- constant hybridization level,
- single-phase hybridization (e.g. climb only),
- battery energy density,
- drivetrain efficiency.

Deliver:
- one plot of a key outcome versus the swept parameter  
  (e.g. battery mass, final SoC margin, constraint violations),
- **6–10 lines** interpreting the trend physically.

---

### Task 3 — Strategy comparison (engineering narrative)

Define **two** hybridization strategies (A and B), for example:

- A: high hybridization during climb, low during cruise  
- B: moderate hybridization throughout the mission

Compare both strategies using the same battery model.

Write **10–12 lines** discussing:
- which strategy is more feasible and why,
- what trade-off it represents (power peaks vs energy consumption),
- which strategy you would recommend if battery mass were constrained.

---

### Task 4 — Critical reflection (mandatory)

In **8–12 lines**, answer:

- which results are robust (architecture-level insights),
- which results are fragile (strongly assumption-dependent),
- what additional data or modeling would be required before design decisions.

---

## Limitations (important)

- This is a simplified test case intended for learning and sensitivity exploration.
- Results depend strongly on:
  - mission definition,
  - battery model fidelity and parameter assumptions,
  - drivetrain efficiency assumptions,
  - chosen power split strategy.
- Structural, thermal, and safety aspects of battery integration are not modeled.
- Aerodynamic and structural effects of battery integration are not modeled (e.g. increase drag). Aircraft structural mass simply scales with take-off mass.
- Results should not be interpreted as optimized or certified designs.

Treat all outputs as **conditional engineering evidence**, not final truth.

---

## Key takeaway

Hybrid-electric feasibility is governed by **mission power profiles** and **constraints**.

Maximizing hybridization is rarely optimal.
The key engineering question is **where and when electrical power is most valuable**,
given realistic limits on battery power, energy, and mass.


