# Exercise 600.3 — Turbofan design trade-offs  
## Pareto analysis: hydrogen vs kerosene

🧪 **Script**  
[`Turbofan_pareto_NSGA_H2_vs_Ker.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/600_hydrogen_combustion/scripts/Turbofan_pareto_NSGA_H2_vs_Ker.py)

---

## Learning objectives

By completing this exercise, you will learn to:

- Interpret **Pareto frontiers** in a turbofan design context
- Distinguish *fuel effects* from *architectural constraints*
- Understand how hydrogen influences **feasible design space**, not just point designs
- Analyze how **design variables evolve along a Pareto frontier**
- Recognize when a new fuel does *not* fundamentally change trade-offs

---

## How to run

From the script folder (chapters/600_hydrogen_combustion/scripts):

```bash
python Turbofan_pareto_NSGA_H2_vs_Ker.py
```

---

## What the script does

The script performs a **multi-objective optimization** of a conventional turbofan engine
using a genetic algorithm (NSGA-type approach).

Two fuels are considered:

- kerosene,
- hydrogen.

For each fuel, the algorithm explores a range of engine designs by varying parameters such as:

- bypass ratio (BPR),
- turbine inlet temperature \( T_4 \),
- pressure ratios,
- other cycle-related design variables.

The optimization seeks **non-dominated solutions**, forming a Pareto frontier.

---

## Objectives and constraints (conceptual)

While details depend on the implementation, the Pareto optimization typically balances:

- **objective 1**: a performance or efficiency-related metric (e.g. TSEC, TSFC, or fuel flow),
- **objective 2**: a thermal or technological constraint proxy (e.g. specific thrust, or maximum temperature).

All designs share:

- the same engine architecture,
- the same cycle assumptions,
- the same optimization bounds.

Only **fuel properties** differ between the two cases.

---

## Key observation from the results

When comparing the Pareto frontiers:

- The **overall shape** of the Pareto frontiers for hydrogen and kerosene is **very similar**.
- Hydrogen designs exhibit a **slightly lower maximum temperature** \(T_\text{max}\) for comparable performance levels.
- No dramatic expansion or contraction of the feasible design space is observed.

This indicates that:
> For a conventional turbofan architecture, fuel choice alone does **not** fundamentally alter the core design trade-offs.

---

## Guided questions

### 1) Interpreting similar Pareto frontiers

- Why might hydrogen and kerosene lead to **nearly overlapping Pareto frontiers**?
- Which aspects of the engine dominate the trade-off structure, regardless of fuel?

Discuss in terms of:

- Brayton-cycle constraints,
- fixed architecture,
- unchanged aerodynamic assumptions.

---

### 2) Maximum temperature as a distinguishing feature

- Why does hydrogen systematically allow for **slightly lower \(T_\text{max}\)**?
- How does this relate to:

  - fuel–air ratio,
  - exhaust composition,
  - thermal efficiency trends observed in Exercise 600.2?

Is this difference likely to be *technologically important*?

---

### 3) Design-variable trends along the frontier

For **each fuel**, examine how design parameters evolve along the Pareto front:

- bypass ratio (BPR),
- turbine inlet temperature \(T_4\),
- pressure ratios.

Questions to address:

- Do optimal designs move toward higher or lower BPR as performance improves?
- Is \(T_4\) always pushed to its upper bound?
- Are trends similar for hydrogen and kerosene?

---

### 4) Pareto dominance and engineering choice

- Does hydrogen dominate kerosene anywhere on the Pareto front?
- If not, what does this imply about “fuel-driven” optimization narratives?

Explain the difference between:

- *incremental advantage*,
- *structural dominance*.

---

## Student tasks

### Task 1 — Pareto front comparison (core)

Plot the Pareto frontiers for:

- kerosene,
- hydrogen,

on the same axes.

Identify:

- regions of overlap,
- any systematic offsets,
- the role of \(T_\text{max}\) as a constraint.

---

### Task 2 — Design-variable mapping

Select **3–5 representative points** along each Pareto frontier
(e.g. low-performance, mid-range, high-performance designs).

For each point, report:

- BPR,
- \(T_4\),
- at least one additional design parameter.

Discuss how these parameters evolve along the frontier and whether trends differ between fuels.

Color the pareto front by values of BPR or \(T_4\) and explain the trends.

---

### Task 3 — Engineering interpretation 

In **10–12 lines**, answer:

- Why does hydrogen not dramatically reshape the Pareto frontier of a conventional turbofan?
- Why is a small reduction in \(T_\text{max}\) still potentially meaningful?
- What types of architectural changes would be required to obtain **qualitatively different Pareto frontiers**?

---

## Limitations (important)

This Pareto analysis assumes:

- a fixed turbofan architecture,
- fixed component efficiency models,
- no heat recovery or cryogenic integration,
- no airframe–engine coupling.

As a result:

- the optimization explores *fuel substitution*, not *fuel-enabled redesign*.

---

## Key takeaway

In a conventional turbofan,
**architecture sets the Pareto structure**.

Hydrogen can shift operating points slightly,
but **only architectural innovation can reshape the frontier itself**.
