# Exercise 600.2 — Turbofan performance  
## Hydrogen vs kerosene in a conventional turbofan cycle

🧪 **Script**  
[`Turbofan_ker_vs_h2.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/600_hydrogen_combustion/scripts/Turbofan_ker_vs_h2.py)

### Reference
This exercise follows the discussion in **Section 6.2 — Effect of hydrogen combustion on cycle performance**
of the lecture notes.

Students are expected to be familiar with:

- TSFC vs TSEC,
- thermal vs propulsive efficiency,
- the Brayton-cycle interpretation of turbofan engines.

---

## How to run

From the script folder (chapters/600_hydrogen_combustion/scripts):

```bash
python Turbofan_ker_vs_h2.py
```

---

## Learning objectives

By completing this exercise, you will learn to:

- Compare turbofan performance **across fuels with different energy densities**
- Understand why **TSFC is misleading** when comparing hydrogen and kerosene
- Use **TSEC** as a physically meaningful comparison metric
- Interpret how fuel properties affect:

    - exhaust velocities,
    - efficiencies,
    - turbine inlet temperature requirements
- Distinguish *mass-flow effects* from *thermodynamic effects*

---

## What the script does

The script compares a **conventional turbofan cycle** fueled with:

- kerosene (reference),
- hydrogen,

under three different comparison logics.

All cases assume:

- identical engine architecture,
- identical bypass ratio,
- identical component efficiencies,
- no unconventional cycle features (no intercooling, no recuperation).

Only **fuel properties and resulting cycle states** are changed.

---

## Comparison modes implemented in the script

### 1) Same design parameters (same TIT, OPR, BPR)

In this mode:

- turbine inlet temperature \(T_4\) is fixed,
- overall pressure ratio and bypass ratio are identical,
- the hydrogen engine is *not retuned*.

This isolates the **pure effect of fuel substitution**.

**Observed trends (see the script output):**

- TSFC decreases dramatically (≈ −64%) due to hydrogen’s high LHV
- TSEC increases slightly (+1.5%)
- thermal efficiency increases
- propulsive efficiency decreases
- specific thrust increases

This confirms a key message from the notes:
> *Lower TSFC does not imply lower energy consumption.*

---

### 2) Same TSEC comparison (energy-fair comparison)

In this mode:

- hydrogen turbine inlet temperature \(T_4\) is **reduced**
- TSEC is forced to be equal to the kerosene case

This answers the question:
> *What hydrogen cycle delivers the same energy efficiency as kerosene?*

**Key observations:**

- required hydrogen \(T_4\) is lower than kerosene
- TSFC is still much lower (mass-flow effect)
- overall efficiency becomes nearly identical
- thermal efficiency remains slightly higher for hydrogen
- propulsive efficiency is slightly lower

This result is **central** to the lecture notes:
> For comparable Brayton cycles, TSEC is broadly similar across fuels.

---

### 3) Same specific thrust comparison (operational equivalence)

In this mode:

- hydrogen \(T_4\) is adjusted so that **specific thrust is identical**
- the engine delivers the same thrust per unit airflow

This corresponds to a practical **engine matching** condition.

**Observed trends:**

- TSEC becomes slightly lower for hydrogen
- TSFC remains much lower
- thermal efficiency increases
- propulsive efficiency decreases marginally

This illustrates that hydrogen can:
- maintain thrust,
- reduce turbine temperatures,
- without improving energy efficiency dramatically.

---

## Guided questions

### 1) TSFC vs TSEC

- Why does TSFC drop by more than 60% in all hydrogen cases?
- Why does TSEC change only marginally?
- Which metric is appropriate for *cycle efficiency* comparisons, and why?

---

### 2) Exhaust velocity and propulsive efficiency

- Why does the **core exhaust velocity** increase with hydrogen?
- How does this affect propulsive efficiency?
- Why does the bypass stream remain unchanged?

Relate your answer to the discussion of exhaust composition and \(c_p\) in the notes.

---

### 3) Thermal efficiency gains

- Why does hydrogen show a systematic increase in thermal efficiency?
- How is this related to:
  - gas composition,
  - turbine expansion,
  - heat capacity effects?

Explain why this does *not* automatically translate into lower TSEC.

---

### 4) Turbine inlet temperature as a design lever

- Why can hydrogen achieve the same TSEC or thrust with a **lower \(T_4\)**?
- What are the implications for:
  - NOx formation,
  - turbine blade life,
  - material limits?

---

## Student tasks

### Task 1 — Comparative table interpretation (core)

Using the script output, summarize in a short table:

- TSFC
- TSEC
- thermal efficiency
- propulsive efficiency
- specific thrust

for:

- same-design,
- same-TSEC,
- same-specific-thrust cases.

For each case, write **2–3 lines** explaining the dominant physical mechanism.

---

### Task 2 — Efficiency decomposition

For one comparison mode of your choice:

- explain qualitatively how:

  - mass flow rate,
  - combustion chamber composition,
  - turbine expansion
contribute to the observed efficiency changes.

You may refer to the Brayton-cycle interpretation from the notes.

---

### Task 3 — Engineering interpretation 

In **10–12 lines**, answer:

- Why does hydrogen *not* dramatically reduce TSEC in a conventional turbofan?
- Why is hydrogen nevertheless attractive from a **cycle-design** perspective?
- Which limitations of this exercise could be overcome only with **unconventional cycles**?

---

## Limitations (important)

This exercise assumes:

- fixed engine architecture,
- no heat exchangers,
- no inlet cooling,
- no recovery of cryogenic fuel exergy,
- identical component efficiencies.

Therefore:

- results apply to **conventional turbofans only**,
- they do not represent the full potential of hydrogen-enabled cycles.

---

## Key takeaway

Hydrogen fundamentally changes **mass flow rates** and **hot fluid properties**,  
but **energy efficiency (TSEC)** remains largely dictated by the Brayton cycle itself.

Meaningful gains require **architectural innovation**, not fuel substitution alone.
