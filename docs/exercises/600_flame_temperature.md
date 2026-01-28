# Exercise 600.1 — Adiabatic flame temperature  
## Hydrogen vs kerosene combustion

🧪 **Script**  
[`equilibrium_calculation.py`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/600_hydrogen_combustion/scripts/equilibrium_calculation.py)

Supporting files (same folder):

- Mechanism: [`kerosene.yaml`](https://github.com/rmalpica/SAP-exercises/blob/main/chapters/600_hydrogen_combustion/scripts/kerosene.yaml)

---

## Learning objectives

- Compute adiabatic flame temperatures for H₂–air and kerosene–air mixtures
- Compare **equilibrium chemistry** with a **simplified main-products model**
- Understand how fuel chemistry affects temperature, dissociation, and heat release

---

## Model description

The script computes adiabatic flame temperatures using two approaches:

1. **Full equilibrium calculation**

    - Chemical equilibrium at fixed pressure and enthalpy
    - Includes dissociation at high temperature

2. **Main products only**

    - Assumes complete combustion to CO₂/H₂O/N₂
    - Neglects dissociation effects

Both are applied to:

- hydrogen–air mixtures
- kerosene–air mixtures

---

## How to run

From the script folder (chapters/600_hydrogen_combustion/scripts):

```bash
python equilibrium_calculation.py
```

---

## Guided questions

### 1) Fuel chemistry and flame temperature

- Why does hydrogen typically exhibit higher adiabatic flame temperatures?
- How does molecular structure influence heat release per unit mass?

### 2) Role of dissociation

- How large is the difference between equilibrium and main-products results?
- Why does dissociation limit peak flame temperature?

### 3) Equivalence ratio effects

- How does flame temperature vary with equivalence ratio?
- Are trends similar for hydrogen and kerosene?

---

## Student tasks

### Task 1 — Temperature comparison

Compute and report:

- adiabatic flame temperature for H₂ and kerosene
- results from both equilibrium and simplified models

Explain the differences physically.

---

### Task 2 — specific heat comparison

Compare specific heats of H₂ and kerosene products at different equivalence ratios.

---

## Limitations

- No flame structure or kinetics
- No pressure losses
- Idealized thermodynamic equilibrium

Results should be interpreted as **upper bounds** on achievable temperatures.
