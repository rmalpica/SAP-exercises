# 300 — Mechanisms

This chapter connects **combustion products** to **atmospheric processes** and introduces how “mechanisms” (in a broad sense) sit between
*engine physics* and *climate-relevant quantities*.

In practice, engineers often need two kinds of models:

1) **Atmospheric formation criteria models**  
   Example: *when do contrails form?* (thermodynamics + saturation curves)

2) **Combustion chemistry / emissions models**  
   Example: *given a fuel/air mixture and residence time, what emission indices do we get?* (chemical kinetics + reactors)

The exercises below expose both, using a minimal but mechanistic workflow.

---

## Exercises in this chapter

### 300.1 — Contrail formation models (Schmidt–Appleman)
A guided notebook implementing the **Schmidt–Appleman criterion** and building a **contrail diagram**, using saturation curves and mixing lines.

➡️ [Exercise page →](../exercises/300_contrail_models.md)

---

### 300.2 — Emission indices from a homogeneous reactor (Cantera)
A Python script using **Cantera** to simulate a constant-pressure homogeneous reactor with a kerosene surrogate mechanism (incl. NOx) and compute time-dependent **emission indices (EI)**.

➡️ [Exercise page →](../exercises/300_emission_indices_cantera.md)

---

## Learning outcomes

After completing this chapter, you should be able to:

- Explain what the **Schmidt–Appleman criterion** is and what assumptions it makes
- Use saturation curves (water/ice) and an isobaric mixing line to determine **contrail formation thresholds**
- Define and compute **emission indices** (EI) on a mass basis (kg species / kg fuel)
- Understand what a **homogeneous reactor model** captures—and what it misses—about real combustors
- Interpret trade-offs between **CO/NO** formation, equivalence ratio, and temperature histories

These skills support later chapters where non-CO₂ effects and engine–mission coupling matter.

