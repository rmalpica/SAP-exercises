from pareto_functions import * 

# -------------------------------------------------------------------------
# Engine factory wiring (decouple optimizer from engine implementation details)
# -------------------------------------------------------------------------
from engine_framework.engines import TurbofanEngine
from engine_framework.core import MixtureProperties
# -------------------------------------------------------------------------
# Consistent objective keys (match whatever TurbofanEngine.calculate returns)
# -------------------------------------------------------------------------
KEY_TSEC = "TSEC [MJ/Nh]"
KEY_ST   = "specific thrust [N/(kg/s)]"




# Configure mixture property evaluation method
MixtureProperties.set_mode("direct")

def make_turbofan_factory(*, fuel_type, nozzle_type="adapted"):
    """
    Returns a closure 'factory' that creates TurbofanEngine instances using:
      - variable design inputs: beta_f, beta_c, BPR, T_max
      - fixed technology assumptions (efficiencies, core_massflow, etc.)
      - externalized choices: fuel_type, nozzle_type

    This lets you re-run the same optimizer with different fuels/nozzles
    without touching the NSGA-II code.
    """
    def factory(beta_f, beta_c, BPR, T_max):
        return TurbofanEngine(
            beta_f=beta_f,
            beta_c=beta_c,
            BPR=BPR,
            T_max=T_max,

            # --- fixed technology choices ---
            eta_d=0.9,
            eta_f=0.90,
            eta_c=0.87,
            eta_mc=0.99,
            eta_b=0.95,
            eta_pb=0.95,
            eta_t=0.98,
            eta_mt=0.99,
            eta_n=0.98,
            core_massflow=1.0,

            # --- scenario choices (swept externally) ---
            fuel_type=fuel_type,
            nozzle_type=nozzle_type,
        )
    return factory

# -------------------------------------------------------------------------
# Example setup: define design space and compare fuels via NSGA-II
# -------------------------------------------------------------------------
space = DesignSpace(
    beta_f=(1.5, 1.8),
    beta_c=(25, 35),
    BPR=(7, 12),
    T4=(1300, 1800)
)

# Two comparable scenarios differing only by fuel properties/model
kerosene_factory = make_turbofan_factory(fuel_type="kerosene")
hydrogen_factory = make_turbofan_factory(fuel_type="hydrogen")

# Additional metrics to store
EXTRA_KEYS = [
    "overall efficiency [-]",
    "thermal efficiency [-]",
    "propulsive efficiency [-]",
    "TSFC [kg/Nh]",
]

# Run optimizer for each fuel (same random seed => comparable search stochasticity)
results_K, pareto_K = nsga2(kerosene_factory, space, pop_size=120, generations=80, h=10000, Ma=0.8, seed=1, extra_keys=EXTRA_KEYS)
results_H, pareto_H = nsga2(hydrogen_factory, space, pop_size=120, generations=80, h=10000, Ma=0.8, seed=1, extra_keys=EXTRA_KEYS)

# Inspect best (lowest TSEC) solutions on each front
print(pareto_K.sort_values(KEY_TSEC).head(15))
print(pareto_H.sort_values(KEY_TSEC).head(15))

# Plot Pareto fronts 
plot_pareto_multi(
    frontiers={"Kerosene": pareto_K, "Hydrogen": pareto_H},
    results_list=[results_K, results_H],
    show_cloud=False,  # set True if you want the background points too
    title="Pareto frontiers in (TSEC, specific thrust)"
)

# Parallel axes to compare multi-metric tradeoffs for each Pareto set
cols = [
    "beta_f", "beta_c", "BPR", "T4",
    "overall efficiency [-]", "thermal efficiency [-]", "propulsive efficiency [-]",
    "TSEC [MJ/Nh]", "specific thrust [N/(kg/s)]"
]

parallel_axes_plot(
    {"Kerosene": pareto_K, "Hydrogen": pareto_H},
    columns=cols,
    invert={"TSEC [MJ/Nh]"},
    alpha=0.25,
    lw=1.0
)
