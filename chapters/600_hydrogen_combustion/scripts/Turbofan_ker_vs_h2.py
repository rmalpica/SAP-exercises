from engine_framework.engines import TurbofanEngine
from engine_framework.core import ISA, MixtureProperties

import copy
import numpy as np

from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTER_DIR = HERE.parent
DATA = CHAPTER_DIR / "data"
OUTPUTS = CHAPTER_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

def solve_by_T4(engine, target_value, metric,
                h=10000, Ma=0.8,
                T4_min=1200, T4_max=2200,
                tol=1e-6, max_iter=60):
    """
    Solves for T_max such that metric(engine) = target_value

    metric must be one of:
        "TSEC"
        "specific_thrust"

    Returns:
        engine_solved, perf_solved
    """

    eng = copy.deepcopy(engine)

    def eval_metric(T4):
        eng.T_max = float(T4)
        perf = eng.calculate(h=h, Ma=Ma, report=False)

        if metric == "TSEC":
            return perf["TSEC [MJ/Nh]"]
        elif metric == "specific_thrust":
            return perf["specific thrust [N/(kg/s)]"]
        else:
            raise ValueError("metric must be 'TSEC' or 'specific_thrust'")

    def resid(T4):
        return eval_metric(T4) - target_value

    # --- bracket check ---
    f_lo = resid(T4_min)
    f_hi = resid(T4_max)

    if np.sign(f_lo) == np.sign(f_hi):
        raise ValueError(
            f"T4 bracket does not contain root:\n"
            f"  T4_min={T4_min}, metric={eval_metric(T4_min)}\n"
            f"  T4_max={T4_max}, metric={eval_metric(T4_max)}\n"
            f"  target={target_value}"
        )

    # --- bisection ---
    for _ in range(max_iter):
        T4_mid = 0.5 * (T4_min + T4_max)
        f_mid = resid(T4_mid)

        if abs(f_mid) < tol:
            break

        if np.sign(f_mid) == np.sign(f_lo):
            T4_min = T4_mid
            f_lo = f_mid
        else:
            T4_max = T4_mid
            f_hi = f_mid

    eng.T_max = T4_mid
    perf = eng.calculate(h=h, Ma=Ma, report=False)

    return eng, perf


def print_perf_comparison(perfA, perfB, nameA="Engine A", nameB="Engine B",
                          tol=1e-12):
    keys = sorted(set(perfA.keys()) & set(perfB.keys()))

    w_name = 35
    w_num  = 18
    w_diff = 14
    w_rel  = 12

    print("\n" + "="*(w_name + 2*w_num + w_diff + w_rel))
    print(f"{'Quantity':<{w_name}}"
          f"{nameA:>{w_num}}"
          f"{nameB:>{w_num}}"
          f"{'Δ(B−A)':>{w_diff}}"
          f"{'Δrel [%]':>{w_rel}}")
    print("="*(w_name + 2*w_num + w_diff + w_rel))

    for k in keys:
        a = perfA[k]
        b = perfB[k]

        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            continue

        d = b - a

        if abs(a) > tol:
            r = 100.0 * d / a
        else:
            r = float("nan")

        print(f"{k:<{w_name}}"
              f"{a:>{w_num}.6g}"
              f"{b:>{w_num}.6g}"
              f"{d:>{w_diff}.4g}"
              f"{r:>{w_rel}.2f}")

    print("="*(w_name + 2*w_num + w_diff + w_rel))




if __name__ == "__main__":
    """Example usage"""
    print("\n" + "="*80)
    print(" KEROSENE TURBOFAN EXAMPLE")
    print("="*80)
    
    # Create turbojet engine
    turbofanKer = TurbofanEngine(
        beta_f=1.5,
        beta_c=35,
        BPR=7.0,
        T_max=1400,
        eta_d=0.9,
        eta_f=0.9,
        eta_mf=0.99,
        eta_c=0.87,
        eta_mc=0.99,
        eta_b=0.95,
        eta_pb=0.95,
        eta_t=0.98,
        eta_mt=0.99,
        eta_n=0.98,
        core_massflow = 1.0,
        fuel_type='kerosene',
        nozzle_type='adapted'
    )
    

    #MixtureProperties.set_mode("table")
    #MixtureProperties._ensure_table_for_fuel("kerosene")
    MixtureProperties.set_mode("direct")

    # Calculate at flight condition
    perfKer = turbofanKer.calculate(h=10000, Ma=0.8, report=True)
    
    print("\n" + "="*80)
    print(" HYDROGEN TURBOFAN EXAMPLE")
    print("="*80)
    
    # Create turbofan engine
    turbofanH2 = TurbofanEngine(
        beta_f=1.5,
        beta_c=35,
        BPR=7.0,
        T_max=1400,
        eta_d=0.9,
        eta_f=0.9,
        eta_mf=0.99,
        eta_c=0.87,
        eta_mc=0.99,
        eta_b=0.95,
        eta_pb=0.95,
        eta_t=0.98,
        eta_mt=0.99,
        eta_n=0.98,
        core_massflow = 1.0,
        fuel_type='hydrogen',
        nozzle_type='adapted'
    )
    
    #MixtureProperties.set_mode("table")
    #MixtureProperties._ensure_table_for_fuel("kerosene")
    MixtureProperties.set_mode("direct")

    # Calculate at flight condition
    perfH2 = turbofanH2.calculate(h=10000, Ma=0.8, report=True)

    print("\n" + "="*80)
    print("SAME-DESIGN-PARAMETERS COMPARISON")
    print("="*80)

    print_perf_comparison(perfKer, perfH2, nameA="Kerosene", nameB="Hydrogen",
                          tol=1e-12)

    target_TSEC = perfKer["TSEC [MJ/Nh]"]
    target_ST   = perfKer["specific thrust [N/(kg/s)]"]

    print("\n" + "="*80)
    print("SAME-TSEC COMPARISON")
    print("="*80)

    engineH2_TSEC, perfH2_TSEC = solve_by_T4(
    turbofanH2,
    target_TSEC,
    metric="TSEC",
    h=10000, Ma=0.8,
    T4_min=1200, T4_max=2200
    )

    print(f"H2-turbofan T4 required for same TSEC: {engineH2_TSEC.T_max:.2f} K")
    print_perf_comparison(perfKer, perfH2_TSEC, nameA="Kerosene", nameB="Hydrogen",
                          tol=1e-12)

    print("\n" + "="*80)
    print("SAME-SPECIFIC-THRUST COMPARISON")
    print("="*80)

    engineH2_ST, perfH2_ST = solve_by_T4(
    turbofanH2,
    target_ST,
    metric="specific_thrust",
    h=10000, Ma=0.8,
    T4_min=1200, T4_max=2200
    )

    print(f"H2-turbofan T4 required for same specific thrust: {engineH2_ST.T_max:.2f} K")
    print_perf_comparison(perfKer, perfH2_ST, nameA="Kerosene", nameB="Hydrogen",
                          tol=1e-12)

