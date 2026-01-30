import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTER_DIR = HERE.parent
DATA = CHAPTER_DIR / "data"
OUTPUTS = CHAPTER_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 1) Tabulated data (roughly from NIST or standard references) near 20-25K
# --------------------------------------------------------------------------
T_data = np.array([20.3, 21.0, 22.0, 23.0, 24.0, 25.0])  # K
P_data = np.array([1.0135e5, 1.40e5, 2.10e5, 3.00e5, 4.30e5, 6.20e5])  # Pa
rhoL_data = np.array([70.8, 70.2, 69.3, 68.5, 67.7, 66.8])   # kg/m^3
rhoV_data = np.array([0.084, 0.105, 0.135, 0.165, 0.198, 0.234])  # kg/m^3
uL_data = np.array([1.5, 6.0, 16.0, 28.0, 42.0, 58.0])      # kJ/kg (internal energy)
uV_data = np.array([448.0, 452.0, 460.0, 469.0, 478.0, 488.0]) # kJ/kg

# Convert internal energy from kJ/kg -> J/kg
uL_data *= 1e3  
uV_data *= 1e3  

# --------------------------------------------------------------------------
# 2) Build interpolators (cubic or linear) - "extrapolate" for T out of range
#    In real code, you might want bounded checks or bigger data sets.
# --------------------------------------------------------------------------
p_sat_interp  = interp1d(T_data, P_data,   kind='cubic', fill_value='extrapolate')
rhoL_interp   = interp1d(T_data, rhoL_data,kind='cubic', fill_value='extrapolate')
rhoV_interp   = interp1d(T_data, rhoV_data,kind='cubic', fill_value='extrapolate')
uL_interp     = interp1d(T_data, uL_data,  kind='cubic', fill_value='extrapolate')
uV_interp     = interp1d(T_data, uV_data,  kind='cubic', fill_value='extrapolate')

# --------------------------------------------------------------------------
# 3) Global parameters and utility
# --------------------------------------------------------------------------
M_total = 200.0 #358.0     # kg H2 total
V_tank  = 5.1       # m^3, rigid tank
# We'll parse a "Q_leak" as an input param in the solver function

# We'll define a small check to see if T is valid:
T_min_valid = min(T_data)
T_max_valid = max(T_data)

def volume_constraint(T, Mvap):
    """
    Returns the sum (liquid volume + vapor volume) 
    minus the total tank volume.
    Should be zero if properly on saturation curve.
    """
    Mliq = M_total - Mvap
    rhoL = rhoL_interp(T)
    rhoV = rhoV_interp(T)
    return (Mliq/rhoL + Mvap/rhoV) - V_tank

def consistent_Mvap_init(T0):
    """
    Solve volume_constraint(T0, Mvap)=0 for Mvap, 
    using a simple 1D root finder or manual approach.
    We'll do a naive bisection from Mvap=0..M_total.
    """
    # If T0 is out of table range, we proceed but might be extrapolating.
    f = lambda x: volume_constraint(T0, x)
    low, high = 0.0, M_total
    f_low, f_high = f(low), f(high)
    if f_low * f_high > 0:
        raise ValueError("No sign change in volume constraint. Check your T0 and tank specs.")
    for _ in range(50):
        mid = 0.5*(low+high)
        val = f(mid)
        if val==0 or abs(high - low)<1e-12:
            return mid
        if val*f_low < 0:
            high = mid
            f_high = val
        else:
            low = mid
            f_low = val
    return 0.5*(low+high)

# --------------------------------------------------------------------------
# 4) Two-phase ODE system
#    y=[T, Mvap]. We solve:
#    (1) dU/dt = Q => dU_dT*dT/dt + dU_dMv*dMv/dt = Q
#    (2) d/dt [ Mliq/rhoL + Mvap/rhoV ]=0 => A*dT/dt + B*dMv/dt=0
# --------------------------------------------------------------------------
def two_phase_odes(t, y, Q_leak):
    T, Mv = y
    Ml = M_total - Mv

    # If T is out of range, we do a clamp or proceed with caution:
    # (In real code, you might do a conditional check: if T> Tcrit => single-phase logic.)
    if T < T_min_valid or T > T_max_valid:
        # We'll just extrapolate (dangerous!). 
        # Ideally, you'd raise an exception or switch model.
        pass

    # Saturation properties
    p_sat = p_sat_interp(T)
    rhoL  = rhoL_interp(T)
    rhoV  = rhoV_interp(T)
    uL    = uL_interp(T)
    uV    = uV_interp(T)

    # Internal energy total
    U = Ml*uL + Mv*uV

    # partial derivatives w.r.t. T
    dT = 1e-4  # small step for finite difference
    uL_dT = uL_interp(T + dT)
    uV_dT = uV_interp(T + dT)
    cL = (uL_dT - uL)/dT
    cV = (uV_dT - uV)/dT
    dU_dT  = Ml*cL + Mv*cV

    # partial derivative w.r.t. Mv => dU/dMv= -uL + uV
    dU_dMv = -uL + uV

    # Volume derivative => A*dT/dt + B*dMv/dt=0
    #   V= Mliq/rhoL + Mvap/rhoV => differentiate wrt time
    inv_rhoL = 1.0/rhoL
    inv_rhoV = 1.0/rhoV

    # finite difference for derivative of (1/rho) wrt T
    inv_rhoL_dT = (1.0/rhoL_interp(T + dT) - inv_rhoL)/dT
    inv_rhoV_dT = (1.0/rhoV_interp(T + dT) - inv_rhoV)/dT

    A = Ml*inv_rhoL_dT + Mv*inv_rhoV_dT    # coefficient of dT/dt
    B = -inv_rhoL + inv_rhoV              # coefficient of dMv/dt

    # ODE system:
    # (1) dU_dT * dT/dt + dU_dMv * dMv/dt = Q_leak
    # (2) A * dT/dt + B * dMv/dt = 0
    mat = np.array([[dU_dT,  dU_dMv],
                    [A,      B     ]])
    rhs = np.array([Q_leak,  0.0])

    # Solve the 2x2
    dvars = np.linalg.solve(mat, rhs)
    dTdt, dMvdt = dvars[0], dvars[1]

    return [dTdt, dMvdt]

# --------------------------------------------------------------------------
# 5) Main solver function
# --------------------------------------------------------------------------
def solve_two_phase(Q_leak=1.3, T0=20.3, tmax=4*3600, n_steps=100):
    """
    Solve the 2-phase system for 'tmax' seconds, with a conduction
    heat leak 'Q_leak' (W). Start with a consistent (T0, Mvap(0)).
    """
    # 1) Find consistent Mvap(0)
    Mvap0 = consistent_Mvap_init(T0)
    print(f"[INFO] Using T0={T0}K => Mvap0={Mvap0:g} kg is consistent with volume=5.1 m^3")

    # 2) Solve ODE
    y0 = [T0, Mvap0]
    t_eval = np.linspace(0, tmax, n_steps)

    def ode_wrapper(t, y):
        return two_phase_odes(t, y, Q_leak)

    sol = solve_ivp(ode_wrapper, [0, tmax], y0, t_eval=t_eval, 
                    method='RK45', rtol=1e-7, atol=1e-9, max_step=10.0)
    return sol

def plot_saturation_curve_and_solution(
    T_data, p_data,
    T_sol, p_sol,
    title="Two-Phase Saturation Check",
    show=True
):
    """
    Plots the saturation curve (from tabulated data) and the ODE solution
    (temperature vs. pressure) on the same axes.

    Parameters
    ----------
    T_data : array-like
        1D array of temperature values [K] from your tabulated or NIST data.
    p_data : array-like
        1D array of corresponding saturation pressures [Pa] from the table.
    T_sol : array-like
        1D array of temperature values [K] from the ODE solution.
    p_sol : array-like
        1D array of the corresponding saturation pressures [Pa] (e.g., p_sat_interp(T_sol)).
    title : str, optional
        Title for the plot.
    show : bool, optional
        If True, calls plt.show() at the end. 
        Otherwise, you can continue to add to the figure before showing.
    """
    fig, ax = plt.subplots(figsize=(7,5))
    
    # Plot the saturation data
    ax.plot(T_data, p_data, 'ko--', label='Saturation data (table)')
    
    # Plot the solution in the same T range
    ax.plot(T_sol, p_sol, 'r.-', label='ODE solution')
    
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    if show:
        plt.savefig(OUTPUTS / "saturation.png", dpi=800, transparent=False)
        plt.show()
# --------------------------------------------------------------------------
# 6) Demo / Plot
# --------------------------------------------------------------------------
if __name__=="__main__":
    # Example usage: try Q_leak=200 W for a bigger heat input
    Q_in = 10.0
    sim_hours = 40.0
    print(f"\nRunning two-phase solver with Q_leak={Q_in} W for {sim_hours} hours...\n")

    sol = solve_two_phase(Q_leak=Q_in, T0=22, tmax=sim_hours*3600, n_steps=50)

    # Results
    t_   = sol.t
    T_   = sol.y[0]
    Mv_  = sol.y[1]
    Ml_  = M_total - Mv_

    # Pressure from saturation
    p_  = p_sat_interp(T_) / 1e5  # in bar
    # check volumes for sanity
    volL_ = Ml_/rhoL_interp(T_)
    volV_ = Mv_/rhoV_interp(T_)
    vol_sum_ = volL_ + volV_

    print("\nFinal time:", t_[-1], " s")
    print("Final T= ", T_[-1], " K")
    print("Final Mvap= ", Mv_[-1], " kg")
    print("Final sum volumes= ", vol_sum_[-1], " m^3  (should be ~5.1 if still 2-phase)")

    # Plot
    fig, ax = plt.subplots(2,2, figsize=(10,8))

    ax[0,0].plot(t_/3600, T_, "b.-", label="T [K]")
    ax[0,0].set_xlabel("Time [h]")
    ax[0,0].set_ylabel("T [K]")
    ax[0,0].grid(True)
    ax[0,0].legend()

    ax[0,1].plot(t_/3600, p_, "m.-", label="Psat [bar]")
    ax[0,1].set_xlabel("Time [h]")
    ax[0,1].set_ylabel("Pressure [bar]")
    ax[0,1].grid(True)
    ax[0,1].legend()

    ax[1,0].plot(t_/3600, Mv_, "r.-", label="Mvap [kg]")
    ax[1,0].set_xlabel("Time [h]")
    ax[1,0].set_ylabel("Mass [kg]")
    ax[1,0].grid(True)
    ax[1,0].legend()

    ax[1,1].plot(t_/3600, Ml_, "g.-", label="Mliq [kg]")
    ax[1,1].set_xlabel("Time [h]")
    ax[1,1].set_ylabel("Mass [kg]")
    ax[1,1].grid(True)
    ax[1,1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUTS / "tank-h2.png", dpi=800, transparent=False)
    plt.show()

    plot_saturation_curve_and_solution(
    T_data, P_data,
    T_, p_ * 1e5,
    title="Saturation Curve vs. Two-Phase Solution"
)