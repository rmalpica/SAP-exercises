import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import emcee
from multiprocessing import Pool
import sys

# RUNNING in conda environment 'chaos'

# -----------------------------
# 1. Define the Climate Model
# -----------------------------
def climate_model(t, y, params, scenario='bau'):
    """
    Full climate model with CO2, CH4, CO2 forcing, CH4 forcing,
    albedo, temperature, and aerosol forcing.

    Arguments:
    ----------
    t : float
        Time (years).
    y : list or array of floats
        State variables in the order:
          y = [co2, ch4, cf, mf, a, T, fa].
    params : list or array of floats
        Model parameters, in the order:

          params = [
            pf0,    #  0: Initial fossil CO2 emission rate
            kf,     #  1: Growth/decline rate for Pf
            pd0,    #  2: Initial deforestation CO2 emission rate
            kd,     #  3: Growth/decline rate for Pd
            betaO,  #  4: Ocean uptake coefficient
            gammaO, #  5: Ocean temperature sensitivity
            betaP,  #  6: Photosynthesis/plant uptake coefficient
            n,      #  7: Nonlinearity exponent for plant uptake
            co2eq,  #  8: Equilibrium CO2 (e.g., pre-industrial)
            lambdaC,#  9: CO2 radiative forcing coefficient
            ch40,   # 10: Baseline CH4 emission rate (wetlands, etc.)
            ch4eq,  # 11: Reference (equilibrium) CH4 level
            gammaM, # 12: Methane-temperature feedback exponent
            tauM,   # 13: Methane atmospheric lifetime
            lambdaM,# 14: CH4 radiative forcing coefficient
            a0,     # 15: Baseline albedo
            deltaA, # 16: Strength of albedo-temperature coupling
            betaA,  # 17: Aerosol forcing to emission coupling
            Tbaseline, # 18: Baseline temperature offset
            S,         # 19: Solar constant
            tau        # 20: Heat capacity (inversely: temperature response rate)
          ]
    """

    # Unpack state variables
    co2, ch4, cf, mf, a, T, fa = y

    # Unpack parameters
    pf0, kf, pd0, kd, betaO, gammaO, betaP, n, co2eq, \
    lambdaC, ch40, ch4eq, gammaM, tauM, lambdaM, \
    a0, deltaA, betaA, Tbaseline, S, tau = params

    # --------------------------------------------------------
    # Emission Scenarios
    # --------------------------------------------------------
    #
    if scenario == 'bau':
        # (1) Business As Usual (BAU)
        Pf = pf0 * np.exp(+kf * t)
        Pd = pd0 * np.exp(-kd * t)
    elif scenario == 'paris':
        # (2) Paris Agreement: Net-Zero by 2050 (around t=30)
        Pf = pf0 /(1 + np.exp(+kf * (t - 70)))
        Pd = pd0 /(1 + np.exp(-kd * (t - 70)))
    elif scenario == 'strong':
        # (3) Strong Climate Action (with Carbon Capture after t=30)
        Pf = pf0 / (1 + np.exp(+kf * (t - 70))) if t < 70 else -0.5  # Carbon Capture
        Pd = pd0 / (1 + np.exp(-kd * (t - 70))) if t > 70 else 0.0

    # --------------------------------------------------------
    # Carbon sinks: Ocean & Land
    # --------------------------------------------------------
    # Ocean uptake
    Ro = betaO * (co2 - co2eq) * np.exp(-gammaO * T)

    # Plant uptake
    Rp = betaP * (co2 ** n)

    # --------------------------------------------------------
    # Methane emissions
    # --------------------------------------------------------
    # Wetlands + microbial sources (baseline) minus atmospheric destruction
    # plus potential Arctic release if T > 2°C above baseline
    #CH4Arctic = 50 * np.exp(0.3 * (T - 2)) if T > 2 else 0

    # --------------------------------------------------------
    # Geoengineering aerosol forcing (turns on after t=30)
    # --------------------------------------------------------
    #GeoAerosol = -2.0 if t > 30 else 0.0

    # --------------------------------------------------------
    # Radiative balance
    # --------------------------------------------------------
    RadIn = (S * (1 - a)) / 4
    RadOut = (240 / Tbaseline**4) * (T + Tbaseline)**4

    # --------------------------------------------------------
    # ODE system
    # --------------------------------------------------------
    # 1) CO2
    dco2_dt = Pf + Pd - Ro - Rp

    # 2) CH4
    dch4_dt = ch40 * np.exp(gammaM * T) - (ch4 / tauM) # + CH4Arctic

    # 3) CO2 radiative forcing
    dcf_dt = lambdaC * np.log(co2 / co2eq) - cf

    # 4) CH4 radiative forcing
    dmf_dt = lambdaM * np.log(ch4 / ch4eq) - mf

    # 5) Albedo
    #    a slowly relaxes toward (a0 - deltaA * tanh(T)), with rate deltaA
    da_dt = -deltaA * (a - (a0 - deltaA * np.tanh(T)))

    # 6) Temperature
    #    Temperature responds to net radiative flux plus forcing from CO2, CH4, and aerosols
    dT_dt = (RadIn - RadOut + cf + mf + fa) / tau

    # 7) Aerosol forcing
    #    Grows with fraction of fossil fuel use (betaA * Pf) but is reduced each step,
    #    plus an extra negative forcing from GeoAerosol after t=30
    dfa_dt = betaA * Pf - fa # + GeoAerosol

    return [
        dco2_dt,  # 0
        dch4_dt,  # 1
        dcf_dt,   # 2
        dmf_dt,   # 3
        da_dt,    # 4
        dT_dt,    # 5
        dfa_dt    # 6
    ]

# Residual and likelihood

def simulate_model(params, t_data, y0, scenario='bau'):
    """
      - Solve the climate_model over the range of t_data

    Arguments:
    ----------
    params : np.array
        Parameter set to test. Must be in the correct order:
        [ pf0, kf, pd0, kd, betaO, gammaO, betaP, n, co2eq,
          lambdaC, ch40, ch4eq, gammaM, tauM, lambdaM,
          a0, deltaA, betaA, Tbaseline, S, tau ]

    t_data : np.array
        1D array of times at which we have historical data.

    y0 : list or np.array
        Initial conditions [co2, ch4, cf, mf, a, T, fa].

    Returns:
    --------
     np.arrays of model predictions at each time point.
    """
    # Solve ODE from t_data[0] to t_data[-1] using solve_ivp
    sol = solve_ivp(
        fun=lambda t, y: climate_model(t, y, params, scenario),
        t_span=[t_data[0], t_data[-1]],
        y0=y0,
        t_eval=t_data,   # direct evaluation at the data times
        rtol=1e-7,
        atol=1e-9
    )

    # Extract the model solutions at those time points
    co2_model = sol.y[0, :]  # the first state variable is CO2
    ch4_model = sol.y[1, :]  # the second state variable is CH4
    co2Forcing_model = sol.y[2, :]  # the second state variable is CH4
    ch4Forcing_model = sol.y[3, :]  # the second state variable is CH4
    albedo_model = sol.y[4, :]  # the second state variable is CH4
    T_model   = sol.y[5, :]  # the sixth state variable is T anomaly
    aerosolForcing_model   = sol.y[6, :]  # the sixth state variable is T anomaly

    return [co2_model, ch4_model, co2Forcing_model, ch4Forcing_model, albedo_model, T_model, aerosolForcing_model]


def scale_params(params, bounds):
    return np.array([(p - low) / (high - low) for p, (low, high) in zip(params, bounds)])

def unscale_params(scaled, bounds):
    return np.array([low + s * (high - low) for s, (low, high) in zip(scaled, bounds)])


def log_likelihood(params, t_data, co2_obs, ch4_obs, T_obs, y0, sigma):
    co2_model, ch4_model , _, _, _, T_model, _ = simulate_model(params, t_data, y0)
    resid = np.concatenate([(co2_model - co2_obs)/sigma[0],
                             (ch4_model - ch4_obs)/sigma[1],
                             (T_model   - T_obs  )/sigma[2]])
    return -0.5 * np.sum(resid**2)


def log_prior(params, bounds):
    # Uniform priors within bounds
    for p, (low, high) in zip(params, bounds):
        if not (low <= p <= high):
            return -np.inf
    return 0.0


#def log_posterior(params, t_data, co2_obs, ch4_obs, T_obs, y0, sigma, bounds):
#    lp = log_prior(params, bounds)
#    if not np.isfinite(lp):
#        return -np.inf
#    return lp + log_likelihood(params, t_data, co2_obs, ch4_obs, T_obs, y0, sigma)

def log_posterior(scaled_params, t_data, co2_obs, ch4_obs, T_obs, y0, sigma, bounds):
    # Transform: scale back from [0, 1] to physical units
    physical_params = unscale_params(scaled_params, bounds)
    
    lp = log_prior(physical_params, bounds)
    if not np.isfinite(lp):
        return -np.inf
    posterior = lp + log_likelihood(physical_params, t_data, co2_obs, ch4_obs, T_obs, y0, sigma)
   
    return posterior


def neg_log_posterior(scaled_params):
    lp = log_posterior(scaled_params,
                       t_data,         # array of times
                       co2_data,       # observed CO2
                       ch4_data,       # observed CH4
                       T_data,         # observed T anomaly
                       y0,             # initial state
                       sigma,          # measurement std-devs
                       bounds)         # prior bounds
    # `log_posterior` can return -inf if outside the prior,
    # so clip it to a big positive penalty
    if not np.isfinite(lp):
        return 1e30
    return -lp

# --------------------------------------------------------------
#                              MAIN
# --------------------------------------------------------------

if __name__ == "__main__":

    # Initial parameter guesses:
    print("Start EXEC")
 
    print("Import Historical Time Series")
# -----------------------------
# 1. Historical (Synthetic) Data
# -----------------------------
    # (A) Load or define your historical data arrays
    year_histo_start = 1984
    year_histo_end   = 2023
    t_data = np.arange(year_histo_start, year_histo_end, 1)  # years from 1984 to 2023
    print("t_data type", type(t_data))  # Should be <class 'numpy.ndarray'>    print(T_data)

   # Load the CO2 CSV file ----------------
    co2_db = pd.read_csv('../data/co2_data.csv')

    # Ensure correct column names
    co2_db.columns = co2_db.columns.str.strip()

    # Extract time and anomaly columns
    time = co2_db['year'].values  # Assuming time is in a column named 'time'
    co2_values = co2_db['co2'].values  # Temperature anomaly values

    # Create an interpolation function (linear by default)
    co2_interp = interp1d(time, co2_values, kind='linear', fill_value="extrapolate")

    co2_data = co2_interp(t_data)
    print("co2_data type", type(co2_data)) 

    # Load the CH4 CSV file ----------------
    ch4_db = pd.read_csv('../data/ch4_data.csv')

    # Ensure correct column names
    ch4_db.columns = ch4_db.columns.str.strip()

    # Extract time and anomaly columns
    time = ch4_db['year'].values  # Assuming time is in a column named 'time'
    ch4_values = ch4_db['ch4'].values  # Temperature anomaly values

    # Create an interpolation function (linear by default)
    ch4_interp = interp1d(time, ch4_values, kind='linear', fill_value="extrapolate")

    ch4_data = ch4_interp(t_data)
    print("ch4_data type", type(ch4_data))

    # Load the Delta T CSV file ----------------
    T_db = pd.read_csv('../data/T_data.csv')

    # Ensure correct column names
    T_db.columns = T_db.columns.str.strip()

    # Extract time and anomaly columns
    time = T_db['year'].values  # Assuming time is in a column named 'time'
    anomaly_values = T_db['anomaly'].values  # Temperature anomaly values

    # Create an interpolation function (linear by default)
    T_interp = interp1d(time, anomaly_values, kind='linear', fill_value="extrapolate")

    T_data = T_interp(t_data) - T_interp(year_histo_start)
    print("T_data type", type(T_data))


    # Define parameter names, initial guesses, and bounds
    param_names = [
            "pf0", "kf", "pd0", "kd", "betaO", "gammaO", "betaP", "n", "co2eq", 
            "lambda", "ch40", "ch4eq", "gammaM", "tauM", "lambdaM", 
            "a0", "deltaA", "betaA", "Tbaseline", "S", "tau"
        ]
    # (C) Initial conditions y0 = [co2, ch4, cf, mf, a, T, fa]
#     Must be appropriate to your earliest data point (year 1850 in this example).
    co2_0 = co2_values[0]  # or a known historical CO2 level in 1850
    ch4_0 = ch4_values[0]  # ...
    cf_0  = 5.35 * np.log(415 / 280)
    mf_0  = 0.36 * np.log(1800 / 800)
    a_0   = 0.3 - 0.1 * np.tanh(T_data[0])
    T_0   = T_data[0]   # T anomaly relative to Tbaseline
    fa_0  =  -0.0025 * 9.8

    y0 = [co2_0, ch4_0, cf_0, mf_0, a_0, T_0, fa_0]
    print("Initial conditions y0 = [co2, ch4, cf, mf, a, T, fa]  ", y0)

    # (B) Provide an initial guess for the 21 parameters

    baseline_params = np.array([
        10.3,   # pf0
        0.008,   # kf
        1.0,    # pd0
        0.001,   # kd
        0.0101,   # betaO
        0.002,   # gammaO
        0.0454,  # betaP
        0.904,    # n
        245.8,  # co2eq
        7.6,    # lambdaC
        710,  # ch40
        763.0,  # ch4eq
        0.1,   # gammaM
        2.4,   # tauM
        0.2,    # lambdaM
        0.3,    # a0
        0.029,   # deltaA
        0.0026,    # betaA
        180.0,   # Tbaseline
        1362.0, # S
        30.2     # tau
    ])

    for name, value in zip(param_names, baseline_params):
        print(f"{name}: {value:.4f}")

    # Time range
    t_span_bl = (0, year_histo_end - year_histo_start + 1)
    t_eval_bl = np.linspace(*t_span_bl, 128)

    print("baseline model t_span  ", t_span_bl) 
    #print("baseline model t_eval  ", t_eval_bl) 

    param_bounds = [
            (9, 11),       # pf0
            (0.0005, 0.04),# kf
            (0.5, 3.5),    # pd0
            (0.0001, 0.01),  # kd
            (0.01, 0.2),   # betaO
            (0.0010, 0.1), # gammaO
            (0.005, 0.1),  # betaP
            (0.5, 1.5),    # n
            (240, 300),    # co2eq
            (4.5, 15),     # lambda_
            (700, 800),   # ch40
            (700, 900),    # ch4eq
            (0.05, 0.2),  # gammaM
            (1.5, 12),     # tauM
            (0.1, 1),    # lambdaM
            (0.1, 1),      # a0
            (0.02, 0.2),   # deltaA
            (0.0001, 0.01),# betaA
            (160, 270),    # Tbaseline
            (1361, 1364),  # S (Solar constant)
            (30, 35)      # tau
        ]
    
    bounds = [(low, high) for low, high in param_bounds]

    # Measurement uncertainties (CO2 ppm, CH4 ppb, T °C) #to be loaded or hypothesized 
    sigma = [0.12, 1.0, 0.1]

    for name, value in zip(param_names, baseline_params):
        print(f"{name}: {value:.4f}")

    # Time range
    t_span_bl = (0, year_histo_end - year_histo_start + 1)
    t_eval_bl = np.linspace(*t_span_bl, 128)

    print("baseline model t_span  ", t_span_bl) 
    #print("baseline model t_eval  ", t_eval_bl) 


    # Run L-BFGS-B with the same bounds you use for priors
    res = minimize(neg_log_posterior,
               baseline_params,
               method='L-BFGS-B',
               bounds=bounds,
               options={'gtol':1e-8, 'ftol':1e-12})
    if not res.success:
        raise RuntimeError("MAP optimization failed: " + res.message)

    map_estimate = res.x
    print("MAP estimate:", dict(zip(param_names, map_estimate)))

    # Solve the MAP model
    co2_MAP, ch4_MAP,_,_,_,T_MAP,_ = simulate_model(map_estimate, t_eval_bl, y0, scenario='bau')
    plt.figure(); plt.plot(t_data, co2_data, 'o'); plt.plot(year_histo_start + t_eval_bl, co2_MAP); plt.title('CO2 Maximum A Posteriori'); plt.show()
    plt.figure(); plt.plot(t_data, ch4_data, 'o'); plt.plot(year_histo_start + t_eval_bl, ch4_MAP); plt.title('CH4 Maximum A Posteriori'); plt.show()
    plt.figure(); plt.plot(t_data, T_data, 'o'); plt.plot(year_histo_start + t_eval_bl, T_MAP); plt.title('Temperature Maximum A Posteriori'); plt.show()

    # MCMC with emcee
    scaled_map_estimate = scale_params(map_estimate, bounds)
    t_eval_mcmc = t_data - t_data[0] 
    n_dim = len(map_estimate)
    n_walkers = 10 * n_dim
    initial = scaled_map_estimate + 1e-1 * np.random.randn(n_walkers, n_dim) 
    initial = np.clip(initial, 0.0, 1)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim,
            log_posterior,
            args=(t_eval_mcmc, co2_data, ch4_data, T_data, y0, sigma, bounds),
            a=5,
            pool=pool
        )

        #print("Running MCMC...")
        sampler.run_mcmc(initial, 3000, progress=True)

    print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))

    #plot chains
    fig, axes = plt.subplots(n_dim, figsize=(10, 30), sharex=True)
    samples = sampler.get_chain()
    labels = param_names
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.axhline(y=scaled_map_estimate[i], color='b', linestyle='-')
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

    # Discard burn-in and flatten chain
    burnin = 0
    samples = sampler.get_chain(discard=burnin, flat=True)

    # Summarize and plot posterior
    import corner
    samples_physical = np.vstack([unscale_params(s, bounds) for s in samples])
    #fig = corner.corner(samples, labels=param_names, truths=scaled_map_estimate)
    fig = corner.corner(samples_physical, labels=param_names, truths=map_estimate)
    plt.show()

    #subset of parameters
    fig = corner.corner(samples_physical[:,[8,13,17]], labels=[param_names[i] for i in [8, 13, 17]], truths=map_estimate[[8,13,17]])
    plt.savefig("../outputs/joint_subset.png", dpi=800, transparent=False)
    plt.show()

    # Extract median parameter estimates
    medians = np.median(samples, axis=0)
    medians = unscale_params(medians, bounds)
    print("Posterior medians:\n", dict(zip(param_names, medians)))

    # Simulate model with posterior medians for comparison
    co2_median, ch4_median,_,_,_,T_median,_ = simulate_model(medians, t_eval_bl, y0, scenario='bau')
    plt.figure(); plt.plot(t_data, co2_data, 'o'); plt.plot(year_histo_start + t_eval_bl, co2_median); plt.title('CO2 with Median'); plt.show()
    plt.figure(); plt.plot(t_data, ch4_data, 'o'); plt.plot(year_histo_start + t_eval_bl, ch4_median); plt.title('CH4 with Median'); plt.show()
    plt.figure(); plt.plot(t_data, T_data, 'o'); plt.plot(year_histo_start + t_eval_bl, T_median); plt.title('Temperature with Median'); plt.show()


    # Plot predictive samples of T anomaly
    # --- Settings ---
    n_draws = 5000
    inds = np.random.choice(len(samples), size=n_draws, replace=False)

    # Historical evaluation grid (already in your code)
    year_pred_start = 2024
    year_pred_end = 2060
    scenario= 'paris'
    t_span_pred = (0, year_pred_end - year_pred_start + 1)
    t_eval_pred = np.linspace(*t_span_pred, 128)

    # --- Preallocate storage ---
    T_all = np.zeros((n_draws, len(t_eval_bl) + len(t_eval_pred)))
    co2_all = np.zeros((n_draws, len(t_eval_bl) + len(t_eval_pred)))

    for i, ind in enumerate(inds):
        # 1) Unscale params
        sample = unscale_params(samples[ind], bounds)

        # 2) Simulate historical BAU
        co2_h, ch4_h, cf_h, mf_h, a_h, T_h, fa_h = simulate_model(
            sample, t_eval_bl, y0, scenario='bau'
        )

        # 3) Get end‐of‐history state as new y0
        y0_now = [co2_h[-1], ch4_h[-1], cf_h[-1], mf_h[-1],
                  a_h[-1],   T_h[-1],   fa_h[-1]]

        # 4) Simulate future scenario
        t_future = t_eval_bl[-1] + t_eval_pred
        co2_f, ch4_f, cf_f, mf_f, a_f, T_f, fa_f = simulate_model(
            sample, t_future, y0_now, scenario=scenario
        )

        # 5) Store concatenated T trajectory
        T_all[i, :] = np.concatenate([T_h, T_f])
        co2_all[i, :] = np.concatenate([co2_h, co2_f])

    # --- Build full year axis ---
    years_hist = year_histo_start + t_eval_bl
    years_fut  = year_pred_start   + t_eval_pred
    years_full = np.concatenate([years_hist, years_fut])

    # --- Compute percentiles ---
    p5_T   = np.percentile(T_all,  5, axis=0)
    p50_T  = np.percentile(T_all, 50, axis=0)
    p95_T  = np.percentile(T_all, 95, axis=0)

    # --- Plot ---
    plt.figure(figsize=(10,6))

    # Shaded 5–95% band
    plt.fill_between(years_full, p5_T, p95_T,
                     color="C1", alpha=0.3,
                     label="5–95% CI")

    # Median trajectory
    plt.plot(years_full, p50_T, color="C1", lw=2, label="Median")

    # Overlay historical data
    plt.errorbar(t_data, T_data, yerr=sigma[2], fmt=".k", capsize=0)

    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.title("Posterior Predictive 5–95% CI for Temperature Anomaly")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../outputs/PP_T-anomaly_"+scenario+".png", dpi=800, transparent=False)
    plt.show()

    # --- Compute percentiles ---
    p5_co2   = np.percentile(co2_all,  5, axis=0)
    p50_co2  = np.percentile(co2_all, 50, axis=0)
    p95_co2  = np.percentile(co2_all, 95, axis=0)

    # --- Plot ---
    plt.figure(figsize=(10,6))

    # Shaded 5–95% band
    plt.fill_between(years_full, p5_co2, p95_co2,
                     color="C1", alpha=0.3,
                     label="5–95% CI")

    # Median trajectory
    plt.plot(years_full, p50_co2, color="C1", lw=2, label="Median")

    # Overlay historical data
    plt.errorbar(t_data, co2_data, yerr=sigma[0], fmt=".k", capsize=0)

    plt.xlabel("Year")
    plt.ylabel("Atmospheric CO2 (ppm)")
    plt.title("Posterior Predictive 5–95% CI for Atmospheric CO2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../outputs/PP_co2_"+scenario+".png", dpi=800, transparent=False)
    plt.show()


    ## Plot predictive samples of T anomaly
    #year_pred_start = 2024
    #year_pred_end = 2060
    #scenario= 'strong'
    #inds = np.random.randint(len(samples), size=500)
    #plt.figure(figsize=(10, 6))
    #t_span_pred = (0, year_pred_end - year_pred_start + 1)
    #t_eval_pred = np.linspace(*t_span_pred, 128)
    #for ind in inds:
    #    sample = unscale_params(samples[ind], bounds)
    #    co2_model, ch4_model, co2Forcing_model, ch4Forcing_model, albedo_model, T_model, aerosolForcing_model= simulate_model(sample, t_eval_bl, y0, scenario='bau')
    #    y0now = [co2_model[-1], ch4_model[-1], co2Forcing_model[-1], ch4Forcing_model[-1], albedo_model[-1], T_model[-1], aerosolForcing_model[-1]]
    #    co2_future, ch4_future,_,_,_, T_future,_= simulate_model(sample, t_eval_bl[-1]+t_eval_pred, y0now, scenario=scenario)
    #    plt.plot(year_histo_start + t_eval_bl, T_model, color="C1", alpha=0.1)
    #    plt.plot(year_pred_start + t_eval_pred, T_future, color="C1", alpha=0.1)

    ## Simulate model with posterior medians for comparison
    #co2_median, ch4_median, co2Forcing_median, ch4Forcing_median, albedo_median, T_median, aerosolForcing_median= simulate_model(medians, t_eval_bl, y0, scenario='bau')
    #y0now = [co2_median[-1], ch4_median[-1], co2Forcing_median[-1], ch4Forcing_median[-1], albedo_median[-1], T_median[-1], aerosolForcing_median[-1]]
    #co2_median_future, ch4_median_future,_,_,_, T_median_future,_= simulate_model(medians, t_eval_bl[-1]+t_eval_pred, y0now, scenario=scenario)
    #plt.plot(year_histo_start + t_eval_bl, T_median, color="red", label="Posterior Median")
    #plt.plot(year_pred_start + t_eval_pred, T_median_future, color="red")


    ## Simulate model with MAP 
    #co2_MAP, ch4_MAP, co2Forcing_MAP, ch4Forcing_MAP, albedo_MAP, T_MAP, aerosolForcing_MAP= simulate_model(map_estimate, t_eval_bl, y0, scenario='bau')
    #y0now = [co2_MAP[-1], ch4_MAP[-1], co2Forcing_MAP[-1], ch4Forcing_MAP[-1], albedo_MAP[-1], T_MAP[-1], aerosolForcing_MAP[-1]]
    #co2_MAP_future, ch4_MAP_future,_,_,_, T_MAP_future,_= simulate_model(map_estimate, t_eval_bl[-1]+t_eval_pred, y0now, scenario=scenario)
    #plt.plot(year_histo_start + t_eval_bl, T_MAP, color="green", label="Posterior MAP")
    #plt.plot(year_pred_start + t_eval_pred, T_MAP_future, color="green")
    
    ## plot measurement data
    #plt.errorbar(t_data, T_data, yerr=sigma[2], fmt=".k", capsize=0)

    #plt.xlabel("Year")
    #plt.ylabel("Temperature Anomaly (°C)")
    #plt.title("Posterior Predictive Temperature Trajectories, scenario "+scenario)
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()



    ## Plot predictive samples of CO2
    #year_pred_start = 2024
    #year_pred_end = 2060
    #scenario= 'strong'
    #inds = np.random.randint(len(samples), size=500)
    #plt.figure(figsize=(10, 6))
    #t_span_pred = (0, year_pred_end - year_pred_start + 1)
    #t_eval_pred = np.linspace(*t_span_pred, 128)
    #for ind in inds:
    #    sample = unscale_params(samples[ind], bounds)
    #    co2_model, ch4_model, co2Forcing_model, ch4Forcing_model, albedo_model, T_model, aerosolForcing_model= simulate_model(sample, t_eval_bl, y0, scenario='bau')
    #    y0now = [co2_model[-1], ch4_model[-1], co2Forcing_model[-1], ch4Forcing_model[-1], albedo_model[-1], T_model[-1], aerosolForcing_model[-1]]
    #    co2_future, ch4_future,_,_,_, T_future,_= simulate_model(sample, t_eval_bl[-1]+t_eval_pred, y0now, scenario=scenario)
    #    plt.plot(year_histo_start + t_eval_bl, co2_model, color="C1", alpha=0.1)
    #    plt.plot(year_pred_start + t_eval_pred, co2_future, color="C1", alpha=0.1)

    ## Simulate model with posterior medians for comparison
    #co2_median, ch4_median, co2Forcing_median, ch4Forcing_median, albedo_median, T_median, aerosolForcing_median= simulate_model(medians, t_eval_bl, y0, scenario='bau')
    #y0now = [co2_median[-1], ch4_median[-1], co2Forcing_median[-1], ch4Forcing_median[-1], albedo_median[-1], T_median[-1], aerosolForcing_median[-1]]
    #co2_median_future, ch4_median_future,_,_,_, T_median_future,_= simulate_model(medians, t_eval_bl[-1]+t_eval_pred, y0now, scenario=scenario)
    #plt.plot(year_histo_start + t_eval_bl, co2_median, color="red", label="Posterior Median")
    #plt.plot(year_pred_start + t_eval_pred, co2_median_future, color="red")


    ## Simulate model with MAP 
    #co2_MAP, ch4_MAP, co2Forcing_MAP, ch4Forcing_MAP, albedo_MAP, T_MAP, aerosolForcing_MAP= simulate_model(map_estimate, t_eval_bl, y0, scenario='bau')
    #y0now = [co2_MAP[-1], ch4_MAP[-1], co2Forcing_MAP[-1], ch4Forcing_MAP[-1], albedo_MAP[-1], T_MAP[-1], aerosolForcing_MAP[-1]]
    #co2_MAP_future, ch4_MAP_future,_,_,_, T_MAP_future,_= simulate_model(map_estimate, t_eval_bl[-1]+t_eval_pred, y0now, scenario=scenario)
    #plt.plot(year_histo_start + t_eval_bl, co2_MAP, color="green", label="Posterior MAP")
    #plt.plot(year_pred_start + t_eval_pred, co2_MAP_future, color="green")
    
    ## plot measurement data
    #plt.errorbar(t_data, co2_data, yerr=sigma[0], fmt=".k", capsize=0)

    #plt.xlabel("Year")
    #plt.ylabel("CO2 ppm")
    #plt.title("Posterior Predictive CO2 Trajectories, scenario "+scenario)
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()


    def plot_variance_decomposition(samples, param_names, t_eval, y0, top_n=5):
        """
        Decompose and plot, over time, the variance in temperature anomaly
        attributable to each parameter (top_n parameters only).
        """

        # 1) Draw a representative subset for speed
        n_draws = min(10000, len(samples))
        inds = np.random.choice(len(samples), size=n_draws, replace=False)

        # 2) Compute T trajectories
        T_trajs = np.zeros((n_draws, len(t_eval)))
        for i, idx in enumerate(inds):
            sample = unscale_params(samples[idx], bounds)
            co2_model, ch4_model, co2Forcing_model, ch4Forcing_model, albedo_model, T_model, aerosolForcing_model= simulate_model(sample, t_eval, y0, scenario='bau')
            T_trajs[i] = T_model  # temperature anomaly

        # 3) Center and compute covariance contributions
        T_mean = T_trajs.mean(axis=0)
        Tc     = T_trajs - T_mean

        P      = samples[inds]                     # shape (n_draws, n_dim)
        Pv     = np.var(P, axis=0)                 # var of each param

        # covariance(p_i, T) over draws, for each time
        cov_pt = np.empty((P.shape[1], len(t_eval)))
        for i in range(P.shape[1]):
            pi_c = P[:, i] - P[:, i].mean()
            cov_pt[i] = (pi_c[:, None] * Tc).mean(axis=0)

        # var contribution var_i(t) = cov_pt^2 / var(p_i)
        var_it = cov_pt**2 / Pv[:, None]

        # 4) Normalize → Sobol indices
        total_var = var_it.sum(axis=0)  # sum over parameters, shape (len(t_data),)
        # Avoid division by zero: replace zeros with np.nan temporarily
        safe_total_var = np.where(total_var == 0, np.nan, total_var)

        # Compute Sobol indices safely
        S_it = var_it / safe_total_var[None, :]  # shape: (n_params, n_times)

        # Replace NaNs (caused by division by zero) with zero
        S_it = np.nan_to_num(S_it, nan=0.0)


        # 5) Pick top_n parameters by average S_i
        avg_S = S_it.mean(axis=1)
        top_idx = np.argsort(avg_S)[-top_n:]
        labels = [param_names[i] for i in top_idx]

        # 6) Generate a 21-color palette (colorblind-friendly)
        import matplotlib.colors as mcolors
        cmap = plt.get_cmap('tab20')  # 20 distinct colors
        extra_color = "#808080"       # Add one more (gray)
        color_list = list(cmap.colors) + [extra_color]

        # 6) Plot
        plt.figure(figsize=(12, 6))
        plt.stackplot(year_histo_start +t_eval, S_it[top_idx], labels=labels, colors=color_list[:top_n],alpha=0.8)
        plt.xlabel("Year")
        plt.ylabel("First‐order Sobol Index")
        plt.title(f"Sobol Sensitivity Indices Over Time (Top {top_n})")
        plt.legend(loc="upper left", fontsize=9)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("../outputs/sobol_T.png", dpi=800, transparent=False)
        plt.show()

        S_avg = np.trapezoid(S_it, t_eval, axis=1) / (t_eval[-1] - t_eval[0])

        # 7. Sort and pick top contributors
        bar_colors = [color_list[i] for i in top_idx]

        top_idx = np.argsort(S_avg)[-top_n:]
        top_names = [param_names[i] for i in top_idx]
        top_vals = S_avg[top_idx]
        # 8. Plot bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_names, top_vals, color=bar_colors)
        plt.xlabel("Time-Integrated Sobol Index")
        plt.title(f"Top {top_n} Parameters by Average Variance Contribution")
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("../outputs/sobol_T_integral.png", dpi=800, transparent=False)
        plt.show()

    plot_variance_decomposition(samples, param_names, t_eval_bl, y0, top_n=21)
