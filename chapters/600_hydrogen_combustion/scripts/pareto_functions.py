import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Optional

# -------------------------------------------------------------------------
# Design-space definition (continuous bounds for each decision variable)
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class DesignSpace:
    beta_f: Tuple[float, float]
    beta_c: Tuple[float, float]
    BPR:    Tuple[float, float]
    T4:     Tuple[float, float]

def sample_designs(space: DesignSpace, n: int, seed: int = 0) -> pd.DataFrame:
    """
    Draw n random designs uniformly within the given DesignSpace bounds.

    Returns a DataFrame with columns: beta_f, beta_c, BPR, T4.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "beta_f": rng.uniform(*space.beta_f, size=n),
        "beta_c": rng.uniform(*space.beta_c, size=n),
        "BPR":    rng.uniform(*space.BPR,    size=n),
        "T4":     rng.uniform(*space.T4,     size=n),
    })

def evaluate(factory: Callable[..., object],
             d: Dict,
             h: float = 10000,
             Ma: float = 0.8) -> Optional[Dict]:
    """
    Evaluate a single design point.

    Parameters
    ----------
    factory:
        Callable that MUST create a *new* TurbofanEngine instance given design vars.
        (Important for avoiding state carry-over between evaluations.)
    d:
        Dict-like object containing beta_f, beta_c, BPR, T4.
    h, Ma:
        Flight condition used for performance calculation.

    Returns
    -------
    dict with design variables + the two objectives, or None if evaluation fails.
    """
    try:
        # Create a new engine instance for this design
        eng = factory(beta_f=float(d["beta_f"]),
                      beta_c=float(d["beta_c"]),
                      BPR=float(d["BPR"]),
                      T_max=float(d["T4"]))
        # Compute engine performance at the selected flight condition
        perf = eng.calculate(h=h, Ma=Ma, report=False)

        # Return a flat record: design vars + objective values
        return {
            "beta_f": float(d["beta_f"]),
            "beta_c": float(d["beta_c"]),
            "BPR":    float(d["BPR"]),
            "T4":     float(d["T4"]),
            KEY_TSEC: float(perf[KEY_TSEC]),
            KEY_ST:   float(perf[KEY_ST]),
        }
    except Exception:
        # Fail-safe: caller can filter out None results
        return None

def pareto_frontier(df: pd.DataFrame,
                    tsec_key: str = KEY_TSEC,
                    st_key: str = KEY_ST) -> pd.DataFrame:
    """
    Compute a 2D Pareto frontier with:
      - minimize tsec_key
      - maximize st_key

    Implementation note
    -------------------
    O(n log n) strategy:
      1) sort points by increasing TSEC
      2) keep the “upper envelope” of specific thrust (monotonically increasing ST)

    This works because for 2 objectives, once sorted by one objective, dominance
    checks reduce to tracking the best achieved value of the other objective.
    """
    if df.empty:
        return df.copy()

    # Sort by objective 1 (TSEC) ascending
    tmp = df[[tsec_key, st_key]].to_numpy()
    order = np.argsort(tmp[:, 0])  # increasing TSEC
    sdf = df.iloc[order].copy()

    # Keep only points that improve the best ST seen so far
    best_st = -np.inf
    keep_idx = []
    for i, row in sdf.iterrows():
        st = row[st_key]
        if st > best_st:
            keep_idx.append(i)
            best_st = st

    return sdf.loc[keep_idx].copy()

# -------------------------------------------------------------------------
# NSGA-II implementation (fast non-dominated sort + crowding distance)
# Objectives: minimize TSEC, maximize ST
# Decision variables: beta_f, beta_c, BPR, T4 (encoded in [0,1]^4)
# -------------------------------------------------------------------------


def dominates(a, b):
    """
    Return True if point a dominates point b under:
      - minimize TSEC
      - maximize ST

    a = (tsec_a, st_a), b = (tsec_b, st_b)
    """
    tsec_a, st_a = a
    tsec_b, st_b = b
    # a is at least as good in both objectives AND strictly better in at least one
    return (tsec_a <= tsec_b and st_a >= st_b) and (tsec_a < tsec_b or st_a > st_b)

def fast_nondominated_sort(obj):
    """
    Classic NSGA-II non-dominated sorting.

    Parameters
    ----------
    obj : ndarray (N, 2)
        Objective values for each individual (TSEC, ST).

    Returns
    -------
    fronts : list[list[int]]
        fronts[0] is the non-dominated set, fronts[1] the next, etc.
    rank : ndarray (N,)
        rank[i] = index of the Pareto front the i-th individual belongs to.
    """

    N = obj.shape[0]

    # S[p] = list of solutions dominated by p
    S = [[] for _ in range(N)]

    # n[p] = number of solutions dominating p
    n = np.zeros(N, dtype=int)

    # rank[p] = Pareto front index of p
    rank = np.zeros(N, dtype=int)

    fronts = [[]] # fronts[0] will be filled first

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(obj[p], obj[q]):
                S[p].append(q)
            elif dominates(obj[q], obj[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    fronts.pop()
    return fronts, rank

def crowding_distance(front, obj):
    """
    Crowding distance for diversity preservation within a front.

    Larger distance => less crowded => preferred in selection (tie-breaker).
    """
    dist = np.zeros(len(front))
    if len(front) == 0:
        return dist
    if len(front) <= 2:
        dist[:] = np.inf
        return dist

    fobj = obj[front]
    m = fobj.shape[1]

    for j in range(m):
        order = np.argsort(fobj[:, j])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        fmin = fobj[order[0], j]
        fmax = fobj[order[-1], j]
        if fmax - fmin < 1e-12:
            continue
        for k in range(1, len(front) - 1):
            dist[order[k]] += (fobj[order[k+1], j] - fobj[order[k-1], j]) / (fmax - fmin)

    return dist

def tournament_select(pop, rank, crowd, rng, k=2):
    """
    Binary (k-way) tournament selection used by NSGA-II.

    Preference order:
      1) lower rank (better Pareto front)
      2) higher crowding distance (more diverse)
    """
    cand = rng.integers(0, len(pop), size=k)
    best = cand[0]
    for c in cand[1:]:
        if rank[c] < rank[best]:
            best = c
        elif rank[c] == rank[best] and crowd[c] > crowd[best]:
            best = c
    return pop[best]

def simulated_binary_crossover(p1, p2, eta=10, rng=None):
    """
    Simulated Binary Crossover (SBX) on real-coded genes in [0,1].

    eta controls offspring spread:
      - small eta => more exploratory (wider spread)
      - large eta => more exploitative (children near parents)

    Notes:
      - Assumes variable bounds are [0,1] here; later you "decode" to physical bounds.
      - Uses clipping to keep children in [0,1].
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(p1)
    c1 = p1.copy()
    c2 = p2.copy()
    for i in range(n):
        if rng.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-14:
            x1 = min(p1[i], p2[i])
            x2 = max(p1[i], p2[i])

            u = rng.random()
            beta = 1.0 + (2.0*(x1 - 0.0)/(x2 - x1))
            alpha = 2.0 - beta**(-(eta+1))
            if u <= 1.0/alpha:
                betaq = (u*alpha)**(1.0/(eta+1))
            else:
                betaq = (1.0/(2.0 - u*alpha))**(1.0/(eta+1))
            child1 = 0.5*((x1+x2) - betaq*(x2-x1))

            u = rng.random()
            beta = 1.0 + (2.0*(1.0 - x2)/(x2 - x1))
            alpha = 2.0 - beta**(-(eta+1))
            if u <= 1.0/alpha:
                betaq = (u*alpha)**(1.0/(eta+1))
            else:
                betaq = (1.0/(2.0 - u*alpha))**(1.0/(eta+1))
            child2 = 0.5*((x1+x2) + betaq*(x2-x1))

            c1[i] = np.clip(child1, 0.0, 1.0)
            c2[i] = np.clip(child2, 0.0, 1.0)
    return c1, c2

def polynomial_mutation(x, pm=0.1, eta=20, rng=None):
    """
    Polynomial mutation for real-coded genes in [0,1].

    pm = per-gene mutation probability
    eta = distribution index (larger => smaller perturbations)
    """
    if rng is None:
        rng = np.random.default_rng()
    y = x.copy()
    for i in range(len(y)):
        if rng.random() < pm:
            u = rng.random()
            if u < 0.5:
                delta = (2*u)**(1/(eta+1)) - 1
            else:
                delta = 1 - (2*(1-u))**(1/(eta+1))
            y[i] = np.clip(y[i] + delta, 0.0, 1.0)
    return y

def nsga2(factory, space, extra_keys=None,
          h=10000, Ma=0.8,
          pop_size=120, generations=60,
          cx_prob=0.9, mut_prob=0.2,
          seed=0, verbose=True):
    """
    Run NSGA-II for 4 design variables: beta_f, beta_c, BPR, T4.

    Objectives:
      - minimize KEY_TSEC
      - maximize KEY_ST

    Parameters
    ----------
    factory:
        Function creating a NEW TurbofanEngine for each candidate design.
    space:
        DesignSpace with physical bounds for each variable.
    extra_keys:
        Optional additional performance metrics to store in the DataFrame (if present in perf).
    h, Ma:
        Evaluation flight condition.
    pop_size, generations:
        NSGA-II controls.
    cx_prob, mut_prob:
        Crossover and mutation probabilities (mutation here is per-gene probability).
    seed:
        RNG seed for reproducibility.
    verbose:
        If True, print progress and first evaluation diagnostics.

    Returns
    -------
    results_df:
        Final population (design vars + objectives + requested extra metrics).
    pareto_df:
        Non-dominated front (front 0) of the final population.
    """
    if extra_keys is None:
        extra_keys = []

    rng = np.random.default_rng(seed)

    lo = np.array([space.beta_f[0], space.beta_c[0], space.BPR[0], space.T4[0]], dtype=float)
    hi = np.array([space.beta_f[1], space.beta_c[1], space.BPR[1], space.T4[1]], dtype=float)

    def decode(z):
        """
        Map normalized genome z in [0,1]^4 to physical design variables.
        """
        x = lo + z*(hi - lo)
        return dict(beta_f=float(x[0]), beta_c=float(x[1]), BPR=float(x[2]), T4=float(x[3]))

    printed_once = False

    def eval_one(z):
        """
        Evaluate one normalized genome z.
        Returns:
          (row_dict, tsec, st)

        Failure handling:
          On exception, assign worst objectives: TSEC=+inf, ST=-inf
          so the individual is dominated and naturally removed by selection.
        """
        nonlocal printed_once
        d = decode(z)
        try:
            eng = factory(beta_f=d["beta_f"], beta_c=d["beta_c"], BPR=d["BPR"], T_max=d["T4"])
            perf = eng.calculate(h=h, Ma=Ma, report=False)

            if verbose and not printed_once:
                print("Example perf keys:", list(perf.keys()))
                printed_once = True

            row = {
                **d,
                KEY_TSEC: float(perf[KEY_TSEC]),
                KEY_ST:   float(perf[KEY_ST]),
            }
            for k in extra_keys:
                if k in perf:
                    row[k] = float(perf[k])

            return row, row[KEY_TSEC], row[KEY_ST]

        except Exception as e:
            if verbose and not printed_once:
                print("First eval error:", repr(e))
                print("At design:", d)
                printed_once = True

            row = {**d, KEY_TSEC: np.inf, KEY_ST: -np.inf}
            for k in extra_keys:
                row[k] = np.nan
            return row, np.inf, -np.inf

    # --- init population in [0,1]^4
    pop = rng.random((pop_size, 4))

    # --- evaluate
    rows = []
    obj = np.zeros((pop_size, 2))
    for i in range(pop_size):
        row, tsec, st = eval_one(pop[i])
        rows.append(row)
        obj[i] = [tsec, st]

    # --- generations
    for gen in range(generations):
        fronts, rank = fast_nondominated_sort(obj)

        crowd = np.zeros(pop_size)
        for f in fronts:
            cd = crowding_distance(f, obj)
            for idx, val in zip(f, cd):
                crowd[idx] = val

        # offspring
        children = []
        while len(children) < pop_size:
            p1 = tournament_select(pop, rank, crowd, rng)
            p2 = tournament_select(pop, rank, crowd, rng)

            if rng.random() < cx_prob:
                c1, c2 = simulated_binary_crossover(p1, p2, eta=10, rng=rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = polynomial_mutation(c1, pm=mut_prob, eta=20, rng=rng)
            c2 = polynomial_mutation(c2, pm=mut_prob, eta=20, rng=rng)

            children.append(c1)
            if len(children) < pop_size:
                children.append(c2)

        children = np.array(children)

        # eval children
        rows_c = []
        obj_c = np.zeros((pop_size, 2))
        for i in range(pop_size):
            row, tsec, st = eval_one(children[i])
            rows_c.append(row)
            obj_c[i] = [tsec, st]

        # combine
        pop_all = np.vstack([pop, children])
        obj_all = np.vstack([obj, obj_c])
        rows_all = rows + rows_c

        # elitist selection
        fronts_all, _ = fast_nondominated_sort(obj_all)

        new_pop, new_obj, new_rows = [], [], []
        for f in fronts_all:
            if len(new_pop) + len(f) <= pop_size:
                new_pop.extend(pop_all[f])
                new_obj.extend(obj_all[f])
                new_rows.extend([rows_all[i] for i in f])
            else:
                cd = crowding_distance(f, obj_all)
                order = np.argsort(-cd)
                needed = pop_size - len(new_pop)
                sel = [f[i] for i in order[:needed]]

                new_pop.extend(pop_all[sel])
                new_obj.extend(obj_all[sel])
                new_rows.extend([rows_all[i] for i in sel])
                break

        pop = np.array(new_pop)
        obj = np.array(new_obj)
        rows = new_rows

        if verbose and (gen % 10 == 0 or gen == generations - 1):
            fronts_now, _ = fast_nondominated_sort(obj)
            f0 = fronts_now[0]
            if len(f0) > 0:
                best_tsec = np.min(obj[f0, 0])
                best_st = np.max(obj[f0, 1])
                print(f"Gen {gen:03d}: |F0|={len(f0)}  best TSEC={best_tsec:.5g}  best ST={best_st:.5g}")
            else:
                print(f"Gen {gen:03d}: |F0|=0")

    # final results + pareto
    results = pd.DataFrame(rows)
    fronts_final, _ = fast_nondominated_sort(obj)
    f0 = fronts_final[0]
    pareto = results.iloc[f0].copy().sort_values(KEY_TSEC)
    return results, pareto


def plot_pareto_multi(frontiers,
                      results_list=None,
                      labels=None,
                      show_cloud=False,
                      tsec_key=KEY_TSEC,
                      st_key=KEY_ST,
                      title=None):
    """
    Plot one or multiple Pareto frontiers in (TSEC, specific thrust).

    Parameters
    ----------
    frontiers : list[pd.DataFrame] or dict[str, pd.DataFrame]
        Each DataFrame must contain columns tsec_key and st_key.
        If dict is provided, dict keys are used as labels (unless labels is provided).

    results_list : list[pd.DataFrame] or None
        Optional list of full sampled populations (same length/order as frontiers).
        Used only if show_cloud=True.

    labels : list[str] or None
        Labels for each frontier (ignored if frontiers is a dict and labels is None).

    show_cloud : bool
        If True, also scatter plot the background samples.

    title : str or None
    """

    # Normalize inputs
    if isinstance(frontiers, dict):
        if labels is None:
            labels = list(frontiers.keys())
        frontier_list = [frontiers[k] for k in labels]
    else:
        frontier_list = list(frontiers)
        if labels is None:
            labels = [f"Frontier {i+1}" for i in range(len(frontier_list))]

    if results_list is None:
        results_list = [None] * len(frontier_list)

    if len(results_list) != len(frontier_list):
        raise ValueError("results_list must have same length as frontiers (or be None).")

    plt.figure()

    # Plot clouds first (so frontiers sit on top)
    if show_cloud:
        for res, lab in zip(results_list, labels):
            if res is None or res.empty:
                continue
            plt.scatter(res[st_key], res[tsec_key], s=8, alpha=0.15, label=f"{lab} samples")

    # Plot frontiers
    for pareto, lab in zip(frontier_list, labels):
        if pareto is None or pareto.empty:
            continue
        # Sort to get a nice curve
        pareto_sorted = pareto.sort_values(st_key)
        plt.scatter(pareto_sorted[st_key], pareto_sorted[tsec_key], s=35, label=lab)

    plt.xlabel(st_key)
    plt.ylabel(tsec_key)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_xy_multi(frontiers,
                  x_key,
                  y_key,
                  results_list=None,
                  labels=None,
                  show_cloud=False,
                  connect=False,
                  invert_y=False,
                  title=None):
    """
    Plot any x_key vs y_key for one or multiple DataFrames (e.g., Pareto sets).

    Parameters
    ----------
    frontiers : list[pd.DataFrame] or dict[str, pd.DataFrame]
    x_key, y_key : str
        Column names to plot.
    results_list : list[pd.DataFrame] or None
        Optional background populations (same length/order as frontiers).
    show_cloud : bool
        If True, plot background populations.
    connect : bool
        If True, connect points in the order they appear (usually False).
    invert_y : bool
        If True, flip y-axis (useful when "lower is better").
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Normalize inputs
    if isinstance(frontiers, dict):
        if labels is None:
            labels = list(frontiers.keys())
        dfs = [frontiers[k] for k in labels]
    else:
        dfs = list(frontiers)
        if labels is None:
            labels = [f"Set {i+1}" for i in range(len(dfs))]

    if results_list is None:
        results_list = [None] * len(dfs)
    if len(results_list) != len(dfs):
        raise ValueError("results_list must match number of frontiers.")

    plt.figure()

    # Plot clouds first
    if show_cloud:
        for res, lab in zip(results_list, labels):
            if res is None or res.empty:
                continue
            X = pd.to_numeric(res[x_key], errors="coerce")
            Y = pd.to_numeric(res[y_key], errors="coerce")
            m = np.isfinite(X) & np.isfinite(Y)
            plt.scatter(X[m], Y[m], s=8, alpha=0.15, label=f"{lab} samples")

    # Plot each frontier / set
    for df, lab in zip(dfs, labels):
        if df is None or df.empty:
            continue
        X = pd.to_numeric(df[x_key], errors="coerce")
        Y = pd.to_numeric(df[y_key], errors="coerce")
        m = np.isfinite(X) & np.isfinite(Y)

        if connect:
            plt.plot(X[m], Y[m], marker="o", linestyle="-", label=lab)
        else:
            plt.scatter(X[m], Y[m], s=35, label=lab)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if invert_y:
        plt.gca().invert_yaxis()
    plt.show()


def parallel_axes_plot(frontiers,
                       columns,
                       labels=None,
                       invert=None,
                       alpha=0.25,
                       lw=1.0):
    """
    Parallel-coordinates plot for comparing multi-dimensional Pareto sets.

    Each row in each frontier is plotted as a polyline across the chosen columns.
    Values are normalized globally (across all frontiers) so scales are comparable.

    Parameters
    ----------
    frontiers:
        list[DataFrame] or dict[label -> DataFrame]
    columns:
        Ordered list of column names to display (design + performance).
    invert:
        Set of column names to invert (useful when "lower is better",
        e.g. invert={'TSEC [MJ/Nh]'} so good solutions are visually higher).
    alpha, lw:
        Line transparency and thickness.
    """
    if invert is None:
        invert = set()

    # Normalize input
    if isinstance(frontiers, dict):
        if labels is None:
            labels = list(frontiers.keys())
        dfs = [frontiers[k] for k in labels]
    else:
        dfs = list(frontiers)
        if labels is None:
            labels = [f"Frontier {i+1}" for i in range(len(dfs))]

    # Collect all values for global normalization
    all_vals = np.vstack([df[columns].to_numpy(dtype=float) for df in dfs])
    vmin = np.nanmin(all_vals, axis=0)
    vmax = np.nanmax(all_vals, axis=0)
    span = np.where(vmax - vmin < 1e-12, 1.0, vmax - vmin)

    ncol = len(columns)
    xs = np.arange(ncol)

    fig, ax = plt.subplots(figsize=(1.35*ncol + 2, 6))

    # Draw vertical axes
    for j in range(ncol):
        ax.plot([xs[j], xs[j]], [0, 1], color="black", linewidth=1)

    # Get matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot each frontier
    for idx, (df, label) in enumerate(zip(dfs, labels)):
        color = colors[idx % len(colors)]

        X = df[columns].to_numpy(dtype=float)
        Xn = (X - vmin) / span

        for j, c in enumerate(columns):
            if c in invert:
                Xn[:, j] = 1.0 - Xn[:, j]

        # Draw all polylines in the same color
        for i in range(Xn.shape[0]):
            ax.plot(xs, Xn[i, :], color=color, alpha=alpha, linewidth=lw)

        # Legend handle
        ax.plot([], [], color=color, label=label)

    # Cosmetics
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["low", "mid", "high"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()



