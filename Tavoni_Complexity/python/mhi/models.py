"""Deterministic inference equations and Gaussian behavioral likelihoods.

Gaussian: Tavoni et al., Methods eqs. 7--27 and codes/MSE_Estim_*.m.
Bernoulli: Glaze et al. (2015), eLife 08825, eqs. 1--3.
"""
import math
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.special import expit

GAUSSIAN_MODELS = ("evidence", "prior", "memoryless", "delta_rule", "sliding_window",
                   "mixture_delta_rules", "mixture_sliding_windows", "bayesian")
BERNOULLI_MODELS = ("constant_prior", "leaky_accumulator", "bayesian")


@njit(cache=True)
def gaussian_bayes(x, h, R):
    """Exact run-length filter in units where prior mean=0 and SD=1."""
    n = len(x)
    means = np.zeros(n + 1)
    weights = np.zeros(n + 1)
    out = np.zeros(n)
    R2 = R*R
    for t in range(n):
        logs = np.empty(t+1)
        for j in range(t+1):
            nu = R2+j
            mu = 0.0 if j == 0 else means[j-1]
            var = R2*(1.0+1.0/nu)
            prior = (1.0 if t == 0 else h) if j == 0 else (1-h)*weights[j-1]
            logs[j] = -0.5*(math.log(var)+(x[t]-mu)**2/var) + math.log(max(prior, 1e-300))
        peak = np.max(logs)
        total = 0.0
        for j in range(t, -1, -1):
            nu = R2+j
            mu = 0.0 if j == 0 else means[j-1]
            means[j] = (nu*mu+x[t])/(nu+1)
            weights[j] = math.exp(logs[j]-peak)
            total += weights[j]
        est = 0.0
        for j in range(t+1):
            weights[j] /= total
            est += means[j]*weights[j]
        out[t] = est
    return out


@njit(cache=True)
def window_units(x, windows, R):
    """Eq. 17; unavailable pre-block observations are padded by prior mean 0."""
    n = len(x)
    cs = np.zeros(n+1)
    for t in range(n):
        cs[t+1] = cs[t]+x[t]
    out = np.zeros((n, len(windows)))
    for j in range(len(windows)):
        r = windows[j]
        k = int(math.floor(r))
        d = r-k
        for t in range(n):
            first = max(0, t-k+1)
            total = cs[t+1]-cs[first]
            if t-k >= 0:
                total += d*x[t-k]
            out[t, j] = total/(R*R+r)
    return out


@njit(cache=True)
def mixture_from_units(x, units, rates, h, R):
    """Eqs. 22--27, using UPDATED unit means, matching supplied MATLAB code."""
    n, m = units.shape
    # Rates must be descending, corresponding to ascending run lengths.
    # A zero learning rate is a frozen prior unit (infinite run length).
    runs = 1.0/np.maximum(rates, 1e-300)-R*R
    transition = np.zeros((m, m))
    transition[0, :] = h
    transition[m-1, m-1] += 1-h
    for j in range(m-1):
        q = 1.0/max(1.0, runs[j+1]-runs[j])
        transition[j+1, j] += (1-h)*q
        transition[j, j] += (1-h)*(1-q)
    weights = np.zeros(m)
    weights[0] = 1.0
    out = np.zeros(n)
    for t in range(n):
        if t > 0:
            logs = np.empty(m)
            for j in range(m):
                probability = 0.0
                for k in range(m):
                    probability += transition[j, k]*weights[k]
                variance = R*R*(1+rates[j])
                logs[j] = math.log(max(probability, 1e-300))-0.5*(math.log(variance)+
                                            (x[t]-units[t, j])**2/variance)
            peak = np.max(logs)
            total = 0.0
            for j in range(m):
                weights[j] = math.exp(logs[j]-peak)
                total += weights[j]
            weights /= total
        for j in range(m):
            out[t] += weights[j]*units[t, j]
    return out


@njit(cache=True)
def delta_units(x, rates):
    units = np.zeros((len(x), len(rates)))
    for j in range(len(rates)):
        mu = 0.0
        for t in range(len(x)):
            mu += rates[j]*(x[t]-mu)
            units[t, j] = mu
    return units


def gaussian_prediction(model, x, h, R, parameters=()):
    x = np.asarray(x, dtype=float)
    if model == "evidence":
        return x.copy()
    if model == "prior":
        return np.zeros(len(x))
    if model == "memoryless":
        return parameters[0]*x
    if model == "delta_rule":
        return delta_units(x, np.array(parameters))[:, 0]
    if model == "sliding_window":
        return window_units(x, np.array(parameters), R)[:, 0]
    if model == "bayesian":
        return gaussian_bayes(x, h, R)
    if model == "mixture_delta_rules":
        rates = np.sort(np.array(parameters, dtype=float))[::-1]
        return mixture_from_units(x, delta_units(x, rates), rates, h, R)
    if model == "mixture_sliding_windows":
        windows = np.sort(np.array(parameters))
        rates = 1/(R*R+windows)
        return mixture_from_units(x, window_units(x, windows, R), rates, h, R)
    raise ValueError(model)


def _fit_window_exact(x, y, mask, R, max_window=1e6):
    """Global SSE optimum for the single fractional window, one segment at a time.

    On r=k+d, prediction=(A+d B)/(R²+k+d); the derivative has one root.
    Beyond the observed history the same rational function has a single optimum.
    """
    best = (np.inf, 1.0)
    cs = np.r_[0.0, np.cumsum(x)]
    t = np.arange(len(x))
    for k in range(1, len(x)+1):
        A = (cs[t+1]-cs[np.maximum(0, t-k+1)])[mask]
        B = np.where(t >= k, x[np.maximum(t-k, 0)], 0)[mask]
        c = R*R+k
        z, w = A-y[mask]*c, B-y[mask]
        q0, q1, q2 = z@z, z@w, w@w
        denom = c*q2-q1
        candidates = [0.0, 1.0]
        if denom != 0:
            d = (q0-c*q1)/denom
            if 0 < d < 1:
                candidates.append(d)
        for d in candidates:
            sse = float(np.sum(((z+d*w)/(c+d))**2))
            if sse < best[0]:
                best = sse, k+d
    A = cs[1:][mask]
    denom = float(A@A)
    u = float(np.clip(A@y[mask]/denom, 1/(R*R+max_window), 1/(R*R+len(x)))) if denom else 1/(R*R+max_window)
    r = 1/u-R*R
    sse = float(np.sum((u*A-y[mask])**2))
    if sse < best[0]:
        best = sse, r
    return best[1]


def fit_gaussian(x, y, mask, h, R, seed=20261107):
    """Fit all eight models; x,y are centered and scaled by prior SD.

    Optimizing SSE also maximizes the profiled Gaussian likelihood, including
    when decision SD reaches its upper bound. Decision noise is fitted later.
    """
    fits = {}
    def add(name, params, success=True, method="closed_form", nfev=0):
        pred = gaussian_prediction(name, x, h, R, params)
        fits[name] = dict(parameters=list(map(float, params)), prediction=pred,
                          optimizer_success=bool(success), optimizer_method=method,
                          nfev=int(nfev))
    for name in ("evidence", "prior", "bayesian"):
        add(name, [])
    denom = x[mask]@x[mask]
    alpha = float(np.clip(x[mask]@y[mask]/denom, 0, 1)) if denom else 0.0
    add("memoryless", [alpha])
    objective = lambda a: float(np.sum((gaussian_prediction("delta_rule", x, h, R, [a])[mask]-y[mask])**2))
    grid = np.unique(np.r_[0, np.geomspace(1e-6, 1, 65), np.linspace(0, 1, 65)])
    vals = np.array([objective(a) for a in grid])
    candidates = [(vals[0], 0.0), (vals[-1], 1.0)]
    for i in range(1, len(grid)-1):
        if vals[i] <= vals[i-1] and vals[i] <= vals[i+1]:
            res = minimize_scalar(objective, bounds=(grid[i-1], grid[i+1]),
                                  method="bounded", options={"xatol": 1e-11})
            candidates.append((res.fun, res.x))
    alpha = min(candidates)[1]
    add("delta_rule", [alpha], method="grid_and_bounded")
    r = _fit_window_exact(x, y, mask, R)
    add("sliding_window", [r], method="analytic_all_fractional_segments")
    for name, single, bounds, transform in (
        ("mixture_delta_rules", "delta_rule", [(0.0, 1.0)]*2, lambda z: np.asarray(z)),
        ("mixture_sliding_windows", "sliding_window", [(0.0, 6.0)]*2, lambda z: 10**np.asarray(z))):
        def obj(z):
            pred = gaussian_prediction(name, x, h, R, transform(z))
            return float(np.sum((pred[mask]-y[mask])**2))
        res = differential_evolution(obj, bounds, seed=seed, popsize=12,
                                     maxiter=150, tol=1e-9, polish=False)
        # Include the nested optimum and widely separated timescales explicitly.
        nested = np.repeat(fits[single]["parameters"][0], 2)
        nested_z = nested if single == "delta_rule" else np.log10(nested)
        starts = [res.x, nested_z]
        starts += ([np.array([0.05, 0.8]), np.array([0.001, 0.2])] if single == "delta_rule"
                   else [np.array([0.0, 1.5]), np.array([0.7, 3.0])])
        solutions = [(obj(nested_z), nested_z, True, "nested_optimum")]
        solutions.append((res.fun, res.x, res.success, "differential_evolution"))
        neval = res.nfev
        for start in starts:
            local = minimize(obj, start, method="Powell", bounds=bounds,
                             options={"xtol": 1e-7, "ftol": 1e-10, "maxiter": 250})
            neval += local.nfev
            solutions.append((local.fun, local.x, local.success, "differential_evolution_and_multistart_powell"))
        _, z, success, method = min(solutions, key=lambda v: v[0])
        params = np.sort(transform(z))
        if single == "delta_rule":
            params = params[::-1]
        add(name, params, success, method, neval)
    return fits


def bernoulli_predictions(beads, h, p):
    """Predict sequence bead t BEFORE it is revealed, in black=1 coordinates.

    Constant Prior means a SIGNED constant in LOG PRIOR ODDS, not probability
    0.5: sign(L)*log((1-h)/h). Leaky: (1-2h)*L. Bayesian uses exact transition.
    Each approximation is also used to project the final posterior one step ahead.
    Outputs are expected bead colors, incorporating both urn transitions and
    bead emission noise; they are not a fitted stochastic choice policy.
    """
    out = {}
    evidence = np.log(p/(1-p))*(2*np.asarray(beads)-1)
    for model in BERNOULLI_MODELS:
        L = 0.0
        pred = np.empty(len(beads))
        for t in range(len(beads)):
            if model == "bayesian":
                prior = h+(1-2*h)*expit(L)
                prior_L = np.log(prior)-np.log1p(-prior)
            elif model == "leaky_accumulator":
                prior_L = (1-2*h)*L
            else:
                prior_L = np.sign(L)*np.log((1-h)/h)
            prior = expit(prior_L)
            pred[t] = (1-p)+(2*p-1)*prior
            L = prior_L+evidence[t]
        out[model] = pred
    return out


def goodness_of_fit(y, prediction, n_parameters, sigma_upper):
    n = len(y)
    residual = y-prediction
    sse = float(residual@residual)
    mse = sse/n
    sd = float(np.clip(np.sqrt(mse), 1e-6, sigma_upper))
    ll = float(-0.5*sse/sd**2-n*np.log(sd)-n/2*np.log(2*np.pi))
    sst = float(np.sum((y-y.mean())**2))
    return dict(n_fit_trials=n, n_parameters=n_parameters, sse=sse, mse=mse,
                rmse=float(np.sqrt(mse)), mae=float(np.mean(abs(residual))),
                r2=(1-sse/sst if sst > 0 else np.nan), decision_noise_sd=sd,
                decision_noise_at_upper_bound=bool(sd >= sigma_upper), log_likelihood=ll,
                aic=2*n_parameters-2*ll, bic=n_parameters*np.log(n)-2*ll,
                paper_log_evidence_score=2*ll-n_parameters*np.log(n),
                bic_log_evidence_approx=ll-0.5*n_parameters*np.log(n))


def gaussian_complexity(model, parameters):
    """Reflective operation counts, unit costs, Supplementary Table 1.

    This is algorithmic cost, distinct from the BIC parameter penalty; no
    reflexive constant is added. Window costs use the fitted window lengths.
    """
    if model == "bayesian":
        return np.inf
    fixed = {"evidence": 0, "prior": 2, "memoryless": 8, "delta_rule": 9,
             "mixture_delta_rules": 129}
    if model in fixed:
        return float(fixed[model])
    if model == "sliding_window":
        return float(19+np.floor(parameters[0]))
    return float(147+np.floor(max(parameters)))
