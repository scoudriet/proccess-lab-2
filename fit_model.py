# fit_model.py
import numpy as np
from scipy.optimize import curve_fit

def _clean_sort(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    idx = np.argsort(t)
    return t[idx], y[idx]

def first_order_response(t, Ka, tau, y0=0.0, t0=0.0):
    """
    First-order step response with Ka treated as ONE parameter:

      y(t) = y0 + Ka*(1 - exp(-(t-t0)/tau)),   t >= t0
      y(t) = y0,                              t <  t0

    Parameters
    ----------
    t : array-like
    Ka : float     (lumped gain*step term)
    tau : float    (time constant, >0)
    y0 : float     (baseline)
    t0 : float     (step time)
    """
    t = np.asarray(t, dtype=float)
    ts = np.maximum(t - t0, 0.0)
    return y0 + Ka * (1.0 - np.exp(-ts / tau))

def _initial_guesses(t, y, t0):
    """
    Robust-ish initial guesses using:
      y0 ~ mean(pre-step) or first few points
      Ka ~ y_inf - y0
      tau ~ time to 63.2% (with interpolation)
    """
    t, y = _clean_sort(t, y)

    # baseline guess
    pre = y[t <= t0]
    y0 = float(np.mean(pre)) if pre.size >= 2 else float(np.mean(y[: min(3, len(y))]))

    # steady-state guess
    n_tail = max(3, int(0.2 * len(y)))
    y_inf = float(np.mean(y[-n_tail:]))

    Ka0 = y_inf - y0

    # tau guess via 63.2% of the total change
    target = y0 + 0.632 * (y_inf - y0)

    after = t >= t0
    t_after = t[after]
    y_after = y[after]

    # fallback
    tau0 = max((t[-1] - t[0]) / 3.0, 1e-6)

    # if step response is negative, use <= for target crossing
    if (y_inf - y0) >= 0:
        idx = np.where(y_after >= target)[0]
    else:
        idx = np.where(y_after <= target)[0]

    if idx.size > 0:
        i = int(idx[0])
        if i == 0:
            tau0 = max(float(t_after[0] - t0), 1e-6)
        else:
            t1, t2 = float(t_after[i - 1]), float(t_after[i])
            y1, y2 = float(y_after[i - 1]), float(y_after[i])
            # linear interpolation for crossing time
            denom = (y2 - y1) if abs(y2 - y1) > 1e-12 else 1e-12
            t_cross = t1 + (target - y1) * (t2 - t1) / denom
            tau0 = max(t_cross - t0, 1e-6)

    return float(Ka0), float(tau0), float(y0)

def fit_first_order(t, y, t0=0.0, fit_y0=True):
    """
    Fit Ka and tau (and optionally y0) to first-order step response.

    Returns dict keys used by your GUI:
      Ka, tau, y0, SSE, R2, y_fit, residuals, plus initial guesses.
    """
    t, y = _clean_sort(t, y)
    if t.size < 4:
        raise ValueError("Need at least 4 valid points to fit.")

    Ka0, tau0, y0_guess = _initial_guesses(t, y, float(t0))

    if fit_y0:
        popt, _ = curve_fit(
            lambda tt, Ka, tau, y0: first_order_response(tt, Ka, tau, y0=y0, t0=float(t0)),
            t, y,
            p0=[Ka0, tau0, y0_guess],
            bounds=([-np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=30000
        )
        Ka_hat, tau_hat, y0_hat = map(float, popt)
    else:
        popt, _ = curve_fit(
            lambda tt, Ka, tau: first_order_response(tt, Ka, tau, y0=y0_guess, t0=float(t0)),
            t, y,
            p0=[Ka0, tau0],
            bounds=([-np.inf, 1e-9], [np.inf, np.inf]),
            maxfev=30000
        )
        Ka_hat, tau_hat = map(float, popt)
        y0_hat = float(y0_guess)

    y_fit = first_order_response(t, Ka_hat, tau_hat, y0=y0_hat, t0=float(t0))
    residuals = y - y_fit

    SSE = float(np.sum(residuals ** 2))
    ybar = float(np.mean(y))
    SStot = float(np.sum((y - ybar) ** 2))
    R2 = float(1.0 - SSE / SStot) if SStot > 0 else float("nan")

    return {
        "t": t,
        "y": y,
        "Ka": Ka_hat,
        "tau": tau_hat,
        "y0": y0_hat,
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residuals,
        "Ka0": Ka0,
        "tau0": tau0,
        "y0_guess": y0_guess,
    }