# fit_model_with_a.py
import numpy as np
from scipy.optimize import curve_fit

def model_with_a(t, K, tau, y0, a):
    t = np.asarray(t, float)
    return y0 + K * a * (1 - np.exp(-t / tau))

def fit_with_a(t, y, a, fit_y0=True):
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # clean + sort
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    # baseline + steady-state guesses
    y0_guess = float(np.mean(y[:max(3, len(y)//10)]))
    yinf = float(np.mean(y[-max(3, len(y)//10):]))

    Ka0 = yinf - y0_guess
    K0 = Ka0 / a if abs(a) > 1e-9 else 1.0
    tau0 = max((t[-1] - t[0]) / 3.0, 1e-6)

    if fit_y0:
        popt, _ = curve_fit(
            lambda tt, K, tau, y0: model_with_a(tt, K, tau, y0, a),
            t, y,
            p0=[K0, tau0, y0_guess],
            bounds=([-np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=20000
        )
        K, tau, y0 = map(float, popt)
    else:
        popt, _ = curve_fit(
            lambda tt, K, tau: model_with_a(tt, K, tau, y0_guess, a),
            t, y,
            p0=[K0, tau0],
            bounds=([-np.inf, 1e-9], [np.inf, np.inf]),
            maxfev=20000
        )
        K, tau = map(float, popt)
        y0 = y0_guess

    yhat = model_with_a(t, K, tau, y0, a)
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - sse/sst) if sst > 0 else float("nan")

    return {
        "t": t,
        "y": y,
        "K": K,
        "tau": tau,
        "y0": y0,
        "SSE": sse,
        "R2": r2,
        "y_fit": yhat,
        "residuals": resid
    }