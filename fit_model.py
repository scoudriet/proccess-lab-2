# fit_model.py
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit, least_squares

def _clean_sort(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    idx = np.argsort(t)
    return t[idx], y[idx]


def _clean_sort_u(t, u, y):
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    t, u, y = t[m], u[m], y[m]
    idx = np.argsort(t)
    return t[idx], u[idx], y[idx]


def _initial_level_guess(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    n = max(3, len(x) // 10)
    return float(np.mean(x[:n]))


def _initial_window_size_from_input(u):
    u = np.asarray(u, dtype=float)
    if u.size == 0:
        return 1
    du = np.abs(np.diff(u))
    tol = 1e-8 * max(1.0, float(np.max(np.abs(u))))
    idx = np.where(du > tol)[0]
    if idx.size > 0:
        return max(3, int(idx[0]) + 1)
    return max(3, len(u) // 10)


def _delay_signal_zoh(t, u, theta, u_init=None):
    """
    Zero-order-hold input delay:
      u_delayed(t_i) = u(t_i - theta)
    using the most recent sampled input value.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    theta = float(max(theta, 0.0))
    u_init = float(u[0] if u_init is None else u_init)

    query_t = t - theta
    idx = np.searchsorted(t, query_t, side="right") - 1
    out = np.full_like(t, u_init, dtype=float)
    m = idx >= 0
    out[m] = u[idx[m]]
    return out


def _infer_full_dataset_guesses(t, u, y):
    t, u, y = _clean_sort_u(t, u, y)
    if t.size < 5:
        raise ValueError("Need at least 5 valid points to fit full-dataset dead-time model.")

    n0 = min(len(t), _initial_window_size_from_input(u))
    y0_guess = float(np.mean(y[:n0]))
    u0_guess = float(np.mean(u[:n0]))
    span_t = float(max(t[-1] - t[0], 1e-6))
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    dt_med = float(np.median(dt)) if dt.size else max(span_t / max(t.size - 1, 1), 1e-6)

    du = u - u0_guess
    dy = y - y0_guess
    du_scale = float(np.max(np.abs(du))) if du.size else 0.0
    dy_scale = float(np.max(np.abs(dy))) if dy.size else 0.0
    if du_scale > 1e-12:
        corr_sign = float(np.sign(np.dot(du, dy)))
        if corr_sign == 0.0:
            corr_sign = 1.0
        K0 = corr_sign * dy_scale / du_scale
    else:
        K0 = 0.0

    tau0 = max(0.1 * span_t, 5.0 * dt_med, 1e-6)
    theta0 = 0.0
    return {
        "y0_guess": float(y0_guess),
        "u0_guess": float(u0_guess),
        "K0": float(K0),
        "tau0": float(tau0),
        "theta0": float(theta0),
        "dt_med": float(dt_med),
        "span_t": float(span_t),
    }


def _estimate_dead_time_guess(t, u, y, theta_max, dt_med):
    """
    Estimate dead time by aligning input changes with output slope magnitude.
    """
    t, u, y = _clean_sort_u(t, u, y)
    if t.size < 5:
        return 0.0

    du_change = np.abs(np.diff(u, prepend=u[0]))
    dy_mag = np.abs(np.gradient(y, t))
    if np.max(du_change) <= 1e-12 or np.max(dy_mag) <= 1e-12:
        return 0.0

    n_grid = max(8, min(60, int(theta_max / max(dt_med, 1e-9)) + 1))
    thetas = np.linspace(0.0, theta_max, n_grid)
    best_theta = 0.0
    best_score = -np.inf
    for theta in thetas:
        delayed = _delay_signal_zoh(t, du_change, theta, u_init=0.0)
        score = float(np.dot(delayed, dy_mag))
        if score > best_score:
            best_score = score
            best_theta = float(theta)
    return best_theta


def _first_order_from_input_response(t, u, K, tau, theta, y0=0.0, u0=0.0):
    """
    FOPDT response to an arbitrary input history with zero-order-hold input.

    Model in deviation variables:
      x' = (-x + K*u'(t-theta)) / tau
      y  = y0 + x
    where u' = u - u0.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    tau = float(max(tau, 1e-12))

    u_dev = _delay_signal_zoh(t, u - float(u0), theta=float(theta), u_init=0.0)
    x = np.zeros_like(t, dtype=float)

    for i in range(1, len(t)):
        dt_i = float(max(t[i] - t[i - 1], 1e-12))
        a = float(np.exp(-dt_i / tau))
        x[i] = x[i - 1] * a + float(K) * u_dev[i - 1] * (1.0 - a)

    return float(y0) + x


def _second_order_from_input_response(t, u, K, tau1, tau2, theta, y0=0.0, u0=0.0):
    """
    SOPDT response to an arbitrary input history with zero-order-hold input.

    Transfer function:
      G(s) = K / ((tau1*s + 1)(tau2*s + 1))

    Implemented with exact ZOH discretization of a 2-state realization for
    irregular sample times. Dead time is applied to the input via ZOH delay.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    tau1, tau2 = sorted([float(max(tau1, 1e-12)), float(max(tau2, 1e-12))], reverse=True)

    u_dev = _delay_signal_zoh(t, u - float(u0), theta=float(theta), u_init=0.0)
    x = np.zeros(2, dtype=float)
    y_dev = np.zeros_like(t, dtype=float)

    A = np.array([
        [-1.0 / tau1, 0.0],
        [1.0 / tau2, -1.0 / tau2],
    ], dtype=float)
    B = np.array([[float(K) / tau1], [0.0]], dtype=float)

    for i in range(1, len(t)):
        dt_i = float(max(t[i] - t[i - 1], 1e-12))
        M = np.zeros((3, 3), dtype=float)
        M[:2, :2] = A
        M[:2, 2:] = B
        expm_M = expm(M * dt_i)
        Ad = expm_M[:2, :2]
        Bd = expm_M[:2, 2]
        x = Ad @ x + Bd * u_dev[i - 1]
        y_dev[i] = x[1]

    return float(y0) + y_dev

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


def second_order_response(t, Ka, tau1, tau2, y0=0.0, t0=0.0):
    """
    Second-order step response for two real poles:

      G(s) = Ka / ((tau1*s + 1)(tau2*s + 1))

    with step at t0 and baseline y0.
    """
    t = np.asarray(t, dtype=float)
    ts = np.maximum(t - t0, 0.0)

    tau1 = float(max(tau1, 1e-12))
    tau2 = float(max(tau2, 1e-12))

    # Numerically-stable branch when time constants are nearly equal.
    if abs(tau1 - tau2) <= 1e-8 * max(tau1, tau2):
        tau = 0.5 * (tau1 + tau2)
        shape = 1.0 - np.exp(-ts / tau) * (1.0 + ts / tau)
    else:
        shape = 1.0 - (
            tau1 * np.exp(-ts / tau1) - tau2 * np.exp(-ts / tau2)
        ) / (tau1 - tau2)

    return y0 + Ka * shape


def fit_second_order(t, y, t0=0.0, fit_y0=True):
    """
    Fit Ka, tau1, tau2 (and optionally y0) to second-order step response.
    """
    t, y = _clean_sort(t, y)
    if t.size < 6:
        raise ValueError("Need at least 6 valid points to fit second-order model.")

    Ka0, tau0, y0_guess = _initial_guesses(t, y, float(t0))
    tau1_0 = max(0.5 * tau0, 1e-6)
    tau2_0 = max(2.0 * tau0, 2e-6)

    if fit_y0:
        popt, _ = curve_fit(
            lambda tt, Ka, tau1, tau2, y0: second_order_response(
                tt, Ka, tau1, tau2, y0=y0, t0=float(t0)
            ),
            t,
            y,
            p0=[Ka0, tau1_0, tau2_0, y0_guess],
            bounds=([-np.inf, 1e-9, 1e-9, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        Ka_hat, tau1_hat, tau2_hat, y0_hat = map(float, popt)
    else:
        popt, _ = curve_fit(
            lambda tt, Ka, tau1, tau2: second_order_response(
                tt, Ka, tau1, tau2, y0=y0_guess, t0=float(t0)
            ),
            t,
            y,
            p0=[Ka0, tau1_0, tau2_0],
            bounds=([-np.inf, 1e-9, 1e-9], [np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        Ka_hat, tau1_hat, tau2_hat = map(float, popt)
        y0_hat = float(y0_guess)

    # Keep tau1 <= tau2 for consistent reporting.
    tau1_hat, tau2_hat = sorted([max(tau1_hat, 1e-9), max(tau2_hat, 1e-9)])

    y_fit = second_order_response(t, Ka_hat, tau1_hat, tau2_hat, y0=y0_hat, t0=float(t0))
    residuals = y - y_fit

    SSE = float(np.sum(residuals ** 2))
    ybar = float(np.mean(y))
    SStot = float(np.sum((y - ybar) ** 2))
    R2 = float(1.0 - SSE / SStot) if SStot > 0 else float("nan")

    return {
        "t": t,
        "y": y,
        "Ka": Ka_hat,
        "tau1": tau1_hat,
        "tau2": tau2_hat,
        "y0": y0_hat,
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residuals,
        "Ka0": Ka0,
        "tau1_0": tau1_0,
        "tau2_0": tau2_0,
        "y0_guess": y0_guess,
    }


def fopdt_response(t, K, tau, theta, y0=0.0):
    """
    First-Order Plus Dead Time (FOPDT) step response:

      y(t) = y0 + K*(1 - exp(-(t - theta)/tau)),   t >= theta
      y(t) = y0,                                  t <  theta

    Parameters
    ----------
    t : array-like
    K : float      (gain)
    tau : float    (time constant, >0)
    theta : float  (dead time, >=0)
    y0 : float     (baseline)
    """
    t = np.asarray(t, dtype=float)
    ts = np.maximum(t - theta, 0.0)
    return y0 + K * (1.0 - np.exp(-ts / tau))


def fit_fopdt(t, y, fit_y0=True):
    """
    Fit K, tau, theta (and optionally y0) to FOPDT step response.

    Assumes step at t=0.

    Returns dict with keys: K, tau, theta, y0, SSE, R2, y_fit, residuals, plus initial guesses.
    """
    t, y = _clean_sort(t, y)
    if t.size < 5:
        raise ValueError("Need at least 5 valid points to fit FOPDT model.")

    # Initial guesses
    y0_guess = float(np.mean(y[:max(3, len(y)//10)]))
    y_inf = float(np.mean(y[-max(3, len(y)//10):]))
    K0 = y_inf - y0_guess
    tau0 = max((t[-1] - t[0]) / 3.0, 1e-6)
    theta0 = 0.0  # start with no dead time

    if fit_y0:
        popt, _ = curve_fit(
            lambda tt, K, tau, theta, y0: fopdt_response(tt, K, tau, theta, y0=y0),
            t, y,
            p0=[K0, tau0, theta0, y0_guess],
            bounds=([-np.inf, 1e-9, 0.0, -np.inf], [np.inf, np.inf, t[-1], np.inf]),
            maxfev=50000
        )
        K_hat, tau_hat, theta_hat, y0_hat = map(float, popt)
    else:
        popt, _ = curve_fit(
            lambda tt, K, tau, theta: fopdt_response(tt, K, tau, theta, y0=y0_guess),
            t, y,
            p0=[K0, tau0, theta0],
            bounds=([-np.inf, 1e-9, 0.0], [np.inf, np.inf, t[-1]]),
            maxfev=50000
        )
        K_hat, tau_hat, theta_hat = map(float, popt)
        y0_hat = float(y0_guess)

    y_fit = fopdt_response(t, K_hat, tau_hat, theta_hat, y0=y0_hat)
    residuals = y - y_fit

    SSE = float(np.sum(residuals ** 2))
    ybar = float(np.mean(y))
    SStot = float(np.sum((y - ybar) ** 2))
    R2 = float(1.0 - SSE / SStot) if SStot > 0 else float("nan")

    return {
        "t": t,
        "y": y,
        "K": K_hat,
        "tau": tau_hat,
        "theta": theta_hat,
        "y0": y0_hat,
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residuals,
        "K0": K0,
        "tau0": tau0,
        "theta0": theta0,
        "y0_guess": y0_guess,
    }


def fopdt_ka_response(t, Ka, tau, theta, y0=0.0):
    """
    First-Order Plus Dead Time (FOPDT) step response with lumped gain Ka:

      y(t) = y0 + Ka*(1 - exp(-(t - theta)/tau)),   t >= theta
      y(t) = y0,                                   t <  theta

    Parameters
    ----------
    t : array-like
    Ka : float     (lumped gain, K * step size)
    tau : float    (time constant, >0)
    theta : float  (dead time, >=0)
    y0 : float     (baseline)
    """
    t = np.asarray(t, dtype=float)
    ts = np.maximum(t - theta, 0.0)
    return y0 + Ka * (1.0 - np.exp(-ts / tau))


def fit_fopdt_ka(t, y, fit_y0=True):
    """
    Fit Ka, tau, theta (and optionally y0) to FOPDT step response.

    Assumes step at t=0.

    Returns dict with keys: Ka, tau, theta, y0, SSE, R2, y_fit, residuals, plus initial guesses.
    """
    t, y = _clean_sort(t, y)
    if t.size < 5:
        raise ValueError("Need at least 5 valid points to fit FOPDT model.")

    # Initial guesses
    y0_guess = float(np.mean(y[:max(3, len(y)//10)]))
    y_inf = float(np.mean(y[-max(3, len(y)//10):]))
    Ka0 = y_inf - y0_guess
    tau0 = max((t[-1] - t[0]) / 3.0, 1e-6)
    theta0 = 0.0  # start with no dead time

    if fit_y0:
        popt, _ = curve_fit(
            lambda tt, Ka, tau, theta, y0: fopdt_ka_response(tt, Ka, tau, theta, y0=y0),
            t, y,
            p0=[Ka0, tau0, theta0, y0_guess],
            bounds=([-np.inf, 1e-9, 0.0, -np.inf], [np.inf, np.inf, t[-1], np.inf]),
            maxfev=50000
        )
        Ka_hat, tau_hat, theta_hat, y0_hat = map(float, popt)
    else:
        popt, _ = curve_fit(
            lambda tt, Ka, tau, theta: fopdt_ka_response(tt, Ka, tau, theta, y0=y0_guess),
            t, y,
            p0=[Ka0, tau0, theta0],
            bounds=([-np.inf, 1e-9, 0.0], [np.inf, np.inf, t[-1]]),
            maxfev=50000
        )
        Ka_hat, tau_hat, theta_hat = map(float, popt)
        y0_hat = float(y0_guess)

    y_fit = fopdt_ka_response(t, Ka_hat, tau_hat, theta_hat, y0=y0_hat)
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
        "theta": theta_hat,
        "y0": y0_hat,
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residuals,
        "Ka0": Ka0,
        "tau0": tau0,
        "theta0": theta0,
        "y0_guess": y0_guess,
    }


def sopdt_response(t, Ka, tau1, tau2, theta, y0=0.0):
    """
    Second-Order Plus Dead Time (SOPDT) step response:

      y(t) = y0 + Ka * (1 - (tau1*exp(-(t-theta)/tau1) - tau2*exp(-(t-theta)/tau2))/(tau1 - tau2)),   t >= theta
      y(t) = y0,                                                                                     t <  theta

    Parameters
    ----------
    t : array-like
    Ka : float     (lumped gain)
    tau1 : float   (time constant 1, >0)
    tau2 : float   (time constant 2, >0)
    theta : float  (dead time, >=0)
    y0 : float     (baseline)
    """
    t = np.asarray(t, dtype=float)
    ts = np.maximum(t - theta, 0.0)

    tau1 = float(max(tau1, 1e-12))
    tau2 = float(max(tau2, 1e-12))

    # Numerically-stable branch when time constants are nearly equal.
    if abs(tau1 - tau2) <= 1e-8 * max(tau1, tau2):
        tau = 0.5 * (tau1 + tau2)
        shape = 1.0 - np.exp(-ts / tau) * (1.0 + ts / tau)
    else:
        shape = 1.0 - (
            tau1 * np.exp(-ts / tau1) - tau2 * np.exp(-ts / tau2)
        ) / (tau1 - tau2)

    return y0 + Ka * shape


def fit_sopdt(t, y, fit_y0=True):
    """
    Fit Ka, tau1, tau2, theta (and optionally y0) to SOPDT step response.

    Assumes step at t=0.

    Returns dict with keys: Ka, tau1, tau2, theta, y0, SSE, R2, y_fit, residuals, plus initial guesses.
    """
    t, y = _clean_sort(t, y)
    if t.size < 7:
        raise ValueError("Need at least 7 valid points to fit SOPDT model.")

    # Initial guesses
    y0_guess = float(np.mean(y[:max(3, len(y)//10)]))
    y_inf = float(np.mean(y[-max(3, len(y)//10):]))
    Ka0 = y_inf - y0_guess
    tau0 = max((t[-1] - t[0]) / 3.0, 1e-6)
    tau1_0 = max(0.5 * tau0, 1e-6)
    tau2_0 = max(2.0 * tau0, 2e-6)
    theta0 = 0.0  # start with no dead time

    if fit_y0:
        popt, _ = curve_fit(
            lambda tt, Ka, tau1, tau2, theta, y0: sopdt_response(tt, Ka, tau1, tau2, theta, y0=y0),
            t, y,
            p0=[Ka0, tau1_0, tau2_0, theta0, y0_guess],
            bounds=([-np.inf, 1e-9, 1e-9, 0.0, -np.inf], [np.inf, np.inf, np.inf, t[-1], np.inf]),
            maxfev=50000
        )
        Ka_hat, tau1_hat, tau2_hat, theta_hat, y0_hat = map(float, popt)
    else:
        popt, _ = curve_fit(
            lambda tt, Ka, tau1, tau2, theta: sopdt_response(tt, Ka, tau1, tau2, theta, y0=y0_guess),
            t, y,
            p0=[Ka0, tau1_0, tau2_0, theta0],
            bounds=([-np.inf, 1e-9, 1e-9, 0.0], [np.inf, np.inf, np.inf, t[-1]]),
            maxfev=50000
        )
        Ka_hat, tau1_hat, tau2_hat, theta_hat = map(float, popt)
        y0_hat = float(y0_guess)

    # Keep tau1 <= tau2 for consistent reporting.
    tau1_hat, tau2_hat = sorted([max(tau1_hat, 1e-9), max(tau2_hat, 1e-9)])

    y_fit = sopdt_response(t, Ka_hat, tau1_hat, tau2_hat, theta_hat, y0=y0_hat)
    residuals = y - y_fit

    SSE = float(np.sum(residuals ** 2))
    ybar = float(np.mean(y))
    SStot = float(np.sum((y - ybar) ** 2))
    R2 = float(1.0 - SSE / SStot) if SStot > 0 else float("nan")

    return {
        "t": t,
        "y": y,
        "Ka": Ka_hat,
        "tau1": tau1_hat,
        "tau2": tau2_hat,
        "theta": theta_hat,
        "y0": y0_hat,
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residuals,
        "Ka0": Ka0,
        "tau1_0": tau1_0,
        "tau2_0": tau2_0,
        "theta0": theta0,
        "y0_guess": y0_guess,
    }


def fit_fopdt_full_dataset(t, u, y, fit_y0=True, u0=None):
    """
    Fit a First-Order Plus Dead Time model to the entire dataset using the full
    input history, so repeated steps/dips are handled in one optimization.

    Parameters
    ----------
    t : array-like
        Sample times.
    u : array-like
        Input signal history.
    y : array-like
        Output signal history.
    fit_y0 : bool
        If True, fit the output baseline y0. Otherwise keep y0 fixed at the
        initial-level estimate.
    u0 : float or None
        Optional input baseline. If None, infer from the first ~10% of samples.
    """
    t, u, y = _clean_sort_u(t, u, y)
    if t.size < 6:
        raise ValueError("Need at least 6 valid points to fit full-dataset FOPDT model.")

    guess = _infer_full_dataset_guesses(t, u, y)
    y0_guess = guess["y0_guess"]
    u0_guess = float(guess["u0_guess"] if u0 is None else u0)
    K0 = guess["K0"]
    tau0 = guess["tau0"]
    theta0 = guess["theta0"]
    dt_med = guess["dt_med"]
    span_t = guess["span_t"]

    tau_min = max(0.25 * dt_med, 1e-9)
    tau_max = max(5.0 * span_t, tau_min * 10.0)
    theta_max = max(dt_med, 0.5 * span_t)
    theta0 = _estimate_dead_time_guess(t, u, y, theta_max=theta_max, dt_med=dt_med)

    if abs(K0) < 1e-12:
        du_span = max(np.max(u) - np.min(u), 1e-12)
        K0 = (np.max(y) - np.min(y)) / du_span

    def unpack(p):
        if fit_y0:
            K_hat, tau_hat, theta_hat, y0_hat = map(float, p)
        else:
            K_hat, tau_hat, theta_hat = map(float, p)
            y0_hat = float(y0_guess)
        return K_hat, tau_hat, theta_hat, y0_hat

    def residuals(p):
        K_hat, tau_hat, theta_hat, y0_hat = unpack(p)
        if tau_hat <= tau_min or theta_hat < 0.0 or theta_hat > theta_max:
            return np.full_like(y, 1e12, dtype=float)
        y_fit = _first_order_from_input_response(
            t, u, K_hat, tau_hat, theta_hat, y0=y0_hat, u0=u0_guess
        )
        if not np.all(np.isfinite(y_fit)):
            return np.full_like(y, 1e12, dtype=float)
        return y - y_fit

    if fit_y0:
        starts = [
            np.array([K0, tau0, theta0, y0_guess], dtype=float),
            np.array([0.5 * K0, max(0.5 * tau0, tau_min), min(0.05 * span_t, theta_max), y0_guess], dtype=float),
            np.array([1.5 * K0, min(2.0 * tau0, tau_max), min(0.10 * span_t, theta_max), y0_guess], dtype=float),
            np.array([K0, max(0.25 * tau0, tau_min), theta0, y0_guess], dtype=float),
            np.array([K0, min(4.0 * tau0, tau_max), min(1.5 * theta0, theta_max), y0_guess], dtype=float),
        ]
        lower = np.array([-np.inf, tau_min, 0.0, -np.inf], dtype=float)
        upper = np.array([np.inf, tau_max, theta_max, np.inf], dtype=float)
    else:
        starts = [
            np.array([K0, tau0, theta0], dtype=float),
            np.array([0.5 * K0, max(0.5 * tau0, tau_min), min(0.05 * span_t, theta_max)], dtype=float),
            np.array([1.5 * K0, min(2.0 * tau0, tau_max), min(0.10 * span_t, theta_max)], dtype=float),
            np.array([K0, max(0.25 * tau0, tau_min), theta0], dtype=float),
            np.array([K0, min(4.0 * tau0, tau_max), min(1.5 * theta0, theta_max)], dtype=float),
        ]
        lower = np.array([-np.inf, tau_min, 0.0], dtype=float)
        upper = np.array([np.inf, tau_max, theta_max], dtype=float)

    best = None
    best_cost = np.inf
    for p0 in starts:
        opt = least_squares(residuals, x0=p0, bounds=(lower, upper), max_nfev=30000)
        if opt.success and np.isfinite(opt.cost) and opt.cost < best_cost:
            best = opt
            best_cost = float(opt.cost)

    if best is None:
        raise ValueError("FOPDT full-dataset fit failed to converge.")

    K_hat, tau_hat, theta_hat, y0_hat = unpack(best.x)
    y_fit = _first_order_from_input_response(
        t, u, K_hat, tau_hat, theta_hat, y0=y0_hat, u0=u0_guess
    )
    residual = y - y_fit
    SSE = float(np.sum(residual ** 2))
    ybar = float(np.mean(y))
    SStot = float(np.sum((y - ybar) ** 2))
    R2 = float(1.0 - SSE / SStot) if SStot > 0 else float("nan")

    return {
        "t": t,
        "u": u,
        "y": y,
        "K": float(K_hat),
        "tau": float(tau_hat),
        "theta": float(theta_hat),
        "y0": float(y0_hat),
        "u0": float(u0_guess),
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residual,
        "u_delayed": _delay_signal_zoh(t, u - float(u0_guess), theta=float(theta_hat), u_init=0.0),
        "K0": float(K0),
        "tau0": float(tau0),
        "theta0": float(theta0),
        "y0_guess": float(y0_guess),
    }


def fit_sopdt_full_dataset(t, u, y, fit_y0=True, u0=None):
    """
    Fit a Second-Order Plus Dead Time model to the entire dataset using the full
    input history, so repeated steps/dips are handled in one optimization.

    tau1 is returned as the slower time constant and tau2 as the faster one.
    """
    t, u, y = _clean_sort_u(t, u, y)
    if t.size < 8:
        raise ValueError("Need at least 8 valid points to fit full-dataset SOPDT model.")

    guess = _infer_full_dataset_guesses(t, u, y)
    y0_guess = guess["y0_guess"]
    u0_guess = float(guess["u0_guess"] if u0 is None else u0)
    K0 = guess["K0"]
    tau0 = guess["tau0"]
    theta0 = guess["theta0"]
    dt_med = guess["dt_med"]
    span_t = guess["span_t"]

    tau_min = max(0.25 * dt_med, 1e-9)
    tau_max = max(5.0 * span_t, tau_min * 10.0)
    theta_max = max(dt_med, 0.5 * span_t)
    theta0 = _estimate_dead_time_guess(t, u, y, theta_max=theta_max, dt_med=dt_med)

    if abs(K0) < 1e-12:
        du_span = max(np.max(u) - np.min(u), 1e-12)
        K0 = (np.max(y) - np.min(y)) / du_span

    tau1_0 = min(max(2.0 * tau0, tau_min), tau_max)
    tau2_0 = min(max(0.5 * tau0, tau_min), tau_max)
    fopdt_seed = None
    try:
        fopdt_seed = fit_fopdt_full_dataset(t, u, y, fit_y0=fit_y0, u0=u0_guess)
    except Exception:
        fopdt_seed = None

    def unpack(p):
        if fit_y0:
            K_hat, tau_a, tau_b, theta_hat, y0_hat = map(float, p)
        else:
            K_hat, tau_a, tau_b, theta_hat = map(float, p)
            y0_hat = float(y0_guess)
        tau1_hat, tau2_hat = sorted([tau_a, tau_b], reverse=True)
        return K_hat, tau1_hat, tau2_hat, theta_hat, y0_hat

    def residuals(p):
        K_hat, tau1_hat, tau2_hat, theta_hat, y0_hat = unpack(p)
        if tau1_hat <= tau_min or tau2_hat <= tau_min or theta_hat < 0.0 or theta_hat > theta_max:
            return np.full_like(y, 1e12, dtype=float)
        y_fit = _second_order_from_input_response(
            t, u, K_hat, tau1_hat, tau2_hat, theta_hat, y0=y0_hat, u0=u0_guess
        )
        if not np.all(np.isfinite(y_fit)):
            return np.full_like(y, 1e12, dtype=float)
        return y - y_fit

    if fit_y0:
        starts = [
            np.array([K0, tau1_0, tau2_0, theta0, y0_guess], dtype=float),
            np.array([0.5 * K0, min(max(3.0 * tau0, tau_min), tau_max), min(max(0.3 * tau0, tau_min), tau_max), min(0.05 * span_t, theta_max), y0_guess], dtype=float),
            np.array([1.5 * K0, min(max(1.5 * tau0, tau_min), tau_max), min(max(0.7 * tau0, tau_min), tau_max), min(0.10 * span_t, theta_max), y0_guess], dtype=float),
            np.array([K0, min(max(4.0 * tau0, tau_min), tau_max), min(max(0.2 * tau0, tau_min), tau_max), theta0, y0_guess], dtype=float),
        ]
        if fopdt_seed is not None:
            starts.append(
                np.array([
                    float(fopdt_seed["K"]),
                    min(max(1.5 * float(fopdt_seed["tau"]), tau_min), tau_max),
                    min(max(0.5 * float(fopdt_seed["tau"]), tau_min), tau_max),
                    min(float(fopdt_seed["theta"]), theta_max),
                    float(fopdt_seed["y0"]),
                ], dtype=float)
            )
        lower = np.array([-np.inf, tau_min, tau_min, 0.0, -np.inf], dtype=float)
        upper = np.array([np.inf, tau_max, tau_max, theta_max, np.inf], dtype=float)
    else:
        starts = [
            np.array([K0, tau1_0, tau2_0, theta0], dtype=float),
            np.array([0.5 * K0, min(max(3.0 * tau0, tau_min), tau_max), min(max(0.3 * tau0, tau_min), tau_max), min(0.05 * span_t, theta_max)], dtype=float),
            np.array([1.5 * K0, min(max(1.5 * tau0, tau_min), tau_max), min(max(0.7 * tau0, tau_min), tau_max), min(0.10 * span_t, theta_max)], dtype=float),
            np.array([K0, min(max(4.0 * tau0, tau_min), tau_max), min(max(0.2 * tau0, tau_min), tau_max), theta0], dtype=float),
        ]
        if fopdt_seed is not None:
            starts.append(
                np.array([
                    float(fopdt_seed["K"]),
                    min(max(1.5 * float(fopdt_seed["tau"]), tau_min), tau_max),
                    min(max(0.5 * float(fopdt_seed["tau"]), tau_min), tau_max),
                    min(float(fopdt_seed["theta"]), theta_max),
                ], dtype=float)
            )
        lower = np.array([-np.inf, tau_min, tau_min, 0.0], dtype=float)
        upper = np.array([np.inf, tau_max, tau_max, theta_max], dtype=float)

    best = None
    best_cost = np.inf
    for p0 in starts:
        opt = least_squares(residuals, x0=p0, bounds=(lower, upper), max_nfev=50000)
        if opt.success and np.isfinite(opt.cost) and opt.cost < best_cost:
            best = opt
            best_cost = float(opt.cost)

    if best is None:
        raise ValueError("SOPDT full-dataset fit failed to converge.")

    K_hat, tau1_hat, tau2_hat, theta_hat, y0_hat = unpack(best.x)
    y_fit = _second_order_from_input_response(
        t, u, K_hat, tau1_hat, tau2_hat, theta_hat, y0=y0_hat, u0=u0_guess
    )
    residual = y - y_fit
    SSE = float(np.sum(residual ** 2))
    ybar = float(np.mean(y))
    SStot = float(np.sum((y - ybar) ** 2))
    R2 = float(1.0 - SSE / SStot) if SStot > 0 else float("nan")

    return {
        "t": t,
        "u": u,
        "y": y,
        "K": float(K_hat),
        "tau1": float(tau1_hat),
        "tau2": float(tau2_hat),
        "theta": float(theta_hat),
        "y0": float(y0_hat),
        "u0": float(u0_guess),
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residual,
        "u_delayed": _delay_signal_zoh(t, u - float(u0_guess), theta=float(theta_hat), u_init=0.0),
        "K0": float(K0),
        "tau1_0": float(tau1_0),
        "tau2_0": float(tau2_0),
        "theta0": float(theta0),
        "y0_guess": float(y0_guess),
    }
