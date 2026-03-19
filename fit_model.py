# fit_model.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit, least_squares, minimize


def _as_1d_float_array(x, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return np.ravel(arr)

def _clean_sort(t, y):
    t = _as_1d_float_array(t, "t")
    y = _as_1d_float_array(y, "y")
    if t.size != y.size:
        raise ValueError("t and y must have the same number of samples.")
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    idx = np.argsort(t)
    return t[idx], y[idx]


def _clean_sort_u(t, u, y):
    t = _as_1d_float_array(t, "t")
    u = _as_1d_float_array(u, "u")
    y = _as_1d_float_array(y, "y")
    if not (t.size == u.size == y.size):
        raise ValueError("t, u, and y must have the same number of samples.")
    m = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    t, u, y = t[m], u[m], y[m]
    idx = np.argsort(t)
    return t[idx], u[idx], y[idx]


def _median_positive_dt(t):
    t = _as_1d_float_array(t, "t")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Need at least two increasing time samples.")
    return float(np.median(dt))


def _delay_input_by_samples(u, n_delay):
    """
    Dead-time delay used by the recursive global FOPDT model.

    For a whole dataset with multiple pump-power changes, delaying the full
    input sequence and simulating sample-by-sample is the correct approach.
    Reusing a single closed-form step response over the entire record is not.
    """
    u = _as_1d_float_array(u, "u")
    n_delay = int(max(n_delay, 0))
    if n_delay == 0:
        return u.copy()
    if n_delay >= u.size:
        return np.zeros_like(u, dtype=float)
    return np.concatenate([np.zeros(n_delay, dtype=float), u[:-n_delay]])


def _detect_first_input_step_time(t, u):
    t, u = _clean_sort(t, u)
    if t.size == 0:
        return None
    if t.size == 1:
        return float(t[0])
    du = np.abs(np.diff(u))
    if du.size == 0:
        return float(t[0])
    tol = max(1e-9, 0.01 * float(np.max(du)))
    idx = np.where(du > tol)[0]
    if idx.size == 0:
        return float(t[0])
    return float(t[int(idx[0]) + 1])


def _find_first_input_step_index(u):
    u = _as_1d_float_array(u, "u")
    if u.size < 2:
        return None
    du = np.abs(np.diff(u))
    if du.size == 0:
        return None
    tol = max(1e-9, 0.01 * float(np.max(du)))
    idx = np.where(du > tol)[0]
    if idx.size == 0:
        return None
    return int(idx[0]) + 1


def _format_metric(value):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.6g}"


def _unpack_global_fopdt_params(params, use_bias=False):
    params = _as_1d_float_array(params, "params")
    expected = 4 if use_bias else 3
    if params.size != expected:
        raise ValueError(f"Expected {expected} FOPDT parameters, got {params.size}.")
    if use_bias:
        K, tau, theta, y_bias = map(float, params)
        return K, tau, theta, y_bias
    K, tau, theta = map(float, params)
    return K, tau, theta, None


def _prepare_global_fopdt_data(t, u, y, use_bias=False):
    """
    Prepare signals for the global recursive FOPDT fit.

    When a pre-step operating region exists, the input is centered around that
    baseline so the delayed-input zero padding represents "no change yet". For
    the no-bias model we also fit output deviation and then add the baseline
    back, which preserves yhat[0] = y[0] while avoiding a false decay toward 0
    for nonzero absolute tank-height baselines.
    """
    t, u, y = _clean_sort_u(t, u, y)
    step_idx = _find_first_input_step_index(u)
    first_step_time = float(t[step_idx]) if step_idx is not None else float(t[0])

    has_pre_step = step_idx is not None and step_idx > 0
    if has_pre_step:
        u0 = float(np.mean(u[:step_idx]))
        y_ref = float(np.mean(y[:step_idx]))
    else:
        u0 = 0.0
        y_ref = 0.0

    u_fit = u - u0
    if use_bias:
        y_target = y.copy()
        y0_sim = float(y[0])
    else:
        y_target = y - y_ref
        y0_sim = float(y_target[0])

    return {
        "t": t,
        "u": u,
        "y": y,
        "u_fit": u_fit,
        "y_target": y_target,
        "y0_sim": y0_sim,
        "u0": u0,
        "y_ref": y_ref,
        "step_idx": step_idx,
        "first_step_time": first_step_time,
        "t_relative": t - first_step_time,
    }


def _predict_fopdt_global_from_prepped(params, prep, use_bias=False):
    yhat_model = simulate_fopdt_global(
        params,
        prep["t"],
        prep["u_fit"],
        y0=float(prep["y0_sim"]),
        use_bias=use_bias,
    )
    if yhat_model.shape != prep["y"].shape or not np.all(np.isfinite(yhat_model)):
        return np.full_like(prep["y"], np.nan, dtype=float)
    if use_bias:
        return yhat_model
    return prep["y_ref"] + yhat_model


def predict_fopdt_global(params, t, u, y, use_bias=False):
    """
    Predict the full output trace for a global FOPDT parameter set.

    This helper applies the same operating-point handling used during the fit:
    the input is centered around its pre-step baseline and, for the no-bias
    model, the output deviation is simulated and shifted back to the measured
    tank-height baseline.
    """
    prep = _prepare_global_fopdt_data(t, u, y, use_bias=use_bias)
    yhat = _predict_fopdt_global_from_prepped(params, prep, use_bias=use_bias)
    if yhat.shape != prep["y"].shape or not np.all(np.isfinite(yhat)):
        return np.full_like(prep["y"], np.nan, dtype=float)
    return yhat


def _residuals_fopdt_global_from_prepped(params, prep, use_bias=False):
    yhat_actual = _predict_fopdt_global_from_prepped(params, prep, use_bias=use_bias)
    if yhat_actual.shape != prep["y"].shape or not np.all(np.isfinite(yhat_actual)):
        return np.full_like(prep["y"], 1e10, dtype=float)
    return prep["y"] - yhat_actual


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


def simulate_fopdt_global(params, t, u, y0, use_bias=False):
    """
    Simulate a global FOPDT model using a non-recursive convolution form.

    This global model must use the real input value u[k] at every time step.
    It must not use the closed-form single-step FOPDT equation over the whole
    dataset, because that would collapse repeated rises and drains into one
    smooth exponential trend.

    For the Euler-discretized model
      y[k+1] = (1 - dt/tau) * y[k] + (dt/tau) * K * u_delayed[k]
    this function evaluates the algebraically equivalent closed-form sum over
    the full delayed input history instead of stepping recursively in Python.
    """
    if use_bias:
        K, tau, theta, y_bias = _unpack_global_fopdt_params(params, use_bias=True)
    else:
        K, tau, theta, _ = _unpack_global_fopdt_params(params, use_bias=False)
        y_bias = None

    t = np.asarray(t, dtype=float).flatten()
    u = np.asarray(u, dtype=float).flatten()

    if len(t) != len(u):
        raise ValueError("t and u must have same length")

    if len(t) == 0:
        return np.array([], dtype=float)

    if tau <= 0 or theta < 0:
        return np.full_like(t, np.nan, dtype=float)

    if len(t) == 1:
        yhat = np.zeros_like(t, dtype=float)
        yhat[0] = float(y0)
        return yhat

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return np.full_like(t, np.nan, dtype=float)

    n_delay = int(round(theta / dt))

    if n_delay > 0:
        u_delayed = np.concatenate([np.zeros(n_delay), u[:-n_delay]])
    else:
        u_delayed = u.copy()

    alpha = float(dt / tau)
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha >= 2.0:
        return np.full_like(t, np.nan, dtype=float)

    a = 1.0 - alpha
    powers = np.power(a, np.arange(len(t), dtype=float))
    if not np.all(np.isfinite(powers)):
        return np.full_like(t, np.nan, dtype=float)

    kernel = (alpha * float(K)) * powers
    conv_term = np.convolve(u_delayed, kernel, mode="full")

    yhat = np.zeros_like(t, dtype=float)
    yhat[0] = float(y0)
    if len(t) > 1:
        yhat[1:] = powers[1:] * float(y0) + conv_term[: len(t) - 1]
        if use_bias:
            yhat[1:] = yhat[1:] + float(y_bias) * (1.0 - powers[1:])

    if not np.all(np.isfinite(yhat)):
        return np.full_like(t, np.nan, dtype=float)

    return yhat


def objective_fopdt_global(params, t, u, y, use_bias=False):
    """
    Sum-of-squared-errors objective for the recursive global FOPDT model.
    """
    prep = _prepare_global_fopdt_data(t, u, y, use_bias=use_bias)
    residuals = _residuals_fopdt_global_from_prepped(params, prep, use_bias=use_bias)
    if np.any(~np.isfinite(residuals)):
        return 1e20
    return float(np.sum(residuals ** 2))


def fit_fopdt_global(
    t,
    u,
    y,
    use_bias=False,
    initial_guess=None,
    bounds=None,
    maxiter=300,
    max_delay_candidates=25,
    refine_delay_window=2,
):
    """
    Fit one global FOPDT model to the full dataset using a non-recursive
    convolution evaluation of the full delayed input sequence.

    This is the correct whole-dataset formulation:
      yhat[k+1] = yhat[k] + (dt/tau) * (-yhat[k] + K * u_delayed[k])

    and, optionally,
      yhat[k+1] = yhat[k] + (dt/tau) * (-(yhat[k] - y_bias) + K * u_delayed[k])
    """
    prep = _prepare_global_fopdt_data(t, u, y, use_bias=use_bias)
    t = prep["t"]
    u = prep["u"]
    y = prep["y"]
    if t.size < 6:
        raise ValueError("Need at least 6 valid points to fit a global FOPDT model.")

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Need strictly increasing time samples for global FOPDT fit.")
    span_t = float(max(t[-1] - t[0], dt))
    guess = _infer_full_dataset_guesses(t, prep["u_fit"], prep["y_target"])
    K0 = float(guess["K0"])
    tau0 = float(max(guess["tau0"], dt))
    theta0 = float(
        _estimate_dead_time_guess(
            t,
            prep["u_fit"],
            prep["y_target"],
            theta_max=max(dt, 0.5 * span_t),
            dt_med=dt,
        )
    )

    if abs(K0) < 1e-12:
        du_span = max(float(np.ptp(prep["u_fit"])), float(np.max(np.abs(prep["u_fit"]))), 1e-9)
        K0 = float((np.max(prep["y_target"]) - np.min(prep["y_target"])) / du_span)

    dy_span = max(
        float(np.ptp(prep["y_target"])),
        abs(float(prep["y_target"][-1] - prep["y_target"][0])),
        float(np.std(prep["y_target"])),
        1e-6,
    )
    du_scale = max(float(np.ptp(prep["u_fit"])), float(np.max(np.abs(prep["u_fit"]))), 1e-6)
    K_bound = max(10.0 * abs(K0), 10.0 * dy_span / du_scale, 1.0)
    tau_min = max(0.51 * dt, 1e-9)
    tau_max = max(5.0 * span_t, 10.0 * dt, 2.0 * tau0)
    theta_max = max(0.0, min(0.5 * span_t, span_t))
    n_delay_max = int(max(0, round(theta_max / dt)))
    first_step_time = prep["first_step_time"]

    if use_bias:
        y_bias0 = float(prep["y_ref"] if prep["step_idx"] is not None and prep["step_idx"] > 0 else y[0])
        y_margin = max(2.0 * dy_span, 1.0)
        lower_full = np.array([-K_bound, tau_min, 0.0, np.min(y) - y_margin], dtype=float)
        upper_full = np.array([K_bound, tau_max, theta_max, np.max(y) + y_margin], dtype=float)
        starts = [
            np.array([K0, tau0, y_bias0], dtype=float),
            np.array([0.5 * K0 if abs(K0) > 1e-12 else np.sign(y[-1] - y[0]) * max(0.1, dy_span / du_scale), max(0.5 * tau0, tau_min), y_bias0], dtype=float),
            np.array([1.5 * K0 if abs(K0) > 1e-12 else max(0.1, dy_span / du_scale), min(2.0 * tau0, tau_max), y_bias0], dtype=float),
        ]
        initial_guess_used = np.array([K0, tau0, theta0, y_bias0], dtype=float)
    else:
        y_bias0 = None
        lower_full = np.array([-K_bound, tau_min, 0.0], dtype=float)
        upper_full = np.array([K_bound, tau_max, theta_max], dtype=float)
        starts = [
            np.array([K0, tau0], dtype=float),
            np.array([0.5 * K0 if abs(K0) > 1e-12 else np.sign(y[-1] - y[0]) * max(0.1, dy_span / du_scale), max(0.5 * tau0, tau_min)], dtype=float),
            np.array([1.5 * K0 if abs(K0) > 1e-12 else max(0.1, dy_span / du_scale), min(2.0 * tau0, tau_max)], dtype=float),
        ]
        initial_guess_used = np.array([K0, tau0, theta0], dtype=float)

    if bounds is not None:
        lower_in, upper_in = bounds
        lower_full = np.asarray(lower_in, dtype=float).flatten()
        upper_full = np.asarray(upper_in, dtype=float).flatten()
        expected = 4 if use_bias else 3
        if lower_full.size != expected or upper_full.size != expected:
            raise ValueError(f"bounds must have {expected} entries for this fit.")
        if not np.all(np.isfinite(lower_full)) or not np.all(np.isfinite(upper_full)) or np.any(lower_full >= upper_full):
            raise ValueError("bounds must be finite and satisfy lower < upper.")
        tau_min = float(lower_full[1])
        tau_max = float(upper_full[1])
        theta0 = float(np.clip(theta0, lower_full[2], upper_full[2]))

    if initial_guess is not None:
        guess_arr = np.asarray(initial_guess, dtype=float).flatten()
        expected = 4 if use_bias else 3
        if guess_arr.size != expected:
            raise ValueError(f"initial_guess must have {expected} entries for this fit.")
        initial_guess_used = guess_arr.copy()
        theta0 = float(np.clip(guess_arr[2], lower_full[2], upper_full[2]))
        if use_bias:
            starts.insert(0, np.array([guess_arr[0], guess_arr[1], guess_arr[3]], dtype=float))
        else:
            starts.insert(0, np.array([guess_arr[0], guess_arr[1]], dtype=float))

    if use_bias:
        lower = np.array([lower_full[0], lower_full[1], lower_full[3]], dtype=float)
        upper = np.array([upper_full[0], upper_full[1], upper_full[3]], dtype=float)
    else:
        lower = np.array([lower_full[0], lower_full[1]], dtype=float)
        upper = np.array([upper_full[0], upper_full[1]], dtype=float)

    theta_min = float(max(0.0, lower_full[2]))
    theta_max = float(max(theta_min, upper_full[2]))
    n_delay_min = int(max(0, round(theta_min / dt)))
    n_delay_max = int(max(n_delay_min, round(theta_max / dt)))

    max_delay_candidates = int(max(5, max_delay_candidates))
    refine_delay_window = int(max(0, refine_delay_window))
    maxiter = int(max(10, maxiter))

    if n_delay_max <= max_delay_candidates:
        delay_candidates = np.arange(n_delay_min, n_delay_max + 1, dtype=int)
    else:
        delay_candidates = np.unique(np.round(np.linspace(n_delay_min, n_delay_max, max_delay_candidates)).astype(int))
    theta_guess_idx = int(np.clip(round(theta0 / dt), n_delay_min, n_delay_max))
    delay_candidates = np.unique(
        np.concatenate([
            delay_candidates,
            np.array(
                [theta_guess_idx, max(theta_guess_idx - 1, n_delay_min), min(theta_guess_idx + 1, n_delay_max)],
                dtype=int,
            ),
        ])
    )

    def _assemble_params(x, theta_value):
        x = np.asarray(x, dtype=float).flatten()
        if use_bias:
            return np.array([float(x[0]), float(x[1]), float(theta_value), float(x[2])], dtype=float)
        return np.array([float(x[0]), float(x[1]), float(theta_value)], dtype=float)

    def _fit_continuous_for_theta(theta_value, extra_starts=None):
        best_local = None
        best_local_sse = np.inf
        use_starts = list(starts)
        if extra_starts:
            use_starts.extend(extra_starts)

        for x0 in use_starts:
            x0 = np.clip(np.asarray(x0, dtype=float), lower, upper)

            def residuals_local(x):
                params_local = _assemble_params(x, theta_value)
                return _residuals_fopdt_global_from_prepped(params_local, prep, use_bias=use_bias)

            opt = least_squares(
                residuals_local,
                x0=x0,
                bounds=(lower, upper),
                method="trf",
                max_nfev=maxiter,
                loss="linear",
            )
            x_hat = np.clip(np.asarray(opt.x, dtype=float), lower, upper)
            params_hat = _assemble_params(x_hat, theta_value)
            residuals_hat = _residuals_fopdt_global_from_prepped(params_hat, prep, use_bias=use_bias)
            sse_hat = float(np.sum(residuals_hat ** 2))
            if np.isfinite(sse_hat) and sse_hat < best_local_sse:
                best_local = params_hat
                best_local_sse = float(sse_hat)

        return best_local, best_local_sse

    best_params = None
    best_sse = np.inf
    best_delay = 0
    for n_delay in delay_candidates:
        theta_value = float(n_delay * dt)
        params_hat, sse_hat = _fit_continuous_for_theta(theta_value)
        if params_hat is not None and sse_hat < best_sse:
            best_params = params_hat
            best_sse = float(sse_hat)
            best_delay = int(n_delay)

    if best_params is None:
        raise ValueError("Global FOPDT fit failed to converge.")

    refine_delays = np.arange(
        max(n_delay_min, best_delay - refine_delay_window),
        min(n_delay_max, best_delay + refine_delay_window) + 1,
        dtype=int,
    )
    if use_bias:
        seeded_start = [np.array([best_params[0], best_params[1], best_params[3]], dtype=float)]
    else:
        seeded_start = [np.array([best_params[0], best_params[1]], dtype=float)]
    for n_delay in refine_delays:
        theta_value = float(n_delay * dt)
        params_hat, sse_hat = _fit_continuous_for_theta(theta_value, extra_starts=seeded_start)
        if params_hat is not None and sse_hat < best_sse:
            best_params = params_hat
            best_sse = float(sse_hat)
            best_delay = int(n_delay)

    best_params[2] = float(best_delay * dt)
    yhat = _predict_fopdt_global_from_prepped(best_params, prep, use_bias=use_bias)
    if yhat.shape != y.shape or not np.all(np.isfinite(yhat)):
        raise ValueError("Global FOPDT simulation became unstable at the fitted parameters.")

    residuals = y - yhat
    SSE = float(np.sum(residuals ** 2))
    RMSE = float(np.sqrt(SSE / len(y)))
    ybar = float(np.mean(y))
    SStot = float(np.sum((y - ybar) ** 2))
    R2 = float(1.0 - SSE / SStot) if SStot > 0 else float("nan")

    K_hat, tau_hat, theta_hat, y_bias_hat = _unpack_global_fopdt_params(best_params, use_bias=use_bias)
    n_delay = int(round(theta_hat / dt))
    u_delayed = _delay_input_by_samples(prep["u_fit"], n_delay)

    print(f"K = {_format_metric(K_hat)}")
    print(f"tau = {_format_metric(tau_hat)} s")
    print(f"theta = {_format_metric(theta_hat)} s")
    if use_bias:
        print(f"y_bias = {_format_metric(y_bias_hat)}")
    print(f"SSE = {_format_metric(SSE)}")
    print(f"RMSE = {_format_metric(RMSE)}")
    print(f"R^2 = {_format_metric(R2)}")

    return {
        "t": t,
        "t_relative": t - float(first_step_time if first_step_time is not None else t[0]),
        "u": u,
        "u_fit": prep["u_fit"],
        "y": y,
        "y_fit": yhat,
        "residuals": residuals,
        "params": best_params.copy(),
        "K": float(K_hat),
        "tau": float(tau_hat),
        "theta": float(theta_hat),
        "y0": float(y[0]),
        "y0_init": float(y[0]),
        "y_bias": float(y_bias_hat) if use_bias else np.nan,
        "use_bias": bool(use_bias),
        "dt": float(dt),
        "n_delay": int(n_delay),
        "u_delayed": u_delayed,
        "u0": float(prep["u0"]),
        "y_ref": float(prep["y_ref"]),
        "SSE": SSE,
        "RMSE": RMSE,
        "R2": R2,
        "K0": float(K0),
        "tau0": float(tau0),
        "theta0": float(theta0),
        "first_step_time": first_step_time,
        "initial_guess_used": initial_guess_used.copy(),
        "bounds_used": (lower_full.copy(), upper_full.copy()),
        "optimizer_maxiter": int(maxiter),
        "max_delay_candidates": int(max_delay_candidates),
    }


def plot_fopdt_global(t, y, yhat, params, r2, first_step_time=None):
    """
    Plot measured data against the global FOPDT fit.
    """
    t, y, yhat = _clean_sort_u(t, y, yhat)

    if isinstance(params, dict):
        K = float(params["K"])
        tau = float(params["tau"])
        theta = float(params["theta"])
        t_rel_from_params = params.get("t_relative")
        if first_step_time is None:
            fst = params.get("first_step_time")
            if fst is not None and np.isfinite(fst):
                first_step_time = float(fst)
    else:
        params = _as_1d_float_array(params, "params")
        t_rel_from_params = None
        K, tau, theta, _ = _unpack_global_fopdt_params(params, use_bias=(params.size == 4))

    if t_rel_from_params is not None and len(np.ravel(t_rel_from_params)) == len(t):
        t_rel = np.ravel(np.asarray(t_rel_from_params, dtype=float))
    elif first_step_time is None:
        first_step_time = float(t[0])
        t_rel = t - float(first_step_time)
    else:
        t_rel = t - float(first_step_time)
    fig, ax = plt.subplots()
    ax.plot(t_rel, y, "k.", markersize=4, label="Measured data")
    ax.plot(t_rel, yhat, color="red", linewidth=2.0, label="Global FOPDT fit")
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.4, label="First step")

    if theta > 0.0:
        ax.axvline(theta, color="tab:blue", linestyle="--", linewidth=1.4, label="Dead time")
        y_min = float(np.nanmin(np.concatenate([y, yhat])))
        y_max = float(np.nanmax(np.concatenate([y, yhat])))
        y_text = y_max - 0.06 * max(y_max - y_min, 1.0)
        ax.annotate(
            f"θ = {theta:.3g} s",
            xy=(theta, y_text),
            xytext=(4, 0),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            color="tab:blue",
        )

    ax.set_title(
        f"Global FOPDT Fit | K = {K:.6g}, tau = {tau:.6g} s, theta = {theta:.6g} s, R² = {r2:.6g}"
    )
    ax.set_xlabel("Time relative to first step (s)")
    ax.set_ylabel("Tank height")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


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


def fit_k_tau_global(t, y, fit_y0=True):
    """
    Fit K and tau (and optionally y0) to first-order step response for entire system.

    Assumes step at t=0, a=1, so K = Ka.

    Returns dict with keys: K, tau, y0, SSE, R2, y_fit, residuals, plus initial guesses.
    """
    t, y = _clean_sort(t, y)
    if t.size < 4:
        raise ValueError("Need at least 4 valid points to fit.")

    # Initial guesses, assuming t0=0
    y0_guess = float(np.mean(y[:max(3, len(y)//10)]))
    y_inf = float(np.mean(y[-max(3, len(y)//10):]))
    K0 = y_inf - y0_guess
    tau0 = max((t[-1] - t[0]) / 3.0, 1e-6)

    if fit_y0:
        popt, _ = curve_fit(
            lambda tt, K, tau, y0: first_order_response(tt, K, tau, y0=y0, t0=0.0),
            t, y,
            p0=[K0, tau0, y0_guess],
            bounds=([-np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=30000
        )
        K_hat, tau_hat, y0_hat = map(float, popt)
    else:
        popt, _ = curve_fit(
            lambda tt, K, tau: first_order_response(tt, K, tau, y0=y0_guess, t0=0.0),
            t, y,
            p0=[K0, tau0],
            bounds=([-np.inf, 1e-9], [np.inf, np.inf]),
            maxfev=30000
        )
        K_hat, tau_hat = map(float, popt)
        y0_hat = float(y0_guess)

    y_fit = first_order_response(t, K_hat, tau_hat, y0=y0_hat, t0=0.0)
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
        "y0": y0_hat,
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residuals,
        "K0": K0,
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
    Legacy compatibility wrapper for the recursive global FOPDT fit.

    For a full experiment with multiple input changes, a single closed-form
    FOPDT step response is not valid over the whole dataset. The global fit now
    uses the recursive delayed-input simulation implemented in
    fit_fopdt_global(...). If u0 is supplied, the input is shifted before
    fitting so the returned gain remains compatible with prior usage.
    """
    t, u, y = _clean_sort_u(t, u, y)
    if u0 is not None:
        u = u - float(u0)

    result = fit_fopdt_global(t, u, y, use_bias=bool(fit_y0))
    if u0 is not None:
        result["u0"] = float(u0)
    return result


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
