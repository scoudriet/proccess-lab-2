import io
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

# Import whichever fitters you created
# 1) Ka models
from fit_model import (
    fit_first_order as fit_Ka_tau,      # returns keys: Ka, tau, y0, ...
    fit_second_order as fit_Ka_tau2,    # returns keys: Ka, tau1, tau2, y0, ...
)

# 2) K model (with a)
# If you created the file I sent: fit_model_with_a.py
try:
    from fit_model_with_a import fit_with_a as fit_K_tau_with_a  # returns keys: K, tau, y0, ...
except Exception:
    fit_K_tau_with_a = None


# ----------------- Helpers -----------------
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    window = int(window)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def clean_sort_xy(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    idx = np.argsort(t)
    return t[idx], y[idx]


def estimate_step_time(t, y, window=5):
    """
    Simple step-time guess: find time where smoothed derivative is max.
    Works decently for lab data.
    """
    t, y = clean_sort_xy(t, y)
    ys = moving_average(y, max(1, window))
    dy = np.gradient(ys, t)
    i = int(np.argmax(np.abs(dy)))
    return float(t[i])


def make_excel_bytes(data_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        data_df.to_excel(writer, index=False, sheet_name="FittedData")
        summary_df.to_excel(writer, index=False, sheet_name="FitSummary")
    return buf.getvalue()


def parse_time_to_elapsed_seconds(time_series: pd.Series) -> np.ndarray:
    """
    Parse mixed time formats into elapsed seconds starting at 0.
    Supports:
      - Excel time-of-day values
      - elapsed strings like 00:01 / 00:06 / 00:00:30
      - datetime-like values
      - numeric values (seconds or Excel day-fractions)
    """
    s = pd.Series(time_series)
    out = np.full(len(s), np.nan, dtype=float)
    saw_clock_time = False

    for i, v in enumerate(s):
        if pd.isna(v):
            continue

        if isinstance(v, pd.Timedelta):
            out[i] = float(v.total_seconds())
            continue
        if isinstance(v, dt.timedelta):
            out[i] = float(v.total_seconds())
            continue
        if isinstance(v, dt.time):
            saw_clock_time = True
            out[i] = float(v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6)
            continue
        if isinstance(v, pd.Timestamp):
            out[i] = float(v.timestamp())
            continue
        if isinstance(v, dt.datetime):
            out[i] = float(v.timestamp())
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[i] = float(v)
            continue

        txt = str(v).strip()
        if not txt:
            continue

        td = pd.to_timedelta(txt, errors="coerce")
        if pd.notna(td):
            out[i] = float(td.total_seconds())
            continue

        parsed = False
        for fmt in ["%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"]:
            try:
                tval = dt.datetime.strptime(txt, fmt).time()
                saw_clock_time = True
                out[i] = float(tval.hour * 3600 + tval.minute * 60 + tval.second)
                parsed = True
                break
            except ValueError:
                continue
        if parsed:
            continue

        dval = pd.to_datetime(txt, errors="coerce")
        if pd.notna(dval):
            out[i] = float(pd.Timestamp(dval).timestamp())

    if np.isfinite(out).sum() < 2:
        raise ValueError("Could not parse time column into elapsed seconds.")

    # Excel often stores time-of-day as fraction of one day.
    if pd.api.types.is_numeric_dtype(s):
        finite = out[np.isfinite(out)]
        if finite.size and np.nanmax(np.abs(finite)) <= 2.0:
            out = out * 86400.0
            saw_clock_time = True

    # Unwrap midnight rollover for clock-style values.
    if saw_clock_time:
        shift = 0.0
        prev = None
        for i in range(len(out)):
            if not np.isfinite(out[i]):
                continue
            cur = out[i] + shift
            if prev is not None and cur < (prev - 43200.0):
                shift += 86400.0
                cur = out[i] + shift
            out[i] = cur
            prev = cur

    first = out[np.isfinite(out)][0]
    out = out - first
    return out


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    w = int(max(3, window))
    if w % 2 == 0:
        w += 1
    return pd.Series(y).rolling(window=w, center=True, min_periods=1).median().to_numpy(dtype=float)


def detect_on_off_times(t: np.ndarray, y: np.ndarray, smooth_window: int = 11, edge_frac: float = 0.05):
    t, y = clean_sort_xy(t, y)
    ys = rolling_median(y, smooth_window)
    dy = np.gradient(ys, t)

    n = len(t)
    edge = max(2, int(edge_frac * n))
    lo, hi = edge, max(edge + 3, n - edge)
    if hi <= lo:
        lo, hi = 1, n - 1

    on_idx = lo + int(np.argmax(dy[lo:hi]))
    min_gap = max(3, int(0.05 * n))
    off_start = min(n - 2, on_idx + min_gap)
    if off_start >= hi:
        off_start = min(n - 2, on_idx + 1)

    if off_start < hi:
        off_idx = off_start + int(np.argmin(dy[off_start:hi]))
    else:
        off_idx = n - 1

    if off_idx <= on_idx:
        off_idx = min(n - 1, on_idx + 1)

    return float(t[on_idx]), float(t[off_idx]), ys, dy


def build_input_profile(t: np.ndarray, t_on: float, t_off: float, u_on: float) -> np.ndarray:
    u = np.zeros_like(t, dtype=float)
    u[(t >= float(t_on)) & (t < float(t_off))] = float(u_on)
    return u


def simulate_first_order_deviation(t: np.ndarray, u: np.ndarray, Kp: float, tau: float, x0: float) -> np.ndarray:
    tau = float(max(tau, 1e-9))
    x = np.zeros_like(t, dtype=float)
    x[0] = float(x0)

    for i in range(1, len(t)):
        dt_i = float(max(t[i] - t[i - 1], 1e-9))
        a = np.exp(-dt_i / tau)
        x[i] = x[i - 1] * a + float(Kp) * float(u[i - 1]) * (1.0 - a)

    return x


def fit_full_model_global(run_rows):
    # Initial guess from observed change and ON-level.
    kp_guess_list, tau_guess_list = [], []
    for rr in run_rows:
        u_on = max(abs(rr["u_on"]), 1e-9)
        kp_guess_list.append((np.nanmax(rr["T_dev"]) - np.nanmin(rr["T_dev"])) / u_on)
        tau_guess_list.append(max(1.0, 0.5 * max(rr["t_off"] - rr["t_on"], 1.0)))

    kp0 = float(np.nanmedian(kp_guess_list)) if kp_guess_list else 0.1
    tau0 = float(np.nanmedian(tau_guess_list)) if tau_guess_list else 30.0

    def sse_obj(p):
        kp, tau = float(p[0]), float(p[1])
        if tau <= 0:
            return 1e30
        sse = 0.0
        for rr in run_rows:
            pred = simulate_first_order_deviation(rr["t"], rr["u"], kp, tau, rr["T_dev"][0])
            err = rr["T_dev"] - pred
            sse += float(np.sum(err ** 2))
        return sse

    opt = minimize(
        sse_obj,
        x0=np.array([kp0, tau0], dtype=float),
        method="L-BFGS-B",
        bounds=[(None, None), (1e-9, None)],
    )
    if not opt.success:
        raise ValueError(f"Optimization failed: {opt.message}")

    kp_hat, tau_hat = map(float, opt.x)

    all_y, all_yhat, per_run = [], [], []
    for rr in run_rows:
        pred_dev = simulate_first_order_deviation(rr["t"], rr["u"], kp_hat, tau_hat, rr["T_dev"][0])
        pred_T = rr["T_base"] + pred_dev
        resid = rr["T"] - pred_T
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((rr["T"] - np.mean(rr["T"])) ** 2))
        r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
        per_run.append({
            "run": rr["run"],
            "R2": r2,
            "t_on": rr["t_on"],
            "t_off": rr["t_off"],
            "T_base": rr["T_base"],
            "pred_T": pred_T,
            "residual": resid,
        })
        all_y.append(rr["T"])
        all_yhat.append(pred_T)

    y_all = np.concatenate(all_y)
    yhat_all = np.concatenate(all_yhat)
    sse_all = float(np.sum((y_all - yhat_all) ** 2))
    sst_all = float(np.sum((y_all - np.mean(y_all)) ** 2))
    r2_all = float(1.0 - sse_all / sst_all) if sst_all > 0 else float("nan")

    return {
        "Kp": kp_hat,
        "tau": tau_hat,
        "SSE": sse_all,
        "R2": r2_all,
        "per_run": per_run,
    }


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Process Fit Tool", layout="wide")
FULL_MODEL_LABEL = "Fit full model (single from full on/off data)"

st.title("Process Fit (Smooth GUI)")
st.caption("Upload Excel → pick model → fit → visualize → download results")

# Sidebar controls (smooth UX)
with st.sidebar:
    st.header("Controls")

    model = st.selectbox(
        "Model",
        [
            "Fit Ka & τ (board style)",
            "Fit K & τ (you enter step size a)",
            "Fit 2nd-order Ka, τ1, τ2",
            FULL_MODEL_LABEL,
        ],
        help="Ka model treats K·a as one parameter (prof example). K model separates K using your step size a. Second-order model fits two real time constants. Full model fits one Kp/tau across full ON/OFF data."
    )

    full_mode = model == FULL_MODEL_LABEL

    fit_y0 = st.checkbox("Fit baseline y₀", value=True)

    st.divider()
    if not full_mode:
        st.subheader("Step timing")
        step_mode = st.radio(
            "Step time t₀",
            ["Assume t₀ = 0", "Enter t₀ manually", "Auto-detect t₀ from data"],
            index=0
        )
        t0_manual = st.number_input("t₀ (if manual)", value=0.0, step=0.1)
    else:
        step_mode = "Auto-detect t₀ from data"
        t0_manual = 0.0

        st.subheader("Full-model input")
        u_on = st.number_input("Heater ON level U_on (%)", value=100.0, step=1.0)
        base_window_sec = st.number_input("Baseline window before ON (s)", value=30.0, step=1.0, min_value=1.0)
        override_on_off = st.checkbox("Override detected ON/OFF times", value=False)
        on_manual = st.number_input("Manual ON time (s)", value=0.0, step=1.0)
        off_manual = st.number_input("Manual OFF time (s)", value=60.0, step=1.0)

    st.divider()
    st.subheader("Noise handling")
    smooth_window = st.slider(
        "Smoothing window (display / detection)",
        min_value=1, max_value=21, value=5, step=2,
        help="Higher smooths choppy data more. Full-model step detection uses rolling median with this window."
    )
    use_smoothed_for_fit = st.checkbox(
        "Use smoothed y for fitting (optional)",
        value=False,
        help="Usually leave OFF for honest fitting. Turn ON if data is extremely choppy."
    )

    st.divider()
    st.subheader("File input")
    sheet = st.text_input("Sheet name (blank = first sheet)", value="")
    header = st.checkbox("First row is header", value=True)

    if model == "Fit K & τ (you enter step size a)":
        a = st.number_input("Step size a", value=1.0, step=0.1)
    else:
        a = None

    st.divider()
    fit_btn = st.button("🚀 Fit model", type="primary", use_container_width=True)


# Main: upload + preview + results
uploaded = st.file_uploader("Upload data (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

# Read file
try:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, header=0 if header else None)
    else:
        df = pd.read_excel(
            uploaded,
            sheet_name=sheet.strip() if sheet.strip() else 0,
            header=0 if header else None,
        )
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df.shape[1] < 2:
    st.error("Need at least 2 columns: time and measured output.")
    st.stop()

# Column selectors
col1, col2, col3 = st.columns([1, 1, 2], gap="large")
with col1:
    t_col = st.selectbox("Time column", options=list(df.columns), index=0)
with col2:
    y_col = st.selectbox("Output column", options=list(df.columns), index=1)
with col3:
    st.write("Preview (first 10 rows):")
    st.dataframe(df[[t_col, y_col]].head(10), use_container_width=True)

t_raw = df[t_col].to_numpy()
y_raw = df[y_col].to_numpy()
t_raw, y_raw = clean_sort_xy(t_raw, y_raw)

y_smooth = moving_average(y_raw, smooth_window)

# Determine t0
if step_mode == "Assume t₀ = 0":
    t0 = 0.0
elif step_mode == "Enter t₀ manually":
    t0 = float(t0_manual)
else:
    t0 = estimate_step_time(t_raw, y_raw, window=smooth_window)

# Choose data for fit (raw vs smoothed)
y_fit_data = y_smooth if use_smoothed_for_fit else y_raw

tabs = st.tabs(["Plot", "Results", "Download / Export"])

# ----------------- Plot tab -----------------
with tabs[0]:
    st.subheader("Data + Fit")

    fig, ax = plt.subplots()
    ax.plot(t_raw, y_raw, marker="o", linestyle="None", label="Raw data")
    ax.plot(t_raw, y_smooth, linestyle="-", label=f"Smoothed (window={smooth_window})")
    ax.axvline(t0, linestyle="--", label=f"t₀ = {t0:g}")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.grid(True)

    # Fit when button is clicked
    result = None
    if fit_btn:
        try:
            if model == "Fit Ka & τ (board style)":
                # Ka fitter expects t0 and fit_y0 per the updated fit_model.py
                result = fit_Ka_tau(t_raw, y_fit_data, t0=float(t0), fit_y0=bool(fit_y0))
                ax.plot(result["t"], result["y_fit"], linewidth=2, label="Fit (Ka, τ)")
            elif model == "Fit 2nd-order Ka, τ1, τ2":
                result = fit_Ka_tau2(t_raw, y_fit_data, t0=float(t0), fit_y0=bool(fit_y0))
                ax.plot(result["t"], result["y_fit"], linewidth=2, label="Fit (Ka, τ1, τ2)")
            else:
                if fit_K_tau_with_a is None:
                    st.error("K-with-a model not available. Create fit_model_with_a.py and restart Streamlit.")
                else:
                    result = fit_K_tau_with_a(t_raw - t0, y_fit_data, a=float(a), fit_y0=bool(fit_y0))
                    # For plotting, reconstruct prediction on original time axis by shifting t to start at t0
                    yhat = result["y_fit"]
                    ax.plot(t_raw, yhat, linewidth=2, label="Fit (K, τ with a)")

            ax.legend()
            st.pyplot(fig, clear_figure=True)

        except Exception as e:
            ax.legend()
            st.pyplot(fig, clear_figure=True)
            st.error(f"Fit failed: {e}")
    else:
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    st.caption(
        "Tip: If your data is super choppy, increase smoothing window and (optionally) enable smoothed fitting."
    )

# Persist result across tabs using session_state
if fit_btn:
    st.session_state["last_result"] = None
    # Re-run same logic (so result exists for other tabs)
    try:
        if model == "Fit Ka & τ (board style)":
            st.session_state["last_result"] = fit_Ka_tau(t_raw, y_fit_data, t0=float(t0), fit_y0=bool(fit_y0))
            st.session_state["last_model"] = "Ka"
        elif model == "Fit 2nd-order Ka, τ1, τ2":
            st.session_state["last_result"] = fit_Ka_tau2(t_raw, y_fit_data, t0=float(t0), fit_y0=bool(fit_y0))
            st.session_state["last_model"] = "Ka2"
        else:
            if fit_K_tau_with_a is not None:
                st.session_state["last_result"] = fit_K_tau_with_a(t_raw - t0, y_fit_data, a=float(a), fit_y0=bool(fit_y0))
                st.session_state["last_model"] = "K"
    except Exception:
        pass

result = st.session_state.get("last_result", None)
mode_key = st.session_state.get("last_model", None)
stale_result_msg = None

# Guard against stale session-state fits from a previous dataset/selection.
if result is not None:
    y_fit_cached = np.asarray(result.get("y_fit", []))
    if y_fit_cached.shape[0] != t_raw.shape[0]:
        st.session_state["last_result"] = None
        st.session_state["last_model"] = None
        result = None
        mode_key = None
        stale_result_msg = "Cached fit does not match current data length. Click **Fit model** again."

# ----------------- Results tab -----------------
with tabs[1]:
    st.subheader("Fit Results")

    if stale_result_msg:
        st.warning(stale_result_msg)

    if result is None:
        st.info("Click **Fit model** in the sidebar to compute parameters.")
    else:
        if mode_key == "Ka":
            st.write(
                f"**Ka** = {result['Ka']:.6g}   |   **τ** = {result['tau']:.6g}   |   "
                f"**y₀** = {result['y0']:.6g}"
            )
            st.write(f"**SSE** = {result['SSE']:.6g}   |   **R²** = {result['R2']:.6g}")
        elif mode_key == "Ka2":
            st.write(
                f"**Ka** = {result['Ka']:.6g}   |   **τ1** = {result['tau1']:.6g}   |   "
                f"**τ2** = {result['tau2']:.6g}   |   **y₀** = {result['y0']:.6g}"
            )
            st.write(f"**SSE** = {result['SSE']:.6g}   |   **R²** = {result['R2']:.6g}")
        else:
            st.write(
                f"**K** = {result['K']:.6g}   |   **τ** = {result['tau']:.6g}   |   "
                f"**y₀** = {result['y0']:.6g}   |   **a** = {a:g}"
            )
            st.write(f"**SSE** = {result['SSE']:.6g}   |   **R²** = {result['R2']:.6g}")

        st.caption(f"Using t₀ = {t0:g}  |  Fit used {'smoothed' if use_smoothed_for_fit else 'raw'} y")

# ----------------- Download tab -----------------
with tabs[2]:
    st.subheader("Download / Export")

    if stale_result_msg:
        st.warning(stale_result_msg)

    if result is None:
        st.info("Fit first, then you can download the fitted Excel.")
    else:
        # Build output dataframe for export
        out = pd.DataFrame({
            "t": t_raw,
            "y_raw": y_raw,
            "y_smooth": y_smooth,
        })

        if mode_key == "Ka":
            out["y_fit"] = result["y_fit"]
            out["residual"] = result["residuals"]
            summary = pd.DataFrame({
                "parameter": ["model", "t0", "Ka", "tau", "y0", "SSE", "R2", "smoothed_fit"],
                "value": ["Ka_tau", t0, result["Ka"], result["tau"], result["y0"], result["SSE"], result["R2"], use_smoothed_for_fit],
            })
        elif mode_key == "Ka2":
            out["y_fit"] = result["y_fit"]
            out["residual"] = result["residuals"]
            summary = pd.DataFrame({
                "parameter": ["model", "t0", "Ka", "tau1", "tau2", "y0", "SSE", "R2", "smoothed_fit"],
                "value": ["Ka_tau1_tau2", t0, result["Ka"], result["tau1"], result["tau2"], result["y0"], result["SSE"], result["R2"], use_smoothed_for_fit],
            })
        else:
            # result["y_fit"] corresponds to (t - t0) fit, but we used same t grid so it matches
            out["y_fit"] = result["y_fit"]
            out["residual"] = result["residuals"]
            summary = pd.DataFrame({
                "parameter": ["model", "t0", "a", "K", "tau", "y0", "SSE", "R2", "smoothed_fit"],
                "value": ["K_tau_with_a", t0, a, result["K"], result["tau"], result["y0"], result["SSE"], result["R2"], use_smoothed_for_fit],
            })

        xlsx_bytes = make_excel_bytes(out, summary)

        st.download_button(
            "⬇️ Download fitted Excel",
            data=xlsx_bytes,
            file_name="process_fit_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.write("**Excel instructions (Solver version):**")
        st.markdown(
            """
- Put your parameters in cells (example): `Ka` (or `K`), `tau`, `y0`, and (if using K-model) `a`.
- Make a `y_fit` column using the formula:
  - Ka model: `y0 + Ka*(1-EXP(-t/tau))`
  - K model:  `y0 + K*a*(1-EXP(-t/tau))`
- 2nd-order model: `y0 + Ka*(1-(tau1*EXP(-t/tau1)-tau2*EXP(-t/tau2))/(tau1-tau2))`
- Make a residual column: `y - y_fit`
- SSE cell: `=SUMSQ(residual_range)`
- Use **Data → Solver**: minimize SSE by changing fit parameters (`Ka/K`, `tau` or `tau1/tau2`, and `y0` if fitting it). Constrain all `tau` values > 0.
            """
        )
