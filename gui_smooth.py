import io
import datetime as dt
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

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
        x[i] = x[i - 1] + dt_i * ((-x[i - 1] + float(Kp) * float(u[i - 1])) / tau)

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
    dt_list = []
    for rr in run_rows:
        dt_rr = np.diff(rr["t"])
        dt_rr = dt_rr[np.isfinite(dt_rr) & (dt_rr > 0)]
        if dt_rr.size:
            dt_list.append(np.median(dt_rr))
    tau_min = max(1e-6, 0.05 * float(np.median(dt_list))) if dt_list else 1e-6

    def sse_obj(p):
        kp, tau = float(p[0]), float(p[1])
        if tau <= tau_min or not np.isfinite(kp) or not np.isfinite(tau):
            return 1e30
        sse = 0.0
        for rr in run_rows:
            pred = simulate_first_order_deviation(rr["t"], rr["u"], kp, tau, rr["T_dev"][0])
            if not np.all(np.isfinite(pred)):
                return 1e30
            err = rr["T_dev"] - pred
            sse_i = float(np.sum(err ** 2))
            if not np.isfinite(sse_i):
                return 1e30
            sse += sse_i
        return sse

    def resid_vec(p):
        kp, tau = float(p[0]), float(p[1])
        if tau <= tau_min or not np.isfinite(kp) or not np.isfinite(tau):
            return np.full(sum(len(rr["t"]) for rr in run_rows), 1e12, dtype=float)
        out = []
        for rr in run_rows:
            pred = simulate_first_order_deviation(rr["t"], rr["u"], kp, tau, rr["T_dev"][0])
            if not np.all(np.isfinite(pred)):
                return np.full(sum(len(r["t"]) for r in run_rows), 1e12, dtype=float)
            out.append(rr["T_dev"] - pred)
        return np.concatenate(out)

    opt = minimize(
        sse_obj,
        x0=np.array([kp0, tau0], dtype=float),
        method="L-BFGS-B",
        bounds=[(None, None), (tau_min, 1e8)],
    )
    if opt.success and np.all(np.isfinite(opt.x)):
        kp_hat, tau_hat = map(float, opt.x)
    else:
        lsq = least_squares(
            resid_vec,
            x0=np.array([kp0, tau0], dtype=float),
            bounds=([-np.inf, tau_min], [np.inf, 1e8]),
            max_nfev=20000,
        )
        if not lsq.success or not np.all(np.isfinite(lsq.x)):
            raise ValueError(f"Optimization failed: {opt.message}")
        kp_hat, tau_hat = map(float, lsq.x)

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


def first_order_segment_response(t, K, tau, y0, t0):
    t = np.asarray(t, dtype=float)
    ts = np.maximum(t - float(t0), 0.0)
    return float(y0) + float(K) * (1.0 - np.exp(-ts / max(float(tau), 1e-9)))


def fit_first_order_segment(t, y, t0):
    t, y = clean_sort_xy(t, y)
    if t.size < 5:
        raise ValueError("Need at least 5 points for first-order segment fit.")

    y0_0 = float(y[0])
    y_inf = float(np.mean(y[-max(3, int(0.2 * len(y))):]))
    K0 = y_inf - y0_0
    tau0 = max((t[-1] - t[0]) / 3.0, 1e-6)

    popt, _ = curve_fit(
        lambda tt, K, tau, y0: first_order_segment_response(tt, K, tau, y0, t0=t0),
        t,
        y,
        p0=[K0, tau0, y0_0],
        bounds=([-np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf]),
        maxfev=40000,
    )
    K, tau, y0 = map(float, popt)
    yhat = first_order_segment_response(t, K, tau, y0, t0=t0)
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"t": t, "y": y, "y_fit": yhat, "residuals": resid, "K": K, "tau": tau, "y0": y0, "SSE": sse, "R2": r2}


def second_order_segment_response(t, K, tau1, tau2, y0, t0):
    t = np.asarray(t, dtype=float)
    ts = np.maximum(t - float(t0), 0.0)
    tau1 = float(max(tau1, 1e-12))
    tau2 = float(max(tau2, 1e-12))

    if abs(tau1 - tau2) <= 1e-8 * max(tau1, tau2):
        tau = 0.5 * (tau1 + tau2)
        shape = 1.0 - np.exp(-ts / tau) * (1.0 + ts / tau)
    else:
        shape = 1.0 - (tau1 * np.exp(-ts / tau1) - tau2 * np.exp(-ts / tau2)) / (tau1 - tau2)

    return float(y0) + float(K) * shape


def fit_second_order_segment(t, y, t0):
    t, y = clean_sort_xy(t, y)
    if t.size < 7:
        raise ValueError("Need at least 7 points for second-order segment fit.")

    y0_0 = float(y[0])
    y_inf = float(np.mean(y[-max(3, int(0.2 * len(y))):]))
    K0 = y_inf - y0_0
    tau0 = max((t[-1] - t[0]) / 3.0, 1e-6)

    popt, _ = curve_fit(
        lambda tt, K, tau1, tau2, y0: second_order_segment_response(tt, K, tau1, tau2, y0, t0=t0),
        t,
        y,
        p0=[K0, 1.5 * tau0, 0.5 * tau0, y0_0],
        bounds=([-np.inf, 1e-9, 1e-9, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=60000,
    )
    K, tau1, tau2, y0 = map(float, popt)
    tau_big, tau_small = sorted([max(tau1, 1e-9), max(tau2, 1e-9)], reverse=True)
    yhat = second_order_segment_response(t, K, tau_big, tau_small, y0, t0=t0)
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {
        "t": t,
        "y": y,
        "y_fit": yhat,
        "residuals": resid,
        "K": K,
        "tau1": tau_big,
        "tau2": tau_small,
        "y0": y0,
        "SSE": sse,
        "R2": r2,
    }


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Process Fit Tool", layout="wide")
FULL_MODEL_LABEL = "Full ON/OFF model (CHEG 330 Part D)"

st.title("Process Fit (Smooth GUI)")
st.caption("Upload CSV/Excel → pick model → fit → visualize → download results")

# Sidebar controls (smooth UX)
with st.sidebar:
    st.header("Controls")

    cheg_auto_enabled = st.toggle("Enable CHEG 330 Auto Analysis", value=False)

    if st.button("About"):
        st.info("Created by Shane Coudriet\nChemical Engineer 2026\nIn memorium to my GPA")

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

    if cheg_auto_enabled:
        st.divider()
        st.subheader("CHEG 330 Auto")
        auto_ignore_edge_pct = st.slider("Ignore edges for step detection (%)", min_value=5, max_value=10, value=8, step=1)
        auto_override_on_off = st.checkbox("Override detected ON/OFF times manually", value=False)
        auto_on_manual = st.number_input("Manual ON time (s)", value=0.0, step=1.0, key="auto_on_manual")
        auto_off_manual = st.number_input("Manual OFF time (s)", value=60.0, step=1.0, key="auto_off_manual")
        auto_base_window_sec = st.number_input("Baseline window before ON (s)", value=30.0, step=1.0, min_value=1.0, key="auto_base_window_sec")
        auto_default_u_on = st.number_input("Default heater ON percent (%)", value=100.0, step=1.0, key="auto_default_u_on")
        show_first_order = st.checkbox("Show first-order overlays", value=True)
        show_second_order = st.checkbox("Show second-order overlays", value=True)
        show_grouped_onoff = st.checkbox("Show grouped ON/OFF overlay", value=True)
    else:
        auto_ignore_edge_pct = 8
        auto_override_on_off = False
        auto_on_manual = 0.0
        auto_off_manual = 60.0
        auto_base_window_sec = 30.0
        auto_default_u_on = 100.0
        show_first_order = True
        show_second_order = True
        show_grouped_onoff = True

    if not full_mode:
        fit_y0 = st.checkbox("Fit baseline y₀", value=True)
    else:
        fit_y0 = True

    st.divider()
    if not full_mode:
        st.subheader("Step timing")
        step_mode = st.radio(
            "Step time t₀",
            ["Assume t₀ = 0", "Enter t₀ manually", "Auto-detect t₀ from data"],
            index=0
        )
        t0_manual = st.number_input("t₀ (if manual)", value=0.0, step=0.1)
        std_auto_split = st.checkbox("Auto split heating/cooling (apply to this mode)", value=True)
        std_segment_choice = st.radio("Analyze segment", ["Heating", "Cooling"], index=0)
        std_override_on_off = st.checkbox("Override ON/OFF manually", value=False)
        std_on_manual = st.number_input("ON time for split (s)", value=0.0, step=1.0)
        std_off_manual = st.number_input("OFF time for split (s)", value=60.0, step=1.0)
    else:
        step_mode = "Auto-detect t₀ from data"
        t0_manual = 0.0
        std_auto_split = True
        std_segment_choice = "Heating"
        std_override_on_off = False
        std_on_manual = 0.0
        std_off_manual = 60.0

        st.subheader("Full-model input")
        u_on = st.number_input("Heater percent while ON", value=100.0, step=1.0)
        base_window_sec = st.number_input("Baseline window before ON (s)", value=30.0, step=1.0, min_value=1.0)
        auto_detect_on_off = st.checkbox("Auto-detect ON/OFF from data derivative", value=True)
        on_manual = st.number_input("Heater ON time (s)", value=0.0, step=1.0)
        off_manual = st.number_input("Heater OFF time (s)", value=60.0, step=1.0)

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
if cheg_auto_enabled:
    st.subheader("CHEG 330 Auto Analysis")
    auto_files = st.file_uploader(
        "Upload one or more CSV/XLSX files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="cheg_auto_files",
    )

    if not auto_files:
        st.info("Enable CHEG 330 Auto Analysis is ON. Upload at least one file to continue.")
        st.stop()

    def _read_uploaded_file(up):
        up.seek(0)
        n = up.name.lower()
        if n.endswith(".csv"):
            return pd.read_csv(up, header=0 if header else None)
        return pd.read_excel(up, sheet_name=sheet.strip() if sheet.strip() else 0, header=0 if header else None)

    try:
        df0 = _read_uploaded_file(auto_files[0])
    except Exception as e:
        st.error(f"Could not read first file: {e}")
        st.stop()

    if df0.shape[1] < 2:
        st.error("Need at least 2 columns in each file: time and tank temperature.")
        st.stop()

    a1, a2, a3 = st.columns([1, 1, 2], gap="large")
    with a1:
        auto_t_col = st.selectbox("Time column", options=list(df0.columns), index=0, key="auto_t_col")
    with a2:
        auto_y_col = st.selectbox("Tank temperature column", options=list(df0.columns), index=1, key="auto_y_col")
    with a3:
        st.write("Preview (first file, first 10 rows):")
        st.dataframe(df0[[auto_t_col, auto_y_col]].head(10), use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        auto_cool_in_col = st.selectbox("Coolant-in column (optional)", options=["(none)"] + list(df0.columns), index=0, key="auto_cool_in_col")
    with b2:
        auto_cool_out_col = st.selectbox("Coolant-out column (optional)", options=["(none)"] + list(df0.columns), index=0, key="auto_cool_out_col")

    hp_default = pd.DataFrame({"file": [f.name for f in auto_files], "heater_percent": [float(auto_default_u_on)] * len(auto_files)})
    hp_editor = st.data_editor(
        hp_default,
        hide_index=True,
        use_container_width=True,
        disabled=["file"],
        key="auto_heater_pct_table",
    )
    heater_map = {str(r["file"]): float(r["heater_percent"]) for _, r in hp_editor.iterrows()}

    run_auto_btn = st.button("Run CHEG 330 Auto Analysis", type="primary", use_container_width=True)
    state_key = "cheg_auto_analysis_result"

    if run_auto_btn:
        analyses = []
        run_rows = []
        warnings = []

        for up in auto_files:
            try:
                dfi = _read_uploaded_file(up)
            except Exception as e:
                warnings.append(f"{up.name}: read failed ({e})")
                continue

            if auto_t_col not in dfi.columns or auto_y_col not in dfi.columns:
                warnings.append(f"{up.name}: missing selected time/tank columns.")
                continue

            try:
                t_all = parse_time_to_elapsed_seconds(dfi[auto_t_col])
            except Exception as e:
                warnings.append(f"{up.name}: time parse failed ({e})")
                continue

            T_all = pd.to_numeric(dfi[auto_y_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(t_all) & np.isfinite(T_all)
            t = t_all[m]
            T = T_all[m]
            if t.size < 10:
                warnings.append(f"{up.name}: not enough valid points.")
                continue

            idx = np.argsort(t)
            t = t[idx]
            T = T[idx]

            if auto_override_on_off:
                t_on = float(auto_on_manual)
                t_off = float(auto_off_manual)
                T_med = rolling_median(T, smooth_window)
            else:
                t_on, t_off, T_med, _ = detect_on_off_times(
                    t,
                    T,
                    smooth_window=smooth_window,
                    edge_frac=float(auto_ignore_edge_pct) / 100.0,
                )

            if not np.isfinite(t_on) or not np.isfinite(t_off) or t_off <= t_on:
                warnings.append(f"{up.name}: invalid detected ON/OFF times.")
                continue

            heat_mask = (t >= t_on) & (t < t_off)
            cool_mask = t >= t_off
            if np.sum(heat_mask) < 5 or np.sum(cool_mask) < 5:
                warnings.append(f"{up.name}: insufficient heating/cooling points after segmentation.")
                continue

            try:
                fo_heat = fit_first_order_segment(t[heat_mask], T[heat_mask], t0=t_on)
            except Exception as e:
                fo_heat = {"K": np.nan, "tau": np.nan, "R2": np.nan, "SSE": np.nan, "y_fit": np.full(np.sum(heat_mask), np.nan), "error": str(e)}
            try:
                fo_cool = fit_first_order_segment(t[cool_mask], T[cool_mask], t0=t_off)
            except Exception as e:
                fo_cool = {"K": np.nan, "tau": np.nan, "R2": np.nan, "SSE": np.nan, "y_fit": np.full(np.sum(cool_mask), np.nan), "error": str(e)}

            try:
                so_heat = fit_second_order_segment(t[heat_mask], T[heat_mask], t0=t_on)
            except Exception as e:
                so_heat = {"K": np.nan, "tau1": np.nan, "tau2": np.nan, "R2": np.nan, "SSE": np.nan, "y_fit": np.full(np.sum(heat_mask), np.nan), "error": str(e)}
            try:
                so_cool = fit_second_order_segment(t[cool_mask], T[cool_mask], t0=t_off)
            except Exception as e:
                so_cool = {"K": np.nan, "tau1": np.nan, "tau2": np.nan, "R2": np.nan, "SSE": np.nan, "y_fit": np.full(np.sum(cool_mask), np.nan), "error": str(e)}

            base_mask = (t >= (t_on - float(auto_base_window_sec))) & (t < t_on)
            if np.sum(base_mask) < 3:
                base_mask = t < t_on
            if np.sum(base_mask) < 3:
                n0 = max(3, int(0.1 * len(T)))
                base_mask = np.zeros(len(T), dtype=bool)
                base_mask[:n0] = True

            T_base = float(np.mean(T[base_mask]))
            T_dev = T - T_base
            u_on_i = float(heater_map.get(up.name, auto_default_u_on))
            u = build_input_profile(t, t_on=t_on, t_off=t_off, u_on=u_on_i)

            cool_in = None
            if auto_cool_in_col != "(none)" and auto_cool_in_col in dfi.columns:
                cin = pd.to_numeric(dfi[auto_cool_in_col], errors="coerce").to_numpy(dtype=float)
                cool_in = cin[m][idx]
            cool_out = None
            if auto_cool_out_col != "(none)" and auto_cool_out_col in dfi.columns:
                cout = pd.to_numeric(dfi[auto_cool_out_col], errors="coerce").to_numpy(dtype=float)
                cool_out = cout[m][idx]

            analyses.append({
                "file": up.name,
                "t": t,
                "T": T,
                "T_med": T_med,
                "t_on": t_on,
                "t_off": t_off,
                "heat_mask": heat_mask,
                "cool_mask": cool_mask,
                "fo_heat": fo_heat,
                "fo_cool": fo_cool,
                "so_heat": so_heat,
                "so_cool": so_cool,
                "u": u,
                "u_on": u_on_i,
                "T_base": T_base,
                "cool_in": cool_in,
                "cool_out": cool_out,
            })
            run_rows.append({
                "run": up.name,
                "t": t,
                "T": T,
                "T_dev": T_dev,
                "u": u,
                "u_on": u_on_i,
                "t_on": t_on,
                "t_off": t_off,
                "T_base": T_base,
            })

        if not analyses:
            st.error("No files were successfully analyzed.")
            if warnings:
                st.warning("\n".join(warnings))
            st.stop()

        grouped = fit_full_model_global(run_rows)
        per_run_map = {pr["run"]: pr for pr in grouped["per_run"]}

        summary_rows = []
        for a in analyses:
            pr = per_run_map.get(a["file"], None)
            if pr is not None:
                a["grouped_pred_T"] = pr["pred_T"]
                a["grouped_R2"] = pr["R2"]
            else:
                a["grouped_pred_T"] = np.full_like(a["T"], np.nan, dtype=float)
                a["grouped_R2"] = float("nan")

            summary_rows.append({
                "file": a["file"],
                "t_on_s": a["t_on"],
                "t_off_s": a["t_off"],
                "first_heat_K": a["fo_heat"]["K"],
                "first_heat_tau": a["fo_heat"]["tau"],
                "first_heat_R2": a["fo_heat"]["R2"],
                "first_heat_SSE": a["fo_heat"]["SSE"],
                "first_cool_K": a["fo_cool"]["K"],
                "first_cool_tau": a["fo_cool"]["tau"],
                "first_cool_R2": a["fo_cool"]["R2"],
                "first_cool_SSE": a["fo_cool"]["SSE"],
                "second_heat_K": a["so_heat"]["K"],
                "second_heat_tau1": a["so_heat"]["tau1"],
                "second_heat_tau2": a["so_heat"]["tau2"],
                "second_heat_R2": a["so_heat"]["R2"],
                "second_heat_SSE": a["so_heat"]["SSE"],
                "second_cool_K": a["so_cool"]["K"],
                "second_cool_tau1": a["so_cool"]["tau1"],
                "second_cool_tau2": a["so_cool"]["tau2"],
                "second_cool_R2": a["so_cool"]["R2"],
                "second_cool_SSE": a["so_cool"]["SSE"],
                "grouped_u_on": a["u_on"],
                "grouped_R2": a["grouped_R2"],
            })

        summary_df = pd.DataFrame(summary_rows)

        avg_dR2_heat = np.nanmean(summary_df["second_heat_R2"] - summary_df["first_heat_R2"]) if len(summary_df) else float("nan")
        avg_dR2_cool = np.nanmean(summary_df["second_cool_R2"] - summary_df["first_cool_R2"]) if len(summary_df) else float("nan")
        better_text = "Second-order is only marginally better than first-order." if (np.nan_to_num(avg_dR2_heat) < 0.02 and np.nan_to_num(avg_dR2_cool) < 0.02) else "Second-order gives a noticeable improvement over first-order."

        part_b_lines = []
        for a in analyses:
            pre = np.mean(a["T"][a["t"] < a["t_on"]]) if np.any(a["t"] < a["t_on"]) else a["T"][0]
            peak = np.max(a["T"])
            endv = a["T"][-1]
            line = f"{a['file']}: tank temperature rises from about {pre:.2f} to {peak:.2f} during heating, then trends toward {endv:.2f} after heater OFF."
            if a["cool_in"] is not None and np.isfinite(a["cool_in"]).sum() > 3:
                line += f" Coolant-in stays near {np.nanmean(a['cool_in']):.2f} on average."
            if a["cool_out"] is not None and np.isfinite(a["cool_out"]).sum() > 3:
                line += f" Coolant-out averages {np.nanmean(a['cool_out']):.2f} and follows the tank trend."
            part_b_lines.append(line)

        hw_text = (
            "Part (b):\n"
            + "\n".join(part_b_lines)
            + "\n\nPart (c):\n"
            + f"Average R² improvement (2nd - 1st) is {avg_dR2_heat:.4f} for heating and {avg_dR2_cool:.4f} for cooling. "
            + better_text + " "
            + "In the second-order model, tau1 is the slower dominant time constant and tau2 is the faster lag. "
            + "When tau2 is much smaller than tau1, the process behaves effectively like first-order."
            + "\n\nPart (d):\n"
            + f"Grouped full ON/OFF fit gives Kp = {grouped['Kp']:.6g} degC/% and tau = {grouped['tau']:.6g} s, with overall R² = {grouped['R2']:.6g}. "
            + "This single model captures both heat-up and cool-down because the tank dynamics are unchanged while only the input level changes from ON to OFF. "
            + "Compared with separate segment fits, grouped fitting trades some local accuracy for one consistent parameter set across all runs."
        )

        st.session_state[state_key] = {
            "analyses": analyses,
            "summary_df": summary_df,
            "grouped": grouped,
            "warnings": warnings,
            "homework_text": hw_text,
        }

    auto_result = st.session_state.get(state_key, None)
    if auto_result is None:
        st.info("Click **Run CHEG 330 Auto Analysis** to compute parts (b), (c), and (d).")
        st.stop()

    if auto_result.get("warnings"):
        st.warning("\n".join(auto_result["warnings"]))

    tabs = st.tabs(["Plots", "Summary", "Homework Text", "Export"])

    with tabs[0]:
        st.subheader("Per-file Plots")
        for i, a in enumerate(auto_result["analyses"]):
            st.markdown(f"**{a['file']}**")
            fig, ax1 = plt.subplots()
            t = a["t"]
            T = a["T"]
            ax1.plot(t, T, label="Measured tank T", linewidth=1.7)
            ax1.axvspan(a["t_on"], a["t_off"], alpha=0.12, color="tab:green", label="Heating segment")
            ax1.axvspan(a["t_off"], t[-1], alpha=0.10, color="tab:blue", label="Cooling segment")

            if show_first_order:
                ax1.plot(t[a["heat_mask"]], a["fo_heat"]["y_fit"], linestyle="--", label="1st-order heating fit")
                ax1.plot(t[a["cool_mask"]], a["fo_cool"]["y_fit"], linestyle="--", label="1st-order cooling fit")
            if show_second_order:
                ax1.plot(t[a["heat_mask"]], a["so_heat"]["y_fit"], linestyle=":", linewidth=2, label="2nd-order heating fit")
                ax1.plot(t[a["cool_mask"]], a["so_cool"]["y_fit"], linestyle=":", linewidth=2, label="2nd-order cooling fit")
            if show_grouped_onoff:
                ax1.plot(t, a["grouped_pred_T"], linewidth=2.1, label="Grouped ON/OFF fit")

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Tank temperature")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(t, a["u"], color="tab:orange", alpha=0.55, label="Heater %")
            ax2.set_ylabel("Heater input (%)")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="best")
            png_bytes = fig_to_png_bytes(fig)
            st.pyplot(fig, clear_figure=True)

            st.download_button(
                f"Download {a['file']} plot (PNG)",
                data=png_bytes,
                file_name=f"{a['file']}_cheg330_plot.png",
                mime="image/png",
                key=f"png_{i}_{a['file']}",
            )

    with tabs[1]:
        st.subheader("Summary Table")
        st.dataframe(auto_result["summary_df"], use_container_width=True)
        g = auto_result["grouped"]
        st.write(f"Grouped Part D fit: **Kp = {g['Kp']:.6g} degC/%**, **tau = {g['tau']:.6g} s**, **overall R² = {g['R2']:.6g}**, **SSE = {g['SSE']:.6g}**")

    with tabs[2]:
        st.subheader("Homework Text")
        st.text_area("Homework-ready answers (Parts b, c, d)", value=auto_result["homework_text"], height=300)

    with tabs[3]:
        st.subheader("Export")
        summary_csv = auto_result["summary_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download summary CSV",
            data=summary_csv,
            file_name="cheg330_auto_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download homework TXT",
            data=auto_result["homework_text"],
            file_name="cheg330_homework_answers.txt",
            mime="text/plain",
            use_container_width=True,
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for a in auto_result["analyses"]:
                fig, ax1 = plt.subplots()
                t = a["t"]
                T = a["T"]
                ax1.plot(t, T, label="Measured tank T", linewidth=1.7)
                ax1.axvspan(a["t_on"], a["t_off"], alpha=0.12, color="tab:green")
                ax1.axvspan(a["t_off"], t[-1], alpha=0.10, color="tab:blue")
                if show_grouped_onoff:
                    ax1.plot(t, a["grouped_pred_T"], linewidth=2.1, label="Grouped ON/OFF fit")
                ax2 = ax1.twinx()
                ax2.plot(t, a["u"], color="tab:orange", alpha=0.55)
                png = fig_to_png_bytes(fig)
                zf.writestr(f"{a['file']}_cheg330_plot.png", png)
                plt.close(fig)
        zip_buf.seek(0)
        st.download_button(
            "⬇️ Download all PNG plots (ZIP)",
            data=zip_buf.getvalue(),
            file_name="cheg330_plots.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.stop()

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
    y_col = st.selectbox("Tank temperature column", options=list(df.columns), index=1)

extra_preview_cols = []
if full_mode:
    c4, c5 = st.columns(2)
    with c4:
        cool_in_col = st.selectbox("Coolant-in column (optional)", options=["(none)"] + list(df.columns), index=0)
    with c5:
        cool_out_col = st.selectbox("Coolant-out column (optional)", options=["(none)"] + list(df.columns), index=0)
    extra_preview_cols = [c for c in [cool_in_col, cool_out_col] if c != "(none)"]
else:
    cool_in_col = "(none)"
    cool_out_col = "(none)"

with col3:
    st.write("Preview (first 10 rows):")
    preview_cols = [t_col, y_col] + [c for c in extra_preview_cols if c not in [t_col, y_col]]
    st.dataframe(df[preview_cols].head(10), use_container_width=True)

if full_mode:
    try:
        t_all = parse_time_to_elapsed_seconds(df[t_col])
    except Exception as e:
        st.error(f"Time parsing failed: {e}")
        st.stop()

    T_all = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t_all) & np.isfinite(T_all)
    t_full = t_all[mask]
    T_full = T_all[mask]

    if t_full.size < 6:
        st.error("Need at least 6 valid time/temperature points for Full ON/OFF model.")
        st.stop()

    order = np.argsort(t_full)
    t_full = t_full[order]
    T_full = T_full[order]

    if auto_detect_on_off:
        t_on_det, t_off_det, T_med, dTdt = detect_on_off_times(t_full, T_full, smooth_window=smooth_window)
        t_on = float(t_on_det)
        t_off = float(t_off_det)
    else:
        t_on = float(on_manual)
        t_off = float(off_manual)
        T_med = rolling_median(T_full, smooth_window)
        dTdt = np.gradient(T_med, t_full)

    if t_off <= t_on:
        st.error("Heater OFF time must be greater than heater ON time.")
        st.stop()

    base_mask = (t_full >= (t_on - float(base_window_sec))) & (t_full < t_on)
    if np.sum(base_mask) < 3:
        base_mask = t_full < t_on
    if np.sum(base_mask) < 3:
        n0 = max(3, int(0.1 * len(T_full)))
        base_mask = np.zeros(len(T_full), dtype=bool)
        base_mask[:n0] = True

    T_base = float(np.mean(T_full[base_mask]))
    T_dev = T_full - T_base
    u_full = build_input_profile(t_full, t_on=t_on, t_off=t_off, u_on=u_on)

    run_rows = [{
        "run": "all",
        "t": t_full,
        "T": T_full,
        "T_dev": T_dev,
        "u": u_full,
        "u_on": float(u_on),
        "t_on": t_on,
        "t_off": t_off,
        "T_base": T_base,
    }]

    if fit_btn:
        try:
            fit_full = fit_full_model_global(run_rows)
            pred_T = fit_full["per_run"][0]["pred_T"]
            resid = fit_full["per_run"][0]["residual"]
            st.session_state["full_last_result"] = {
                "t": t_full,
                "T": T_full,
                "u": u_full,
                "T_base": T_base,
                "t_on": t_on,
                "t_off": t_off,
                "Kp": fit_full["Kp"],
                "tau": fit_full["tau"],
                "R2": fit_full["R2"],
                "SSE": fit_full["SSE"],
                "T_pred": pred_T,
                "residual": resid,
            }
        except Exception as e:
            st.error(f"Fit failed: {e}")

    full_result = st.session_state.get("full_last_result", None)
    if full_result is not None and len(full_result.get("t", [])) != len(t_full):
        st.session_state["full_last_result"] = None
        full_result = None

    tabs = st.tabs(["Plot", "Results", "Download / Export"])

    with tabs[0]:
        st.subheader("Measured vs Full ON/OFF Model")
        fig, ax1 = plt.subplots()
        ax1.plot(t_full, T_full, label="Measured tank T", linewidth=1.8)
        if full_result is not None:
            ax1.plot(t_full, full_result["T_pred"], label="Model-predicted T", linewidth=2.2)
        ax1.axvline(t_on, linestyle="--", label=f"ON @ {t_on:.2f}s")
        ax1.axvline(t_off, linestyle="--", label=f"OFF @ {t_off:.2f}s")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Tank temperature")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(t_full, u_full, color="tab:orange", alpha=0.6, label="Heater input u(t) [%]")
        ax2.set_ylabel("Heater input (%)")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
        st.pyplot(fig, clear_figure=True)

    with tabs[1]:
        st.subheader("Fit Results")
        if full_result is None:
            st.info("Click **Fit model** to estimate Kp and tau for the full ON/OFF dataset.")
        else:
            st.write(
                f"**Kp** = {full_result['Kp']:.6g} °C/%   |   **tau** = {full_result['tau']:.6g} s   |   "
                f"**R²** = {full_result['R2']:.6g}"
            )
            st.write(
                f"**Detected/used ON time** = {full_result['t_on']:.6g} s   |   "
                f"**Detected/used OFF time** = {full_result['t_off']:.6g} s"
            )
            st.caption(f"Baseline temperature T_base = {full_result['T_base']:.6g}")

    with tabs[2]:
        st.subheader("Download / Export")
        if full_result is None:
            st.info("Fit first, then download CSV/text outputs.")
        else:
            out = pd.DataFrame({
                "time_s": t_full,
                "tank_temp_meas": T_full,
                "heater_u_percent": u_full,
                "tank_temp_pred": full_result["T_pred"],
                "residual": full_result["residual"],
            })
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results CSV",
                data=csv_bytes,
                file_name="full_on_off_fit_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            model_text = (
                f"G(s) = {full_result['Kp']:.6g} / ({full_result['tau']:.6g} s + 1)\n"
                f"Equivalent form: G(s) = Kp / (tau s + 1), with Kp = {full_result['Kp']:.6g}, tau = {full_result['tau']:.6g} s.\n\n"
                "A single first-order model is reasonable because the same tank dynamics govern both heating and cooling. "
                "The heater ON/OFF profile is handled through the input u(t), while Kp and tau remain constant system properties. "
                "The full-dataset fit captures both transients with one parameter set and provides a consistent R² measure."
            )
            st.text_area("Model summary text", value=model_text, height=170)
            st.download_button(
                "⬇️ Download model text",
                data=model_text,
                file_name="full_on_off_model_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.stop()

# Robust time parsing for all standard modes.
try:
    t_raw = parse_time_to_elapsed_seconds(df[t_col])
except Exception:
    t_raw = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)

y_raw = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
m_std = np.isfinite(t_raw) & np.isfinite(y_raw)
t_raw = t_raw[m_std]
y_raw = y_raw[m_std]
t_raw, y_raw = clean_sort_xy(t_raw, y_raw)

y_smooth = moving_average(y_raw, smooth_window)

# Split detection (heating/cooling) for standard modes.
t_on_std, t_off_std = float(t_raw[0]), float(t_raw[-1])
if std_auto_split:
    if std_override_on_off:
        t_on_std = float(std_on_manual)
        t_off_std = float(std_off_manual)
    else:
        t_on_std, t_off_std, _, _ = detect_on_off_times(t_raw, y_raw, smooth_window=smooth_window)
    if t_off_std <= t_on_std:
        st.warning("Auto split produced invalid ON/OFF; using full data.")
        std_auto_split = False

fit_mask = np.ones_like(t_raw, dtype=bool)
if std_auto_split:
    if std_segment_choice == "Heating":
        fit_mask = (t_raw >= t_on_std) & (t_raw < t_off_std)
    else:
        fit_mask = t_raw >= t_off_std
    if np.sum(fit_mask) < 5:
        st.warning(f"{std_segment_choice} split has too few points; using full data.")
        fit_mask = np.ones_like(t_raw, dtype=bool)
        std_auto_split = False

# Determine t0
if step_mode == "Assume t₀ = 0":
    t0 = 0.0
elif step_mode == "Enter t₀ manually":
    t0 = float(t0_manual)
else:
    t0 = estimate_step_time(t_raw, y_raw, window=smooth_window)

# Choose data for fit (raw vs smoothed)
y_fit_data_full = y_smooth if use_smoothed_for_fit else y_raw
t_fit = t_raw[fit_mask]
y_fit_data = y_fit_data_full[fit_mask]
if std_auto_split:
    t0 = float(t_on_std if std_segment_choice == "Heating" else t_off_std)

tabs = st.tabs(["Plot", "Results", "Download / Export"])

# ----------------- Plot tab -----------------
with tabs[0]:
    st.subheader("Data + Fit")

    fig, ax = plt.subplots()
    ax.plot(t_raw, y_raw, marker="o", linestyle="None", label="Raw data")
    ax.plot(t_raw, y_smooth, linestyle="-", label=f"Smoothed (window={smooth_window})")
    ax.axvline(t0, linestyle="--", label=f"t₀ = {t0:g}")
    if std_auto_split:
        ax.axvspan(t_on_std, t_off_std, alpha=0.12, color="tab:green", label="Heating split")
        ax.axvspan(t_off_std, t_raw[-1], alpha=0.08, color="tab:blue", label="Cooling split")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.grid(True)

    # Fit when button is clicked
    result = None
    if fit_btn:
        try:
            if model == "Fit Ka & τ (board style)":
                # Ka fitter expects t0 and fit_y0 per the updated fit_model.py
                result = fit_Ka_tau(t_fit, y_fit_data, t0=float(t0), fit_y0=bool(fit_y0))
                ax.plot(result["t"], result["y_fit"], linewidth=2, label="Fit (Ka, τ)")
            elif model == "Fit 2nd-order Ka, τ1, τ2":
                result = fit_Ka_tau2(t_fit, y_fit_data, t0=float(t0), fit_y0=bool(fit_y0))
                ax.plot(result["t"], result["y_fit"], linewidth=2, label="Fit (Ka, τ1, τ2)")
            else:
                if fit_K_tau_with_a is None:
                    st.error("K-with-a model not available. Create fit_model_with_a.py and restart Streamlit.")
                else:
                    result = fit_K_tau_with_a(t_fit - t0, y_fit_data, a=float(a), fit_y0=bool(fit_y0))
                    # For plotting, reconstruct prediction on original time axis by shifting t to start at t0
                    yhat = result["y_fit"]
                    ax.plot(t_fit, yhat, linewidth=2, label="Fit (K, τ with a)")

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
            rr = fit_Ka_tau(t_fit, y_fit_data, t0=float(t0), fit_y0=bool(fit_y0))
            y_full = np.full_like(t_raw, np.nan, dtype=float)
            r_full = np.full_like(t_raw, np.nan, dtype=float)
            y_full[fit_mask] = rr["y_fit"]
            r_full[fit_mask] = rr["residuals"]
            rr["y_fit_full"] = y_full
            rr["residuals_full"] = r_full
            rr["n_raw"] = int(len(t_raw))
            rr["t_on"] = float(t_on_std)
            rr["t_off"] = float(t_off_std)
            rr["split_used"] = bool(std_auto_split)
            rr["segment"] = std_segment_choice if std_auto_split else "Full"
            st.session_state["last_result"] = rr
            st.session_state["last_model"] = "Ka"
        elif model == "Fit 2nd-order Ka, τ1, τ2":
            rr = fit_Ka_tau2(t_fit, y_fit_data, t0=float(t0), fit_y0=bool(fit_y0))
            y_full = np.full_like(t_raw, np.nan, dtype=float)
            r_full = np.full_like(t_raw, np.nan, dtype=float)
            y_full[fit_mask] = rr["y_fit"]
            r_full[fit_mask] = rr["residuals"]
            rr["y_fit_full"] = y_full
            rr["residuals_full"] = r_full
            rr["n_raw"] = int(len(t_raw))
            rr["t_on"] = float(t_on_std)
            rr["t_off"] = float(t_off_std)
            rr["split_used"] = bool(std_auto_split)
            rr["segment"] = std_segment_choice if std_auto_split else "Full"
            st.session_state["last_result"] = rr
            st.session_state["last_model"] = "Ka2"
        else:
            if fit_K_tau_with_a is not None:
                rr = fit_K_tau_with_a(t_fit - t0, y_fit_data, a=float(a), fit_y0=bool(fit_y0))
                y_full = np.full_like(t_raw, np.nan, dtype=float)
                r_full = np.full_like(t_raw, np.nan, dtype=float)
                y_full[fit_mask] = rr["y_fit"]
                r_full[fit_mask] = rr["residuals"]
                rr["y_fit_full"] = y_full
                rr["residuals_full"] = r_full
                rr["n_raw"] = int(len(t_raw))
                rr["t_on"] = float(t_on_std)
                rr["t_off"] = float(t_off_std)
                rr["split_used"] = bool(std_auto_split)
                rr["segment"] = std_segment_choice if std_auto_split else "Full"
                st.session_state["last_result"] = rr
                st.session_state["last_model"] = "K"
    except Exception:
        pass

result = st.session_state.get("last_result", None)
mode_key = st.session_state.get("last_model", None)
stale_result_msg = None

# Guard against stale session-state fits from a previous dataset/selection.
if result is not None:
    n_cached = int(result.get("n_raw", len(result.get("y_fit_full", []))))
    if n_cached != t_raw.shape[0]:
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

        seg = result.get("segment", "Full")
        split_txt = f"split {seg.lower()} segment" if result.get("split_used", False) else "full dataset"
        st.caption(
            f"Using t₀ = {t0:g}  |  Fit used {'smoothed' if use_smoothed_for_fit else 'raw'} y  |  "
            f"Data region: {split_txt}"
        )
        if result.get("split_used", False):
            st.caption(f"Detected/used ON = {result.get('t_on', float('nan')):.6g} s, OFF = {result.get('t_off', float('nan')):.6g} s")

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
            out["y_fit"] = result.get("y_fit_full", result["y_fit"])
            out["residual"] = result.get("residuals_full", result["residuals"])
            summary = pd.DataFrame({
                "parameter": ["model", "t0", "Ka", "tau", "y0", "SSE", "R2", "smoothed_fit", "split_used", "segment", "t_on", "t_off"],
                "value": ["Ka_tau", t0, result["Ka"], result["tau"], result["y0"], result["SSE"], result["R2"], use_smoothed_for_fit, result.get("split_used", False), result.get("segment", "Full"), result.get("t_on", np.nan), result.get("t_off", np.nan)],
            })
        elif mode_key == "Ka2":
            out["y_fit"] = result.get("y_fit_full", result["y_fit"])
            out["residual"] = result.get("residuals_full", result["residuals"])
            summary = pd.DataFrame({
                "parameter": ["model", "t0", "Ka", "tau1", "tau2", "y0", "SSE", "R2", "smoothed_fit", "split_used", "segment", "t_on", "t_off"],
                "value": ["Ka_tau1_tau2", t0, result["Ka"], result["tau1"], result["tau2"], result["y0"], result["SSE"], result["R2"], use_smoothed_for_fit, result.get("split_used", False), result.get("segment", "Full"), result.get("t_on", np.nan), result.get("t_off", np.nan)],
            })
        else:
            # result["y_fit"] corresponds to (t - t0) fit, but we used same t grid so it matches
            out["y_fit"] = result.get("y_fit_full", result["y_fit"])
            out["residual"] = result.get("residuals_full", result["residuals"])
            summary = pd.DataFrame({
                "parameter": ["model", "t0", "a", "K", "tau", "y0", "SSE", "R2", "smoothed_fit", "split_used", "segment", "t_on", "t_off"],
                "value": ["K_tau_with_a", t0, a, result["K"], result["tau"], result["y0"], result["SSE"], result["R2"], use_smoothed_for_fit, result.get("split_used", False), result.get("segment", "Full"), result.get("t_on", np.nan), result.get("t_off", np.nan)],
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
