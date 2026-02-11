import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Import whichever fitters you created
# 1) Ka model (board style)
from fit_model import fit_first_order as fit_Ka_tau  # returns keys: Ka, tau, y0, ...

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


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="First-Order Fit Tool", layout="wide")

st.title("First-Order Process Fit (Smooth GUI)")
st.caption("Upload Excel → pick model → fit → visualize → download results")

# Sidebar controls (smooth UX)
with st.sidebar:
    st.header("Controls")

    model = st.selectbox(
        "Model",
        [
            "Fit Ka & τ (board style)",
            "Fit K & τ (you enter step size a)"
        ],
        help="Ka model treats K·a as one parameter (prof example). K model separates K using your step size a."
    )

    fit_y0 = st.checkbox("Fit baseline y₀", value=True)

    st.divider()
    st.subheader("Step timing")
    step_mode = st.radio(
        "Step time t₀",
        ["Assume t₀ = 0", "Enter t₀ manually", "Auto-detect t₀ from data"],
        index=0
    )
    t0_manual = st.number_input("t₀ (if manual)", value=0.0, step=0.1)

    st.divider()
    st.subheader("Noise handling")
    smooth_window = st.slider(
        "Smoothing window (for display / optional guessing)",
        min_value=1, max_value=21, value=5, step=2,
        help="Higher smooths choppy data more. Does not change the raw-fit unless you enable it below."
    )
    use_smoothed_for_fit = st.checkbox(
        "Use smoothed y for fitting (optional)",
        value=False,
        help="Usually leave OFF for honest fitting. Turn ON if data is extremely choppy."
    )

    st.divider()
    st.subheader("Excel input")
    sheet = st.text_input("Sheet name (blank = first sheet)", value="")
    header = st.checkbox("First row is header", value=True)

    if model == "Fit K & τ (you enter step size a)":
        a = st.number_input("Step size a", value=1.0, step=0.1)
    else:
        a = None

    st.divider()
    fit_btn = st.button("🚀 Fit model", type="primary", use_container_width=True)


# Main: upload + preview + results
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

# Read Excel
try:
    df = pd.read_excel(
        uploaded,
        sheet_name=sheet.strip() if sheet.strip() else 0,
        header=0 if header else None
    )
except Exception as e:
    st.error(f"Could not read Excel: {e}")
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
        else:
            if fit_K_tau_with_a is not None:
                st.session_state["last_result"] = fit_K_tau_with_a(t_raw - t0, y_fit_data, a=float(a), fit_y0=bool(fit_y0))
                st.session_state["last_model"] = "K"
    except Exception:
        pass

result = st.session_state.get("last_result", None)
mode_key = st.session_state.get("last_model", None)

# ----------------- Results tab -----------------
with tabs[1]:
    st.subheader("Fit Results")

    if result is None:
        st.info("Click **Fit model** in the sidebar to compute parameters.")
    else:
        if mode_key == "Ka":
            st.write(
                f"**Ka** = {result['Ka']:.6g}   |   **τ** = {result['tau']:.6g}   |   "
                f"**y₀** = {result['y0']:.6g}"
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
            file_name="first_order_fit_results.xlsx",
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
- Make a residual column: `y - y_fit`
- SSE cell: `=SUMSQ(residual_range)`
- Use **Data → Solver**: minimize SSE by changing `Ka` (or `K`) and `tau` (and y0 if fitting it). Constrain `tau > 0`.
            """
        )