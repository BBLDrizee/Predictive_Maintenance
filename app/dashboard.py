"""TurboRUL dashboard (FR-11): fleet overview, engine detail and model comparison.

    streamlit run app/dashboard.py                                        # in-process
    TURBORUL_API_URL=http://localhost:8000 streamlit run app/dashboard.py   # against the API

Red, amber and green are reserved for risk tiers and always come with a text label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from client import get_client  # noqa: E402

TIER_COLOR = {"Critical": "#d03b3b", "Watch": "#fab219", "Healthy": "#0ca30c"}
TIER_ICON = {"Critical": "▲", "Watch": "●", "Healthy": "✓"}
MODEL_LABEL = {"linear": "Linear regression", "xgb": "XGBoost", "lstm": "LSTM", "cnn": "1D-CNN"}
MODEL_COLOR = {"linear": "#2a78d6", "xgb": "#eb6834", "lstm": "#1baf7a", "cnn": "#eda100"}
BLUE, MAGENTA, INK, MUTED, GRID = "#2a78d6", "#e87ba4", "#1f1f1e", "#6b6a64", "#e4e3dc"
PAGES = ["Fleet overview", "Engine detail", "Model comparison", "About"]

st.set_page_config(page_title="TurboRUL", page_icon="✈", layout="wide")
client = get_client()


@st.cache_data(show_spinner=False)
def subsets() -> list[dict]:
    return client.subsets()


@st.cache_data(show_spinner="Scoring the fleet…")
def fleet(subset: str) -> pd.DataFrame:
    return pd.DataFrame(client.fleet(subset))


@st.cache_data(show_spinner="Explaining the prediction…")
def engine(subset: str, unit: int) -> dict:
    return client.engine(subset, unit)


@st.cache_data(show_spinner=False)
def results() -> dict:
    return client.results()


@st.cache_data(show_spinner=False)
def test_predictions(subset: str, model: str) -> pd.DataFrame:
    return pd.DataFrame(client.test_predictions(subset, model))


def tier_badge(tier: str) -> str:
    return (f"<span style='background:{TIER_COLOR[tier]};color:#fff;padding:2px 10px;border-radius:10px;"
            f"font-weight:600;font-size:0.9em'>{TIER_ICON[tier]} {tier}</span>")


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=INK), legend=dict(orientation="h", y=1.08, x=0), hovermode="closest")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# --- sidebar -----------------------------------------------------------------

available = subsets()
if not available:
    st.error("No exported models found. Run `python -m turborul.export` first.")
    st.stop()

with st.sidebar:
    st.markdown("## ✈ TurboRUL")
    st.caption("Remaining Useful Life for turbofan engines")
    subset = st.selectbox("Engine fleet (C-MAPSS subset)", [s["subset"] for s in available],
                          format_func=lambda s: {"FD001": "FD001 · 1 condition, 1 fault", "FD002": "FD002 · 6 conditions, 1 fault",
                                                 "FD003": "FD003 · 1 condition, 2 faults", "FD004": "FD004 · 6 conditions, 2 faults"}[s],
                          key="subset")
    page = st.radio("View", PAGES, key="page")
    meta = next(s for s in available if s["subset"] == subset)
    st.caption(f"Model: **{meta['model_label']}** ({meta['params_source']}), window {meta['window_len']} cycles, "
               f"RUL cap {meta['rul_cap']}. Data via {client.mode}.")
    st.info("Simulated data (NASA C-MAPSS). A research prototype, not certified maintenance software.", icon="ℹ️")


# --- fleet overview -------------------------------------------------------------

def page_fleet() -> None:
    df = fleet(subset)
    st.title("Which engines need attention?")
    st.caption(f"{len(df)} in-service engines in {subset}, sorted by predicted Remaining Useful Life (RUL, in flight cycles). "
               "Ranges are 90% prediction intervals.")
    cols = st.columns(3)
    for col, tier in zip(cols, ["Critical", "Watch", "Healthy"]):
        n = int((df.tier == tier).sum())
        rule = {"Critical": "RUL ≤ 30 cycles", "Watch": "31–60 cycles", "Healthy": "> 60 cycles"}[tier]
        col.markdown(
            f"<div style='border-left:6px solid {TIER_COLOR[tier]};padding:8px 14px;background:#f7f7f5;border-radius:6px'>"
            f"<div style='font-size:0.95em;color:{MUTED}'>{TIER_ICON[tier]} {tier} · {rule}</div>"
            f"<div style='font-size:2.2em;font-weight:700;color:{INK}'>{n}</div></div>", unsafe_allow_html=True)
    st.write("")

    f1, f2 = st.columns([3, 2])
    tiers = f1.multiselect("Show tiers", ["Critical", "Watch", "Healthy"], default=["Critical", "Watch", "Healthy"])
    show_truth = f2.toggle("Show true RUL (known for these test engines)", value=False)
    view = df[df.tier.isin(tiers)].copy()
    table = pd.DataFrame({
        "Engine": view.unit, "Current cycle": view.cycle, "Predicted RUL": view.rul.round(0).astype(int),
        "90% range": [f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(view.low, view.high)],
        "Tier": view.tier, "Recommended action": view.action,
    })
    if show_truth:
        table.insert(4, "True RUL", view.true_rul.astype(int))

    def color_tier(v: str) -> str:
        return f"background-color:{TIER_COLOR[v]};color:white;font-weight:600" if v in TIER_COLOR else ""

    st.dataframe(table.style.map(color_tier, subset=["Tier"]), hide_index=True, use_container_width=True, height=420)

    c1, c2 = st.columns([3, 1])
    options = view.unit.tolist()
    if options:
        pick = c1.selectbox("Open an engine", options,
                            format_func=lambda u: f"Engine {u} — {view.set_index('unit').loc[u, 'tier']}, "
                                                  f"{view.set_index('unit').loc[u, 'rul']:.0f} cycles")
        c2.write("")
        c2.button("Open engine detail →", use_container_width=True, on_click=open_engine, args=(int(pick),))


def open_engine(unit: int) -> None:
    """Button callback: runs before the next rerun, so it may change the page widget's state."""
    st.session_state["unit"] = unit
    st.session_state["page"] = "Engine detail"


# --- engine detail ---------------------------------------------------------------

def page_engine() -> None:
    df = fleet(subset)
    units = df.unit.tolist()
    default = st.session_state.get("unit", units[0])
    unit = st.selectbox("Engine", units, index=units.index(default) if default in units else 0,
                        format_func=lambda u: f"Engine {u} — {df.set_index('unit').loc[u, 'tier']}, "
                                              f"{df.set_index('unit').loc[u, 'rul']:.0f} cycles")
    st.session_state["unit"] = int(unit)
    e = engine(subset, int(unit))

    st.markdown(f"## Engine {unit} &nbsp; {tier_badge(e['tier'])}", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted RUL", f"{e['rul']:.0f} cycles", help="Remaining Useful Life in flight cycles.")
    m2.metric(f"{e['confidence']:.0%} range", f"{e['low']:.0f} – {e['high']:.0f}")
    m3.metric("Current cycle", e["cycle"])
    m4.metric("True RUL (test data)", f"{e['true_rul']:.0f} cycles", delta=f"error {e['rul'] - e['true_rul']:+.0f}",
              delta_color="off")
    {"Critical": st.error, "Watch": st.warning, "Healthy": st.success}[e["tier"]](f"**Recommended action:** {e['action']}")

    left, right = st.columns([3, 2])
    traj = pd.DataFrame(e["trajectory"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.concat([traj.cycle, traj.cycle[::-1]]), y=pd.concat([traj.high, traj.low[::-1]]),
                             fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="90% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=traj.cycle, y=traj.rul, name="Predicted RUL", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=traj.cycle, y=traj.true_rul, name="True RUL", line=dict(color=MUTED, width=2, dash="dash")))
    for tier, y in [("Critical", 30), ("Watch", 60)]:
        fig.add_hline(y=y, line=dict(color=TIER_COLOR[tier], width=1, dash="dot"),
                      annotation_text=f"{tier} ≤ {y}", annotation_position="top left", annotation_font_color=MUTED)
    fig.update_layout(title="Predicted RUL over the engine's life (as it would have been seen live)",
                      xaxis_title="Cycle", yaxis_title="RUL (cycles)")
    left.plotly_chart(style_fig(fig), use_container_width=True)

    top = pd.DataFrame(e["top_sensors"]).iloc[::-1]
    labels = [f"{r.sensor} · {r.symbol} ({r['name']})" for _, r in top.iterrows()]
    fig = go.Figure(go.Bar(x=top.contribution, y=labels, orientation="h",
                           marker_color=[BLUE if v > 0 else MAGENTA for v in top.contribution],
                           text=[f"{v:+.1f}" for v in top.contribution], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.1f} cycles<extra></extra>"))
    span = float(top.contribution.abs().max()) or 1.0
    lo = -1.3 * span if (top.contribution < 0).any() else -0.1 * span
    hi = 1.3 * span if (top.contribution > 0).any() else 0.1 * span
    fig.update_layout(title="Why this prediction", xaxis_title="Contribution to RUL (cycles)", xaxis_range=[lo, hi])
    right.plotly_chart(style_fig(fig), use_container_width=True)
    right.caption(f"Starts from an average-engine baseline of {e['base_value']:.0f} cycles. Magenta bars pull the prediction "
                  "towards failure; blue bars push it up. Sensors are correlated, so credit is shared among them.")

    trends = e["sensor_trends"]
    names = list(trends["sensors"])
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{s} · {trends['sensors'][s]['symbol']}" for s in names],
                        shared_xaxes=True, vertical_spacing=0.18)
    for i, s in enumerate(names):
        r, c = i // 3 + 1, i % 3 + 1
        fig.add_trace(go.Scatter(x=trends["cycle"], y=trends["sensors"][s]["value"], mode="lines",
                                 line=dict(color="#b7d3f6", width=1), showlegend=False, hoverinfo="skip"), r, c)
        fig.add_trace(go.Scatter(x=trends["cycle"], y=trends["sensors"][s]["smoothed"], mode="lines", name=s,
                                 line=dict(color=BLUE, width=2), showlegend=False), r, c)
    fig.update_layout(title=f"Most influential sensors — {trends['units']}")
    fig.update_xaxes(title_text="Cycle", row=2)
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)


# --- model comparison -------------------------------------------------------------

def pm(mean: float, std: float | None, fmt: str = ".2f") -> str:
    return f"{mean:{fmt}}" if std is None or pd.isna(std) or std == 0 else f"{mean:{fmt}} ± {std:{fmt}}"


def page_models() -> None:
    res = results()
    s = pd.DataFrame(res["summary"])
    st.title("Model comparison")
    st.caption("Final configuration of each model on each subset's official test set: mean ± std over five seeds. "
               "Lower is better for RMSE and the NASA score (which penalises late predictions more).")
    order = [m for m in MODEL_LABEL if m in set(s.model)]
    for metric, col, std, fmt in [("Test RMSE (cycles)", "test_rmse", "test_rmse_std", ".2f"),
                                  ("NASA score", "nasa", "nasa_std", ".0f")]:
        table = s.assign(v=[pm(a, b, fmt) for a, b in zip(s[col], s[std])]).pivot(index="model", columns="subset", values="v")
        table = table.loc[order].rename(index=MODEL_LABEL)
        st.markdown(f"**{metric}**")
        st.dataframe(table, use_container_width=True)
    cover = s.dropna(subset=["coverage"]).pivot(index="model", columns="subset", values="coverage")
    if not cover.empty:
        st.markdown("**90% range coverage** (share of test engines whose true RUL falls inside the predicted range)")
        st.dataframe(cover.loc[[m for m in order if m in cover.index]].rename(index=MODEL_LABEL).map(lambda v: f"{v:.0%}"),
                     use_container_width=True)

    c1, c2 = st.columns(2)
    model = c1.selectbox("Model", order, format_func=MODEL_LABEL.get, index=order.index(meta["model"]) if meta["model"] in order else 0)
    preds = test_predictions(subset, model)
    lim = float(max(preds.true_rul.max(), preds.high.max() if "high" in preds else preds.pred_rul.max())) + 10
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=lim, y0=0, y1=30, fillcolor="rgba(208,59,59,0.07)", line_width=0)
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=MUTED, dash="dash"), name="Perfect"))
    err = dict(type="data", symmetric=False, array=preds.high - preds.pred_rul, arrayminus=preds.pred_rul - preds.low,
               color="rgba(0,0,0,0.15)", thickness=1, width=0) if "high" in preds else None
    fig.add_trace(go.Scatter(x=preds.true_rul, y=preds.pred_rul, mode="markers", error_y=err, name=MODEL_LABEL[model],
                             marker=dict(color=MODEL_COLOR[model], size=8, line=dict(color="white", width=1)),
                             customdata=preds.unit, hovertemplate="Engine %{customdata}<br>true %{x:.0f}, predicted %{y:.0f}<extra></extra>"))
    fig.update_layout(title=f"Predicted vs true RUL — {subset} test set (seed 0)", xaxis_title="True RUL (cycles)",
                      yaxis_title="Predicted RUL (cycles)", xaxis_range=[0, lim], yaxis_range=[0, lim])
    c1.plotly_chart(style_fig(fig, 420), use_container_width=True)

    b = pd.DataFrame(res["buckets"])
    b = b[b.subset == subset]
    fig = go.Figure()
    for m in order:
        d = b[b.model == m]
        fig.add_trace(go.Bar(x=d.bucket, y=d.rmse, name=MODEL_LABEL[m], marker_color=MODEL_COLOR[m],
                             hovertemplate="%{x}: %{y:.1f} cycles<extra>" + MODEL_LABEL[m] + "</extra>"))
    fig.update_layout(title="Error by true-RUL bucket (smallest near failure)", barmode="group",
                      xaxis_title="True RUL bucket (cycles)", yaxis_title="RMSE (cycles)")
    c2.plotly_chart(style_fig(fig, 420), use_container_width=True)


def page_about() -> None:
    st.title("About this project")
    st.markdown(
        """
TurboRUL predicts how many flight cycles a turbofan engine has left (its **Remaining Useful Life**) from its sensor
history, explains which sensors drove each prediction, and ranks engines so a maintenance planner knows which to
schedule first.

**Data.** NASA's C-MAPSS turbofan degradation dataset. **It is simulated, not from a real fleet**: failure is defined by a
health-margin threshold, and there is no cost, schedule or parts data. The "fleet" shown here is each subset's official
test engines, whose true RUL is known, so errors can be shown honestly.

**Method.** Sensors are normalised within each operating condition, sliced into windows, and scored by linear regression,
XGBoost, an LSTM and a 1D-CNN, tuned with engine-level cross-validation. Ranges are 90% prediction intervals calibrated on
held-out engines. Explanations are SHAP values (XGBoost) or integrated gradients (deep models), in cycles.

**Risk tiers.** Critical: RUL ≤ 30 cycles · Watch: 31–60 · Healthy: > 60.

This is a course research prototype, not certified maintenance software.
        """
    )


{"Fleet overview": page_fleet, "Engine detail": page_engine, "Model comparison": page_models, "About": page_about}[page]()
