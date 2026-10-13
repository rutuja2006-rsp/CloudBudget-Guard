
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="CloudBudget Guard",
    page_icon="☁️",
    layout="wide"
)

BG = "#0B1020"
PANEL = "#111A2E"
BLUE = "#60A5FA"
PURPLE = "#A78BFA"
TEAL = "#2DD4BF"
CORAL = "#FB7185"
GOLD = "#FBBF24"
TEXT = "#E5E7EB"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"] {{font-family:Inter,sans-serif;}}
.stApp {{background:{BG};color:{TEXT};}}
[data-testid="stSidebar"] {{
  background:linear-gradient(180deg,#17163F,#1B2854 60%,#075985);
}}
[data-testid="stSidebar"] * {{color:#F8FAFC;}}
.block-container {{padding-top:1.1rem;max-width:1600px;}}
div[data-testid="stMetric"] {{
  background:{PANEL};border:1px solid #26334D;
  border-radius:16px;padding:16px;
}}
div[data-testid="stMetricValue"] {{color:#FFFFFF;font-weight:800;}}
.hero {{
  background:linear-gradient(110deg,#151D38,#18274C,#20204A);
  border:1px solid #334155;border-radius:18px;
  padding:22px;margin-bottom:18px;
}}
.hero h1 {{color:white;margin:0;font-size:30px;}}
.hero p {{color:#A5B4FC;margin:5px 0 0 0;}}
</style>
""", unsafe_allow_html=True)

ROOT = os.path.dirname(os.path.abspath(__file__))

def load_csv(name):
    path = os.path.join(ROOT, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

daily = load_csv("team_daily_summary.csv")
usage = load_csv("daily_usage.csv")
monthly = load_csv("monthly_summary.csv")
forecasts = load_csv("monthly_cost_forecasts.csv")
anomalies = load_csv("anomaly_detection_results.csv")
services = load_csv("services.csv")
metrics = load_csv("model_metrics.csv")
drivers = load_csv("top_cost_drivers.csv")
service_summary = load_csv("service_cost_summary.csv")
events = load_csv("events.csv")
incidents = load_csv("incidents_ground_truth.csv")
actions = load_csv("optimization_actions.csv")

if daily.empty:
    st.error("Dashboard CSV files are missing. Keep app.py beside the CSV files.")
    st.stop()

daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
if "date" in usage:
    usage["date"] = pd.to_datetime(usage["date"], errors="coerce")
if "date" in anomalies:
    anomalies["date"] = pd.to_datetime(anomalies["date"], errors="coerce")

st.sidebar.markdown("## ☁️ CloudBudget Guard")
st.sidebar.caption("SMART CLOUD COST MONITORING")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Cost Analysis",
        "Budget Tracking",
        "Anomaly Detection",
        "Forecasting",
        "Recommendations",
        "Data Explorer"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### ✨ Smarter Spending")
st.sidebar.caption("Monitor. Predict. Optimize.")

st.markdown("""
<div class="hero">
<h1>CloudBudget Guard</h1>
<p>AI-assisted cloud cost monitoring, forecasting and anomaly detection</p>
</div>
""", unsafe_allow_html=True)

team_col = "team_name" if "team_name" in daily else "team_id"
team_options = ["All teams"] + sorted(
    daily[team_col].dropna().astype(str).unique().tolist()
)

left_filter, right_filter = st.columns(2)
with left_filter:
    selected_team = st.selectbox("Select team", team_options)
with right_filter:
    min_date = daily["date"].min().date()
    max_date = daily["date"].max().date()
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

filtered = daily.copy()
if selected_team != "All teams":
    filtered = filtered[
        filtered[team_col].astype(str) == selected_team
    ]

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    filtered = filtered[
        (filtered["date"] >= pd.Timestamp(selected_dates[0])) &
        (filtered["date"] <= pd.Timestamp(selected_dates[1]))
    ]

def theme(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=35, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter", size=11),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(showgrid=False, zeroline=False, color="#9CA3AF"),
        yaxis=dict(gridcolor="#28344B", zeroline=False, color="#9CA3AF")
    )
    return fig

total_cost = float(filtered["daily_cost_inr"].sum())
budget_util = (
    float(filtered["mtd_budget_pct"].mean())
    if "mtd_budget_pct" in filtered else 0
)
forecast_value = (
    float(forecasts["forecast_month_end_inr"].mean())
    if not forecasts.empty else np.nan
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cost · Selected Period", f"₹{total_cost:,.0f}")
k2.metric("Avg. Budget Utilization", f"{budget_util:.1f}%")
k3.metric(
    "Avg. Month-end Forecast",
    f"₹{forecast_value:,.0f}" if np.isfinite(forecast_value) else "—"
)
k4.metric("Flagged Anomalies", f"{len(anomalies):,}")

if page in ["Overview", "Cost Analysis"]:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Cost Breakdown by Service")
        if not usage.empty:
            u = usage.copy()
            if selected_team != "All teams" and "team_name" in u:
                u = u[u["team_name"].astype(str) == selected_team]
            if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                u = u[
                    (u["date"] >= pd.Timestamp(selected_dates[0])) &
                    (u["date"] <= pd.Timestamp(selected_dates[1]))
                ]
            by_service = u.groupby("service", as_index=False)["cost_inr"].sum()
            fig = px.pie(
                by_service, names="service", values="cost_inr",
                hole=0.60,
                color_discrete_sequence=[
                    BLUE, PURPLE, TEAL, CORAL, GOLD,
                    "#818CF8", "#38BDF8", "#94A3B8"
                ]
            )
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(
                height=330, paper_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT,
                legend=dict(orientation="h", y=-0.15)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Service-level data is unavailable.")

    with col2:
        st.subheader("Daily Cost Trend")
        trend = (
            filtered.groupby("date", as_index=False)["daily_cost_inr"]
            .sum().sort_values("date")
        )
        fig = px.bar(
            trend, x="date", y="daily_cost_inr",
            color_discrete_sequence=[BLUE], opacity=0.8
        )
        fig.add_scatter(
            x=trend["date"],
            y=trend["daily_cost_inr"].rolling(7, min_periods=1).mean(),
            mode="lines", name="7-day average",
            line=dict(color=PURPLE, width=3)
        )
        theme(fig)
        fig.update_yaxes(title="Cost (₹)")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Budget vs Forecast by Team")
        if not forecasts.empty:
            f = forecasts.copy()
            if selected_team != "All teams":
                f = f[f["team_name"].astype(str) == selected_team]
            f = f.sort_values("forecast_month_end_inr", ascending=False).head(12)
            fig = go.Figure()
            fig.add_bar(
                x=f["team_name"], y=f["monthly_budget_inr"],
                name="Monthly budget", marker_color="#64748B"
            )
            fig.add_bar(
                x=f["team_name"], y=f["forecast_month_end_inr"],
                name="Forecast", marker_color=TEAL
            )
            fig.update_layout(barmode="group")
            theme(fig, 360)
            fig.update_xaxes(tickangle=-25, title="")
            fig.update_yaxes(title="INR")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Forecast Risk Overview")
        if not forecasts.empty:
            fig = px.scatter(
                forecasts,
                x="monthly_budget_inr",
                y="forecast_month_end_inr",
                color="risk_level",
                hover_name="team_name",
                color_discrete_map={
                    "High": CORAL, "Medium": GOLD, "Low": TEAL
                }
            )
            theme(fig, 360)
            fig.update_xaxes(title="Budget (₹)")
            fig.update_yaxes(title="Forecast (₹)")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Budget Tracking":
    st.subheader("🎯 Budget Tracking")
    if not monthly.empty:
        m = monthly.copy()
        m["budget_status"] = np.where(
            m["total_spend_inr"] > m["monthly_budget_inr"],
            "Over budget", "Within budget"
        )
        fig = px.scatter(
            m, x="monthly_budget_inr", y="total_spend_inr",
            color="budget_status",
            color_discrete_map={
                "Over budget": CORAL, "Within budget": TEAL
            },
            hover_name="team_name"
        )
        limit = float(max(m["monthly_budget_inr"].max(),
                          m["total_spend_inr"].max()))
        fig.add_shape(
            type="line", x0=0, y0=0, x1=limit, y1=limit,
            line=dict(color=PURPLE, dash="dash")
        )
        theme(fig, 400)
        fig.update_xaxes(title="Budget (₹)")
        fig.update_yaxes(title="Actual Spend (₹)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            m.sort_values("overrun_pct", ascending=False),
            use_container_width=True, hide_index=True
        )

elif page == "Anomaly Detection":
    st.subheader("🚨 Anomaly Detection")
    st.caption("Isolation Forest flags unusual costs for human review.")
    if not anomalies.empty:
        st.dataframe(anomalies, use_container_width=True, hide_index=True)
        fig = px.scatter(
            anomalies, x="date", y="daily_cost_inr",
            color_discrete_sequence=[CORAL],
            hover_name="team_name"
        )
        theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies were flagged.")

elif page == "Forecasting":
    st.subheader("🔮 Month-end Forecasting")
    st.caption("Risk scores are prototype estimates, not calibrated guarantees.")
    if not forecasts.empty:
        st.dataframe(
            forecasts.sort_values(
                "overrun_probability_pct", ascending=False
            ),
            use_container_width=True, hide_index=True
        )
    if not metrics.empty:
        st.subheader("Chronological Holdout Model Evaluation")
        st.dataframe(metrics, use_container_width=True, hide_index=True)

elif page == "Recommendations":
    st.subheader("💡 Cost Optimization Recommendations")
    if not service_summary.empty:
        for _, row in service_summary.iterrows():
            st.markdown(
                f"**{row['service']}** · {row['priority']} priority  \n"
                f"Observed cost: ₹{row['total_cost_inr']:,.0f}  \n"
                f"{row['recommendation']}"
            )
            st.divider()
    else:
        st.write("Review idle compute, GPU, storage, database and network costs.")
    st.caption("Recommendations are suggestions, not automatic cloud-resource changes.")

elif page == "Data Explorer":
    st.subheader("🧭 Data Explorer")
    datasets = {
        "Daily summary": daily,
        "Daily service usage": usage,
        "Monthly summary": monthly,
        "Forecasts": forecasts,
        "Anomalies": anomalies,
        "Service cost summary": service_summary,
        "Top cost drivers": drivers,
        "Events": events,
        "Incident ground truth": incidents,
        "Optimization actions": actions,
        "Model metrics": metrics
    }
    name = st.selectbox("Choose dataset", list(datasets.keys()))
    frame = datasets[name]
    search = st.text_input("Search rows")
    if search:
        frame = frame[
            frame.astype(str).apply(
                lambda col: col.str.contains(
                    search, case=False, na=False
                )
            ).any(axis=1)
        ]
    st.caption(f"{len(frame):,} rows · {len(frame.columns)} columns")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=name.lower().replace(" ", "_") + ".csv",
        mime="text/csv"
    )

st.markdown(
    '<p style="text-align:center;color:#64748B;font-size:11px">'
    'CloudBudget Guard · Synthetic-data research prototype</p>',
    unsafe_allow_html=True
)
