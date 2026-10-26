import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MachinePulse AI", page_icon="⚙️", layout="wide")
st.title("⚙️ MachinePulse AI")
st.subheader("Explainable anomaly monitoring for industrial sensor data")
st.caption("Independent student project • Uses synthetic demonstration data, not live factory telemetry")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("sample_sensor_data.csv")
    required = ["timestamp", "temperature_c", "vibration_mm_s", "pressure_bar", "energy_kw"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    return data

uploaded = st.sidebar.file_uploader("Upload sensor CSV (optional)", type=["csv"])
contamination = st.sidebar.slider("Expected anomaly fraction", 0.01, 0.15, 0.04, 0.01)
try:
    data = load_data(uploaded)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

features = ["temperature_c", "vibration_mm_s", "pressure_bar", "energy_kw"]
if len(data) < 20:
    st.warning("Upload at least 20 valid rows for meaningful demonstration.")
    st.stop()

# Fit on standardized readings; this is a demonstration baseline, not a validated industrial model.
scaler = StandardScaler()
X = scaler.fit_transform(data[features])
model = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
data["anomaly_label"] = model.fit_predict(X)
data["is_anomaly"] = data["anomaly_label"] == -1
data["anomaly_score"] = -model.decision_function(X)

# Transparent evidence: mark readings outside robust IQR bounds for each sensor.
evidence = []
for col in features:
    q1, q3 = data[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    evidence.append(data[col].lt(low) | data[col].gt(high))
evidence_frame = pd.concat(evidence, axis=1)
evidence_frame.columns = features
data["evidence"] = evidence_frame.apply(
    lambda row: ", ".join([c.replace("_", " ") for c, flag in row.items() if flag]) or "Multivariate pattern", axis=1
)
data["severity"] = np.where(data["anomaly_score"] >= data["anomaly_score"].quantile(.95), "High",
                    np.where(data["is_anomaly"], "Medium", "Normal"))

a, b, c, d = st.columns(4)
a.metric("Valid readings", f"{len(data):,}")
b.metric("Flagged anomalies", f"{int(data['is_anomaly'].sum()):,}")
c.metric("Anomaly rate", f"{data['is_anomaly'].mean():.1%}")
d.metric("High-severity alerts", f"{int((data['severity'] == 'High').sum()):,}")

st.markdown("### Sensor trends")
chosen = st.selectbox("Sensor to visualize", features, format_func=lambda x: x.replace("_", " ").title())
fig = px.line(data, x="timestamp", y=chosen, title=f"{chosen.replace('_',' ').title()} over time")
anom = data[data["is_anomaly"]]
if not anom.empty:
    fig.add_scatter(x=anom["timestamp"], y=anom[chosen], mode="markers", name="Flagged anomaly",
                    marker=dict(size=9, symbol="x"))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("### Sensor relationship")
    corr = data[features].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix"), use_container_width=True)
with right:
    st.markdown("### Alert severity")
    counts = data.loc[data["is_anomaly"], "severity"].value_counts().rename_axis("severity").reset_index(name="count")
    if counts.empty:
        st.info("No anomalies were flagged with this setting.")
    else:
        st.plotly_chart(px.bar(counts, x="severity", y="count", color="severity", title="Flagged readings by severity"), use_container_width=True)

st.markdown("### Explainable alert log")
alerts = data.loc[data["is_anomaly"], ["timestamp"] + features + ["severity", "anomaly_score", "evidence"]].copy()
alerts["timestamp"] = alerts["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
alerts = alerts.sort_values("anomaly_score", ascending=False)
st.dataframe(alerts, use_container_width=True, hide_index=True)
st.download_button("Download alert log CSV", alerts.to_csv(index=False).encode("utf-8"),
                   file_name="machinepulse_alerts.csv", mime="text/csv")

st.markdown("### Method and limitations")
st.write("The demo validates columns, removes rows with missing required values, standardizes sensor features, and uses Isolation Forest to flag unusual multivariate readings. The evidence column lists sensor values outside IQR-based bounds when available.")
st.warning("This is a learning prototype using synthetic data. It is not validated for real equipment, does not diagnose a specific fault, and should not be used for operational safety decisions.")
