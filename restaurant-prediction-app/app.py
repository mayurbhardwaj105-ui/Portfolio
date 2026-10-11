"""Streamlit dashboard.  Run with:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.features import load_history
from src.predict import load_model, predict_day
from src.train import DATA_PATH, MODEL_PATH, train

st.set_page_config(page_title="Restaurant Prediction App", page_icon="🍽️", layout="wide")
st.title("🍽️ Restaurant Prediction App")
st.caption("Predict how many people will visit tomorrow, using your past data and AI.")

# ---------- Data ----------
with st.sidebar:
    st.header("1. Your data")
    upload = st.file_uploader("Upload history CSV", type="csv")
    st.markdown(
        "Columns needed: `date, customers, temperature, rain_mm, is_holiday, promotion`"
    )
    if not DATA_PATH.exists():
        st.error("No data found. Run `python generate_sample_data.py` or upload a CSV.")
        st.stop()
    source = upload if upload is not None else DATA_PATH

try:
    history = load_history(source)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read data: {e}")
    st.stop()

# ---------- Model ----------
with st.sidebar:
    st.header("2. AI model")
    if st.button("Train / retrain model", use_container_width=True):
        with st.spinner("Training..."):
            try:
                st.session_state["metrics"] = train(history, model_path=MODEL_PATH)
                st.success("Model trained.")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

if not Path(MODEL_PATH).exists():
    st.info("Train the model from the sidebar first.")
    st.stop()
bundle = load_model()

# ---------- Tomorrow's inputs ----------
tomorrow = history["date"].max() + pd.Timedelta(days=1)
st.subheader(f"Forecast for {tomorrow:%A, %d %B %Y}")

c1, c2, c3, c4 = st.columns(4)
temp = c1.number_input("Temperature (°C)", value=float(history["temperature"].tail(7).mean().round(1)))
rain = c2.number_input("Rain (mm)", min_value=0.0, value=0.0)
holiday = c3.checkbox("Holiday / festival")
promo = c4.checkbox("Running a promotion")

try:
    pred = predict_day(history, tomorrow, temp, rain, holiday, promo, bundle)
except Exception as e:  # noqa: BLE001
    st.error(str(e))
    st.stop()

last7 = int(history["customers"].tail(7).mean())
m1, m2, m3 = st.columns(3)
m1.metric("Expected customers", pred)
m2.metric("Last 7-day average", last7, delta=pred - last7)
m3.metric("Same day last week", int(history.set_index("date")["customers"].get(tomorrow - pd.Timedelta(days=7), 0)))

st.markdown("**Staffing / stock hint:** prepare for roughly "
            f"**{int(pred * 0.9)}–{int(pred * 1.1)}** guests (±10%).")

# ---------- History ----------
st.subheader("Recent history")
recent = history.tail(60).set_index("date")["customers"]
st.line_chart(recent)

if "metrics" in st.session_state:
    m = st.session_state["metrics"]
    st.subheader("Model accuracy (last 60 days held out)")
    a, b = st.columns(2)
    a.metric("Average error", f"{m['mae']:.1f} customers", f"{m['mape']:.1f}%")
    b.metric("Naive baseline error", f"{m['baseline_mae']:.1f} customers")
    st.bar_chart(pd.Series(m["importances"]).head(8))
