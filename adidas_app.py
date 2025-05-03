import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
import io
import time

# Set page config
st.set_page_config(page_title="Adidas Revenue Forecast", layout="wide")

# Title and description
st.title("👟 Adidas Sales Forecasting App")
st.markdown("A stylish interactive app to explore and forecast Adidas revenue using time series models.")

# Theme switcher
theme = st.selectbox("Choose a theme 🌗", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            body {background-color: #0e1117; color: white;}
            .stApp {background-color: #0e1117;}
        </style>
    """, unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your Adidas CSV file", type=["csv"])

# Main logic
if uploaded_file is not None:
    with st.spinner("Loading data..."):
        time.sleep(1.5)
        data = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded successfully!")

    st.subheader("📈 Preview of your data")
    st.write(data.head())

    # Revenue line plot
    fig = px.line(data, x="Time Period", y="Revenue", title="Adidas Revenue Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Decompose series
    with st.expander("🔍 Seasonal Decomposition"):
        result = seasonal_decompose(data["Revenue"], model='multiplicative', period=4)
        fig_decomp = result.plot()
        st.pyplot(fig_decomp.figure)

    # Show PACF
    with st.expander("🔁 Partial Autocorrelation Plot"):
        fig_pacf, ax = plt.subplots()
        plot_pacf(data["Revenue"], lags=20, ax=ax)
        st.pyplot(fig_pacf)

    # ARIMA Model Controls
    st.subheader("⚙️ ARIMA Model Settings")
    p = st.slider("AR term (p)", 0, 5, 1)
    d = st.slider("Differencing (d)", 0, 2, 1)
    q = st.slider("MA term (q)", 0, 5, 1)

    if st.button("🚀 Run Forecast"):
        with st.spinner("Training ARIMA model..."):
            model = sm.tsa.statespace.SARIMAX(data["Revenue"],
                                              order=(p, d, q),
                                              seasonal_order=(p, d, q, 4))
            model_fit = model.fit()
            st.balloons()
            st.success("✅ Model trained and forecast complete!")

            forecast = model_fit.predict(len(data), len(data)+4)
            st.subheader("📊 Forecasted Revenue")
            st.write(forecast)

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(data["Revenue"], label="Historical Revenue")
            plt.plot(range(len(data), len(data)+5), forecast, label="Forecast", color="red", marker="o")
            plt.title("Revenue Forecast")
            plt.xlabel("Quarter")
            plt.ylabel("Revenue")
            plt.legend()
            st.pyplot(plt)

else:
    st.info("👈 Upload a CSV file to get started.")
