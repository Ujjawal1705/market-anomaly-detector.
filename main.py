import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Streamlit Page Setup
st.set_page_config(page_title="Market Anomaly Detector", page_icon="📊", layout="wide")
st.title("📊 Market Anomaly Detector")
st.write("Detect anomalies in **stock/crypto market data** using AI (Isolation Forest).")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
contamination = st.sidebar.slider(
    "Anomaly Sensitivity (contamination)", 0.01, 0.5, 0.1, 0.01
)
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk", "1mo"])

# User input for ticker
st.subheader("📥 Fetch Market Data")
ticker = st.text_input(
    "Enter Stock/Crypto Ticker (e.g., AAPL, TSLA, BTC-USD, RELIANCE.NS)", "AAPL"
)

if st.button("Fetch Data"):
    try:
        # Fetch data from Yahoo Finance
        df = yf.download(ticker, period=period, interval=interval)
        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Close": "value"})

        if df.empty:
            st.error("⚠️ No data found. Try another ticker or period.")
        else:
            # Apply anomaly detection
            model = IsolationForest(contamination=contamination, random_state=42)
            df["anomaly"] = model.fit_predict(df[["value"]])
            anomalies = df[df["anomaly"] == -1]

            # Show raw data
            st.subheader("📊 Raw Data")
            st.dataframe(df[["date", "value"]])

            # Show anomalies
            st.subheader("🚨 Detected Anomalies")
            st.write(f"Found **{len(anomalies)} anomalies** out of {len(df)} records.")
            st.dataframe(anomalies[["date", "value"]])

            # Plot results
            st.subheader("📈 Visualization")
            fig, ax = plt.subplots()
            ax.plot(df["date"], df["value"], label="Market Value", color="blue")
            ax.scatter(anomalies["date"], anomalies["value"], color="red", label="Anomalies")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)

            # Download anomalies
            csv = anomalies.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Anomalies CSV",
                data=csv,
                file_name=f"{ticker}_anomalies.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
