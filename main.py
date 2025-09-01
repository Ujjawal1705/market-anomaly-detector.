import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Market Anomaly Detector")
st.write("Upload a CSV file with **date** and **value** columns to detect anomalies.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(data.head())

    if "date" in data.columns and "value" in data.columns:
        # Convert date
        data["date"] = pd.to_datetime(data["date"])

        # Model for anomaly detection
        model = IsolationForest(contamination=0.1, random_state=42)
        data["anomaly"] = model.fit_predict(data[["value"]])

        # Mark anomalies
        anomalies = data[data["anomaly"] == -1]

        st.subheader("Detected Anomalies")
        st.write(anomalies)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x="date", y="value", data=data, ax=ax, label="Normal")
        sns.scatterplot(x="date", y="value", data=anomalies, ax=ax, color="red", label="Anomaly")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        st.error("CSV must have 'date' and 'value' columns.")
else:
    st.info("Please upload a CSV file to continue.")
