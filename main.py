if st.button("Fetch Data"):
    try:
        # Fetch data from Yahoo Finance
        df = yf.download(ticker, period=period, interval=interval)

        # Fallback if empty
        if df.empty:
            st.warning("⚠️ No data found with period. Retrying with fixed start/end dates...")
            df = yf.download(ticker, start="2023-01-01", end="2025-01-01", interval=interval)

        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Close": "value"})

        if df.empty:
            st.error("⚠️ Still no data found. Try another ticker.")
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
