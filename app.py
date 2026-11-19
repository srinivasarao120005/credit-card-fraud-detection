import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("Credit Card Fraud Detection")
p = Path("models/fraud_pipeline.joblib")
if not p.exists():
    st.warning("Train the model first: python train.py")
    st.stop()
bundle = joblib.load(p); model = bundle["model"]; features = bundle["features"]
st.write("Upload a CSV containing the model input features. Do not upload real customer-identifying information.")
f = st.file_uploader("Transaction CSV", type="csv")
if f:
    df = pd.read_csv(f)
    missing = sorted(set(features) - set(df.columns))
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        probs = model.predict_proba(df[features])[:, 1]
        result = df.copy(); result["fraud_probability"] = probs
        result["predicted_fraud_at_0_5"] = (probs >= 0.5).astype(int)
        st.dataframe(result.head(100), use_container_width=True)
        st.download_button("Download scored CSV", result.to_csv(index=False), "scored_transactions.csv")
