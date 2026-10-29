"""Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from src.predict import predict_df

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")

tab1, tab2 = st.tabs(["Single customer", "Bulk CSV"])

with tab1:
    c1, c2 = st.columns(2)
    tenure = c1.slider("Tenure (months)", 1, 72, 12)
    monthly = c1.number_input("Monthly charges", 18.0, 120.0, 70.0)
    calls = c1.slider("Support calls", 0, 10, 2)
    contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = c2.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    pay = c2.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    sec = c2.selectbox("Online security", ["Yes", "No"])
    sup = c2.selectbox("Tech support", ["Yes", "No"])
    senior = c2.selectbox("Senior citizen", ["No", "Yes"])
    if st.button("Predict"):
        row = pd.DataFrame([{
            "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
            "SupportCalls": calls, "gender": "Male", "SeniorCitizen": senior, "Partner": "No",
            "Dependents": "No", "InternetService": internet, "OnlineSecurity": sec,
            "TechSupport": sup, "Contract": contract, "PaperlessBilling": "Yes", "PaymentMethod": pay}])
        r = predict_df(row).iloc[0]
        st.metric("Churn probability", f"{r.churn_probability:.1%}")
        st.write(f"Risk level: **{r.risk}**")

with tab2:
    f = st.file_uploader("Upload CSV", type="csv")
    if f:
        res = predict_df(pd.read_csv(f))
        st.dataframe(res.sort_values("churn_probability", ascending=False))
        st.download_button("Download", res.to_csv(index=False), "predictions.csv")
