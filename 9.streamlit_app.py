# STEP 9 — STREAMLIT WEB APP (UPDATED)

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Enterprise AI Customer Analytics",
    layout="wide"
)

st.title("Enterprise AI Customer Analytics Dashboard")

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():

    churn_model = joblib.load("models/churn_model.pkl")

    revenue_model = joblib.load("models/revenue_model.pkl")

    kmeans_model = joblib.load("models/kmeans_model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    return churn_model,revenue_model,kmeans_model,scaler


churn_model,revenue_model,kmeans_model,scaler = load_models()

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.header("Customer Input")

Tenure = st.sidebar.number_input("Tenure (months)",1,120,12)

Revenue_Amount = st.sidebar.number_input("Revenue Amount",0.0,100000.0,5000.0)

Login_Count = st.sidebar.number_input("Login Count",0,1000,50)

Feature_Usage_Count = st.sidebar.number_input("Feature Usage Count",0,1000,200)

Usage_Intensity_Score = st.sidebar.slider("Usage Intensity Score",0,100,60)

Support_Tickets = st.sidebar.number_input("Support Tickets",0,50,2)

Engagement_Score = st.sidebar.slider("Engagement Score",0,100,70)

# =============================
# CREATE INPUT DATAFRAME
# =============================

input_dict = {

"Tenure":[Tenure],
"Revenue_Amount":[Revenue_Amount],
"Login_Count":[Login_Count],
"Feature_Usage_Count":[Feature_Usage_Count],
"Usage_Intensity_Score":[Usage_Intensity_Score],
"Support_Tickets":[Support_Tickets],
"Engagement_Score":[Engagement_Score]

}

input_df = pd.DataFrame(input_dict)

# =============================
# DISPLAY INPUT
# =============================

st.subheader("Input Data")

st.dataframe(input_df)

# =============================
# SCALE INPUT
# =============================

try:

    scaled_input = scaler.transform(input_df)

except:

    st.error("Column mismatch between training and app")

    st.stop()

# =============================
# PREDICTION
# =============================

if st.button("Predict"):

    # churn probability
    churn_prob = churn_model.predict_proba(input_df)[0][1]

    churn_pred = churn_model.predict(input_df)[0]

    # revenue prediction
    predicted_revenue = revenue_model.predict(input_df)[0]

    # segmentation
    segment = kmeans_model.predict(scaled_input)[0]

    # =============================
    # OUTPUT DISPLAY
    # =============================

    st.subheader("Prediction Result")

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Churn Probability",
        f"{churn_prob:.2%}"
    )

    col2.metric(
        "Predicted Revenue",
        f"{predicted_revenue:.2f}"
    )

    col3.metric(
        "Customer Segment",
        segment
    )

    # =============================
    # RISK LEVEL
    # =============================

    if churn_prob > 0.7:

        st.error("High Risk Customer")

    elif churn_prob > 0.4:

        st.warning("Medium Risk Customer")

    else:

        st.success("Low Risk Customer")

    # =============================
    # PROGRESS BAR
    # =============================

    st.subheader("Risk Score")

    st.progress(float(churn_prob))

    # =============================
    # BUSINESS INSIGHT
    # =============================

    st.subheader("AI Recommendation")

    if churn_prob > 0.7:

        st.write("""
        • Assign account manager immediately  
        • Offer discount renewal  
        • Improve onboarding support  
        """)

    elif churn_prob > 0.4:

        st.write("""
        • Increase product usage training  
        • Promote additional features  
        """)

    else:

        st.write("""
        • Upsell premium plan  
        • Offer long-term contract  
        """)

# =============================
# FOOTER
# =============================

st.write("---")

st.caption("AI Powered CRM Analytics Dashboard")