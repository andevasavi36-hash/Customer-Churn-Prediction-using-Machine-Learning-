"""
Customer Churn Prediction - Streamlit Application
====================================================
Loads a trained ML model (plus its label encoders and scaler) and provides
an interactive form for predicting whether a telecom customer is likely
to churn.

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
]

NUMERICAL_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "numAdminTickets",
    "numTechTickets",
    "AvgMonthlySpend",
    "TotalSupportTickets",
    "CustomerLifetimeValue",
]

# Business thresholds used during feature engineering (must match training)
HIGH_MONTHLY_CHARGE_THRESHOLD = 70.0
HIGH_SUPPORT_TICKET_THRESHOLD = 3
LONG_TERM_TENURE_MONTHS = 24


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the trained model, scaler, and label encoders from disk."""
    missing = [p for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, encoders, []


# ----------------------------------------------------------------------
# Feature engineering (must mirror src/feature_engineering.py)
# ----------------------------------------------------------------------
def get_tenure_group(tenure: int) -> str:
    if tenure <= 12:
        return "0-12"
    elif tenure <= 24:
        return "13-24"
    elif tenure <= 48:
        return "25-48"
    elif tenure <= 60:
        return "49-60"
    else:
        return "61-72"


def engineer_features(raw: dict) -> dict:
    """Create the derived business features from raw user input."""
    data = dict(raw)

    tenure = data["tenure"]
    monthly_charges = data["MonthlyCharges"]
    total_charges = data["TotalCharges"]
    admin_tickets = data["numAdminTickets"]
    tech_tickets = data["numTechTickets"]

    data["AvgMonthlySpend"] = total_charges / (tenure + 1)
    data["TotalSupportTickets"] = admin_tickets + tech_tickets
    data["CustomerLifetimeValue"] = tenure * monthly_charges
    data["LongTermCustomer"] = int(tenure > LONG_TERM_TENURE_MONTHS)
    data["HighMonthlyCharges"] = int(monthly_charges > HIGH_MONTHLY_CHARGE_THRESHOLD)
    data["HighSupportCustomer"] = int(data["TotalSupportTickets"] > HIGH_SUPPORT_TICKET_THRESHOLD)
    data["TenureGroup"] = get_tenure_group(tenure)

    return data


def preprocess_input(raw: dict, scaler, encoders) -> pd.DataFrame:
    """Apply label encoding and scaling to match the training pipeline."""
    engineered = engineer_features(raw)
    df = pd.DataFrame([engineered])

    # Label-encode categorical columns using the saved encoders
    for col in CATEGORICAL_COLUMNS:
        if col in encoders:
            le = encoders[col]
            value = df.at[0, col]
            if value in le.classes_:
                df[col] = le.transform([value])[0]
            else:
                # Unseen category fallback -> most frequent class (index 0)
                df[col] = 0
        else:
            df[col] = 0

    # Ensure the full expected column order exists before scaling
    expected_order = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [
        "LongTermCustomer",
        "HighMonthlyCharges",
        "HighSupportCustomer",
        "SeniorCitizen",
    ]
    for col in expected_order:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_order]

    # Scale numerical columns
    df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])

    return df


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("📊 About")
    st.sidebar.info(
        "This application predicts the likelihood that a telecom customer "
        "will churn, using a model trained on demographics, subscribed "
        "services, billing details, and support-interaction history."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline**")
    st.sidebar.markdown(
        "- Label Encoding\n"
        "- Standard Scaling\n"
        "- Engineered business features\n"
        "- Best of Logistic Regression / Random Forest / XGBoost"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit")


# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
def render_input_form():
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**Billing & Support**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 65.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 20000.0, 800.0, step=10.0)
        num_admin_tickets = st.number_input("Admin Support Tickets", 0, 50, 0, step=1)
        num_tech_tickets = st.number_input("Tech Support Tickets", 0, 50, 0, step=1)

    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "numAdminTickets": num_admin_tickets,
        "numTechTickets": num_tech_tickets,
    }
    return raw_input


# ----------------------------------------------------------------------
# Result rendering
# ----------------------------------------------------------------------
def business_recommendation(churn: bool, probability: float, raw_input: dict) -> str:
    if not churn:
        return (
            "This customer shows a healthy engagement profile. Continue standard "
            "retention touchpoints and monitor for changes in usage or billing."
        )

    tips = []
    if raw_input["Contract"] == "Month-to-month":
        tips.append("offer an incentive to switch to a 1- or 2-year contract")
    if raw_input["TechSupport"] == "No" or raw_input["OnlineSecurity"] == "No":
        tips.append("promote add-on services such as Tech Support or Online Security")
    if raw_input["numAdminTickets"] + raw_input["numTechTickets"] >= HIGH_SUPPORT_TICKET_THRESHOLD:
        tips.append("prioritize a proactive support/service-quality follow-up")
    if raw_input["MonthlyCharges"] > HIGH_MONTHLY_CHARGE_THRESHOLD:
        tips.append("consider a loyalty discount or plan review")
    if not tips:
        tips.append("reach out with a personalized retention offer")

    if probability >= 0.75:
        urgency = "High risk — immediate retention action recommended."
    elif probability >= 0.5:
        urgency = "Moderate-to-high risk — proactive outreach advised."
    else:
        urgency = "Borderline risk — keep on a watch list."

    return f"{urgency} Suggested actions: " + "; ".join(tips) + "."


def render_prediction(model, scaler, encoders, raw_input):
    processed_df = preprocess_input(raw_input, scaler, encoders)

    prediction = model.predict(processed_df)[0]
    try:
        probability = model.predict_proba(processed_df)[0][1]
    except AttributeError:
        probability = float(prediction)

    is_churn = bool(prediction == 1)

    st.markdown("---")
    st.subheader("Prediction Result")

    result_col, prob_col = st.columns(2)

    with result_col:
        if is_churn:
            st.error("⚠️ **Prediction: Customer is likely to CHURN**")
        else:
            st.success("✅ **Prediction: Customer is likely to STAY**")

    with prob_col:
        st.metric("Churn Probability", f"{probability * 100:.2f}%")
        st.progress(min(max(probability, 0.0), 1.0))

    st.markdown("### 💡 Business Recommendation")
    st.info(business_recommendation(is_churn, probability, raw_input))

    with st.expander("View engineered feature values"):
        st.dataframe(pd.DataFrame([engineer_features(raw_input)]).T.rename(columns={0: "Value"}))


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------
def main():
    st.title("📉 Customer Churn Prediction")
    st.caption(
        "Predict the likelihood of customer churn based on demographics, "
        "service usage, billing, and support history."
    )

    render_sidebar()

    model, scaler, encoders, missing = load_artifacts()

    if missing:
        st.error(
            "Required model artifacts were not found. Please ensure the "
            "following files exist before running the app:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
        )
        st.stop()

    raw_input = render_input_form()

    st.markdown("---")
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        with st.spinner("Running prediction..."):
            render_prediction(model, scaler, encoders, raw_input)


if __name__ == "__main__":
    main()
