import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="ChurnSense | Customer Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #18283b;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #3d5875;
}

/* Dropdown selectors */
div[data-baseweb="select"] > div {
    background-color: #26374d !important;
    border: 1px solid #6483a5 !important;
    border-radius: 8px !important;
}

/* Dropdown text */
div[data-baseweb="select"] span {
    color: #f1f5f9 !important;
}

/* Dropdown options menu */
div[data-baseweb="popover"] ul {
    background-color: #1e293b !important;
}

div[data-baseweb="popover"] li {
    background-color: #1e293b !important;
    color: #f8fafc !important;
}

/* Hover on dropdown options */
div[data-baseweb="popover"] li:hover {
    background-color: #2563eb !important;
    color: #ffffff !important;
}

/* Number inputs */
div[data-testid="stNumberInput"] input {
    background-color: #202c3c !important;
    color: #ffffff !important;
    border: 1px solid #526b88 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    min-height: 45px;
    background-color: #2563eb;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📊 ChurnSense")
st.caption("Customer Churn Prediction & Retention Analytics")
st.divider()

st.subheader("Customer Information")
st.write("Enter customer details to estimate their churn risk.")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=2000.0,
            value=70.0,
            step=5.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=50.0
        )

    submitted = st.form_submit_button("🔍 Predict Customer Churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
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
        "TotalCharges": total_charges
    }])

    try:
        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])
        risk = "High" if probability >= 0.60 else (
            "Medium" if probability >= 0.30 else "Low"
        )

        st.divider()
        st.subheader("Prediction Results")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Churn Probability", f"{probability:.1%}")
        metric2.metric("Predicted Outcome", "Churn" if prediction == 1 else "No Churn")
        metric3.metric("Risk Indicator", risk)

        st.progress(probability, text=f"Estimated churn probability: {probability:.1%}")

        if prediction == 1:
            st.error("This customer is predicted to churn. Consider reviewing retention options.")
        else:
            st.success("This customer is predicted not to churn.")

        st.caption(
            "Risk bands are illustrative thresholds. Predictions are estimates, "
            "not guarantees of future customer behavior."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption("ChurnSense | Machine Learning Demo | Built with Python and Streamlit")
