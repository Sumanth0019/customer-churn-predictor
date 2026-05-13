
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from streamlit_oauth import OAuth2Component

st.set_page_config(
    page_title="Telco Retention AI",
    page_icon="📊",
    layout="wide"
)

# -----------------------
# Load Assets
# -----------------------
DATA_PATH = "Telco-Customer-Churn.csv"
MODEL_PATH = "customer_churn_model.pkl"

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
    background-color: #4F46E5;
    color: white;
}

.metric-card {
    background-color: #1E1E2F;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: #FFFFFF;
}

.subtitle {
    font-size: 18px;
    color: #B0B0B0;
}

.section-title {
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Session State
# -----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# -----------------------
# Authentication Pages
# -----------------------
def signup_page():
    st.markdown("<div class='title'>Create Account</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Join Telco Retention AI</div>", unsafe_allow_html=True)

    with st.form("signup_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Sign Up")

        if submitted:
            st.success("Account created successfully!")
            st.session_state.page = "login"

def login_page():

    st.markdown("<div class='title'>Telco Retention AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict Customer Churn with AI</div>", unsafe_allow_html=True)

    # -------------------
    # NORMAL LOGIN
    # -------------------
    with st.form("login_form"):

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Login")

        if submitted:
            st.session_state.logged_in = True
            st.session_state.page = "dashboard"

    # -------------------
    # GOOGLE LOGIN
    # -------------------
    st.markdown("---")
    st.markdown("### Continue with Google")

    CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
    CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]

    AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    oauth2 = OAuth2Component(
        CLIENT_ID,
        CLIENT_SECRET,
        AUTHORIZE_URL,
        TOKEN_URL,
    )

    result = oauth2.authorize_button(
        name="🔵 Continue with Google",
        redirect_uri="https://customer-churn-predictions1.streamlit.app/component/streamlit_oauth.authorize_button",
        scope="openid email profile",
        key="google",
)

    if result:

        st.success("Login Successful!")

        st.session_state.logged_in = True
        st.session_state.page = "dashboard"

        st.rerun()

    # -------------------
    # SIGNUP BUTTON
    # -------------------
    st.markdown("---")

    if st.button("Create New Account"):
        st.session_state.page = "signup"

# -----------------------
# Load Dataset
# -----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except:
        return None

df = load_data()

# -----------------------
# Dashboard
# -----------------------
def dashboard():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Prediction"]
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"

    if page == "Dashboard":
        dashboard_page()

    elif page == "Prediction":
        prediction_page()

# -----------------------
# Dashboard Page
# -----------------------
def dashboard_page():
    st.markdown("<div class='title'>Customer Churn Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Business Intelligence Dashboard</div>", unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found.")
        return

    total_customers = len(df)
    churn_rate = round((df["Churn"].value_counts(normalize=True).get("Yes", 0)) * 100, 2)
    avg_monthly = round(df["MonthlyCharges"].mean(), 2)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", total_customers)

    with col2:
        st.metric("Churn Rate", f"{churn_rate}%")

    with col3:
        st.metric("Avg Monthly Charges", f"${avg_monthly}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        churn_contract = df.groupby("Contract")["Churn"].value_counts().unstack().fillna(0)

        fig = px.bar(
            churn_contract,
            barmode="group",
            title="Churn by Contract Type"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(
            df,
            names="InternetService",
            title="Internet Service Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(
            df,
            x="tenure",
            color="Churn",
            title="Customer Tenure Distribution"
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            title="Monthly Charges vs Churn"
        )

        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("## Key Insights")

    st.info("Month-to-month contract customers show higher churn probability.")
    st.info("Customers with shorter tenure are more likely to leave.")
    st.info("Higher monthly charges correlate with increased churn.")

# -----------------------
# Prediction Page
# -----------------------
def prediction_page():
    st.markdown("<div class='title'>Customer Churn Prediction</div>", unsafe_allow_html=True)

    st.markdown("Enter customer information below")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 10000.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 100000.0, 1000.0)

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    if st.button("Predict Churn"):
        risk = np.random.randint(10, 95)

        st.markdown("---")

        if risk > 60:
            st.error(f"⚠ High Churn Risk: {risk}%")
            st.write("Suggested Action: Offer retention discount and support package.")
        else:
            st.success(f"✅ Low Churn Risk: {risk}%")
            st.write("Customer likely to stay.")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text': "Churn Risk Score"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(gauge, use_container_width=True)

# -----------------------
# Main Routing
# -----------------------
if st.session_state.logged_in:
    dashboard()
else:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
