import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer 360 Financial Health", layout="centered")

st.title("🏦 Customer 360 Financial Health Score")
st.write("Evaluate a customer's financial risk based on their behavior")

# ==============================
# USER INPUT
# ==============================
st.header("Enter Customer Details")

income = st.number_input("Income (₹)", min_value=1000, value=50000)
monthly_spend = st.number_input("Monthly Spend (₹)", min_value=0, value=20000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=100000)
missed_payments = st.slider("Missed Payments", 0, 10, 2)
credit_utilization = st.slider("Credit Utilization (0 to 1)", 0.0, 1.0, 0.5)
account_balance = st.number_input("Account Balance (₹)", min_value=0, value=20000)

employment_status = st.selectbox(
    "Employment Status",
    ["employed", "self-employed", "unemployed"]
)

# ==============================
# FEATURE ENGINEERING
# ==============================
savings_ratio = min(account_balance / income, 1)
spend_ratio = monthly_spend / income
income_stability = 1 if employment_status in ["employed", "self-employed"] else 0

# ==============================
# SCORE CALCULATION
# ==============================
def calculate_score():
    score = 0

    # Income stability (20)
    score += income_stability * 20

    # Savings (20)
    score += savings_ratio * 20

    # Spending (15)
    score += max(0, (1 - spend_ratio)) * 15

    # Credit utilization (15)
    score += (1 - credit_utilization) * 15

    # Missed payments (30)
    score += max(0, (1 - missed_payments / 10)) * 30

    return round(score, 2)

# ==============================
# SEGMENT
# ==============================
def get_segment(score):
    if score >= 75:
        return "Excellent"
    elif score >= 50:
        return "Good"
    elif score >= 30:
        return "Risky"
    else:
        return "High Risk"

# ==============================
# BUTTON ACTION
# ==============================
if st.button("Calculate Financial Health Score"):

    score = calculate_score()
    segment = get_segment(score)

    st.subheader("📊 Results")

    st.metric("Financial Health Score", score)
    st.write(f"### Segment: {segment}")

    # ==============================
    # INTERPRETATION
    # ==============================
    st.subheader("📌 Insights")

    if missed_payments > 5:
        st.warning("⚠️ High number of missed payments")

    if credit_utilization > 0.8:
        st.warning("⚠️ High credit utilization")

    if savings_ratio < 0.2:
        st.warning("⚠️ Low savings")

    if income_stability == 0:
        st.warning("⚠️ Unstable income source")

    if segment == "Excellent":
        st.success("✅ Customer is financially strong")
    elif segment == "High Risk":
        st.error("❌ Customer is high risk")
