import streamlit as st
import pickle
import pandas as pd
import sklearn

# Load the trained model
with open("my_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #cce5ff; padding: 10px; color: #003366;'><b>Loan Approval Predictor</b></h1>",
    unsafe_allow_html=True
)

# Lender payouts from the case study
LENDER_PAYOUT = {"A": 250, "B": 350, "C": 150}

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numeric values
fico = st.slider("FICO Score", min_value=300, max_value=850, step=10)
income = st.slider("Monthly Gross Income ($)", min_value=0, max_value=50000, step=100)
housing = st.slider("Monthly Housing Payment ($)", min_value=0, max_value=20000, step=50)
loan = st.slider("Granted Loan Amount ($)", min_value=1000, max_value=1000000, step=1000)

# Categorical inputs with options
reason = st.selectbox("Loan Reason (REASON)", ["debt_conslidation", "credit_card_refinancing",
                                                "home_improvement", "major_purchase",
                                                "cover_an_unexpected_cost", "other"])
employment = st.selectbox("Employment Status", ["full_time", "part_time", "unemployed"])
sector = st.selectbox("Employment Sector", ["information_technology", "financials", "health_care",
                                              "industrials", "consumer_discretionary",
                                              "consumer_staples", "communication_services",
                                              "energy", "materials", "real_estate",
                                              "utilities", "Unknown"])
bankrupt_label = st.selectbox("Ever Bankrupt or Foreclosed?", ["No", "Yes"])
bankrupt = 1 if bankrupt_label == "Yes" else 0


def get_fico_group(score):
    if score < 580:  return "poor"
    if score < 670:  return "fair"
    if score < 740:  return "good"
    if score < 800:  return "very_good"
    return "excellent"


def build_input(lender):
    input_data = pd.DataFrame({
        "Granted_Loan_Amount":        [loan],
        "FICO_score":                 [fico],
        "Monthly_Gross_Income":       [income],
        "Monthly_Housing_Payment":    [housing],
        "Ever_Bankrupt_or_Foreclose": [bankrupt],
        "Reason":                     [reason],
        "Fico_Score_group":           [get_fico_group(fico)],
        "Employment_Status":          [employment],
        "Employment_Sector":          [sector],
        "Lender":                     [lender]
    })

    input_data_encoded = pd.get_dummies(input_data, columns=['Reason', 'Fico_Score_group',
                                                              'Employment_Status',
                                                              'Employment_Sector', 'Lender'])

    bool_cols = input_data_encoded.select_dtypes('bool').columns
    input_data_encoded[bool_cols] = input_data_encoded[bool_cols].astype(int)

    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    input_data_encoded = input_data_encoded[model_columns]
    return input_data_encoded


# Predict button
if st.button("Evaluate Loan"):
    results = []
    for lender in ["A", "B", "C"]:
        X_input = build_input(lender)
        proba = model.predict_proba(X_input)[0, 1]
        payout = LENDER_PAYOUT[lender]
        expected_revenue = proba * payout
        results.append({
            "Lender": lender,
            "Approval Probability": f"{proba * 100:.1f}%",
            "Bounty": f"${payout}",
            "Expected Revenue": f"${expected_revenue:.2f}",
            "_revenue_value": expected_revenue
        })

    results_df = pd.DataFrame(results)

    best_idx = results_df["_revenue_value"].idxmax()
    best_lender = results_df.loc[best_idx, "Lender"]
    best_revenue = results_df.loc[best_idx, "Expected Revenue"]

    st.subheader("Recommendation")
    st.write(f"Route this applicant to **Lender {best_lender}** — Expected Revenue: **{best_revenue}**")

    st.dataframe(results_df.drop(columns=["_revenue_value"]), hide_index=True)
