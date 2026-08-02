import streamlit as st
import joblib
import numpy as np

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏭 Factory Reallocation System")

st.write("Predict Gross Profit using Machine Learning")


# Region
region_map = {
    "East": 0,
    "West": 1,
    "North": 2,
    "South": 3
}

selected_region = st.selectbox(
    "📍 Region",
    list(region_map.keys())
)
region = region_map[selected_region]


# Division
division_map = {
    "Division 1": 0,
    "Division 2": 1,
    "Division 3": 2
}

selected_division = st.selectbox(
    "🏢 Division",
    list(division_map.keys())
)
division = division_map[selected_division]


# Product
product_map = {
    "Product A": 0,
    "Product B": 1,
    "Product C": 2
}

selected_product = st.selectbox(
    "📦 Product",
    list(product_map.keys())
)
product = product_map[selected_product]


# State
state_map = {
    "California": 0,
    "Texas": 1,
    "New York": 2
}

selected_state = st.selectbox(
    "🗺️ State",
    list(state_map.keys())
)
state = state_map[selected_state]
units = st.number_input(
    "📦 Units",
    min_value=1,
    step=1
)

cost = st.number_input(
    "💰 Cost",
    min_value=0,
    step=10
)

month = st.number_input(
    "📅 Month",
    min_value=1,
    max_value=12,
    step=1
)

if st.button("Predict Gross Profit"):

    data = np.array([[region,
                      division,
                      product,
                      state,
                      units,
                      cost,
                      month]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Predicted Gross Profit: £{prediction[0]:.2f}")
    st.write(prediction)
    st.metric("Gross Profit", f"${prediction[0]:.2f}")
    st.markdown("---")
    if prediction[0] > 100:

     st.success("✅ Recommendation: High Profit Order")

    elif prediction[0] > 50:

     st.warning("⚠️ Recommendation: Medium Profit Order")

    else:

     st.error("❌ Recommendation: Low Profit Order")

     st.metric(
    "Predicted Gross Profit",
    f"${prediction[0]:.2f}"
)


     if prediction[0] > 100:
      st.success("✅ High Profit Order\n\nRecommended for processing.")
     elif prediction[0] > 50:
      st.warning("⚠️ Medium Profit Order")
     else:
      st.error("❌ Low Profit Order")

with st.expander("ℹ About this Project"):
    st.write("""
This application predicts Gross Profit using a Random Forest model.
Based on the predicted Gross Profit, it recommends whether the order
is High, Medium, or Low Profit.
""")

    
st.sidebar.title("Factory Reallocation")
st.sidebar.info("AI-powered Gross Profit Prediction")



st.markdown("---")

st.caption(
    "Developed by THOUFEEQ AHMED | Machine Learning Internship Project | 2026"
)