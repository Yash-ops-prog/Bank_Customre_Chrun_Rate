
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
# Make sure the model file 'logistic_regression_model.pkl' is in the same directory as app.py
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'logistic_regression_model.pkl' not found. Please make sure it's in the same directory.")
    st.stop()

# Load the StandardScaler fitted on the training data
# Make sure the scaler file 'scaler.pkl' is in the same directory as app.py
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please make sure it's in the same directory.")
    st.stop()


st.title("Bank Customer Churn Prediction")

st.write("Enter customer details to predict churn:")

credit_score = st.slider("Credit Score", 350, 850, 700)
country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 18, 92, 45)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Balance", value=100000.00)
products_number = st.slider("Number of Products", 1, 4, 2)
credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.number_input("Estimated Salary", value=90000.00)

# Mapping categorical features to numerical as done during training
country_mapping = {"France": 0, "Germany": 1, "Spain": 2}
gender_mapping = {"Female": 0, "Male": 1}

country_encoded = country_mapping[country]
gender_encoded = gender_mapping[gender]

# Create a DataFrame with the new customer data
new_customer_data = pd.DataFrame({
    'credit_score': [credit_score],
    'country': [country_encoded],
    'gender': [gender_encoded],
    'age': [age],
    'tenure': [tenure],
    'balance': [balance],
    'products_number': [products_number],
    'credit_card': [credit_card],
    'active_member': [active_member],
    'estimated_salary': [estimated_salary]
})

# Scale the new customer data
scaled_new_customer_data = scaler.transform(new_customer_data)

# Predict churn
if st.button("Predict Churn"):
    churn_prediction = model.predict(scaled_new_customer_data)

    if churn_prediction[0] == 1:
        st.error("Predicted Churn: Yes")
    else:
        st.success("Predicted Churn: No")
