import streamlit as st
import joblib
import numpy as np
import pandas as pd
from iv_statistics import GroupedMedianImputer, ColumnDropper

model = joblib.load('catboost_model.pkl')

st.title("Stroke Prediction ML Model App")

age = st.number_input("Enter age:", min_value=0, max_value=120, step=1)
bmi = st.number_input("Enter BMI:", min_value=10.0, max_value=100.0, step=0.1)
avg_glucose_level = st.number_input("Enter average glucose level:", min_value=0.0, max_value=300.0, step=0.1)

heart_disease = st.radio("Do you have heart disease?", options=[0, 1])
hypertension = st.radio("Do you have hypertension?", options=[0, 1])

gender = st.selectbox("Select gender:", ['Male', 'Female', 'Other'])
ever_married = st.selectbox("Have you ever been married?", ['Yes', 'No'])
Residence_type = st.selectbox("Select residence type:", ['Urban', 'Rural'])
work_type = st.selectbox("Select work type:", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
smoking_status = st.selectbox("Select smoking status:", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [Residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

input_data['age_bin'] = pd.cut(input_data['age'], 
                                bins=[0, 18, 30, 45, 60, 100], 
                                labels=['0-18', '19-30', '31-45', '46-60', '60+'])

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"The predicted result is: {prediction[0]}")