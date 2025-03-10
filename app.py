import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('titanic_logistic_regression_model.pkl', 'rb'))

st.title("Predict Titanic Passenger Survival")

pclass = st.selectbox("Passenger Class)", [1, 2, 3], index=2)
sex = st.radio("Sex", ['Male', 'Female'])
age = st.number_input("Age")
sibsp = st.number_input("Number of Siblings/Spouses Aboard")
parch = st.number_input("Number of Parents/Children Aboard (Parch)")
fare = st.number_input("Fare Paid")
embarked = st.radio("Port of Embarkation", ['C', 'Q', 'S'])

# One-hot encoding transformation
sex_male = 1 if sex == 'Male' else 0
sex_female = 1 if sex == 'Female' else 0
embarked_C = 1 if embarked == 'C' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Creating a DataFrame for prediction
new_data = np.array([[pclass, age, sibsp, parch, fare, sex_female, sex_male, embarked_C, embarked_Q, embarked_S]])

# Prediction button
if st.button("Predict Survival"):
    prediction = model.predict(new_data)
    result = "Survived!" if prediction[0] == 1 else "Did Not Survive"
    st.subheader(f"Prediction: {result}")
