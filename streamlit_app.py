import streamlit as st
import numpy as np
import pickle

# Load the trained diabetes model
model = pickle.load(open('diabetes_model.sav', 'rb'))

# Streamlit app title
st.title('Diabetes Prediction App')

# Input fields for the diabetes prediction model
pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
glucose = st.number_input('Glucose Level', min_value=0.0)
blood_pressure = st.number_input('Blood Pressure Level', min_value=0.0)
skin_thickness = st.number_input('Skin Thickness', min_value=0.0)
insulin = st.number_input('Insulin Level', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0, step=1)

# Collecting input data in a list
input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

# Code for prediction
result = ''

# Creating a button for prediction
if st.button('Predict'):
    # Converting input data to a NumPy array and reshaping it for prediction
    input_data = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    
    # Displaying the result based on the prediction
    if prediction[0] == 1:
        result = 'The model predicts that the person has diabetes.'
    else:
        result = 'The model predicts that the person does not have diabetes.'

# Displaying the prediction result
st.success(result)
