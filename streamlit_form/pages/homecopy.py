import streamlit as st
from datetime import datetime

# Title for the form
st.title('SARS Suspected Case Form')

# Patient Identification Section
st.header('Patient Identification')
name = st.text_input('Name')
age = st.number_input('Age', step=1, format='%d')
sex = st.selectbox('Sex', ['Male', 'Female', 'Other'])
contact_info = st.text_input('Contact Information')

# Symptoms Section
st.header('Symptoms')
symptoms_list = ['Dyspnea (difficulty breathing)', 'Persistent pressure in the chest', 'Oxygen saturation below 95% in room air', 'Bluish coloration of the lips or face']
symptoms = st.multiselect('Check all that apply', symptoms_list)

# Date of Symptom Onset
symptom_onset = st.date_input('Date of Symptom Onset', datetime.now())

# Hospitalization Details
st.header('Hospitalization Details')
hospitalization_date = st.date_input('Date of Hospitalization', datetime.now())
hospital_info = st.text_input('Hospital Information')

# Underlying Conditions
st.header('Underlying Conditions')
conditions_list = ['Diabetes', 'Hypertension', 'Heart Disease', 'Others']
conditions = st.multiselect('Check all that apply', conditions_list)

# Epidemiological Information
st.header('Epidemiological Information')
travel_history = st.text_input('Travel History')
contact_with_cases = st.text_input('Contact with Confirmed Cases')