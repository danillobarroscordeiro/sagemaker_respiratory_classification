import streamlit as st
import requests


endpoint = "https://4us3v5rmw2yh4sokm77udggh4q0zqrmd.lambda-url.us-east-1.on.aws/"

st.set_page_config(
        page_title="SARS Suspected Case Form"
)
st.title('SARS Suspected Case Form')

with st.form(key='form1'):
        # Patient Identification Section
        st.header('Demographic Section')
        nu_idade_n = st.number_input('Age', step=1, format='%d')

        cs_raca = st.selectbox(
        "Race",
        (1, 2, 3, 4, 5),
        )
        st.write('1 - White')
        st.write('2 - Black')
        st.write('3 - Yellow')
        st.write('4 - Mixed Race')
        st.write('5 - Brazilian Indian')

        cs_zona = st.selectbox(
        "Geografic area of living",
        (1, 2, 3, 9)
        )
        st.write('1 - Urban')
        st.write('2 - Rural')
        st.write('3 - Peri-urban')
        st.write('9 - Other')


        # Symptoms Section
        st.header('Symptoms')
        saturacao = st.selectbox(
        "Blood oxygen saturation is below 95%?",
        (1, 2)
        )
        st.write('1 - Yes')
        st.write('2 - No')

        antiviral = st.selectbox(
        "Patient took flu antiviral?",
        (1, 2)
        )
        st.write('1 - Yes')
        st.write('2 - No')

        tp_antivir = st.selectbox(
        "Which antiviral flu?",
        (1, 2, 3)
        )
        st.write('1 - Oseltamivir')
        st.write('2 - Zanamivir')
        st.write('3 - Other')

        raiox_res = st.selectbox(
        "Chest X-ray result",
        (1, 2, 3, 4, 5, 6)
        )
        st.write('1 - Normal')
        st.write('2 - Interstitial infiltrate')
        st.write('3 - Consolidation')
        st.write('4 - Mixed')
        st.write('5 - Other')
        st.write('6 - Not realized')

        dor_abd = st.selectbox(
        "Has abdominal pain?",
        (1, 2)
        )
        st.write('1 - Yes')
        st.write('2 - No')

        perd_olft = st.selectbox(
        "Loss or change in sense of smell?",
        (1, 2)
        )
        st.write('1 - Yes')
        st.write('2 - No')

        perd_pala= st.selectbox(
        "Loss or change in sense of taste?",
        (1, 2)
        )
        st.write('1 - Yes')
        st.write('2 - No')

        tomo_res = st.selectbox(
        "Torax Tomography result",
        (1, 2, 3, 4, 5, 6)
        )
        st.write('1 - Typical Covid 19')
        st.write('2 - Undetermined Covid 19')
        st.write('3 - Atypical Covid 19')
        st.write('4 - Negative for pneumonia')
        st.write('5 - Other')
        st.write('6 - Not realized')

        vacina_cov= st.selectbox(
        "Took covid vaccine?",
        (1, 2)
        )
        st.write('1 - Yes')
        st.write('2 - No')

        # Hospitalization Details
        st.header('Hospitalization Details')
        hospital = st.selectbox(
        "Pacient was hospitalized?",
        (1, 2)
        )
        st.write('1 - Yes')
        st.write('2 - No')

        uti = st.selectbox(
        "Pacient is or was in Intensive Care Unit (ICU)?",
        (1, 2)
        )
        st.write('1 - Yes')
        st.write('2 - No')

        # Epidemiological Information
        st.header('Epidemiological Information')
        sem_pri = st.number_input(
        'Epidemiological week of the first symptoms', 
        step=1, 
        format='%d'
        )

        if st.form_submit_button():
                input_data = {
                        'sem_pri': sem_pri,
                        'Age': nu_idade_n,
                        'saturacao': saturacao,
                        'antiviral': antiviral,
                        'tp_antivir': tp_antivir,
                        'hospital': hospital,
                        'uti': uti,
                        'raiox_res': raiox_res,
                        'dor_abd': dor_abd,
                        'perd_olft': perd_olft,
                        'tomo_res': tomo_res,
                        'cs_raca': cs_raca,
                        'cs_zona': cs_zona,
                        'perd_pala': perd_pala,
                        'vacina_cov': vacina_cov
                }
        
                response = requests.post(endpoint, json=input_data)
        
                if response.status_code == 200:
                        output_data = response.json()
                        st.write("Result: SARS caused by")
                        st.write(output_data)
                        print(output_data)
                        if output_data == 1:
                                st.write("Result: SARS caused by Influenza")
                        elif output_data == 2:
                                st.write("Result: SARS caused by respiratory virus")
                        elif output_data == 3:
                                st.write("Result: SARS caused by other etiologic agent")
                        elif output_data == 4:
                                st.write("Result: SARS does not specified")
                        else:
                                st.write("Result: SARS caused by COVID-19")                                
                        
                else:
                        st.write("Error in API request")