import streamlit as st
import requests


endpoint = "https://4us3v5rmw2yh4sokm77udggh4q0zqrmd.lambda-url.us-east-1.on.aws/"

st.set_page_config(
    page_title="Medical Form",
)

with st.form(key='form1'):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
            sem_pri =  st.text_input("sem_pri")
            nu_idade_n =  st.text_input("nu_idade_n")
            saturacao =  st.text_input("saturacao")
            antiviral =  st.text_input("antiviral")
            tp_antivir =  st.text_input("tp_antivir")
    with col2:
            hospital =  st.text_input("hospital")
            dose_ref =  st.text_input("dose_ref")
            fnt_in_cov =  st.text_input("fnt_in_cov")
            uti =  st.text_input("uti")
    with col3:
            raiox_res =  st.text_input("raiox_res")
            dor_abd =  st.text_input("dor_abd")
            perd_olft =  st.text_input("perd_olft")
            tomo_res =  st.text_input("tomo_res")
            cs_raca =  st.text_input("cs_raca")
    with col4:
            cs_zona =  st.text_input("cs_zona")
            perd_pala =  st.text_input("perd_pala")
            vacina_cov =  st.text_input("vacina_cov")
    if st.form_submit_button():
        input_data = {
                'sem_pri': sem_pri,
                'nu_idade_n': nu_idade_n,
                'saturacao': saturacao,
                'antiviral': antiviral,
                'tp_antivir': tp_antivir,
                'hospital': hospital,
                'dose_ref': dose_ref,
                'fnt_in_cov': fnt_in_cov,
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
                st.write("Result:")
                st.write(output_data)
        else:
                st.write("Error in API request")