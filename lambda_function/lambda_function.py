import boto3
import csv
import json
import pickle
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
logging.basicConfig(level=logging.INFO)

s3 = boto3.client('s3')
sm_client = boto3.client('sagemaker-runtime', region_name='us-east-1')
endpoint_name = 'respiratory-classification-prod'

def read_preprocessor():
    bucket_name = "sagemaker-project-p-hnneqf6paono"
    object_key = "estimator/preprocessor/preprocessor.pkl"
    
    try:
        # Download the preprocessor file from S3
        preprocessor_object = s3.get_object(Bucket=bucket_name, Key=object_key)
        preprocessor_content = preprocessor_object['Body'].read()

        # Load the preprocessor from the downloaded bytes
        preprocessor = pickle.load(BytesIO(preprocessor_content))
        print("Preprocessor downloaded and loaded successfully.")

    except Exception as e:
        print(f"Failed to download or load the preprocessor. Error: {e}")
        preprocessor = None

    return preprocessor


def handler(event, context):

    preprocessor = read_preprocessor()

    try:
        data_json = json.loads(event['body'])
        
        features = [
            'saturacao', 'antiviral','tp_antivir', 
            'hospital', 'uti','raiox_res', 'dor_abd', 
            'perd_olft', 'tomo_res', 'cs_raca', 'cs_zona', 
            'perd_pala','vacina_cov', 'sem_pri', 'nu_idade_n'
        ]

        for feature in features:
            if feature not in data_json:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': f'Feature "{feature}" is required.'})
                }

        df = pd.DataFrame([data_json], columns=features)
        df[
            [
                'saturacao', 'antiviral', 'tp_antivir',
                'hospital', 'uti', 'raiox_res', 'dor_abd', 'perd_olft', 
                'tomo_res', 'cs_raca', 'cs_zona', 'perd_pala', 'vacina_cov'
            ]
        ] = df[
            [
                'saturacao', 'antiviral', 'tp_antivir',
                'hospital', 'uti', 'raiox_res', 'dor_abd', 'perd_olft', 
                'tomo_res', 'cs_raca', 'cs_zona', 'perd_pala', 'vacina_cov'
            ]
        ].astype('category')
        
        preprocessed_df = preprocessor.transform(df)
        print(preprocessed_df)
        preprocessed_df = ",".join(str(value) for value in preprocessed_df[0]) + "\n"
        print(preprocessed_df)


        # features = "sem_pri,nu_idade_n,saturacao,antiviral,\
        # tp_antivir,hospital,uti,raiox_res,\
        # dor_abd,perd_olft,tomo_res,cs_raca,\
        # cs_zona,perd_pala,vacina_cov"

        #csv_data = ','.join() ','.join(value for value in data_json.values()) + "\n"
        
        
        response = sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=preprocessed_df
        )
        
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            return {
                'statusCode': response['ResponseMetadata']['HTTPStatusCode'],
                'body': json.dumps({'error': 'Error in calling SageMaker endpoint.'})
            }
    
        output_data = response['Body'].read().decode().strip()
        probabilities = np.fromstring(output_data, sep=',')
        predicted_label = int(np.argmax(probabilities) + 1)
        output_data_json = {'output_data': predicted_label}
        
        return {
            'statusCode': 200,
            'body': json.dumps(output_data_json)
        }
        
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Error in decoding JSON'})
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Server error: {str(e)}'})
        }