import argparse
import pandas as pd
from numpy import genfromtxt
import os
from io import StringIO
from xgboost import XGBClassifier 
from pickle import load, dump
import logging
import json
import boto3
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from xgboost import DMatrix
logging.basicConfig(level=logging.INFO)

# s3 = boto3.client("s3")
# base_dir = "/opt/ml/processing"
# base_dir_evaluation = f"{base_dir}/evaluation"
#base_dir_jobinfo =  "/opt/ml/input/data/jobinfo"

def train(train=None, hyperparams=None):
    """
        Trains a model using the specified algorithm with given parameters.
        Args:
            train : location on the filesystem for training dataset
        Returns:
            trained model object
    """


    X = pd.read_csv(f'{train}/X.csv', sep=',')
    y = genfromtxt(f'{train}/y.csv', delimiter=',')

    clf = XGBClassifier(
        eta=float(hyperparams['eta']), 
        max_depth=int(hyperparams['max_depth']), 
        gamma=float(hyperparams['gamma']),
        min_child_weight=int(hyperparams['min_child_weight']),
        subsample=float(hyperparams['subsample']),
        verbose=1,
        objective="multi:softmax",
        random_state=42
    )
    
    clf.fit(X, y)
    
    booster = clf.get_booster()

    return booster

def read_hyperparameters(jobinfo=None):
    with open(f"{jobinfo}/jobinfo.json", "rb") as file:
        jobinfo_file = json.load(file)
    return jobinfo_file['hyperparams']

# def read_preprocessor():
#     bucket_name = "sagemaker-traintest-respiratory-classification"
#     path = "estimator/preprocessor/preprocessor.pkl"
#     local_filename = "preprocessor.pkl"
    
#     try:
#         # Download the preprocessor file from S3
#         with open(local_filename, 'wb') as f:
#             s3.download_fileobj(bucket_name, path, f)

#         # Load the preprocessor from the downloaded file
#         with open(local_filename, 'rb') as f:
#             preprocessor = load(f)
#         print(f"File {local_filename} downloaded and loaded successfully.")
#         logging.info(f"File {local_filename} downloaded and loaded successfully.")

#     except Exception as e:
#         print(f"Failed to download or load {local_filename}. Error: {e}")
#         preprocessor = None


#     return preprocessor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--jobinfo', type=str, default=os.environ['SM_CHANNEL_JOBINFO'])

    args = parser.parse_args()
    hyperparams = read_hyperparameters(jobinfo=args.jobinfo)
    
    ## Load preprocessor to save inside the model
    #preprocessor = read_preprocessor()

    booster = train(train=args.train, hyperparams=hyperparams)
    #dump(preprocessor, os.path.join(args.model_dir, "preprocessor.joblib"))
    #dump(model, os.path.join(args.model_dir, "model.pkl"))
    model_location = f"{args.model_dir}/xgboost-model.pkl"
    #save the xgboost.Booster file
    with open(model_location, "wb") as model_file:
        dump(booster, model_file)
    logging.info("Stored trained model at {}".format(model_location))

def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    with open(os.path.join(model_dir, "xgboost-model.pkl"), "rb") as model_file:
        booster = load(model_file)
    return booster

# def input_fn(request_body, request_content_type):
#     """
#     The SageMaker XGBoost model server receives the request data body and the content type,
#     and invokes the `input_fn`.

#     Return a DMatrix (an object that can be passed to predict_fn).
#     """
#     if request_content_type == "text/csv":
#         data = pd.read_csv(StringIO(request_body), header=0)
#     else:
#         raise ValueError(f"Unsupported content type: {request_content_type}")
        
#     preprocessor = read_preprocessor()
    
#     transformed_data = preprocessor.transform(data)
#     print(transformed_data)
#     transformed_data_csv = transformed_data.to_csv(header=False, index=False)
#     print(transformed_data_csv)
    
#     return DMatrix(transformed_data_csv)
    
#     # return xgb_encoders.csv_to_dmatrix(
#     #     pd.to_csv(transformed_data, header=False, index=False)
#     # )

# def predict_fn(input_object, model):
#     """
#     SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

#     Return a two-dimensional NumPy array where the first columns are predictions
#     and the remaining columns are the feature contributions (SHAP values) for that prediction.
#     """
#     predictions = model.predict(input_object)
    
#     return predictions

# def output_fn(predictions, content_type):
#     """
#     After invoking predict_fn, the model server invokes `output_fn`.
#     """
#     if content_type == "text/csv":
#         return ",".join(str(x) for x in predictions)
#     else:
#         raise ValueError("Content type {} not supported.".format(content_type))