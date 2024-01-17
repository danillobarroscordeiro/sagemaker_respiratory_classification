import argparse
import pandas as pd
from numpy import genfromtxt
import os
from xgboost import XGBClassifier 
from joblib import load, dump
import logging
import json
import boto3
from sklearn.metrics import cohen_kappa_score, f1_score

logging.basicConfig(level=logging.INFO)

s3 = boto3.client("s3")
base_dir = "/opt/ml/processing"
base_dir_evaluation = f"{base_dir}/evaluation"
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
    ).fit(X, y)

    return clf

def read_hyperparameters(jobinfo=None):
    with open(f"{jobinfo}/jobinfo.json", "rb") as file:
        jobinfo_file = json.load(file)
    return jobinfo_file['hyperparams']

def read_preprocessor():
    bucket_name = "sagemaker-traintest-respiratory-classification"
    path = "estimator/preprocessor/preprocessor.joblib"
    
    with open('encoder.joblib', 'wb') as f:
        s3.download_fileobj(bucket_name, path, f)
        preprocessor = load("preprocessor.joblib")
    
    return preprocessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--jobinfo', type=str, default=os.environ['SM_CHANNEL_JOBINFO'])

    args = parser.parse_args()
    hyperparams = read_hyperparameters(jobinfo=args.jobinfo)
    
    # Load preprocessor to save inside the model
    preprocessor = read_preprocessor()

    model = train(train=args.train, hyperparams=hyperparams)
    dump(preprocessor, os.path.join(args.model_dir, "preprocessor.joblib"))
    dump(model, os.path.join(args.model_dir, "model.joblib"))

# def model_fn(model_dir):
#     """Deserialized and return fitted model
#     Note that this should have the same name as the serialized model in the main method
#     """
#     clf = load(os.path.join(model_dir, "model.joblib"))
#     return clf