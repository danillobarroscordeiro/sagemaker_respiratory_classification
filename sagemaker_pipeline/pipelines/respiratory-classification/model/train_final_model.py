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

logging.basicConfig(level=logging.INFO)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--jobinfo', type=str, default=os.environ['SM_CHANNEL_JOBINFO'])

    args = parser.parse_args()
    hyperparams = read_hyperparameters(jobinfo=args.jobinfo)

    booster = train(train=args.train, hyperparams=hyperparams)
    model_location = f"{args.model_dir}/xgboost-model.pkl"
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