import argparse
import pandas as pd
import os
import xgboost as xgb
from joblib import load, dump
import logging
import json
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score as calculate_f1_score

logging.basicConfig(level=logging.INFO)

base_dir = "/opt/ml/processing"
base_dir_evaluation = f"{base_dir}/evaluation" 

def train(train=None, test=None):
    """
        Trains a model using the specified algorithm with given parameters.
    
        Args:
            train : location on the filesystem for training dataset
            test: location on the filesystem for test dataset

        Returns:
            trained model object
    """
    # Take the set of files and read them all into a single pandas dataframe
    train_files = [os.path.join(train, file) for file in os.listdir(train)]
    if test:
        test_files = [os.path.join(test, file) for file in os.listdir(test)]
    if len(train_files) == 0 or (test and len(test_files)) == 0:
        raise ValueError(
            (f'There are no files in {train}.\n' +
            'This usually indicates that the channel train was incorrectly specified,\n' +
            'the data specification in S3 was incorrectly specified or the role specified\n' +
            'does not have permission to access the data.')
        )

    X_train = pd.read_csv(f'{train}/train_x.csv', sep=',', header=0)
    y_train = pd.read_csv(f'{train}/train_y.csv', sep=',', header=None)

    clf = xgb.XGBClassifier(
        eta=args.eta, 
        max_depth=args.max_depth, 
        gamma=args.gamma,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        verbose=1,
        objective="multi:softmax",
        random_state=42
    ).fit(X_train, y_train)

    return clf
    
def evaluate(test=None, model=None):
    """
        Evaluates the performance for the given model.
    
        Args:
        test: location on the filesystem for test dataset 
    """
    if test:
        X_test = pd.read_csv(f'{test}/test_x.csv', delimiter=',', header=0)
        y_test = pd.read_csv(f'{test}/test_y.csv', delimiter=',', header=None)
        y_predicted = model.predict(X_test)
        cohen_score = cohen_kappa_score(y_test, y_predicted)
        f1_score = calculate_f1_score(y_test, y_predicted, average='macro')
        
        
        logging.info(f'model cohen score:{cohen_score};')
        logging.info(f"model F1 score:{f1_score};")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()
    model = train(train=args.train, test=args.test)

    evaluate(test=args.test, model=model)
    dump(model, os.path.join(args.model_dir, "model.joblib"))
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = load(os.path.join(model_dir, "model.joblib"))
    return clf