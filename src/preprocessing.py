import numpy as np
import os
import argparse
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
import logging


# Defines the base target directory for datasets.
base_dir = "/opt/ml/processing"

def read_data(base_dir):
    path = f"{base_dir}/input/df_cleaned.csv"
    df = pd.read_csv(path, delimiter=",")
    return df

def preprocess_data(df):

    int_cols = df.select_dtypes(['int64']).drop(['sem_pri','nu_idade_n'], axis=1).columns
    df[int_cols] = df[int_cols].astype('category')
    
    df['sem_pri'] = df['sem_pri'].astype('int8')
    df['nu_idade_n'] = df['nu_idade_n'].astype('int8')

    X = df.drop(['id','classi_fin'], axis=1)
    y = df['classi_fin'].astype('int8')

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y), index=y.index)
    
    return X, y


def save_kfold_datasets(X, y, k):
    """ Splits the datasets (X,y) k folds and saves the output from 
    each fold into separate directories.

    Args:
        X : numpy array represents the features
        y : numpy array represents the target
        k : int value represents the number of folds to split the given datasets
    """

    # Shuffles and Split dataset into k folds. Using fixed random state 
    # for repeatable dataset splits.
    kf = KFold(n_splits=k, random_state=42, shuffle=True)

    numerical_cols = X.select_dtypes(['int8']).columns.tolist()
    categorical_cols = X.select_dtypes(['category']).columns.tolist()
    columns_name = categorical_cols + numerical_cols
    

    fold_idx = 0
    for train_index, test_index in kf.split(X, y=y, groups=None):    
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        categorical_transformer = Pipeline(
            steps=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        numerical_transformer = Pipeline(
            steps=[
                ('encoder', MinMaxScaler())
            ]    
        )

        preprocessor = ColumnTransformer(
            [
                ('cat', categorical_transformer, categorical_cols),
                ('num', numerical_transformer, numerical_cols)
            ]   
        )

        X_train_transformed = pd.DataFrame(
            preprocessor.fit_transform(X_train)
        )

        feature_names = preprocessor.get_feature_names_out()
        X_train_transformed.columns = feature_names

        X_test_transformed = pd.DataFrame(
            preprocessor.transform(X_test)
        )
        X_test_transformed.columns = feature_names
        
        os.makedirs(f'{base_dir}/data/train/{fold_idx}', exist_ok=True)
        X_train_transformed.to_csv(f"{base_dir}/data/train/{fold_idx}/train_x.csv", index=False)
        np.savetxt(f'{base_dir}/data/train/{fold_idx}/train_y.csv', y_train, delimiter=',')

        os.makedirs(f'{base_dir}/data/test/{fold_idx}', exist_ok=True)
        X_test_transformed.to_csv(f"{base_dir}/data/test/{fold_idx}/test_x.csv", index=False)
        np.savetxt(f'{base_dir}/data/test/{fold_idx}/test_y.csv', y_test, delimiter=',')
        fold_idx += 1
    
def process(k):
    """Performs preprocessing by splitting the datasets into k folds.
    """
    df = read_data(base_dir=base_dir)
    X, y = preprocess_data(df)
    save_kfold_datasets(X,y,k)
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k', type=int, default=3)    
    args = parser.parse_args()
    process(k=args.k)
    sys.exit(0)