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
from pickle import dump


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
    
    numerical_cols = X.select_dtypes(['int8']).columns.tolist()
    categorical_cols = X.select_dtypes(['category']).columns.tolist()
    columns_name = categorical_cols + numerical_cols

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

    X_transformed = pd.DataFrame(
        preprocessor.fit_transform(X)
    )
    
    feature_names = preprocessor.get_feature_names_out()
    X_transformed.columns = feature_names
    
    #save preprocessor to use in deployment
    os.makedirs(f'{base_dir}/output/preprocessor', exist_ok=True)
    with open(f'{base_dir}/output/preprocessor/preprocessor.pkl', "wb") as f:
        dump(preprocessor, f)
    
    return X_transformed, y

def save_data(X, y):

    os.makedirs(f'{base_dir}/output/X_transformed', exist_ok=True)
    os.makedirs(f'{base_dir}/output/y_transformed', exist_ok=True)

    X.to_csv(f"{base_dir}/output/X_transformed/X.csv", index=False)
    np.savetxt(f"{base_dir}/output/y_transformed/y.csv", y, delimiter=',')


if __name__ =='__main__':
    df = read_data(base_dir=base_dir)
    X, y = preprocess_data(df)
    save_data(X, y)
    sys.exit(0)