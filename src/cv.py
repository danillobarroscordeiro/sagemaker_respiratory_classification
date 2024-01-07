#!/usr/bin/env python
import os
import boto3
import re
import json
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import argparse
import time
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO)

def fit_model(
        instance_type, 
        output_path,
        s3_train_base_dir,
        s3_test_base_dir,
        f, 
        eta, 
        max_depth, 
        gamma,
        min_child_weight,
        subsample
    ):
    """Fits a model using the specified algorithm.
    
        Args:
            instance_type: instance to use for Sagemaker Training job
            output_path: S3 URI as the location for the trained model artifact
            s3_train_base_dir: S3 URI for train datasets
            s3_test_base_dir: S3 URI for test datasets
            f: index represents a fold number in the K fold cross validation
            c: regularization parameter for SVM
            gamma: kernel coefficiency value
            kernel: kernel type for SVM algorithm

        Returns: 
            Sagemaker Estimator created with given input parameters.
    """
    sklearn_framework_version='1.2-1'
    script_path = 'train.py'

    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    sklearn_estimator = SKLearn(
        entry_point=script_path,
        instance_type=instance_type,
        framework_version=sklearn_framework_version,
        role=role,
        keep_alive_period_in_seconds=600,
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        hyperparameters={
            'eta': eta,
            'max_depth' : max_depth,
            'gamma': gamma,
            "min_child_weight": min_child_weight,
            "subsample": subsample
        },
        metric_definitions=[
            { "Name": "test:f1_score", "Regex": "model F1 score:(.*?);"},
            { "Name": "test:cohen_score", "Regex": "model cohen score:(.*?);"}
        ]
    )
    sklearn_estimator.fit(
        inputs = { 'train': f'{s3_train_base_dir}/{f}',
        'test':  f'{s3_test_base_dir}/{f}'},
        wait=False
    )
    return sklearn_estimator

def monitor_training_jobs(training_jobs, sm_client):
    """Monitors the submit training jobs for completion.
    
        Args: 
            training_jobs: array of submitted training jobs
            sm_client: boto3 sagemaker client

    """
    all_jobs_done = False
    while not all_jobs_done:
        completed_jobs = 0
        for job in training_jobs:
            job_detail = sm_client.describe_training_job(TrainingJobName=job._current_job_name)
            job_status = job_detail['TrainingJobStatus']
            if job_status.lower() in ('completed', 'failed', 'stopped'):
                completed_jobs += 1
        if completed_jobs == len(training_jobs):
            all_jobs_done = True
        else:
            time.sleep(30)

def evaluation(training_jobs, sm_client):
    """
    Evaluates and calculate the performance for the cross validation training jobs.
    
        Args:
            training_jobs: array of submitted training jobs
            sm_client: boto3 sagemaker client

        Returns:
            Average score from the training jobs collection in the given input
    """
    scores = {'test:f1_score': [], 'test:cohen_score': []}
    for job in training_jobs:
        job_detail = sm_client.describe_training_job(TrainingJobName=job._current_job_name)
        metrics = job_detail['FinalMetricDataList']
        f1_score = [x['Value'] for x in metrics if x['MetricName'] == 'test:f1_score'][0]
        scores['test:f1_score'].append(f1_score)
        cohen_score = [x['Value'] for x in metrics if x['MetricName'] == 'test:cohen_score'][0]
        scores['test:cohen_score'].append(cohen_score)
        
    np_f1_scores = np.array(scores['test:f1_score'])
    np_cohen_scores = np.array(scores['test:cohen_score'])
    
    # Calculate the score by taking the average score across the performance of the training job
    score_f1_avg = np.average(np_f1_scores)
    score_cohen_avg = np.average(np_cohen_scores)
    score_f1_std = np.std(np_f1_scores)
    score_cohen_std = np.std(np_cohen_scores)
    logging.info(f'average model f1 test score:{score_f1_avg};')
    logging.info(f'average model cohen test score:{score_cohen_avg};')
    logging.info(f'std model f1 test score:{score_f1_std};')
    logging.info(f'std model cohen test score:{score_cohen_std};')    
    return score_f1_avg, score_cohen_avg, score_f1_std, score_cohen_std

def train():
    """
    Trains a Cross Validation Model with the given parameters.
    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('-k', '--k', type=int, default=3)
    parser.add_argument('--train_src', type=str)
    parser.add_argument('--test_src', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--instance_type', type=str, default="ml.c4.xlarge")
    parser.add_argument('--region', type=str, default="us-east-1")
    
    args = parser.parse_args()

    os.environ['AWS_DEFAULT_REGION'] = args.region
    sm_client = boto3.client("sagemaker")
    training_jobs = []
    # Fit k training jobs with the specified parameters.
    for f in range(args.k):
        sklearn_estimator = fit_model(
            instance_type=args.instance_type,
            output_path=args.output_path,
            s3_train_base_dir=args.train_src,
            s3_test_base_dir=args.test_src,
            f=f,
            eta=args.eta,
            max_depth=args.max_depth, 
            gamma=args.gamma,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
        )
        training_jobs.append(sklearn_estimator)
        time.sleep(5) # avoid Sagemaker Training Job API throttling

    monitor_training_jobs(training_jobs=training_jobs, sm_client=sm_client)
    f1_score, cohen_score = evaluation(training_jobs=training_jobs, sm_client=sm_client)
    return f1_score, cohen_score

if __name__ == '__main__':
    train()
    sys.exit(0)