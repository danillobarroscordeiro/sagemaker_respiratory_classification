from sagemaker.estimator import Estimator
import sagemaker
import argparse
import os
import json
import boto3
from sagemaker.tuner import ParameterRange, CategoricalParameter, ContinuousParameter, HyperparameterTuner

base_dir = "/opt/ml/processing"
base_dir_evaluation = f"{base_dir}/evaluation" 
base_dir_jobinfo = f"{base_dir}/jobinfo" 

def train(
    train=None, 
    test=None, 
    image_uri=None, 
    instance_type="ml.c5.2xlarge", 
    instance_count=1, 
    output_path=None,
    k = 3,
    max_jobs=30,
    max_parallel_jobs=3,
    eta = 0.1,
    max_depth = 3,
    gamma=1,
    min_child_weight=5,
    subsample=1,
    region = "us-east-1",
    role=None):
    
    """
    Triggers a sagemaker automatic hyperparameter tuning optimization job to train and evaluate a given algorithm. 
    Hyperparameter tuner job triggers maximum number of training jobs with the given maximum parallel jobs per batch. 
    Each training job triggered by the tuner would trigger k cross validation model training jobs.

    Args:
        train: S3 URI where the training dataset is located
        test: S3 URI where the test dataset is located
        image_uri: ECR repository URI for the training image
        instance_type: Instance type to be used for the Sagemaker Training Jobs.
        instance_count: number of instances to be used for the Sagemaker Training Jobs.
        output_path: S3 URI for the output artifacts generated in this script.
        k: number of k in Kfold cross validation
        max_jobs: Maximum number of jobs the HyperparameterTuner triggers
        max_parallel_jobs: Maximum number of parallel jobs the HyperparameterTuner trigger in one batch.
    """
    sagemaker_session = sagemaker.session.Session()
    sm_client = boto3.client("sagemaker")
    # role = sagemaker.get_execution_role()

    # An Estimator object to be associated with the HyperparameterTuner job. 
    cv_estimator = Estimator(
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=instance_count,
        role=role,
        sagemaker_session=sagemaker_session,
        output_path=output_path
    )

    cv_estimator.set_hyperparameters(
        train_src = train,
        test_src = test,
        k = k,
        instance_type = instance_type,
        region = region)

    hyperparameter_ranges = {
        'eta': sagemaker.tuner.ContinuousParameter(0.1, 0.5),
        'max_depth': sagemaker.tuner.IntegerParameter(2, 9),
        'gamma': sagemaker.tuner.IntegerParameter(3, 10),
        'min_child_weight': sagemaker.tuner.IntegerParameter(8, 15),
        'subsample': sagemaker.tuner.ContinuousParameter(0.5, 0.7)
    }

    objective_metric_name = "test:F1_score_average"
    tuner = HyperparameterTuner(
        cv_estimator,
        objective_metric_name,
        hyperparameter_ranges,
        objective_type="Maximize",
        max_jobs=max_jobs,
        strategy="Bayesian",
        base_tuning_job_name="respiratory-classifi",
        max_parallel_jobs=max_parallel_jobs,
        metric_definitions=[
            {
                "Name": objective_metric_name, 
                "Regex": "average model f1 test score:(.*?);"
            },
            {
                "Name": 'test:Cohen_score_average', 
                "Regex": "average model cohen test score:(.*?);"
            },
            {
                "Name": 'test:F1_score_std', 
                "Regex": "std model f1 test score:(.*?);"
            },
            {
                "Name": 'test:Cohen_score_std', 
                "Regex": "std model cohen test score:(.*?);"
            }
        ]
    )

    tuner.fit({"train": train, "test": test})

    best_traning_job_name = tuner.best_training_job()
    tuner_job_name = tuner.latest_tuning_job.name  
    best_performing_job = sm_client.describe_training_job(TrainingJobName=best_traning_job_name)

    hyper_params = best_performing_job['HyperParameters']
    best_hyperparams = {k:v for k,v in hyper_params.items() if not k.startswith("sagemaker_")}

    jobinfo = {}
    jobinfo['name'] = tuner_job_name
    jobinfo['best_training_job'] = best_traning_job_name
    jobinfo['hyperparams'] = best_hyperparams
    f1_value_avg = [
        x['Value'] for x in best_performing_job['FinalMetricDataList'] 
        if x['MetricName'] == objective_metric_name
    ][0]
    cohen_value_avg = [
        x['Value'] for x in best_performing_job['FinalMetricDataList'] 
        if x['MetricName'] == 'test:Cohen_score'
    ][0]
    f1_value_std = [
    x['Value'] for x in best_performing_job['FinalMetricDataList'] 
    if x['MetricName'] == 'test:F1_score_std'
    ][0]
    cohen_value_std = [
    x['Value'] for x in best_performing_job['FinalMetricDataList'] 
    if x['MetricName'] == 'test:cohen_score_std'
    ][0]

    evaluation_metrics = {
        "multiclass_classification_metrics": {
            "F1 score" : {
                "value" : f1_value_avg,
                "standard_deviation" : f1_value_std
            },
            "Cohen score" : {
                "value" : cohen_value_avg,
                "standard_deviation" : cohen_value_std
            },
        }
    }
    os.makedirs(base_dir_evaluation, exist_ok=True) 
    with open(f'{base_dir_evaluation}/evaluation.json', 'w') as f:
        f.write(json.dumps(evaluation_metrics))

    with open(f'{base_dir_jobinfo}/jobinfo.json', 'w') as f:
        f.write(json.dumps(jobinfo))

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--k', type=int, default=3)
    parser.add_argument('--image-uri', type=str)    
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--instance-type', type=str, default="ml.c4.xlarge")
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--max-jobs', type=int, default=30)
    parser.add_argument('--max-parallel-jobs', type=int, default=3)
    parser.add_argument('--region', type=str, default="us-east-1")
    parser.add_argument('--role', type=str)
    
    args = parser.parse_args()
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    train(
        train=args.train, 
        test=args.test, 
        image_uri=args.image_uri, 
        instance_type=args.instance_type, 
        instance_count=args.instance_count,
        output_path=args.output_path,
        k=args.k,
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel_jobs,
        region = args.region,
        role=args.role
    )
