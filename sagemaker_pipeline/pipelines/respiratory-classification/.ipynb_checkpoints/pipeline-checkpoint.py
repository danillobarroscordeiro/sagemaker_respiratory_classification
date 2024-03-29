from typing import List
import os
import boto3
import io
import json
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.sklearn.processing import SKLearnProcessor, ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import ContinuousParameter, IntegerParameter
from sagemaker import session
from sagemaker.workflow.properties import PropertyFile
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep 
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.execution_variables import ExecutionVariables

base_dir = os.path.dirname(os.path.realpath(__file__))

processing_instance_count = 1
processing_instance_type = f"ml.m5.xlarge"
training_instance_type = f"ml.m5.xlarge"
training_instance_count = 1
inference_instance_type = "ml.c5.xlarge"
hpo_tuner_instance_type = "ml.m5.xlarge"
inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
transform_instances=["ml.m5.xlarge"]
baseline_model_objective_value = 0.4
image_uri_tuning = f"513734873949.dkr.ecr.us-east-1.amazonaws.com/respiratory_classification_pipeline:latest"
image_uri_model = f"513734873949.dkr.ecr.us-east-1.amazonaws.com/respiratory_classification_model:latest"
k = 3
max_tuning_jobs = 15
max_parallel_jobs = 2
model_package_group_name = f"respiratory-virus-classification"
framework_version_sklearn = f"1.2-1"
pipeline_name = f"RespiratoryClassificationPipeline"
preprocessing_cv_job_name = f"prepocessing_kfold_split"
hyper_tunning_job_name = f"KFoldCrossValidationHyperParameterTuner"
preprocessing_cv_step_name = f"PreprocessingCVStep"
hyper_tunning_cv_step_name = f"HyperParameterTuningStep",
preprocessing_step_name = f"PreprocessingStep",
model_training_step_name = f"ModelTrainingStep"
register_model_step_name = f"RegisterModelStep"
model_evaluation_step_name = f"ModelEvaluationStep"


bucket_name = f"sagemaker-project-p-hnneqf6paono"
default_bucket = f"s3://{bucket_name}"
s3_bucket_base_path_train = os.path.join(default_bucket, "/data/train")
s3_bucket_base_path_test = os.path.join(default_bucket, "/data/test")
s3_bucket_base_path_cleaned = os.path.join(default_bucket, "/data/cleaned")
s3_bucket_base_path_processed = os.path.join(default_bucket, "/data/processed")
s3_bucket_base_path_preprocessor = os.path.join(default_bucket, "/estimator/preprocessor")
s3_bucket_base_path_evaluation = os.path.join(default_bucket, "/evaluation")
s3_bucket_base_path_jobinfo = os.path.join(default_bucket, "/jobinfo")
s3_bucket_base_path_output = os.path.join(default_bucket, "/output")

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    
    Args:
        region: The aws region to start the session.
        default_bucket: The bucket to use for storing the artifacts.
        
    Returns:
        `sagemaker.session.Session` instance.
    """


    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")

    return session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_sagemaker_client(region):
    """Gets the sagemaker client.
        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    """Gets the pipeline custom tags.
    
    Args:
        new_tags: Project tags.
        region: The aws region to start the session.
        sagemaker_project_arn: Amazon Resource Name.
        
    Returns:
        Tags.
    """
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region: str,
    role: str = None,
    default_bucket: str = None,
    s3_bucket_base_path_train: str = s3_bucket_base_path_train,
    s3_bucket_base_path_test: str = s3_bucket_base_path_test,
    s3_bucket_base_path_cleaned: str = s3_bucket_base_path_cleaned,
    s3_bucket_base_path_processed: str = s3_bucket_base_path_processed,
    s3_bucket_base_path_preprocessor: str = s3_bucket_base_path_preprocessor,
    s3_bucket_base_path_evaluation: str = s3_bucket_base_path_evaluation,
    s3_bucket_base_path_jobinfo: str = s3_bucket_base_path_jobinfo,
    s3_bucket_base_path_output: str = s3_bucket_base_path_output,
    image_uri_model: str = image_uri_model,
    image_uri_tuning: str = image_uri_tuning,
    sagemaker_project_arn: str = None,
    processing_instance_count: str = processing_instance_count,
    processing_instance_type: str = processing_instance_type,
    transform_instances: List[str] = transform_instances,
    training_instance_count: int = training_instance_count,
    inference_instances: List[str] = inference_instances,
    hpo_tuner_instance_type: str = hpo_tuner_instance_type,
    model_package_group_name: str = model_package_group_name,
    pipeline_name: str = pipeline_name,
    preprocessing_cv_job_name: str = preprocessing_cv_job_name,
    hyper_tunning_job_name: str = hyper_tunning_job_name,
    preprocessing_cv_step_name: str = preprocessing_cv_step_name,
    hyper_tunning_cv_step_name: str = hyper_tunning_cv_step_name,
    preprocessing_step_name: str = preprocessing_step_name,
    model_training_step_name: str = model_training_step_name,
    register_model_step_name: str = register_model_step_name,
    model_evaluation_step_name: str = model_evaluation_step_name,
    baseline_model_objective_value: float = baseline_model_objective_value,
    k: int = k,
    max_tuning_jobs: int = max_tuning_jobs,
    max_parallel_jobs: int = max_parallel_jobs
    ):
    """Gets a SageMaker ML Pipeline instance working with respiratory disease data.
    
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts.
        sagemaker_project_arn: ARN of the project.
        s3_bucket_base_path_train: the folder to use for storing the k train folds.
        s3_bucket_base_path_test: the folder to use for storing the k test folds.
        s3_bucket_base_path_cleaned: the folder to use for storing the cleaned dataset.
        s3_bucket_base_path_processed: the folder to use for storing processed X and y data.
        s3_bucket_base_path_preprocessor: the folder to use for storing the preprocessor encoders.
        s3_bucket_base_path_evaluation: the folder to use for storing the evaluation metric.
        s3_bucket_base_path_jobinfo: the folder to use for storing the jobinfo.
        s3_bucket_base_path_output: the folder to use for storing the model.
        processing_instance_count: number of processing instances to run.
        image_uri_tuning: Docker image to run the cross-validation jobs with the range of hyperparameters.
        image_uri_model: Docker image to fit the XGBoost classifier.
        processing_instance_type: Instance type for processing.
        transform_instances: Instance type for transforming.
        inference_instances: Instance type for inference.
        model_package_group_name: Model package collection name.
        pipeline_name: Name for the pipeline.
        preprocessing_cv_job_name: Preprocessor of cross-validation step job name.
        hyper_tunning_job_name: Hyperparameter tunning cross-validation job name.
        preprocessing_cv_step_name: Name for Preprocessor of cross-validation pipeline step.
        hyper_tunning_cv_step_name: Name for Hyperparameter tunning cross-validation pipeline step.
        preprocessing_step_name: Name for the preprocessing pipeline step.
        model_training_step_name: Name for model training pipeline step.
        register_model_step_name: Name for register model pipeline step.
        model_evaluation_step_name: Name for model evaluation pipeline step.
        baseline_model_objective_value: Minimum metric value required for model to register. Used in condition step.
        k: number of k fold used to split the data in train and test dataset.
        max_tuning_jobs: Maximum number of hyperparameter tuning jobs to run.
        max_parallel_jobs: Maximum number of parallel hyperparameter tuning jobs to run.

    Returns:
        An instance of a pipeline.
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = session.get_execution_role(sagemaker_session)

    # Print the role for debugging    
    print(f"SageMaker assumes role: {role}.")
    
    model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval",  # We want manual approval
    )


    #preprocessing step to create the k folds datasets to use in cross-validation step
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version_sklearn,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=preprocessing_cv_job_name,
        sagemaker_session=sagemaker_session,
        role=role
    )

    preprocessing_kfold_split_step = ProcessingStep(
        name=preprocessing_cv_step_name,
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                input_name="cleaned_data",
                source=s3_bucket_base_path_cleaned,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/data/train/",
                destination=s3_bucket_base_path_train
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/data/test/",
                destination=s3_bucket_base_path_test
            )
        ],
        code=os.path.join(base_dir,"/hyperparameter_tuning/preprocessing_cv.py"),
        job_arguments=["--k", k]
    )


    #Cross-validation to model selection with hyperparameter tuning
    script_tuner = ScriptProcessor(
    image_uri=image_uri_tuning,
    command=["python3"],
    instance_type=hpo_tuner_instance_type,
    instance_count=processing_instance_count,
    base_job_name=hyper_tunning_job_name,
    role=role
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    jobinfo = PropertyFile(
        name="JobInfo", output_name="jobinfo", path="jobinfo.json"
    )


    hyperparam_tunning_cv_step = ProcessingStep(
        name=hyper_tunning_cv_step_name,
        processor=script_tuner,
        code=os.path.join(
            base_dir,"hyperparameter_tuning/cv_hyperparameter_tunning.py"
        ),
        outputs=[
            ProcessingOutput(
                output_name="evaluation", 
                source="/opt/ml/processing/evaluation", 
                destination=s3_bucket_base_path_evaluation
            ),
            ProcessingOutput(
                output_name="jobinfo", 
                source="/opt/ml/processing/jobinfo", 
                destination=s3_bucket_base_path_jobinfo
            )
        ],
        job_arguments=[
            "-k", k,
            "--image-uri-tuning", image_uri_tuning,
            "--image-uri-model", image_uri_model,
            "--train", s3_bucket_base_path_train, 
            "--test", s3_bucket_base_path_test,
            "--instance-type", training_instance_type,
            "--instance-count", training_instance_count,
            "--output-path", s3_bucket_base_path_output,
            "--max-tuning-jobs", max_tuning_jobs,
            "--max-parallel-jobs" , max_parallel_jobs,
            "--region", region,
            "--role", role
        ],
        property_files=[evaluation_report],
        depends_on=[preprocessing_cv_step_name]
    )


    #preprocessing step to train the model with the hyperparameters tuned
    preprocessing_step = ProcessingStep(
        name=preprocessing_step_name,
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                input_name="CleanedData",
                source=s3_bucket_base_path_cleaned,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="X_train_transformed",
                source="/opt/ml/processing/output/X_transformed",
                destination=s3_bucket_base_path_processed
            ),
            ProcessingOutput(
                output_name="y_train_transformed",
                source="/opt/ml/processing/output/y_transformed",
                destination=s3_bucket_base_path_processed           
            ),
            ProcessingOutput(
                output_name="preprocessor",
                source="/opt/ml/processing/output/preprocessor",
                destination=s3_bucket_base_path_preprocessor             
            )
        ],
        code=os.path.join(base_dir,"/model/preprocessing.py")
    )


    #Training the model with all the dataset
    model_train_estimator = SKLearn(
    entry_point="train_final_model.py",
    image_uri=image_uri_model,
    instance_type=training_instance_type,
    source_dir=os.path.join(base_dir,"/model"),
    output_path=s3_bucket_base_path_output,
    role=role
    )

    model_training_step = TrainingStep(
        name=model_training_step_name,
        estimator=model_train_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=s3_bucket_base_path_processed,
                content_type="text/csv"
            ),
            "jobinfo": TrainingInput(
                s3_data=s3_bucket_base_path_jobinfo,
                content_type="application/json"
            )
        },
        depends_on=[preprocessing_step_name]
    )


    # Register Model step
    model = Model(
    image_uri=model_train_estimator.image_uri,
    model_data=model_training_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=sagemaker_session,
    role=role
    )

    s3_uri = s3_uri = Join(
        on="", 
        values=[hyperparam_tunning_cv_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"], 
                "/evaluation.json"]
        )
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=s3_uri,
            content_type="application/json"
        )
    )


    register_model_step = RegisterModel(
        name=register_model_step_name,
        estimator=model_train_estimator,
        model_data=hyperparam_tunning_cv_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=inference_instances,
        transform_instances=transform_instances,
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )


    #Condition Step
    condition_evaluate_model = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=hyper_tunning_cv_step_name,
        property_file=evaluation_report,
        json_path="multiclass_classification_metrics.F1_score.value"
    ),
    right=baseline_model_objective_value
    )

    condition_step = ConditionStep(
        name=model_evaluation_step_name,
        conditions=[condition_evaluate_model],
        if_steps=[preprocessing_step, model_training_step, register_model_step],
        else_steps=[]
    )


    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[model_approval_status],
        pipeline_experiment_config=PipelineExperimentConfig(
            ExecutionVariables.PIPELINE_NAME,
            ExecutionVariables.PIPELINE_EXECUTION_ID
        ),
        steps=[preprocessing_kfold_split_step, hyperparam_tunning_cv_step, condition_step]
    )

    return pipeline