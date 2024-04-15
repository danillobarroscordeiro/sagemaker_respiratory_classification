# Respiratory Classification

## 1. Business problem

This project predicts which type of SARS virus a patient has contracted using an XGBoost model. The model is deployed on AWS using SageMaker, and a medical form for patient data input is available via a Streamlit web API.

## 2. Business Assumptions
* We started with a balanced random sample of the dataset.
* Patients who did not specify their gender (marked 'I' in the gender column) were removed.
* Data entries with errors (like out-of-range values) were corrected to the most common value (mode).
* Missing values were filled with the mode.


## 3. Dataset
The dataCreating an AWS crawler to retrieve and catalog dataset metadata stored in an S3 bucket using AWS Glue.set includes about 700,000 records from Brazilian hospital patients showing symptoms of the SARS virus. It has 76 features covering demographic details, symptoms, test results (e.g., Chest X-ray, CT scans), vaccination history, and conditions like diabetes and obesity. The "CLASSI_FIN" field specifies the type of SARS virus diagnosed. The patients were diagnosed with one of the following labels:

1-SARS by influenza \
2-SARS by other respiratory virus \
3-SARS by other etiological agent \
4-SARS unspecified \
5-SARS by covid-19

The data dictionary is inside the jupyter notebook.

## 4. Tools used

* Python
* Jupyter notebook
* Git and Github
* AWS Sagemaker
* AWS Cloud
* Streamlit
* Docker
* Sklearn

## 5. Planning Soluction

This project was developed based on CRISP-DM (Cross-Industry Standard Process - Data Science) framework but with additional steps, especially in deployment:

Step 1. Create an AWS crawler to retrieve metadata of the dataset stored in a S3 bucket and insert this metadata in AWS Glue catalog \
Step 2. Querying data in S3 using AWS Athena and saving the raw, training, and testing data. \
Step 3. Describing and cleaning data: analyzing dimensions, checking for NAs, performing descriptive statistics, handling outliers, and imputing missing values. \
Step 4. Selecting features to remove irrelevant data. \
Step 5. Setting up a project in SageMaker Studio Classic using the MLOps template for model building, training, and deployment. \
Step 6. Writing Python scripts and creating their Docker images for pipeline steps. \
Developing and configuring the pipeline steps: Preprocessing CV, Hyperparameter Tuning, Preprocessing, Model Training, Model Registration, and Condition Step. \
Committing the pipeline to trigger execution via CodePipeline. \
Manually approving the registered model in SageMaker Studio Classic, which triggers model deployment using CodeBuild. \
Manually approving the staging endpoint to promote model deployment to the production endpoint. \
Creating and deploying a medical form on Streamlit via an ECS service running a Docker image registered in ECR.

## 6. Project Architecture

The architecture diagram below shows the data flow and infrastructure setup in AWS:

![](figures/aws_project_infrastructure.jpeg)

The historical data is stored in an S3 bucket, and an AWS Glue Crawler catalogs this data. We use AWS Athena to query the data within a SageMaker notebook. Following data querying, the pipeline and model were developed and subsequently deployed using AWS CloudFormation.

The medical form, hosted on a web app via Streamlit, operates on an ECS server running a Docker image registered in the ECR. This form collects patient information and sends it as an HTTP request to a Lambda function. This function receives the data in JSON format, processes it by applying the same encoders and transformations used on the training data, and forwards the transformed data to the model's endpoint. The model evaluates the data to predict which SARS virus the patient may have and sends the results back to the Lambda function, which in turn sends them back to the website.


The Sagemaker Data Science project template for model building, training, and deployment (MLOps) consists of two main phases: model building and model deployment. The model building process is outlined below:

![](figures/sagemaker_project_1.png)

When you commit the pipeline code to CodeCommit in Sagemaker Studio Classic, it triggers an Amazon EventBridge event. This event initiates the build process in CodeBuild, which in turn runs CodePipeline, the training pipeline. If the trained model achieves an F1 score higher than 0.5 during the Condition Step, it is registered in the model registry, and its artifacts are stored in S3.

Following manual approval of the trained model, the deployment phase begins. The infrastructure for model deployment is shown below:

![](figures/sagemaker_project_2.png)

CodeBuild executes the scripts to build and deploy the model in the staging area. This deployment process is managed by CloudFormation, which orchestrates the necessary infrastructure to establish the inference endpoint with the trained model. Once the staging endpoint is fully set up, manual approval is required to proceed with deploying the model to production. Finally, once approved, the model endpoint becomes available in production for making predictions.

## 7. Machine Learning Model Applied and performance

The model applied in the first circle was an XGBoost. To train a generalized model it was applied a k-fold cross-validation to avoid underfitting or overfitting. This cross-validation also was used to do a hyperparameter tuning. The architecture of Hyperparameter Tuner step is shown below:

![](figures/hyperparameter_tuning.png)

The metric used to check performance was F1 score. The score of the first model was 53.81%. It was a reasonable score for the first MLOps circle, since this is the baseline score.

## 8. Web Application

The web application processes and displays the classification results from the model based on patient data:
![](figures/web_app.gif)

## 9. Next steps

Improvements for the next phase include:

* Adding more features to the model.
* Testing different algorithms like Random Forest, LightGBM, and SVM.
* Revising how missing values are handled.
* Training the model with more historical data to enhance its accuracy.