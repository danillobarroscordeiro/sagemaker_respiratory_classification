FROM python:3.10
RUN apt-get update && pip install sagemaker boto3 numpy sagemaker-training pandas scikit-learn xgboost==1.5.1 python-dotenv
COPY cv.py /opt/ml/code/cv.py
COPY train_cv.py /opt/ml/code/train.py
COPY .env /opt/ml/code/.env
ENV SAGEMAKER_PROGRAM cv.py
