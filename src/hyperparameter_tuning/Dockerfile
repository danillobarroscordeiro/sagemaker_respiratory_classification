FROM python:3.10
RUN apt-get update && pip install sagemaker boto3 numpy sagemaker-training pandas scikit-learn xgboost
COPY cv.py /opt/ml/code/cv.py
COPY train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM cv.py