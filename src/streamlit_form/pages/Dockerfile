FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install streamlit pandas boto3

EXPOSE 8501

CMD ["streamlit", "run", "--server.port", "8501", "home.py"]

