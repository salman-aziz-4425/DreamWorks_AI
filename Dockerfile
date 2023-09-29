FROM python:3.9


RUN pip install --no-cache-dir fastapi uvicorn sagemaker matplotlib
RUN pip install --no-cache-dir diffusers==0.19.3 accelerate==0.17.0  boto3 transformers==4.30.1


ENV PATH="/opt/program:${PATH}"
COPY code /opt/program
COPY sagemaker-logo-small.png /opt/program

RUN chmod 755 /opt/program
WORKDIR /opt/program
RUN chmod 755 serve

