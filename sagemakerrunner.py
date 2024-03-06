import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker import Session
import json
# Create a boto3 session
boto_session = boto3.Session(region_name='us-east-1')

# Create a SageMaker session using the boto3 session
sagemaker_session = Session(boto_session=boto_session)

role = "arn:aws:iam::428549984681:role/ai-dreamworks"
image_uri = "428549984681.dkr.ecr.us-east-1.amazonaws.com/dreamworks:latest"


model = Model(
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="instance type",
)

# client = boto3.client('sagemaker-runtime')
#  # An example of a trace ID.
# endpoint_name = "dreamworks-2023-09-07-22-21-19-351/ping"

# # Create a SageMaker client
# client = boto3.client('sagemaker-runtime')

# # Define your JSON data as a dictionary
# request_data = {
#     "text": "boy eating chocolate",
#     "Negative_Prompt": "bad quality, poor",
#     "sub_model": "runwayml/stable-diffusion-v1-5",
#     "id": "model_v1"
# }

# # Convert the dictionary to a JSON string
# request_body = json.dumps(request_data)

# # Specify the SageMaker endpoint name and other parameters
# content_type = 'application/json'
# accept = 'application/json'
# # Send the request to the endpoint
# response = client.invoke_endpoint(
#     EndpointName=endpoint_name,
#     ContentType=content_type,
#     Accept=accept,
#     Body=request_body
# )

# # Parse and process the response
# response_body = response['Body'].read().decode('utf-8')
# result = json.loads(response_body)

# # Process the result as needed
# print(result)



# import requests

# image_server_url = "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/dreamworks-2023-09-07-22-21-19-351/invocations"
# payload= {
#     "text": "boy eating chocolate",
#     "Negative_Prompt": "bad quality, poor",
#     "sub_model": "runwayml/stable-diffusion-v1-5",
#     "id": "model_v1"
# }
#   # Your request data as a dictionary

# response = requests.post(image_server_url, json=payload)

# if response.status_code == 200:
#     # Process the response from the image server
#     result = response.json()
#     print(result)
# else:
#     print(f"Request failed with status code: {response.status_code}")

# import sagemaker

# # Create a Batch Transform job object
# job = sagemaker.transformer.TransformJob(
#     name="stable-diffusion-job",
#     image="<your-ecr-image-uri>",
#     instance_count=1,
#     instance_type="ml.p3.2xlarge",
#     max_concurrent_transforms=10,
#     input_path="<your-input-data-path>",
#     output_path="<your-output-data-path>",
# )

# # Run the Batch Transform job
# job.run()
