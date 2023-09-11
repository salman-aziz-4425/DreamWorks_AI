import torch
import boto3
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler,DiffusionPipeline,AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.models.attention_processor import AttnProcessor
import logging
import os
import io
import gc
import uuid
from PIL import Image
import datetime


logging.basicConfig(level=logging.INFO)
class ImageGenerator:
    def __init__(self, model_id,id):
        self.isAlreadyAvailable=False
        self.modeid=model_id
        self.path=model_id.split("/")
        self.path=self.path[1]
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")
            logging.info("CUDA is not available. Using CPU.")

        try:
            self.pipe = DiffusionPipeline.from_pretrained(
               model_id,
               torch_dtype=torch.float16,
               use_safetensors=True,
               cache_dir="/tmp/"
            ).to(self.device)
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config,timestep_spacing="trailing")
            self.pipe.enable_attention_slicing()
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(self.device)
            self.pipe.vae = vae
            logging.info("Model successfully loaded.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            
    def get_inputs(self,batch_size,prompt):
        generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
        prompts = batch_size * [prompt]
        num_inference_steps = 20
        return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}
        
    def generate_image(self, prompt):
        try:
          batch_size=4
          images = self.pipe(**self.get_inputs(batch_size,prompt)).images
          if isinstance(images, torch.Tensor):
            images = images.to("cpu").numpy()
          return images
        except Exception as e:
            return str(e)

        

    @staticmethod
    def display_image(image):
        # Display the image using Matplotlib
        plt.imshow(image)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

    def save_model(self, model_path):
        try:
            model_file_path = os.path.join(model_path)
            self.pipe.save_pretrained(model_file_path)
            logging.info(f"Model saved successfully at {model_path}.")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def free_gpu_memory(self):
        try:
            del self.pipe
            gc.collect()
            logging.info("GPU memory freed successfully.")
        except Exception as e:
            logging.error(f"Error freeing GPU memory: {e}")
            
    def get_bucket_and_key(self,s3uri):
        s3uri_parts = s3uri.replace("s3://", "").split("/")
        bucket = s3uri_parts[0]
        key = "/".join(s3uri_parts[1:])
        return bucket, key
    
    def write_image_to_s3(self,image, output_s3uri):
        s3_client = boto3.client('s3',aws_access_key_id='',aws_secret_access_key='')
        bucket, key = self.get_bucket_and_key(output_s3uri)
        key = f'{key}{uuid.uuid4()}.png'
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return image
    #     s3_client.put_object(
    #     Body=buf.getvalue(),
    #     Bucket=bucket,
    #     Key=key,
    #     ContentType='image/png',
    #     Metadata={
    #         "seed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     }
    # )
    #     uploaded_s3_uri = f'https://{bucket}.s3.amazonaws.com/{key}'
    #     return uploaded_s3_uri

   


