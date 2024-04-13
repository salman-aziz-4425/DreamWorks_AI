import torch
import boto3
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler,DiffusionPipeline,AutoencoderKL,DPMSolverMultistepScheduler,UNet2DConditionModel,LCMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.models.attention_processor import AttnProcessor
import logging
import os
import io
import gc
import uuid
from PIL import Image
import datetime
import cloudinary
import cloudinary.uploader
import cloudinary.api
import time
import base64

cloudinary.config( 
  cloud_name = "", 
  api_key = "", 
  api_secret = "" 
)

lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
logging.basicConfig(level=logging.INFO)
class ImageGenerator:
    def __init__(self, model_id):
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
            vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
            self.pipe = DiffusionPipeline.from_pretrained(
               model_id,
               vae=vae,
               unet=unet,
               torch_dtype=torch.float16,
               use_safetensors=True,
               cache_dir="/tmp/",
            )
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.to(device="cuda", dtype=torch.float16)
            logging.info("Model successfully loaded.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            
    def get_inputs(self,batch_size,prompt):
        prompts = batch_size * [prompt]
        num_inference_steps = 8
        return {"prompt": prompts,"num_inference_steps": num_inference_steps,"height":1024,"width":1024,"guidance_scale":1.0,"negative_prompt":"Hd"}
    
    def convert_to_b64(self, image: Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64
        
    def generate_image(self, prompt):
           
        try:
          start_time = time.time() 
          batch_size=1
          images = self.pipe(**self.get_inputs(batch_size,prompt)).images
          if isinstance(images, torch.Tensor):
            images = images.to(self.device).numpy()
          end_time = time.time()

          print("Time taken by image generator:", end_time - start_time, "seconds")
            # Convert the numpy array image to a PIL Image
          print(self.write_image_to_cloudinary(images[0]))
          b64_results = self.convert_to_b64(images[0])
          return  b64_results
        except Exception as e:
            return str(e)

    @staticmethod
    def display_image(image):
        plt.imshow(image)
        plt.axis('off') 
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
        s3_client.put_object(
        Body=buf.getvalue(),
        Bucket=bucket,
        Key=key,
        ContentType='image/png',
        Metadata={
            "seed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
        uploaded_s3_uri = f'https://{bucket}.s3.amazonaws.com/{key}'
        return uploaded_s3_uri
    
    def write_image_to_cloudinary(self,image):
     print(image)
     buf = io.BytesIO()
     image.save(buf, format='PNG')
     buf.seek(0)
     image_bytes = buf.getvalue()
     upload_result = cloudinary.uploader.upload(image_bytes, public_id=str(uuid.uuid4()))
     return upload_result['secure_url']

   
# image_generator=I
# image_generator=ImageGenerator("runwayml/stable-diffusion-v1-5")
image_generator=ImageGenerator("stabilityai/stable-diffusion-xl-base-1.0")


