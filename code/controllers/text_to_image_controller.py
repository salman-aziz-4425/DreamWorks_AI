import sys
import os
sys.path.append("../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import HTTPException
from inference import ImageGenerator
from pydantic import BaseModel


class TextToImageInput(BaseModel):
    text: str
    sub_model: str
    Negative_Prompt:str
    id:str

class ImageToImageInput(BaseModel):
    input_image_url: str

class UpscaleInput(BaseModel):
    input_image_url: str
    upscale_factor: float

class MLSDInput(BaseModel):
    input_image_url: str
    
async def text_to_image_sd1_5(input: TextToImageInput):
    try:
       image_generator=ImageGenerator(input.sub_model,input.id)
       output_loc=image_generator.generate_image(input.text+" image quality should be "+input.Negative_Prompt)
       output_Image=[]
       for image in output_loc:
          output_Image.append(image_generator.write_image_to_s3(image,"s3://dw-stablediffusion/SD15/"))
    #    if image!='NSFW Detected':
    #        output_loc=image_generator.write_image_to_s3(image,"s3://dw-stablediffusion/SD15/")
    #        print("output data")
       return output_Image
    #    else:
    #        return "NSFW Detected"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


