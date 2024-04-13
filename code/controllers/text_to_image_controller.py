import sys
import os
sys.path.append("../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import HTTPException
from inference import ImageGenerator
from pydantic import BaseModel
from inference import image_generator
from typing import List
import base64

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
    
async def text_to_image_sd1_5(input: TextToImageInput) -> List[str]:
    try:
        image_generator=image_generator=ImageGenerator(input.id)
        output_loc = image_generator.generate_image(input.text)
        image_generator.free_gpu_memory()
        return output_loc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

