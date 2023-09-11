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