from io import BytesIO
from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Exact evaluation transform from the training notebook
eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

def load_image_as_rgb(image_source: Union[str, Path, bytes, bytearray, BytesIO, Image.Image]) -> np.ndarray:
    """
    Safely loads and validates an image from various input formats,
    converting it into a standard uint8 RGB numpy array.
    """
    if isinstance(image_source, Image.Image):
        return np.array(image_source.convert("RGB"))
        
    if isinstance(image_source, (str, Path)):
        img_path = str(image_source)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image from path: {img_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    if isinstance(image_source, (bytes, bytearray, BytesIO)):
        if isinstance(image_source, BytesIO):
            byte_data = image_source.getvalue()
        else:
            byte_data = bytes(image_source)
            
        np_arr = np.frombuffer(byte_data, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from byte buffer.")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    raise TypeError(f"Unsupported image source type: {type(image_source)}")

def preprocess_image(image_source: Union[str, Path, bytes, bytearray, BytesIO, Image.Image]) -> torch.Tensor:
    """
    Preprocesses an input image according to the exact HAM10000 notebook evaluation pipeline:
    1. Reads RGB numpy array.
    2. Resizes to 384 x 384.
    3. Normalizes using ImageNet mean & std.
    4. Converts to float32 PyTorch tensor with shape (1, 3, 384, 384).
    """
    rgb_img = load_image_as_rgb(image_source)
    transformed = eval_transform(image=rgb_img)["image"] # shape: [3, 384, 384]
    tensor = transformed.unsqueeze(0).to(dtype=torch.float32) # shape: [1, 3, 384, 384]
    return tensor
