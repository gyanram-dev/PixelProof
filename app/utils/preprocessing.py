import torch
from torchvision import transforms
from PIL import Image

def get_preprocessing_transform(img_size: int = 224):
    """
    Returns the standard preprocessing transform for PixelProof.
    Matches the pipeline used during training.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image: Image.Image, img_size: int = 224) -> torch.Tensor:
    """
    Preprocesses a PIL Image for inference.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    transform = get_preprocessing_transform(img_size)
    return transform(image).unsqueeze(0)
