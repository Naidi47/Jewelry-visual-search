"""
Image preprocessing pipeline optimized for CLIP inference.
Handles format validation, resizing, normalization, and tensor conversion.
"""

import io
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms


class ImageValidationError(Exception):
    """Raised when uploaded image fails validation checks."""
    pass


class ImagePreprocessor:
    """
    Production-grade image preprocessor for CLIP models.
    
    CLIP expects:
    - RGB color space (3 channels)
    - Spatial size 224x224 (ViT-B/32)
    - Normalized with CLIP-specific statistics (not standard ImageNet)
    
    Reference: OpenAI CLIP preprocessing
    https://github.com/openai/CLIP/blob/main/clip/clip.py
    """
    
    # CLIP's specific normalization constants
    # Different from standard ImageNet: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    def __init__(
        self,
        target_size: int = 224,
        max_file_size_mb: int = 10
    ):
        self.target_size = target_size
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Preprocessing pipeline (matches CLIPProcessor behavior)
        self.transform = transforms.Compose([
            # Bicubic interpolation preserves fine jewelry details better than bilinear
            transforms.Resize(
                target_size,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            # Center crop to exact target size
            transforms.CenterCrop(target_size),
            # [0, 255] -> [0.0, 1.0]
            transforms.ToTensor(),
            # Normalize with CLIP-specific statistics
            transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
        ])
    
    def validate_upload(self, file_content: bytes, content_type: str) -> None:
        """
        Validate uploaded image before processing.
        
        Args:
            file_content: Raw bytes from multipart upload
            content_type: MIME type from HTTP header
            
        Raises:
            ImageValidationError: If any check fails
        """
        # File size check
        if len(file_content) > self.max_file_size_bytes:
            size_mb = len(file_content) / (1024 * 1024)
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            raise ImageValidationError(
                f"Image too large: {size_mb:.1f}MB. Maximum: {max_mb:.0f}MB"
            )
        
        # Format whitelist
        supported_types = ["image/jpeg", "image/png", "image/webp"]
        if content_type not in supported_types:
            raise ImageValidationError(
                f"Unsupported format: {content_type}. "
                f"Supported: {', '.join(supported_types)}"
            )
    
    def preprocess(self, file_content: bytes) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert raw image bytes to model-ready tensor.
        
        Processing steps:
        1. Decode bytes to PIL Image
        2. Convert to RGB (handles RGBA, P, L modes)
        3. Apply CLIP transforms (resize, crop, normalize)
        4. Add batch dimension for inference
        
        Args:
            file_content: Raw image bytes from upload
            
        Returns:
            (batched_tensor [1, 3, 224, 224], original_size (w, h))
        """
        # Decode image from bytes
        try:
            image = Image.open(io.BytesIO(file_content))
        except Exception as e:
            raise ImageValidationError(f"Cannot decode image: {str(e)}")
        
        # Store original dimensions for metadata
        original_size = image.size  # (width, height)
        
        # Ensure RGB - critical for CLIP which expects 3 channels
        # Handles: RGBA (4-ch), P (palette), L (grayscale), LA, etc.
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply preprocessing pipeline
        # Output: tensor [3, 224, 224]
        tensor = self.transform(image)
        
        # Add batch dimension for neural network
        # Output: [1, 3, 224, 224]
        batched_tensor = tensor.unsqueeze(0)
        
        return batched_tensor, original_size
