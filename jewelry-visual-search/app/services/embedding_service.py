"""
Production CLIP embedding service with model caching,
GPU optimization, and memory management.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from app.config import Settings, get_settings


class EmbeddingService:
    """
    Singleton-pattern embedding service for CLIP inference.
    
    Design decisions:
    1. Lazy initialization: Model loaded once at startup, not per-request
    2. Device affinity: Model pinned to GPU for entire lifetime
    3. No-gradient context: Disables autograd to save ~40% memory
    4. Torch.compile: ~20% speedup on GPU (PyTorch 2.0+)
    
    Memory lifecycle:
    - Load: ~2GB GPU memory (ViT-B/32)
    - Inference: ~100MB temporary per batch
    - Cleanup: torch.cuda.empty_cache() releases temporaries
    """
    
    _instance: Optional["EmbeddingService"] = None
    _lock = asyncio.Lock()
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.device: Optional[torch.device] = None
        self._model_loaded = False
        
        # Performance metrics
        self._inference_count = 0
        self._total_inference_time = 0.0
    
    @classmethod
    async def get_instance(cls, settings: Optional[Settings] = None) -> "EmbeddingService":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings)
        return cls._instance
    
    def _get_device(self) -> torch.device:
        """
        Determine compute device with validation.
        
        Priority:
        1. Explicit config (cuda/cpu)
        2. Auto-detect CUDA availability
        """
        config_device = self.settings.DEVICE.lower()
        
        if config_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config_device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA requested but not available. "
                    f"Available: {torch.cuda.device_count()} devices"
                )
        
        return device
    
    def load_model(self) -> None:
        """
        Initialize CLIP model and processor.
        
        This is the slow operation (~30s first time, ~5s cached).
        Called explicitly during FastAPI startup to fail fast.
        """
        if self._model_loaded:
            return
        
        self.device = self._get_device()
        print(f"[EmbeddingService] Loading {self.settings.MODEL_NAME}")
        print(f"[EmbeddingService] Device: {self.device}")
        
        # Ensure cache directory exists for model weights
        if self.settings.MODEL_CACHE_DIR:
            cache_path = Path(self.settings.MODEL_CACHE_DIR)
            cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load processor (handles tokenizer and image preprocessing config)
        self.processor = CLIPProcessor.from_pretrained(
            self.settings.MODEL_NAME,
            cache_dir=self.settings.MODEL_CACHE_DIR
        )
        
        # Determine dtype based on device
        # float16 on GPU for 2x memory efficiency, float32 on CPU for compatibility
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Load model with optimizations
        self.model = CLIPModel.from_pretrained(
            self.settings.MODEL_NAME,
            cache_dir=self.settings.MODEL_CACHE_DIR,
            torch_dtype=dtype,
            # Use Scaled Dot Product Attention (faster than default on modern GPUs)
            attn_implementation="sdpa"
        )
        
        self.model.to(self.device)
        self.model.eval()  # CRITICAL: Disable training-specific layers
        
        # Compile model for additional optimization (PyTorch 2.0+)
        if hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[EmbeddingService] Model compiled with torch.compile")
            except Exception as e:
                print(f"[EmbeddingService] torch.compile skipped: {e}")
        
        self._model_loaded = True
        
        # Warm-up: dummy inference to initialize CUDA kernels
        if self.device.type == "cuda":
            self._warmup()
        
        print("[EmbeddingService] Model ready")
    
    def _warmup(self) -> None:
        """
        Execute dummy forward pass to trigger CUDA kernel compilation.
        
        First inference on GPU is slow (~2s) due to kernel compilation.
        Warmup ensures real requests are fast (~50ms).
        """
        dummy = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            _ = self.model.get_image_features(dummy)
        torch.cuda.synchronize()
        print("[EmbeddingService] GPU warmup complete")
    
    @torch.no_grad()
    def embed_image(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate image embedding with CLIP vision encoder.
        
        Mathematical operations:
        1. Vision Transformer: image patches -> latent representation
        2. Projection head: latent space -> joint image-text space
        3. L2 normalization: v = v / ||v||_2 (if normalize=True)
        
        With normalization:
            cosine_similarity(a, b) = dot(a, b)  # since ||a|| = ||b|| = 1
        
        Args:
            pixel_values: Preprocessed image [batch, 3, 224, 224]
            normalize: Whether to L2-normalize output
            
        Returns:
            (embedding tensor [batch, 512], inference_time_ms)
        """
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start = time.perf_counter()
        
        # Ensure tensor on correct device
        pixel_values = pixel_values.to(self.device)
        
        # Forward pass through vision model
        # get_image_features = vision_backbone + projection_head
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        
        # L2 normalize to unit hypersphere
        if normalize:
            # F.normalize: x / sqrt(sum(x^2) + eps)
            # Result: ||x||_2 = 1.0
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        # Synchronize for accurate GPU timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000
        
        # Update rolling metrics
        self._inference_count += 1
        self._total_inference_time += inference_time
        
        return image_features, inference_time
    
    def get_average_latency(self) -> float:
        """Rolling average inference latency in milliseconds."""
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_time / self._inference_count
    
    def get_embedding_dimension(self) -> int:
        """Return expected output dimension."""
        return self.settings.VECTOR_DIMENSIONS
    
    def cleanup(self) -> None:
        """Release GPU memory and model resources."""
        if self.device and self.device.type == "cuda":
            # Move model to CPU first to free GPU memory
            if self.model:
                self.model.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self._model_loaded = False
        self.model = None
        self.processor = None
        print("[EmbeddingService] Cleanup complete")
