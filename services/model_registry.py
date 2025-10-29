"""
Model Registry Service
Loads ML model artifacts from various sources with fallback priorities:
1. Hugging Face Hub
2. Direct URL
3. Local path
"""
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Handles loading of ML model artifacts from multiple sources."""
    
    def __init__(self):
        self.cache_dir = tempfile.mkdtemp(prefix="model_cache_")
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def load_model(self) -> Optional[str]:
        """
        Load model with priority fallback.
        Returns path to the loaded model file or None if all methods fail.
        """
        # Try Hugging Face Hub first
        model_path = self._load_from_huggingface()
        if model_path:
            return model_path
        
        # Try direct URL
        model_path = self._load_from_url()
        if model_path:
            return model_path
        
        # Try local path
        model_path = self._load_from_local()
        if model_path:
            return model_path
        
        logger.warning("All model loading methods failed. Model not available.")
        return None
    
    def _load_from_huggingface(self) -> Optional[str]:
        """Load model from Hugging Face Hub."""
        try:
            from huggingface_hub import hf_hub_download
            
            model_id = os.getenv("HUGGINGFACE_MODEL_ID")
            artifact_name = os.getenv("MODEL_ARTIFACT_NAME", "startup_model.joblib")
            
            if not model_id:
                logger.debug("HUGGINGFACE_MODEL_ID not set, skipping HF Hub")
                return None
            
            logger.info(f"Attempting to load from Hugging Face: {model_id}/{artifact_name}")
            
            # Download to cache
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=artifact_name,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Successfully loaded model from Hugging Face: {model_path}")
            return model_path
            
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            return None
    
    def _load_from_url(self) -> Optional[str]:
        """Load model from direct URL."""
        try:
            import requests
            
            url = os.getenv("MODEL_ASSET_URL")
            if not url:
                logger.debug("MODEL_ASSET_URL not set, skipping URL download")
                return None
            
            logger.info(f"Attempting to load from URL: {url}")
            
            # Download to cache
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Save to cache directory
            filename = os.path.basename(url) or "model.joblib"
            model_path = os.path.join(self.cache_dir, filename)
            
            with open(model_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully loaded model from URL: {model_path}")
            return model_path
            
        except Exception as e:
            logger.warning(f"Failed to load from URL: {e}")
            return None
    
    def _load_from_local(self) -> Optional[str]:
        """Load model from local path."""
        try:
            local_path = os.getenv("LOCAL_MODEL_PATH")
            if not local_path:
                logger.debug("LOCAL_MODEL_PATH not set, skipping local load")
                return None
            
            if not os.path.exists(local_path):
                logger.warning(f"Local model path does not exist: {local_path}")
                return None
            
            logger.info(f"Successfully found local model: {local_path}")
            return local_path
            
        except Exception as e:
            logger.warning(f"Failed to load from local path: {e}")
            return None
