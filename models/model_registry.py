import os
import tempfile
import requests
from typing import Optional

def _download_to(path: str, url: str, chunk: int = 8192) -> str:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for c in r.iter_content(chunk):
                if c:
                    f.write(c)
    return path

class ModelRegistry:
    """
    Load a joblib artifact with priority:
    1) Hugging Face Hub (HUGGINGFACE_MODEL_ID, MODEL_ARTIFACT_NAME)
    2) Direct URL (MODEL_ASSET_URL)
    3) Local fallback (LOCAL_MODEL_PATH)
    """
    def __init__(self,
                 hf_model_id: Optional[str] = None,
                 artifact_name: Optional[str] = None,
                 asset_url: Optional[str] = None,
                 local_path: Optional[str] = None):
        self.hf_model_id = hf_model_id or os.getenv("HUGGINGFACE_MODEL_ID", "")
        self.artifact_name = artifact_name or os.getenv("MODEL_ARTIFACT_NAME", "startup_model.joblib")
        self.asset_url = asset_url or os.getenv("MODEL_ASSET_URL", "")
        self.local_path = local_path or os.getenv("LOCAL_MODEL_PATH", "")
        self.cache_dir = os.path.join(tempfile.gettempdir(), "anthill_model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_joblib_path(self) -> Optional[str]:
        # Try Hugging Face Hub
        if self.hf_model_id:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=self.hf_model_id, filename=self.artifact_name, local_dir=self.cache_dir)
                return path
            except Exception:
                pass

        # Try direct URL
        if self.asset_url:
            try:
                target = os.path.join(self.cache_dir, os.path.basename(self.asset_url.split("?")[0]) or self.artifact_name)
                return _download_to(target, self.asset_url)
            except Exception:
                pass

        # Try local
        if self.local_path and os.path.exists(self.local_path):
            return self.local_path

        return None
