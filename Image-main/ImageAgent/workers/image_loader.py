from pathlib import Path
from PIL import Image
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
from tools.image_utils import resize_image_max_dim

@registry.register_worker()
class ImageLoader(BaseWorker):
    """Worker to load, validate, and cache images in workflow short-term memory."""
    
    def _run(self, image_path: str, *args, **kwargs):
        if not image_path:
            return {"status": "error", "message": "No image path provided."}
        
        try:
            # Load the image
            img = Image.open(image_path)
            # Resize if necessary
            img = resize_image_max_dim(img, max_dim=1024)
            
            # Cache the PIL Image object in STM
            self.stm(self.workflow_instance_id)["image_cache"] = {"<image_0>": img}
            
            return {"status": "success", "image_path": image_path}
        except Exception as e:
            return {"status": "error", "message": f"Failed to load image: {str(e)}"}
