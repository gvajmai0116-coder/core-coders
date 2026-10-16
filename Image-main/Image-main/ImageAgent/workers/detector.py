from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
from tools.detector import YOLODetectorTool

@registry.register_worker()
class DetectorWorker(BaseWorker):
    """Detector Worker that detects objects using YOLOv11 if run_detector is enabled."""
    
    _tool: YOLODetectorTool = None

    def _run(self, run_detector: bool, *args, **kwargs):
        if not run_detector:
            return {"objects": []}
            
        # Lazy initialization of the tool
        if self._tool is None:
            self._tool = YOLODetectorTool()
            
        # Retrieve cached image from short-term memory (STM)
        img_cache = self.stm(self.workflow_instance_id).get("image_cache")
        if not img_cache or "<image_0>" not in img_cache:
            return {"objects": []}
            
        img = img_cache["<image_0>"]
        try:
            objects = self._tool.detect_objects(img)
            return {"objects": objects}
        except Exception as e:
            return {"objects": []}
