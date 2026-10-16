from ultralytics import YOLO
from PIL import Image

class YOLODetectorTool:
    def __init__(self):
        self.model = None

    def detect_objects(self, img: Image.Image) -> list:
        """Run YOLOv11 nano model on the PIL image and return list of detected object labels."""
        if self.model is None:
            # Load YOLOv11 nano model. Downloads weights on first use (~15MB).
            self.model = YOLO("yolo11n.pt")
        
        # Run prediction
        results = self.model(img, verbose=False)
        
        detected_objects = []
        for result in results:
            if result.boxes is not None:
                for cls_id in result.boxes.cls:
                    label = result.names[int(cls_id)]
                    detected_objects.append(label)
        return detected_objects
