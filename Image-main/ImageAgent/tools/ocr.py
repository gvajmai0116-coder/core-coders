import torch
import os
os.environ["FLAGS_use_onednn"] = "0"

import paddle
paddle.set_flags({'FLAGS_use_onednn': False})

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from typing import Dict, Any, List

class PaddleOCRTool:
    def __init__(self):
        self.ocr = None

    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Enhance image using OpenCV prior to running PaddleOCR."""
        # Convert PIL image to BGR numpy array
        img_np = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]

        # 1. Resize to 2x if the image is small
        if w < 1200 or h < 1200:
            img_np = cv2.resize(img_np, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # 3. Apply CLAHE (contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Apply a light bilateral filter to remove noise while preserving text edges
        filtered = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)

        # 5. Adaptive thresholding for optimal OCR edge contrast
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )

        # Merge thresholded binary and enhanced gray to keep subtle gradients usable for OCR
        combined = cv2.addWeighted(filtered, 0.7, binary, 0.3, 0)
        
        # Convert back to 3-channel RGB as expected by PaddleOCR
        return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)

    def _reconstruct_lines(self, first_res: list) -> tuple[str, list, list, list]:
        """Group detected text boxes into lines by Y-position and sort left-to-right."""
        items = []
        
        # Parse PaddleOCR detection entries
        for entry in first_res:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                box = entry[0]
                text_info = entry[1]
                text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                conf = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0
                
                # Box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                ys = [pt[1] for pt in box]
                xs = [pt[0] for pt in box]
                y_center = sum(ys) / len(ys)
                x_min = min(xs)
                height = max(ys) - min(ys)
                
                items.append({
                    "box": box,
                    "text": text,
                    "confidence": conf,
                    "y_center": y_center,
                    "x_min": x_min,
                    "height": height
                })

        if not items:
            return "", [], [], []

        # Determine baseline height tolerance for line grouping
        avg_height = sum(item["height"] for item in items) / len(items)
        y_tolerance = max(12.0, avg_height * 0.5)

        # Sort items primarily by Y center
        items_sorted_y = sorted(items, key=lambda item: item["y_center"])

        # Cluster items into visual lines
        lines_clusters = []
        for item in items_sorted_y:
            placed = False
            for cluster in lines_clusters:
                cluster_y_avg = sum(elem["y_center"] for elem in cluster) / len(cluster)
                if abs(item["y_center"] - cluster_y_avg) <= y_tolerance:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                lines_clusters.append([item])

        # Sort lines top-to-bottom
        lines_clusters = sorted(lines_clusters, key=lambda c: sum(e["y_center"] for e in c) / len(c))

        # Sort each line left-to-right and build structured output
        formatted_lines = []
        structured_lines = []
        boxes = []
        confidences = []

        for line_idx, cluster in enumerate(lines_clusters, 1):
            sorted_cluster = sorted(cluster, key=lambda e: e["x_min"])
            line_text = " ".join(e["text"] for e in sorted_cluster)
            formatted_lines.append(line_text)

            cluster_boxes = [e["box"] for e in sorted_cluster]
            cluster_confs = [e["confidence"] for e in sorted_cluster]
            
            boxes.extend(cluster_boxes)
            confidences.extend(cluster_confs)

            structured_lines.append({
                "line_no": line_idx,
                "text": line_text,
                "boxes": cluster_boxes,
                "confidences": cluster_confs
            })

        full_text = "\n".join(formatted_lines)
        return full_text, structured_lines, boxes, confidences

    def perform_ocr(self, img: Image.Image) -> Dict[str, Any]:
        """Run PaddleOCR on preprocessed image and return structured OCR results."""
        if self.ocr is None:
            # Initialize PaddleOCR (uses English, disables MKL-DNN)
            self.ocr = PaddleOCR(lang='en', enable_mkldnn=False)
        
        # Preprocess image with OpenCV
        enhanced_np = self.preprocess_image(img)
        
        # Run OCR on the enhanced image
        result = self.ocr.ocr(enhanced_np)
        
        # Extract text and line structure
        if result and len(result) > 0 and result[0] is not None:
            first_res = result[0]
            if isinstance(first_res, dict):
                texts = first_res.get('rec_texts', [])
                full_text = "\n".join(texts)
                lines = [{"line_no": i+1, "text": t} for i, t in enumerate(texts)]
                boxes = first_res.get('boxes', [])
                confidences = first_res.get('rec_scores', [])
            else:
                full_text, lines, boxes, confidences = self._reconstruct_lines(first_res)
        else:
            full_text, lines, boxes, confidences = "", [], [], []
                
        return {
            "text": full_text,
            "lines": lines,
            "boxes": boxes,
            "confidences": confidences
        }
