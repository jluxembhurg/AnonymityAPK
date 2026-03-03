import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

class FaceDetector:
    """
    Wrapper for YOLOv8 face detection.
    Optimized for CPU and includes box buffering logic.
    """
    
    def __init__(self, model_path: str = "models/detector.onnx"):
        """
        Initializes the YOLOv8 detector.
        
        Args:
            model_path (str): Path to the YOLOv8 model file.
        """
        # Load the model. Ultralytics handles CPU/ONNX automatically.
        self.model = YOLO(model_path, task='detect')
        
        # Bounding box buffer percentage (15% as requested)
        self.buffer_pct = 0.15

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in a frame and returns buffered bounding boxes.
        
        Args:
            frame (np.ndarray): Image frame from camera.
            
        Returns:
            List[Tuple[int, int, int, int]]: List of (x1, y1, x2, y2) coordinates.
        """
        results = self.model(frame, verbose=False)
        boxes = []
        
        h, w = frame.shape[:2]
        
        for result in results:
            for box in result.boxes:
                # Get raw coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Apply 10-15% buffer
                bw = x2 - x1
                bh = y2 - y1
                
                dx = int(bw * self.buffer_pct)
                dy = int(bh * self.buffer_pct)
                
                # Expand and clip to frame boundaries
                nx1 = max(0, x1 - dx)
                ny1 = max(0, y1 - dy)
                nx2 = min(w, x2 + dx)
                ny2 = min(h, y2 + dy)
                
                boxes.append((nx1, ny1, nx2, ny2))
                
        return boxes
