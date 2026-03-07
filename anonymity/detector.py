import cv2
import numpy as np
from typing import List, Tuple

class FaceDetector:
    """
    Wrapper for YOLOv8 face detection using OpenCV DNN module.
    Eliminates dependency on onnxruntime for better Android compatibility.
    """
    
    def __init__(self, model_path: str = "models/detector.onnx"):
        # Load the model with OpenCV DNN
        self.net = cv2.dnn.readNetFromONNX(model_path)
        # Use CPU for Android (standard Buildozer builds)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # YOLOv8 face model typically expects 640x640
        self.input_width = 640
        self.input_height = 640
        
        # Bounding box buffer percentage (15% as requested)
        self.buffer_pct = 0.15
        self.conf_threshold = 0.25

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess: RGB, 640x640, normalized 0-1, NCHW
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.input_width, self.input_height), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run inference
        outputs = self.net.forward()
        
        # YOLOv8 output: [1, 5, 8400] -> (x, y, w, h, score) for each anchor
        # Note: Depending on the specific export, it might be [1, 5, 8400] or [1, 8400, 5]
        if outputs.shape[1] < outputs.shape[2]:
            outputs = outputs[0] # [5, 8400]
        else:
            outputs = outputs[0].T # [5, 8400]

        boxes = []
        for i in range(outputs.shape[1]):
            score = outputs[4, i]
            if score > self.conf_threshold:
                # x, y are center coordinates
                cx, cy, w, h = outputs[0:4, i]
                
                # Scale to original frame
                x1 = (cx - w/2) * (orig_w / self.input_width)
                y1 = (cy - h/2) * (orig_h / self.input_height)
                x2 = (cx + w/2) * (orig_w / self.input_width)
                y2 = (cy + h/2) * (orig_h / self.input_height)

                # Apply buffer
                bw = x2 - x1
                bh = y2 - y1
                dx = int(bw * self.buffer_pct)
                dy = int(bh * self.buffer_pct)
                
                nx1 = max(0, int(x1 - dx))
                ny1 = max(0, int(y1 - dy))
                nx2 = min(orig_w, int(x2 + dx))
                ny2 = min(orig_h, int(y2 + dy))
                
                boxes.append((nx1, ny1, nx2, ny2))
                
        return boxes
