import cv2
import numpy as np
from typing import List, Tuple

class FaceRecognizer:
    def __init__(self, original_path: str = "models/recognizer.onnx", 
                 vlogger_path: str = "models/mobilefacenet.onnx"):
        # Load Original Model (Spatial/GAP) with OpenCV DNN
        self.spatial_net = cv2.dnn.readNetFromONNX(original_path)
        self.spatial_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.spatial_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load MobileFaceNet (Primary Vlogger Identification)
        self.vlogger_net = cv2.dnn.readNetFromONNX(vlogger_path)
        self.vlogger_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.vlogger_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def get_spatial_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract spatial GAP descriptor from original model."""
        # Original model: 368x368, RGB, NCHW normalization
        blob = cv2.dnn.blobFromImage(face_img, 1/128.0, (368, 368), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.spatial_net.setInput(blob)
        output = self.spatial_net.forward()
        # GAP pooling
        embedding = np.mean(output, axis=(2, 3)).flatten()
        return self._l2_normalize(embedding)

    def get_vlogger_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract 128D identity vector from MobileFaceNet."""
        # MobileFaceNet: 112x112, RGB, NHWC normalization
        # Note: cv2.dnn.blobFromImage defaults to NCHW. 
        # If the model strictly needs NHWC, we might need manual preprocessing or a transpose.
        # However, most ONNX exports for MobileFaceNet expect NCHW or are flexible.
        # Let's check the original code: it was NHWC. 
        # To get NHWC with blobFromImage, we'd need to transpose the result.
        
        # Manual preprocessing to ensure NHWC if needed:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (112, 112))
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        blob = np.expand_dims(face_img, axis=0)
        
        self.vlogger_net.setInput(blob)
        embedding = self.vlogger_net.forward()
        return self._l2_normalize(embedding.flatten())

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm > 1e-6:
            return x / norm
        return x

    def calculate_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine Distance (1 - Cosine Similarity)."""
        return 1.0 - np.dot(emb1, emb2)

    def mean_top_k_distance(self, emb: np.ndarray, profile_embs: list, k: int = 3) -> float:
        if profile_embs is None or len(profile_embs) == 0:
            return 1.0
        distances = sorted([self.calculate_distance(emb, p_emb) for p_emb in profile_embs])
        top_k = distances[:min(k, len(distances))]
        return sum(top_k) / len(top_k)

    def is_vlogger(self, emb: np.ndarray, profile_embs: list, threshold: float = 0.35) -> tuple[bool, float]:
        """Verify against vlogger gallery."""
        if profile_embs is None or len(profile_embs) == 0:
            return False, 1.0
        dist = self.mean_top_k_distance(emb, profile_embs)
        return dist < threshold, dist
