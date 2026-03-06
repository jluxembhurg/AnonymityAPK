import cv2
import numpy as np
import onnxruntime as ort

class FaceRecognizer:
    def __init__(self, original_path: str = "models/recognizer.onnx", 
                 vlogger_path: str = "models/mobilefacenet.onnx"):
        # Load Original Model (Spatial/GAP)
        self.spatial_session = ort.InferenceSession(original_path, providers=['CPUExecutionProvider'])
        self.spatial_input = self.spatial_session.get_inputs()[0].name
        
        # Load MobileFaceNet (Primary Vlogger Identification)
        self.vlogger_session = ort.InferenceSession(vlogger_path, providers=['CPUExecutionProvider'])
        self.vlogger_input = self.vlogger_session.get_inputs()[0].name
        
    def preprocess_spatial(self, face_img: np.ndarray) -> np.ndarray:
        """Original model: 368x368, RGB, NCHW normalization."""
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (368, 368))
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def preprocess_vlogger(self, face_img: np.ndarray) -> np.ndarray:
        """MobileFaceNet: 112x112, RGB, NHWC normalization."""
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (112, 112))
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def get_spatial_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract spatial GAP descriptor from original model."""
        blob = self.preprocess_spatial(face_img)
        output = self.spatial_session.run(None, {self.spatial_input: blob})[0]
        # GAP pooling
        embedding = np.mean(output, axis=(2, 3)).flatten()
        return self._l2_normalize(embedding)

    def get_vlogger_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract 128D identity vector from MobileFaceNet."""
        blob = self.preprocess_vlogger(face_img)
        embedding = self.vlogger_session.run(None, {self.vlogger_input: blob})[0]
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
