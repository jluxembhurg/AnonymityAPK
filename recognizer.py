import cv2
import numpy as np
import onnxruntime as ort

class FaceRecognizer:
    def __init__(self, model_path: str = "models/recognizer.onnx"):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the face image for FaceNet.
        Model expects: 368x368, normalized.
        """
        face_img = cv2.resize(face_img, (368, 368))
        face_img = face_img.astype(np.float32)
        
        # Standard normalization: (x - 127.5) / 128
        face_img = (face_img - 127.5) / 128.0
        
        # Convert to NCHW
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract 128D or 512D embedding from a face image.
        """
        blob = self.preprocess(face_img)
        embedding = self.session.run(None, {self.input_name: blob})[0]
        # Normalize the embedding to unit length
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def calculate_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        """
        return np.linalg.norm(emb1 - emb2)

    def is_vlogger(self, emb: np.ndarray, profile_embs: list, threshold: float = 0.70) -> tuple[bool, float]:
        """
        Compare identity against vlogger profile.
        Returns (is_match, best_distance).
        """
        if profile_embs is None or len(profile_embs) == 0:
            return False, 1.0
            
        distances = [self.calculate_distance(emb, p_emb) for p_emb in profile_embs]
        min_dist = min(distances)
        return min_dist < threshold, min_dist
