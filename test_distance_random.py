import numpy as np
import cv2
from recognizer import FaceRecognizer

rec = FaceRecognizer()
path = "data/vlogger_profiles.npy"
galleries = [np.load(path, allow_pickle=True)]

v_profile = [p["vlogger"] for p in galleries[0]]

# Compare against a real face
test_img = cv2.imread("test_face.jpg")
if test_img is not None:
    v_emb_test = rec.get_vlogger_embedding(test_img)
    dist = rec.mean_top_k_distance(v_emb_test, v_profile, k=5)
    print(f"Distance for REAL STRANGER: {dist:.4f}")
else:
    print("Failed to load test_face.jpg")
