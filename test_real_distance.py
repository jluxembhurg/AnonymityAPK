import cv2
import numpy as np
from recognizer import FaceRecognizer

rec = FaceRecognizer()

def get_face_emb(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    # Assuming the whole image is a face or generic crop
    roi_proc = cv2.resize(img, (112, 112))
    return rec.get_vlogger_embedding(roi_proc)

emb1 = get_face_emb("test_face.jpg")
emb2 = get_face_emb("test_face2.jpg")

if emb1 is not None and emb2 is not None:
    dist = rec.calculate_distance(emb1, emb2)
    print(f"Distance between two DIFFERENT real faces: {dist:.4f}")
    
    dist_same = rec.calculate_distance(emb1, emb1)
    print(f"Distance between SAME face: {dist_same:.4f}")
