import numpy as np
import cv2
import onnxruntime as ort

def test_gap():
    model_path = "models/recognizer.onnx"
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Generate two slightly shifted dummy faces
    # (Simulated shift in the 368x368 crop)
    img1 = np.random.randint(0, 255, (368, 368, 3), dtype=np.uint8)
    img2 = np.roll(img1, shift=5, axis=1) # Shifted 5 pixels
    
    def get_gap_emb(img):
        face_img = img.astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        
        output = session.run(None, {input_name: face_img})[0]
        # output is (1, 71, 46, 46)
        gap = np.mean(output, axis=(2, 3)).flatten()
        return gap / (np.linalg.norm(gap) + 1e-6)

    emb1 = get_gap_emb(img1)
    emb2 = get_gap_emb(img2)
    
    dist = np.linalg.norm(emb1 - emb2)
    print(f"Shifted Image Euclidean Distance (GAP): {dist:.4f}")
    
    # Compare with flattened
    def get_flat_emb(img):
        face_img = img.astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        output = session.run(None, {input_name: face_img})[0].flatten()
        return output / (np.linalg.norm(output) + 1e-6)
        
    emb1_f = get_flat_emb(img1)
    emb2_f = get_flat_emb(img2)
    dist_f = np.linalg.norm(emb1_f - emb2_f)
    print(f"Shifted Image Euclidean Distance (Flat): {dist_f:.4f}")

if __name__ == "__main__":
    test_gap()
