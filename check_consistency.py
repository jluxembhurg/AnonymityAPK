import numpy as np
import os
import cv2

# Mock recognizer for distance calculation
class MockRecognizer:
    def calculate_distance(self, emb1, emb2):
        return 1.0 - np.dot(emb1, emb2)

path = "data/vlogger_profiles.npy"
if os.path.exists(path):
    data = np.load(path, allow_pickle=True)
    rec = MockRecognizer()
    
    for i, profile in enumerate(data):
        print(f"\nProfile {i+1} Consistency Check:")
        spatial_embs = [p["spatial"] for p in profile]
        vlogger_embs = [p["vlogger"] for p in profile]
        
        # Check Spatial consistency
        s_dists = []
        for j in range(len(spatial_embs)-1):
            s_dists.append(rec.calculate_distance(spatial_embs[j], spatial_embs[j+1]))
        print(f"  Spatial Distances: Avg {np.mean(s_dists):.4f}, Max {np.max(s_dists):.4f}")
        
        # Check Vlogger consistency
        v_dists = []
        for j in range(len(vlogger_embs)-1):
            v_dists.append(rec.calculate_distance(vlogger_embs[j], vlogger_embs[j+1]))
        print(f"  Vlogger Distances: Avg {np.mean(v_dists):.4f}, Max {np.max(v_dists):.4f}")
        
        if np.mean(v_dists) > 0.5:
            print("  WARNING: Vlogger embeddings are very inconsistent!")
        if np.mean(s_dists) > 0.6:
            print("  WARNING: Spatial embeddings are very inconsistent!")
else:
    print(f"{path} does not exist.")
