import numpy as np
import os

def dist(e1, e2):
    return 1.0 - np.dot(e1, e2)

path = "data/vlogger_profiles.npy"
if os.path.exists(path):
    data = np.load(path, allow_pickle=True)
    for i, profile in enumerate(data):
        s_embs = [p["spatial"] for p in profile]
        v_embs = [p["vlogger"] for p in profile]
        
        s_dists = [dist(s_embs[j], s_embs[j+1]) for j in range(len(s_embs)-1)]
        v_dists = [dist(v_embs[j], v_embs[j+1]) for j in range(len(v_embs)-1)]
        
        print(f"PROFILE_{i+1}_SPATIAL_AVG={np.mean(s_dists):.4f}")
        print(f"PROFILE_{i+1}_VLOGGER_AVG={np.mean(v_dists):.4f}")
        print(f"PROFILE_{i+1}_COUNT={len(profile)}")
else:
    print("FILE_NOT_FOUND")
