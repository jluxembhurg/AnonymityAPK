import numpy as np
import os
from recognizer import FaceRecognizer

def diagnose():
    recognizer = FaceRecognizer()
    profiles_path = "data/vlogger_profiles.npy"
    
    if not os.path.exists(profiles_path):
        print("No profiles found.")
        return
        
    galleries = np.load(profiles_path, allow_pickle=True)
    print(f"Loaded {len(galleries)} vlogger galleries.")
    
    for i, profile in enumerate(galleries):
        print(f"\n--- Profile {i+1} ({len(profile)} embeddings) ---")
        if len(profile) > 0:
            print(f"Shape of first embedding: {profile[0].shape}")
        
        # Check internal distances (Intra-class variance)
        distances = []
        for j in range(len(profile)):
            for k in range(j+1, len(profile)):
                dist = recognizer.calculate_distance(profile[j], profile[k])
                distances.append(dist)
        
        if distances:
            print(f"Mean Internal Distance: {np.mean(distances):.4f}")
            print(f"Max Internal Distance: {np.max(distances):.4f}")
            print(f"Min Internal Distance: {np.min(distances):.4f}")
        else:
            print("Single embedding profile. Internal distance N/A.")

    # Inter-profile distance if 2 profiles
    if len(galleries) == 2:
        profile1 = galleries[0]
        profile2 = galleries[1]
        inter_dist = []
        for p1 in profile1:
            for p2 in profile2:
                inter_dist.append(recognizer.calculate_distance(p1, p2))
        print(f"\n--- Inter-Profile distance (P1 vs P2) ---")
        print(f"Mean: {np.mean(inter_dist):.4f}")
        print(f"Min: {np.min(inter_dist):.4f}")

if __name__ == "__main__":
    diagnose()
