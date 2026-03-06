import numpy as np
import os

path = "data/vlogger_profiles.npy"
if os.path.exists(path):
    data = np.load(path, allow_pickle=True)
    print(f"Data type: {type(data)}")
    print(f"Number of profiles (slots): {len(data)}")
    for i, profile in enumerate(data):
        print(f"Profile {i+1}: Type {type(profile)}, Length {len(profile)}")
        if len(profile) > 0:
            first_item = profile[0]
            print(f"  First item type: {type(first_item)}")
            if isinstance(first_item, dict):
                print(f"  Keys: {first_item.keys()}")
else:
    print(f"{path} does not exist.")
