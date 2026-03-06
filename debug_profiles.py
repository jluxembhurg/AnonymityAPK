import numpy as np
path = "data/vlogger_profiles.npy"
data = np.load(path, allow_pickle=True)
print("Data type:", type(data))
print("Data shape:", getattr(data, 'shape', 'No shape'))

if data.ndim == 0:
    data = data.item()
elif data.ndim == 1 and isinstance(data[0], dict):
    pass
elif data.ndim == 2:
    data = data[0]

# Check first item
if hasattr(data, '__len__') and len(data) > 0:
    first = data[0]
    print("First item type:", type(first))
    if isinstance(first, dict):
        v = first.get("vlogger")
        print("Vlogger emb type:", type(v), "shape:", getattr(v, 'shape', 'No shape'))
        
        if len(data) > 1:
            v2 = data[1].get("vlogger")
            dist = 1.0 - np.dot(v, v2)
            print("Distance between sample 1 and 2:", dist)
    else:
        print("First item:", first)
