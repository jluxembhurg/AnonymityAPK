import onnxruntime as ort
import sys

def inspect_model(path):
    try:
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        print(f"Model: {path}")
        print("\n--- Inputs ---")
        for i in session.get_inputs():
            print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        
        print("\n--- Outputs ---")
        for o in session.get_outputs():
            print(f"Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
    except Exception as e:
        print(f"Error inspecting {path}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_model(sys.argv[1])
    else:
        print("Usage: python inspect_onnx.py <model_path>")
