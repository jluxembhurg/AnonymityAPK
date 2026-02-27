import onnxruntime as ort

def inspect_model(model_path):
    print(f"Inspecting {model_path}...")
    session = ort.InferenceSession(model_path)
    for model_input in session.get_inputs():
        print(f"Input Name: {model_input.name}")
        print(f"Input Shape: {model_input.shape}")
        print(f"Input Type: {model_input.type}")
    
    for model_output in session.get_outputs():
        print(f"Output Name: {model_output.name}")
        print(f"Output Shape: {model_output.shape}")
        print(f"Output Type: {model_output.type}")

if __name__ == "__main__":
    inspect_model("models/recognizer.onnx")
