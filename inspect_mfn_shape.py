import onnxruntime as ort

session = ort.InferenceSession("models/mobilefacenet.onnx", providers=['CPUExecutionProvider'])
for input_node in session.get_inputs():
    print(f"Input name: {input_node.name}, shape: {input_node.shape}, type: {input_node.type}")
