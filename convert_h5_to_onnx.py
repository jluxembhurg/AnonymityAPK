import tensorflow as tf
import tf2onnx
import onnx
import sys

def convert_h5_to_onnx(h5_path, onnx_path):
    print(f"Loading {h5_path}...")
    try:
        model = tf.keras.models.load_model(h5_path)
        spec = (tf.TensorSpec((None, 112, 112, 3), tf.float32, name="input"),)
        
        print("Converting to ONNX...")
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        
        onnx.save(model_proto, onnx_path)
        if os.path.exists(onnx_path):
            print(f"SUCCESS: Saved to {os.path.abspath(onnx_path)}")
        else:
            print(f"FAILURE: File not found after save at {onnx_path}")
    except Exception as e:
        print(f"CRITICAL Error during conversion: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_script.py input.h5 output.onnx")
    else:
        convert_h5_to_onnx(sys.argv[1], sys.argv[2])
