import urllib.request
import os

def download_model():
    url = "https://huggingface.co/doguscank/facenet-onnx/resolve/main/facenet.onnx"
    output_path = "models/recognizer.onnx"
    
    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        return

    print(f"Downloading model from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    download_model()
