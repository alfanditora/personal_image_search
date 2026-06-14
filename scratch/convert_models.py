import tensorflow as tf
import tf2onnx
from deepface import DeepFace

try:
    print("Building VGG-Face Keras model...")
    vgg_model = DeepFace.build_model("VGG-Face").model
    print("Converting VGG-Face to ONNX with large_model=True...")
    spec_vgg = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        vgg_model, 
        input_signature=spec_vgg, 
        output_path="models/vgg_face.onnx",
        large_model=True
    )
    print("VGG-Face ONNX conversion completed successfully!")
except Exception as e:
    print(f"Failed to convert VGG-Face: {str(e)}")
