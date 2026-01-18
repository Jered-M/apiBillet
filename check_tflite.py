import tensorflow as tf
import os

tflite_path = r'c:\Users\HP\Downloads\model (1).tflite'

if not os.path.exists(tflite_path):
    print(f"Fichier non trouvé: {tflite_path}")
    exit(1)

try:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    output_details = interpreter.get_output_details()
    input_details = interpreter.get_input_details()
    
    print("=" * 50)
    print("TFLite Model Info: model (1).tflite")
    print("=" * 50)
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    num_classes = output_details[0]['shape'][-1]
    print(f"Nombre de classes: {num_classes}")
    print("=" * 50)
    
except Exception as e:
    print(f"Erreur: {e}")
    import traceback
    traceback.print_exc()
