"""
Flask API pour reconnaissance de billets - VERSION SavedModel
"""
import os
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Classes de dénominations
CLASS_NAMES = [
    "100_CDF", "50_CDF", "200_CDF", "500_CDF", "1000_CDF", 
    "5000_CDF", "10000_CDF", "20000_CDF",
    "1_USD", "5_USD", "10_USD", "20_USD"
]

# Variables globales
MODEL = None
SAVED_MODEL_PATH = "model_saved"

def load_model():
    """Charge le SavedModel"""
    global MODEL
    logger.info(f"📦 Chargement du modèle depuis {SAVED_MODEL_PATH}...")
    try:
        MODEL = tf.saved_model.load(SAVED_MODEL_PATH)
        logger.info("✅ SavedModel chargé")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False

def preprocess_image(image, target_size=(224, 224)):
    """Prétraite une image pour la prédiction"""
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionner
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convertir en array et normaliser
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Ajouter dimension batch
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_from_array(image_array):
    """Fait une prédiction à partir d'un array"""
    if MODEL is None:
        return None, "Model not loaded"
    
    try:
        # Utiliser la signature serving_default
        concrete_func = MODEL.signatures['serving_default']
        
        # Préparer l'input
        input_tensor = tf.constant(image_array)
        
        # Faire la prédiction
        output = concrete_func(input_layer_1=input_tensor)
        
        # Extraire les logits
        logits = output['output_0'].numpy()[0]
        
        # Softmax pour obtenir les probabilités
        probabilities = tf.nn.softmax(logits).numpy()
        
        # Top 5 prédictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_probs = probabilities[top_indices]
        
        result = {
            'predictions': [
                {
                    'class': CLASS_NAMES[idx],
                    'confidence': float(prob)
                }
                for idx, prob in zip(top_indices, top_probs)
            ],
            'all_confidences': {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(CLASS_NAMES))
            }
        }
        
        return result, None
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {e}")
        return None, str(e)

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'model_loaded': MODEL is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Lire l'image
        image_bytes = file.read()
        
        # Prétraiter
        image_array = preprocess_image(image_bytes)
        
        # Prédire
        result, error = predict_from_array(image_array)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Information sur le modèle"""
    return jsonify({
        'model_path': SAVED_MODEL_PATH,
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'model_loaded': MODEL is not None
    })

if __name__ == '__main__':
    if not load_model():
        logger.error("❌ Impossible de charger le modèle - API non fonctionnelle")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
