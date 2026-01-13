import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging
import time

# Configuration TensorFlow pour optimiser la performance CPU
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.set_visible_devices([], 'GPU')  # Désactiver GPU s'il existe

# Configuration
app = Flask(__name__)

# Configuration CORS avancée
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,
        "max_age": 3600
    }
})
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max au cas où
app.config['UPLOAD_FOLDER'] = 'uploads'

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer le dossier uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variables globales pour le modèle
MODEL = None
MODEL_LOADED = False

# Dictionnaire de mapping des classes aux billets
# 14 classes : Ordre exact du dataset (alphabétique)
BILL_LABELS = {
    0: "100 CDF",      # 100FC
    1: "50 CDF",       # 50FC
    2: "200 CDF",      # 200FC
    3: "500 CDF",      # 500FC
    4: "1000 CDF",     # 1000FC
    5: "5000 CDF",     # 5000FC
    6: "10000 CDF",    # 10000FC
    7: "20000 CDF",    # 20000FC
    8: "100 USD",      # 100$
    9: "5 USD",        # 5$
    10: "10 USD",      # 10$
    11: "50 USD",      # 50$
    12: "20 USD",      # 20$
    13: "1 USD",       # 1$
}

def load_model_on_startup():
    """Charge le modèle au démarrage"""
    global MODEL, MODEL_LOADED
    try:
        # Obtenir le répertoire du script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Chemin vers le modèle dans le même répertoire que app.py
        model_path = os.path.join(script_dir, 'model (1).h5')
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Modèle non trouvé à {model_path}")
            logger.info(f"📁 Répertoire courant: {script_dir}")
            logger.info(f"📁 Fichiers présents: {os.listdir(script_dir)}")
            return False
        
        logger.info(f"📂 Chargement du modèle depuis: {model_path}")
        MODEL = tf.keras.models.load_model(model_path)
        MODEL_LOADED = True
        
        # Afficher les infos du modèle
        logger.info("=" * 50)
        logger.info(f"✅ Modèle chargé avec succès!")
        logger.info(f"   - Input shape: {MODEL.input_shape}")
        logger.info(f"   - Output shape: {MODEL.output_shape}")
        logger.info(f"   - Nombre de paramètres: {MODEL.count_params()}")
        logger.info("=" * 50)
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def preprocess_image(image_path, target_size=(224, 224)):
    """Prétraite l'image pour le modèle - taille 224x224 comme lors de l'entraînement"""
    try:
        start_time = time.time()
        logger.info(f"📖 Ouverture de l'image: {image_path}")
        img = Image.open(image_path).convert('RGB')
        logger.info(f"✅ Image ouverte: {img.size}")
        
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        logger.info(f"✅ Image redimensionnée: {target_size}")
        
        img_array = np.array(img, dtype=np.float32)
        
        # IMPORTANT: Utilisez la fonction officielle au lieu de / 255.0
        # Elle convertit les pixels de [0, 255] à [-1, 1]
        img_array = preprocess_input(img_array)
        logger.info(f"✅ Image normalisée: min={img_array.min()}, max={img_array.max()}")
        
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter dimension batch
        logger.info(f"✅ Dimension batch ajoutée: {img_array.shape}")
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️ Prétraitement: {elapsed:.2f}s")
        
        return img_array
    except Exception as e:
        logger.error(f"❌ Erreur prétraitement image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Endpoint de vérification de santé"""
    if request.method == 'OPTIONS':
        return '', 204
    
    logger.info("✓ Health check reçu")
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'message': 'API Bill Recognition prête',
        'max_content_length': app.config['MAX_CONTENT_LENGTH']
    }), 200

@app.route('/', methods=['GET', 'HEAD'])
def index():
    """Route racine pour les health checks Render"""
    return 'Bill Recognition API is running', 200

@app.route('/test-upload', methods=['POST', 'OPTIONS'])
def test_upload():
    """Endpoint de test pour vérifier les uploads"""
    logger.info("=== TEST UPLOAD ===")
    logger.info(f"Content-Length: {request.content_length}")
    logger.info(f"Content-Type: {request.content_type}")
    
    if 'file' in request.files:
        file = request.files['file']
        logger.info(f"✓ Fichier reçu: {file.filename}")
        return jsonify({
            'status': 'ok',
            'filename': file.filename,
            'size': request.content_length
        }), 200
    else:
        logger.warning("✗ Pas de fichier reçu")
        return jsonify({'error': 'Pas de fichier'}), 400
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'message': 'API Bill Recognition prête'
    }), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Endpoint pour prédire le billet
    Attendu: Image multipart/form-data avec clé 'file'
    Retour: { "result": "100 USD", "confidence": 0.95 }
    """
    # Gérer les requêtes OPTIONS (CORS preflight)
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        logger.info("=" * 50)
        logger.info("🚀 NOUVELLE REQUÊTE /predict")
        logger.info("=" * 50)
        logger.info(f"📋 Content-Type: {request.content_type}")
        logger.info(f"📊 Content-Length: {request.content_length} bytes")
        
        # Vérifier la présence du fichier
        if 'file' not in request.files:
            logger.error("❌ Aucun fichier 'file' trouvé dans la requête")
            logger.error(f"   Fichiers présents: {list(request.files.keys())}")
            return jsonify({'error': 'Aucun fichier fourni. Clé attendue: "file"'}), 400
        
        file = request.files['file']
        logger.info(f"📦 Fichier trouvé: {file.filename}")
        
        if file.filename == '':
            logger.error("❌ Nom de fichier vide")
            return jsonify({'error': 'Fichier vide'}), 400
        
        # Vérifier l'extension
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            logger.error(f"❌ Extension non autorisée: .{file_ext}")
            return jsonify({'error': f'Format non autorisé. Autorisés: {allowed_extensions}'}), 400
        
        logger.info(f"✅ Extension autorisée: .{file_ext}")
        
        # Sauvegarder temporairement
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"💾 Fichier sauvegardé: {filepath}")
        
        # Vérifier que le modèle est chargé
        if not MODEL_LOADED:
            logger.info("🔄 Chargement du modèle...")
            if not load_model_on_startup():
                logger.error("❌ Impossible de charger le modèle")
                os.remove(filepath)
                return jsonify({'error': 'Modèle non disponible'}), 500
        
        # Prétraiter l'image
        logger.info("🖼️  Prétraitement de l'image...")
        img_array = preprocess_image(filepath)
        logger.info(f"✅ Image prétraitée: shape {img_array.shape}")
        
        # Prédire
        logger.info("🤖 Exécution de la prédiction...")
        try:
            pred_start = time.time()
            predictions = MODEL.predict(img_array, verbose=0)
            pred_time = time.time() - pred_start
            logger.info(f"✅ Prédictions reçues: {predictions.shape} en {pred_time:.2f}s")
        except Exception as pred_error:
            logger.error(f"❌ Erreur lors de la prédiction: {str(pred_error)}")
            import traceback
            logger.error(traceback.format_exc())
            os.remove(filepath)
            return jsonify({'error': f'Erreur prédiction: {str(pred_error)}'}), 500
        
        # Obtenir la classe prédite
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Afficher toutes les prédictions pour déboguer
        logger.info("📊 Toutes les prédictions:")
        for idx, prob in enumerate(predictions[0]):
            label = BILL_LABELS.get(idx, f"Classe {idx}")
            logger.info(f"   {label}: {float(prob):.2%}")
        
        # Vérifier le seuil de confiance (minimum 50%)
        MIN_CONFIDENCE = 0.50
        if confidence < MIN_CONFIDENCE:
            logger.warning(f"⚠️ Confiance trop basse: {confidence:.2%} < {MIN_CONFIDENCE:.0%}")
            os.remove(filepath)
            return jsonify({
                'error': f'Image peu claire - Confiance: {confidence:.2%}',
                'confidence': confidence,
                'top_guess': BILL_LABELS.get(predicted_class, "Inconnu")
            }), 400
        
        # Obtenir le label
        bill_label = BILL_LABELS.get(predicted_class, f"Billet inconnu (classe {predicted_class})")
        
        # Nettoyer
        os.remove(filepath)
        logger.info(f"🗑️  Fichier temporaire supprimé")
        
        logger.info(f"✅ SUCCÈS: {bill_label} (confiance: {confidence:.2%})")
        logger.info("=" * 50)
        
        # Parser le label pour extraire montant et devise
        # Format: "100 USD" ou "50000 CDF"
        parts = bill_label.split()
        amount = parts[0] if parts else "?"
        currency = parts[1] if len(parts) > 1 else "?"
        
        return jsonify({
            'result': bill_label,
            'amount': amount,
            'currency': currency,
            'confidence': confidence,
            'class': int(predicted_class)
        }), 200
        
    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"❌ ERREUR: {str(e)}")
        logger.error("=" * 50)
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Teste si le modèle fonctionne"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modèle non chargé'}), 503
    
    try:
        # Créer une image test aléatoire
        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        logger.info("🧪 Test du modèle avec image aléatoire")
        pred1 = MODEL.predict(test_image, verbose=0)
        
        # Deuxième test avec la même image
        pred2 = MODEL.predict(test_image, verbose=0)
        
        # Vérifier si les résultats sont identiques
        are_same = np.allclose(pred1, pred2)
        
        logger.info(f"Résultats identiques: {are_same}")
        logger.info(f"Prédiction 1: {pred1[0]}")
        logger.info(f"Prédiction 2: {pred2[0]}")
        
        return jsonify({
            'model_loaded': True,
            'test_results_identical': are_same,
            'prediction_1': pred1[0].tolist(),
            'prediction_2': pred2[0].tolist()
        }), 200
    except Exception as e:
        logger.error(f"❌ Erreur test modèle: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

@app.route('/model-info', methods=['GET', 'OPTIONS'])
def model_info():
    """Retourne les informations sur le modèle"""
    if request.method == 'OPTIONS':
        return '', 204
    
    if MODEL_LOADED:
        return jsonify({
            'model_loaded': True,
            'input_shape': str(MODEL.input_shape),
            'output_shape': str(MODEL.output_shape),
            'classes': len(BILL_LABELS),
            'labels': BILL_LABELS
        }), 200
    else:
        return jsonify({
            'model_loaded': False,
            'message': 'Modèle non chargé'
        }), 503

if __name__ == '__main__':
    logger.info("Démarrage de l'API Bill Recognition...")
    load_model_on_startup()
    
    # Force la désactivation du debug mode
    os.environ['FLASK_ENV'] = 'production'
    os.environ['FLASK_DEBUG'] = '0'
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Démarrage sur le port {port} en mode production")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
