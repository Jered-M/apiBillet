import os
import time
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

import tensorflow as tf

# Support optionnel pour TFLite
try:
    import tensorflow.lite as tflite
    HAS_TFLITE = True
except:
    HAS_TFLITE = False

# =========================
# CONFIGURATION DU LOGGER
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURATION GLOBALE
# =========================

IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "uploads"
MIN_CONFIDENCE = 0.50

# Labels (14 classes - selon le modèle Colab)
# Le modèle sur Colab a 14 classes, le model.h5 local en a 12
# Adapter selon le modèle réel chargé
BILL_LABELS = {
    0: "100 CDF",
    1: "50 CDF",
    2: "200 CDF",
    3: "500 CDF",
    4: "1000 CDF",
    5: "5000 CDF",
    6: "10000 CDF",
    7: "20000 CDF",
    8: "100 USD",
    9: "5 USD",
    10: "10 USD",
    11: "50 USD",
    12: "20 USD",
    13: "1 USD",
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# TensorFlow CPU safe
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# =========================
# FLASK APP
# =========================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB max

CORS(app)

# =========================
# LOAD MODEL
# =========================

MODEL = None
TFLITE_INTERPRETER = None

def load_model_simple():
    """
    Charge le modèle avec priorité:
    1. TFLite (optimisé, plus rapide)
    2. H5 Keras (fallback)
    """
    global MODEL, TFLITE_INTERPRETER
    
    logger.info("📦 Chargement du modèle...")
    
    # Essayer TFLite d'abord (plus rapide et optimisé)
    if os.path.exists("model (1).tflite"):
        try:
            logger.info("📍 Tentative TFLite: model (1).tflite")
            TFLITE_INTERPRETER = tf.lite.Interpreter(model_path="model (1).tflite")
            TFLITE_INTERPRETER.allocate_tensors()
            
            output_details = TFLITE_INTERPRETER.get_output_details()
            num_classes = output_details[0]['shape'][-1]
            
            logger.info(f"✅ TFLite chargé: model (1).tflite")
            logger.info(f"  Output shape: {output_details[0]['shape']}")
            logger.info(f"  Nombre de classes: {num_classes}")
            return True
        except Exception as e:
            logger.warning(f"⚠️  Erreur TFLite: {e}")
            TFLITE_INTERPRETER = None
    
    # Fallback sur H5 Keras
    model_paths = ["model.h5", "model (1).h5", "best_model.h5"]
    
    for model_file in model_paths:
        if os.path.exists(model_file):
            try:
                logger.info(f"📍 Tentative H5: {model_file}")
                MODEL = tf.keras.models.load_model(model_file)
                logger.info(f"✅ H5 chargé: {model_file}")
                logger.info(f"  Input shape : {MODEL.input_shape}")
                logger.info(f"  Output shape: {MODEL.output_shape}")
                logger.info(f"  Params: {MODEL.count_params():,}")
                return True
            except Exception as e:
                logger.warning(f"⚠️  Erreur H5 {model_file}: {e}")
                continue
    
    logger.error("❌ Aucun modèle n'a pu être chargé")
    return False

try:
    if load_model_simple():
        logger.info("🚀 Modèle chargé et prêt!")
    else:
        logger.error("L'API va démarrer mais retournera une erreur pour les prédictions")
except Exception as e:
    logger.error(f"❌ Erreur au démarrage: {e}")
    MODEL = None
    TFLITE_INTERPRETER = None

# =========================
# IMAGE PREPROCESS (COMPATIBLE MODEL)
# =========================

def preprocess_image(image_path):
    """
    Prétraitement IDENTIQUE au modèle entraîné.
    
    Le modèle a été entraîné avec:
    - ImageDataGenerator(rescale=1./255)
    - flow_from_directory avec target_size=(224, 224)
    - PIL Image.load_img (utilise LANCZOS par défaut)
    
    Pipeline:
    1. Charger l'image
    2. Corriger l'orientation EXIF (pour iPhone)
    3. Convertir en RGB
    4. Redimensionner à 224x224 avec LANCZOS (COMME ImageDataGenerator)
    5. Normaliser par 255.0 (EXACTEMENT comme rescale=1./255)
    """
    img = Image.open(image_path)
    
    # Corriger l'orientation EXIF (important pour les photos iPhone)
    img = ImageOps.exif_transpose(img)
    
    # Convertir en RGB (ImageDataGenerator le fait automatiquement)
    img = img.convert('RGB')
    
    # Redimensionner avec LANCZOS (algorithme par défaut de PIL pour downsampling)
    # C'est ce qu'utilise ImageDataGenerator/Keras par défaut
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Convertir en array
    img_array = np.array(img, dtype=np.float32)
    
    # Normaliser par 255.0 (EXACTEMENT rescale=1./255)
    # Convertit [0, 255] → [0, 1]
    img_array = img_array / 255.0
    
    # Ajouter dimension batch (comme model.predict() l'attend)
    img_array = np.expand_dims(img_array, axis=0)
    
    logger.info(f"✅ Image prétraitée - Shape: {img_array.shape}, Range: [{img_array.min():.2f}, {img_array.max():.2f}]")
    
    return img_array

# =========================
# INFERENCE FUNCTION
# =========================

def predict_tflite(img_array):
    """Prédit avec le modèle TFLite"""
    try:
        input_details = TFLITE_INTERPRETER.get_input_details()
        output_details = TFLITE_INTERPRETER.get_output_details()
        
        # Convertir en float32 si nécessaire
        if input_details[0]['dtype'] == np.float32:
            img_array = img_array.astype(np.float32)
        
        TFLITE_INTERPRETER.set_tensor(input_details[0]['index'], img_array)
        TFLITE_INTERPRETER.invoke()
        
        predictions = TFLITE_INTERPRETER.get_tensor(output_details[0]['index'])
        num_classes = output_details[0]['shape'][-1]
        
        return predictions[0], num_classes
    except Exception as e:
        logger.error(f"Erreur TFLite: {e}")
        raise

def predict_keras(img_array):
    """Prédit avec le modèle Keras H5"""
    try:
        predictions = MODEL.predict(img_array, verbose=0)
        num_classes = predictions.shape[-1]
        return predictions[0], num_classes
    except Exception as e:
        logger.error(f"Erreur Keras: {e}")
        raise

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET"])
def index():
    return "Bill Recognition API running", 200


@app.route("/health", methods=["GET"])
def health():
    model_info = {
        "model_loaded": TFLITE_INTERPRETER is not None or MODEL is not None,
        "model_type": "tflite" if TFLITE_INTERPRETER is not None else ("keras_h5" if MODEL is not None else "none"),
    }
    
    if TFLITE_INTERPRETER is not None:
        try:
            output_details = TFLITE_INTERPRETER.get_output_details()
            model_info["output_shape"] = str(output_details[0]['shape'])
            model_info["num_classes"] = output_details[0]['shape'][-1]
            model_info["file"] = "model (1).tflite"
        except:
            pass
    elif MODEL is not None:
        try:
            model_info["input_shape"] = str(MODEL.input_shape)
            model_info["output_shape"] = str(MODEL.output_shape)
            model_info["params"] = MODEL.count_params()
            model_info["num_classes"] = MODEL.output_shape[-1] if MODEL.output_shape else "unknown"
        except:
            pass
    
    is_ready = TFLITE_INTERPRETER is not None or MODEL is not None
    return jsonify({
        "status": "ok" if is_ready else "model_missing",
        "model": model_info,
        "port": 5000
    }), 200 if is_ready else 503


@app.route("/debug/upload", methods=["POST"])
def debug_upload():
    """Endpoint de debug pour tester les uploads"""
    logger.info("🔍 DEBUG: Request reçue")
    logger.info(f"  Content-Type: {request.content_type}")
    logger.info(f"  Form keys: {list(request.form.keys())}")
    logger.info(f"  Files keys: {list(request.files.keys())}")
    logger.info(f"  Args keys: {list(request.args.keys())}")
    
    if "file" in request.files:
        file = request.files["file"]
        logger.info(f"  File name: {file.filename}")
        logger.info(f"  File size: {len(file.read())} bytes")
        file.seek(0)
        return jsonify({
            "debug": "File reçu avec succès",
            "filename": file.filename,
            "size": len(file.read())
        }), 200
    else:
        return jsonify({
            "error": "Pas de fichier détecté",
            "files_keys": list(request.files.keys()),
            "content_type": request.content_type
        }), 400


@app.route("/debug/save-raw", methods=["POST"])
def debug_save_raw():
    """Sauvegarde l'image brute SANS preprocessing pour test Colab"""
    logger.info("💾 DEBUG: Sauvegarde image brute pour test")
    
    if "file" not in request.files:
        return jsonify({"error": "Pas de fichier"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nom vide"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"raw_{filename}")
    
    try:
        file.save(filepath)
        logger.info(f"✅ Image brute sauvegardée: {filepath}")
        return jsonify({
            "message": "Image sauvegardée",
            "path": filepath,
            "instruction": "Télécharge cette image et teste dans Colab"
        }), 200
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint de prédiction - EXACTEMENT comme dans le notebook
    
    Logique:
    1. Reçoit une image (file)
    2. Prétraite (redimensionner 224x224, normaliser /255)
    3. Prédit la classe (TFLite priorité, puis H5)
    4. Retourne la classe et la confiance
    """
    start_time = time.time()
    
    # Vérifier les modèles
    if TFLITE_INTERPRETER is None and MODEL is None:
        logger.error("❌ Aucun modèle disponible")
        return jsonify({"error": "Modèle non chargé"}), 503

    # Vérifier le fichier
    if "file" not in request.files:
        logger.error("❌ Pas de fichier")
        return jsonify({"error": "Pas de fichier envoyé"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.error("❌ Filename vide")
        return jsonify({"error": "Filename vide"}), 400

    # Vérifier l'extension
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"jpg", "jpeg", "png", "gif", "bmp"}:
        return jsonify({"error": f"Format non supporté: {ext}"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        # Sauvegarder temporairement
        file.save(filepath)
        logger.info(f"✅ Image reçue: {filename}")
        
        # Prétraiter (comme dans le notebook)
        img_array = preprocess_image(filepath)
        logger.info(f"✅ Image prétraitée - Shape: {img_array.shape}")
        
        # Prédire avec TFLite (priorité) ou H5 (fallback)
        if TFLITE_INTERPRETER is not None:
            logger.info("🔮 Utilisation: TFLite")
            predictions, num_classes = predict_tflite(img_array)
        else:
            logger.info("🔮 Utilisation: Keras H5")
            predictions, num_classes = predict_keras(img_array)
        
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        
        # Obtenir le label
        if predicted_class_idx < len(BILL_LABELS):
            predicted_label = BILL_LABELS.get(predicted_class_idx, f"Unknown ({predicted_class_idx})")
        else:
            predicted_label = f"Unknown ({predicted_class_idx})"
        
        logger.info(f"🎯 Prédiction: {predicted_label} ({confidence:.2%}) [Classes: {num_classes}]")
        
        # Retourner le résultat (format comme notebook)
        return jsonify({
            "prediction": predicted_label,
            "confidence": f"{confidence:.2%}",
            "confidence_value": confidence,
            "class_index": predicted_class_idx,
            "num_classes": num_classes,
            "model_type": "tflite" if TFLITE_INTERPRETER is not None else "keras_h5",
            "processing_time": round(time.time() - start_time, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {e}")
        return jsonify({"error": f"Erreur: {str(e)}"}), 500
    
    finally:
        # Nettoyer
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"🗑️  Image temporaire supprimée")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
