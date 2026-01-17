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

# Try multiple model paths (in order of preference)
MODEL_PATHS = [
    "model.tflite",       # TFLite (preferred - 4x smaller)
    "model (1).h5",       # Primary model (Keras 3 compatible)
    "best_model.h5",      # Fallback H5 model
    "model.h5",           # Fallback H5 model
    "model_saved",        # SavedModel format (last resort - needs tf.saved_model.load)
]

MIN_CONFIDENCE = 0.50

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
# LABELS (ORDRE DATASET)
# =========================

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

# =========================
# LOAD MODEL
# =========================

MODEL = None
TFLITE_INTERPRETER = None

def load_model():
    """
    Essayer de charger le modèle en testant chaque fichier séquentiellement.
    Passer au suivant si le chargement échoue.
    """
    global MODEL, TFLITE_INTERPRETER
    logger.info("📦 Chargement du modèle...")
    
    # Liste de tous les fichiers H5/TFLite/SavedModel à essayer
    all_models_to_try = []
    
    # Vérifier les fichiers locaux
    for path in MODEL_PATHS:
        if path.endswith(".tflite") and os.path.exists(path):
            all_models_to_try.append((path, "tflite"))
        elif path == "model_saved" and os.path.isdir(path):
            all_models_to_try.append((path, "saved_model"))
        elif os.path.exists(path) and not path.endswith(".tflite"):
            all_models_to_try.append((path, "h5"))
    
    if not all_models_to_try:
        logger.error("❌ Aucun fichier modèle trouvé!")
        raise FileNotFoundError("Aucun modèle valide trouvé")
    
    # Essayer chaque fichier
    last_error = None
    for model_file, model_format in all_models_to_try:
        try:
            logger.info(f"📍 Tentative avec: {model_file}")
            
            if model_format == "tflite":
                logger.info("⚡ Chargement TFLite...")
                TFLITE_INTERPRETER = tf.lite.Interpreter(model_path=model_file)
                TFLITE_INTERPRETER.allocate_tensors()
                logger.info(f"✅ TFLite chargé: {model_file}")
                return  # Succès
                
            elif model_format == "saved_model":
                logger.info("❌ SavedModel non supporté par Keras 3")
                continue  # Sauter au suivant
                
            else:  # H5
                logger.info("📦 Chargement H5...")
                MODEL = tf.keras.models.load_model(model_file)
                logger.info(f"✅ H5 chargé: {model_file}")
                
                # Afficher les infos du modèle
                try:
                    logger.info(f"  Input shape : {MODEL.input_shape}")
                    logger.info(f"  Output shape: {MODEL.output_shape}")
                    logger.info(f"  Params: {MODEL.count_params():,}")
                except Exception as e:
                    logger.warning(f"⚠️  Impossible d'accéder aux infos: {e}")
                
                return  # Succès
                
        except Exception as e:
            last_error = e
            error_type = type(e).__name__
            logger.warning(f"⚠️  {error_type} avec {model_file}: {str(e)[:100]}")
            continue  # Essayer le suivant
    
    # Tous les modèles ont échoué
    logger.error(f"❌ Impossible de charger tous les modèles disponibles")
    logger.error(f"Dernière erreur: {type(last_error).__name__}: {str(last_error)}")
    raise last_error if last_error else FileNotFoundError("Aucun modèle n'a pu être chargé")

try:
    load_model()
    logger.info("🚀 Modèle chargé et prêt!")
except Exception as e:
    logger.error(f"❌ Impossible de charger le modèle au démarrage: {type(e).__name__}: {str(e)}")
    logger.error("L'API va démarrer mais retournera une erreur pour les prédictions")
    MODEL = None

# =========================
# IMAGE PREPROCESS (COMPATIBLE MODEL)
# =========================

def preprocess_image(image_path):
    """
    Prétraitement compatible avec le modèle entraîné :
    Le modèle utilise rescale=1./255 (normalisation [0, 1])
    
    1. Charger l'image
    2. Corriger l'orientation EXIF (CRITIQUE pour iPhone)
    3. Convertir en RGB
    4. Redimensionner avec BICUBIC
    5. Normaliser par 255.0 (COMME LE MODÈLE)
    """
    img = Image.open(image_path)
    
    # 🔥 ÉTAPE CRITIQUE : Corriger l'orientation EXIF
    img = ImageOps.exif_transpose(img)
    
    # Convertir en RGB
    img = img.convert('RGB')
    
    # Crop automatique (optionnel, pour cenrer le billet)
    # img = img.crop(img.getbbox())
    
    # Redimensionner avec BICUBIC (meilleure qualité)
    img = img.resize(IMG_SIZE, Image.Resampling.BICUBIC)
    
    # Convertir en array
    img_array = np.array(img, dtype=np.float32)
    
    # 🔥 NORMALISATION: diviser par 255.0 (comme le modèle entraîné)
    # Le modèle a été entraîné avec rescale=1./255 → [0, 1]
    img_array = img_array / 255.0
    
    # Ajouter dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    logger.info(f"✅ Image prétraitée - Shape: {img_array.shape}, Min: {img_array.min():.2f}, Max: {img_array.max():.2f}")
    
    return img_array

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET"])
def index():
    return "Bill Recognition API running", 200


@app.route("/health", methods=["GET"])
def health():
    model_info = {
        "model_loaded": MODEL is not None or TFLITE_INTERPRETER is not None,
        "model_type": "keras_h5" if MODEL is not None else ("tflite" if TFLITE_INTERPRETER is not None else "none"),
    }
    
    if MODEL is not None:
        try:
            model_info["input_shape"] = str(MODEL.input_shape)
            model_info["output_shape"] = str(MODEL.output_shape)
            model_info["params"] = MODEL.count_params()
        except:
            pass
    
    return jsonify({
        "status": "ok" if (MODEL is not None or TFLITE_INTERPRETER is not None) else "model_missing",
        "model": model_info,
        "port": 5000
    }), 200 if (MODEL is not None or TFLITE_INTERPRETER is not None) else 503


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
    start_time = time.time()
    
    # Log de debug
    logger.info(f"📨 Request reçue - Content-Type: {request.content_type}")
    logger.info(f"   Form keys: {list(request.form.keys())}")
    logger.info(f"   Files keys: {list(request.files.keys())}")

    # Vérifier que le modèle est chargé
    if MODEL is None and TFLITE_INTERPRETER is None:
        error_msg = "Modèle non disponible"
        logger.error(f"❌ {error_msg}")
        return jsonify({
            "error": error_msg,
            "message": "Aucun modèle n'a pu être chargé au démarrage"
        }), 503

    if "file" not in request.files:
        logger.error(f"❌ Erreur 400: Pas de fichier 'file'")
        logger.error(f"   Files reçus: {list(request.files.keys())}")
        return jsonify({
            "error": "Aucun fichier envoyé",
            "expected_key": "file",
            "received_keys": list(request.files.keys())
        }), 400

    file = request.files["file"]

    if file.filename == "":
        logger.error("❌ Erreur 400: Nom de fichier vide")
        return jsonify({"error": "Nom de fichier vide"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"jpg", "jpeg", "png"}:
        logger.error(f"❌ Erreur 400: Format non supporté: {ext}")
        return jsonify({"error": f"Format non supporté: {ext}"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    logger.info(f"📥 Réception fichier: {filename}")
    logger.info(f"📥 Extension: {ext}")
    logger.info(f"📥 Taille: {len(file.read())} bytes")
    file.seek(0)  # Reset file pointer après lecture
    
    try:
        file.save(filepath)
        logger.info(f"✅ Fichier sauvegardé: {filepath}")
        file_size = os.path.getsize(filepath)
        logger.info(f"✅ Taille sauvegardée: {file_size} bytes")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde: {e}")
        return jsonify({"error": f"Erreur sauvegarde: {e}"}), 500

    try:
        img = preprocess_image(filepath)

        # Prédiction avec TFLite ou Keras
        if TFLITE_INTERPRETER is not None:
            # TFLite inference
            input_details = TFLITE_INTERPRETER.get_input_details()
            output_details = TFLITE_INTERPRETER.get_output_details()
            
            # Adapter l'input
            input_data = img.astype(input_details[0]['dtype'])
            TFLITE_INTERPRETER.set_tensor(input_details[0]['index'], input_data)
            TFLITE_INTERPRETER.invoke()
            
            # Récupérer l'output
            preds = TFLITE_INTERPRETER.get_tensor(output_details[0]['index'])[0]
        else:
            # Keras inference
            preds = MODEL.predict(img, verbose=0)[0]

        predicted_class = int(np.argmax(preds))
        confidence = float(preds[predicted_class])

        logger.info("📊 Prédictions:")
        for i, p in enumerate(preds):
            logger.info(f"{BILL_LABELS[i]} → {p:.2%}")

        if confidence < MIN_CONFIDENCE:
            return jsonify({
                "error": "Confiance trop faible",
                "confidence": confidence
            }), 400

        label = BILL_LABELS[predicted_class]
        amount, currency = label.split()

        return jsonify({
            "result": label,
            "amount": amount,
            "currency": currency,
            "confidence": confidence,
            "class": predicted_class,
            "processing_time": round(time.time() - start_time, 2)
        }), 200

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

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
