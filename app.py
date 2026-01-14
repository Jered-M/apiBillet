import os
import time
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CONFIGURATION GLOBALE
# =========================

IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "uploads"

# Try multiple model paths
MODEL_PATHS = [
    "model.h5",           # Primary model
    "model (1).h5",       # Fallback model
    "best_model.h5",      # Alternative model
]

MODEL_PATH = None
for path in MODEL_PATHS:
    if os.path.exists(path) and os.path.getsize(path) > 1000000:  # > 1MB
        MODEL_PATH = path
        break

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BillAPI")

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

def load_model():
    global MODEL
    logger.info("📦 Chargement du modèle...")
    
    if not MODEL_PATH:
        logger.error("❌ Aucun fichier modèle valide trouvé!")
        logger.error("Fichiers détectés:")
        for path in MODEL_PATHS:
            size = os.path.getsize(path) if os.path.exists(path) else 0
            logger.error(f"  - {path}: {size} bytes")
        raise FileNotFoundError("Aucun modèle valide trouvé")
    
    try:
        logger.info(f"Chargement depuis: {MODEL_PATH}")
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        logger.info("✅ Modèle chargé avec succès")
        logger.info(f"Input shape : {MODEL.input_shape}")
        logger.info(f"Output shape: {MODEL.output_shape}")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        raise

try:
    load_model()
except Exception as e:
    logger.error(f"⚠️  Impossible de charger le modèle au démarrage: {str(e)}")
    MODEL = None

# =========================
# IMAGE PREPROCESS (CORRECT PIPELINE)
# =========================

def preprocess_image(image_path):
    """
    Prétraitement CORRECT pour MobileNetV2 :
    1. Charger l'image
    2. Corriger l'orientation EXIF (CRITIQUE pour iPhone)
    3. Convertir en RGB
    4. Redimensionner avec BICUBIC
    5. Appliquer preprocess_input MobileNetV2
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
    
    # 🔥 ÉTAPE CRITIQUE : preprocess_input MobileNetV2
    img_array = preprocess_input(img_array)
    
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
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    # Vérifier que le modèle est chargé
    if MODEL is None:
        return jsonify({
            "error": "Modèle non disponible",
            "message": "Le modèle n'a pas pu être chargé. Vérifiez que le fichier model.h5 existe et est valide."
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Nom de fichier vide"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"jpg", "jpeg", "png"}:
        return jsonify({"error": "Format non supporté"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        img = preprocess_image(filepath)

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
