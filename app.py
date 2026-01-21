import os
import time
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import ClientDisconnected
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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

# Labels - 14 classes pour MobileNetV2
BILL_LABELS = {
    0: "1 USD",
    1: "10 USD",
    2: "100 USD",
    3: "10000 CDF",
    4: "1000 CDF",
    5: "100 CDF",
    6: "20 USD",
    7: "20000 CDF",
    8: "200 CDF",
    9: "5 USD",
    10: "50 CDF",
    11: "5000 CDF",
    12: "500 CDF",
    13: "50 CDF",
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
    Charge et recompile le modèle Keras H5 correctement.
    
    La recompilation est CRITIQUE pour garantir:
    - L'optimiseur correct (Adam avec learning_rate=0.0001)
    - La fonction de perte (categorical_crossentropy)
    - Les métriques identiques à l'entraînement
    
    Cela assure la cohérence entre Colab et l'API
    """
    global MODEL, TFLITE_INTERPRETER
    
    logger.info("📦 Chargement du modèle...")
    
    # Charger model.h5 (format Keras HDF5)
    if os.path.exists("model.h5"):
        try:
            logger.info("📍 Chargement: model.h5 (Keras format)")
            MODEL = tf.keras.models.load_model("model.h5")
            logger.info(f"✅ Modèle Keras chargé")
            logger.info(f"  Input shape : {MODEL.input_shape}")
            logger.info(f"  Output shape: {MODEL.output_shape}")
            logger.info(f"  Classes: {MODEL.output_shape[-1]}")
            
            # ===== RECOMPILATION CRITIQUE =====
            # Recompiler avec les MÊMES paramètres que l'entraînement
            # Cela garantit la cohérence avec Colab
            try:
                MODEL.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("✅ Modèle recompilé avec Adam(lr=0.0001)")
                logger.info("   ✓ Loss: categorical_crossentropy")
                logger.info("   ✓ Metrics: accuracy")
            except Exception as e:
                logger.warning(f"⚠️  Recompilation impossible (SavedModel?): {e}")
                logger.info("   ℹ️  Continuant avec le modèle tel quel...")
            
            return True
        except Exception as e:
            logger.error(f"❌ Erreur model.h5: {e}", exc_info=True)
            return False
    
    logger.error("❌ model.h5 non trouvé - API non fonctionnelle")
    return False

try:
    if load_model_simple():
        logger.info("🚀 Modèle chargé et prêt pour prédictions!")
    else:
        logger.error("L'API va démarrer mais retournera une erreur pour les prédictions")
except Exception as e:
    logger.error(f"❌ Erreur au démarrage: {e}", exc_info=True)
    MODEL = None
    TFLITE_INTERPRETER = None

# =========================
# IMAGE PREPROCESS (COMPATIBLE MODEL)
# =========================

def preprocess_image(image_path):
    """
    Prétraitement IDENTIQUE au Colab pour garantir cohérence avec MobileNetV2.

    Points critiques pour la cohérence Colab ↔ API:
    1. ✓ Conversion RGB (PIL default)
    2. ✓ Resize 224x224 avec LANCZOS (ImageDataGenerator default)
    3. ✓ preprocess_input MobileNetV2 (normalise [-1, 1])
    4. ✓ Dimension batch [1, 224, 224, 3]
    5. ✓ Float32 precision (modèle attend float32)

    Pipeline:
    1. Charger l'image
    2. Valider le format
    3. Corriger l'orientation EXIF (photos iPhone)
    4. Convertir en RGB
    5. Redimensionner à 224x224
    6. Appliquer preprocess_input MobileNetV2
    7. Ajouter dimension batch
    """
    try:
        import io
        
        # Vérifier que le fichier existe et est lisible
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Fichier non trouvé: {image_path}")
        
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            raise ValueError(f"Fichier vide: {image_path}")
        
        logger.info(f"📸 Ouverture du fichier: {image_path} ({file_size} bytes)")
        
        # Ouvrir et valider l'image
        img = Image.open(image_path)
        img.verify()  # Vérifier que c'est une image valide
        
        # Réouvrir après verify() (qui ferme le fichier)
        img = Image.open(image_path)
        logger.info(f"✅ Image valide - Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        
        # ===== ÉTAPE 1: CORRECTION EXIF =====
        # Important pour les photos prises avec iPhone qui ont des métadonnées EXIF
        img = ImageOps.exif_transpose(img)
        logger.debug(f"  ✓ EXIF transposé - Nouveau size: {img.size}")
        
        # ===== ÉTAPE 2: CONVERSION RGB =====
        # ImageDataGenerator convertit automatiquement en RGB
        # C'est CRITIQUE pour cohérence avec Colab
        img = img.convert('RGB')
        logger.debug(f"  ✓ Converti en RGB - Mode: {img.mode}")
        
        # ===== ÉTAPE 3: REDIMENSIONNEMENT =====
        # Utiliser LANCZOS (algorithme par défaut de PIL pour downsampling)
        # C'est EXACTEMENT ce qu'utilise ImageDataGenerator
        img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        logger.debug(f"  ✓ Redimensionné à {IMG_SIZE}")
        
        # ===== ÉTAPE 4: CONVERSION EN ARRAY =====
        # Float32 (type attendu par le modèle Keras)
        img_array = np.array(img_resized, dtype=np.float32)
        logger.debug(f"  ✓ Converti en array - dtype: {img_array.dtype}, shape: {img_array.shape}")
        
        # ===== ÉTAPE 5: NORMALISATION =====
        # Appliquer preprocess_input MobileNetV2 (normalise [-1, 1])
        # Convertit [0, 255] → [-1, 1] selon MobileNetV2
        img_array = preprocess_input(img_array)
        logger.debug(f"  ✓ preprocess_input MobileNetV2 appliqué - Range: [{img_array.min():.4f}, {img_array.max():.4f}]")

        # ===== ÉTAPE 6: DIMENSION BATCH =====
        # model.predict() attend (batch_size, height, width, channels)
        # Transformer (224, 224, 3) → (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        logger.info(f"✅ Prétraitement complet - Shape final: {img_array.shape}")
        logger.info(f"   Data type: {img_array.dtype}, Range: [{img_array.min():.4f}, {img_array.max():.4f}]")
        
        return img_array
        
    except Image.UnidentifiedImageError as e:
        logger.error(f"❌ Format image non reconnu: {e}")
        raise ValueError(f"Format image invalide ou corrompu: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"❌ Fichier non trouvé: {e}")
        raise
    except ValueError as e:
        logger.error(f"❌ Fichier vide ou invalide: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Erreur inattendue lors du prétraitement: {e}", exc_info=True)
        raise ValueError(f"Erreur prétraitement: {str(e)}")

# =========================
# INFERENCE FUNCTION
# =========================

def predict_model(img_array):
    """
    Prédit avec le modèle Keras H5 (cohérent avec Colab).
    
    Le modèle.h5 est un modèle Keras classique.
    - Input: (1, 224, 224, 3) - array normalisé [0, 1]
    - Output: (1, 14) - logits pour 14 classes
    - Utilise softmax pour obtenir les probabilités
    """
    try:
        if MODEL is None:
            raise ValueError("Modèle non chargé")
        
        logger.debug(f"🔮 Input array shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # ===== PRÉDICTION =====
        # model.predict() retourne les probabilités directement
        # (contrairement à model(x) qui retourne les logits)
        predictions = MODEL.predict(img_array, verbose=0)
        
        logger.debug(f"  ✓ Predictions shape: {predictions.shape}")
        logger.debug(f"  ✓ Predictions sum: {predictions.sum():.4f} (should be ~1.0)")
        
        num_classes = predictions.shape[-1]
        
        # Retourner les prédictions pour la première image du batch
        # predictions[0] = array de 14 probabilités
        return predictions[0], num_classes
        
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {e}", exc_info=True)
        raise ValueError(f"Erreur lors de la prédiction: {str(e)}")

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET"])
def index():
    return "Bill Recognition API running", 200


@app.route("/health", methods=["GET"])
def health():
    model_info = {
        "model_loaded": MODEL is not None,
        "model_type": "SavedModel",
        "source": "model_saved/"
    }
    
    if MODEL is not None:
        try:
            model_info["signatures"] = list(MODEL.signatures.keys())
            model_info["num_classes"] = 14  # MobileNetV2 a 14 classes
            model_info["file"] = "model.h5"
        except:
            pass
    
    is_ready = MODEL is not None
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
    Endpoint de prédiction - Robuste avec validation complète
    
    Logique:
    1. Valide la présence et le format du fichier
    2. Prétraite (redimensionner 224x224, normaliser /255)
    3. Prédit la classe
    4. Retourne la classe et la confiance
    """
    start_time = time.time()
    filepath = None
    
    try:
        # ===== VALIDATION DU MODÈLE =====
        if TFLITE_INTERPRETER is None and MODEL is None:
            logger.error("❌ Aucun modèle disponible")
            return jsonify({
                "error": "Modèle non chargé",
                "status": "model_missing"
            }), 503

        # ===== VALIDATION DE LA REQUÊTE =====
        logger.debug(f"🔍 Content-Type: {request.content_type}")
        try:
            logger.debug(f"🔍 Files keys: {list(request.files.keys())}")
        except ClientDisconnected:
            logger.warning("⚠️  Client déconnecté lors du traitement de la requête")
            return jsonify({
                "error": "Client déconnecté - veuillez réessayer"
            }), 400
        
        try:
            files_keys = list(request.files.keys())
        except ClientDisconnected:
            logger.warning("⚠️  Client déconnecté lors de l'accès à request.files")
            return jsonify({
                "error": "Client déconnecté - veuillez réessayer"
            }), 400
        
        if "file" not in request.files:
            logger.warning("⚠️  Clé 'file' manquante dans request.files")
            return jsonify({
                "error": "Clé 'file' manquante. Utilisez: files={'file': open('image.jpg', 'rb')}",
                "received_keys": files_keys,
                "content_type": request.content_type
            }), 400

        try:
            file = request.files["file"]
        except ClientDisconnected:
            logger.warning("⚠️  Client déconnecté lors de la récupération du fichier")
            return jsonify({
                "error": "Client déconnecté - veuillez réessayer"
            }), 400
        
        if file.filename == "":
            logger.warning("⚠️  Filename vide")
            return jsonify({
                "error": "Filename vide - impossible de traiter"
            }), 400

        # ===== VALIDATION DE L'EXTENSION =====
        filename = file.filename
        if "." not in filename:
            logger.warning(f"⚠️  Extension manquante: {filename}")
            return jsonify({
                "error": f"Extension manquante. Formats acceptés: jpg, jpeg, png, gif, bmp"
            }), 400
        
        ext = filename.rsplit(".", 1)[-1].lower()
        allowed_extensions = {"jpg", "jpeg", "png", "gif", "bmp"}
        
        if ext not in allowed_extensions:
            logger.warning(f"⚠️  Format non supporté: {ext}")
            return jsonify({
                "error": f"Format '{ext}' non supporté",
                "allowed_formats": list(allowed_extensions)
            }), 400

        # ===== SAUVEGARDE TEMPORAIRE =====
        filename_safe = secure_filename(filename)
        # Ajouter timestamp pour éviter les collisions
        import uuid
        filename_unique = f"{uuid.uuid4().hex}_{filename_safe}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename_unique)
        
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        logger.info(f"✅ Fichier sauvegardé: {filename_unique} ({file_size} bytes)")
        
        # Vérifier que le fichier a bien été écrit
        if file_size == 0:
            logger.error("❌ Fichier vide après la sauvegarde")
            return jsonify({
                "error": "Fichier vide - impossible de traiter"
            }), 400
        
        # ===== PRÉTRAITEMENT =====
        try:
            img_array = preprocess_image(filepath)
            logger.info(f"✅ Image prétraitée - Shape: {img_array.shape}")
        except ValueError as e:
            logger.error(f"❌ Erreur prétraitement (ValueError): {e}")
            return jsonify({
                "error": f"Image invalide: {str(e)}"
            }), 400
        except Exception as e:
            logger.error(f"❌ Erreur prétraitement (Exception): {e}")
            return jsonify({
                "error": f"Erreur traitement image: {str(e)}"
            }), 500
        
        # ===== PRÉDICTION =====
        try:
            logger.info("🔮 Prédiction en cours avec model.h5...")
            predictions, num_classes = predict_model(img_array)
            logger.info(f"✅ Prédiction réussie - {num_classes} classes détectées")
        except ValueError as e:
            logger.error(f"❌ Erreur prédiction (ValueError): {e}")
            return jsonify({
                "error": f"Erreur prédiction: {str(e)}"
            }), 500
        except Exception as e:
            logger.error(f"❌ Erreur prédiction (Exception): {e}")
            return jsonify({
                "error": f"Erreur serveur prédiction: {str(e)}"
            }), 500
        
        # ===== ANALYSE DES RÉSULTATS =====
        predicted_class_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class_idx])
        num_classes = int(num_classes)
        
        logger.debug(f"  ✓ Classe prédite: {predicted_class_idx}")
        logger.debug(f"  ✓ Confiance: {confidence:.4f}")
        logger.debug(f"  ✓ Top 3 prédictions:")
        top_3_idx = np.argsort(predictions)[::-1][:3]
        for i, idx in enumerate(top_3_idx):
            logger.debug(f"    {i+1}. Classe {idx}: {predictions[idx]:.4f} ({BILL_LABELS.get(idx, '?')})")
        
        # ===== RÉCUPÉRATION DU LABEL =====
        predicted_label = BILL_LABELS.get(predicted_class_idx, f"Unknown ({predicted_class_idx})")
        
        logger.info(f"🎯 RÉSULTAT FINAL: {predicted_label} ({confidence*100:.2f}%) [Classe {predicted_class_idx}/{num_classes}]")
        
        # ===== RÉPONSE JSON =====
        # Format cohérent avec les attentes de l'app mobile
        response = {
            "result": predicted_label,
            "prediction": predicted_label,
            "confidence": float(confidence),
            "confidence_percent": round(float(confidence) * 100, 2),
            "class": int(predicted_class_idx),
            "class_index": int(predicted_class_idx),
            "num_classes": num_classes,
            "model": "model.h5 (Keras)",
            "model_source": "Colab training",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(f"✅ Réponse préparée: {response}")
        return jsonify(response), 200
        
    except FileNotFoundError as e:
        logger.error(f"❌ Fichier non trouvé: {e}")
        return jsonify({
            "error": f"Erreur fichier: {str(e)}",
            "error_type": "file_not_found"
        }), 500
    except ValueError as e:
        logger.error(f"❌ Erreur validation: {e}")
        return jsonify({
            "error": f"Format image invalide: {str(e)}",
            "error_type": "invalid_image"
        }), 400
    except Exception as e:
        logger.error(f"❌ Erreur serveur: {e}", exc_info=True)
        return jsonify({
            "error": f"Erreur serveur: {str(e)}",
            "error_type": "server_error"
        }), 500
    
    finally:
        # ===== NETTOYAGE =====
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.debug(f"🗑️  Fichier temporaire supprimé: {filepath}")
            except Exception as e:
                logger.warning(f"⚠️  Impossible de supprimer {filepath}: {e}")

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
