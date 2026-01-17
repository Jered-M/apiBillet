"""
TEST PIPELINE COLAB VS BACKEND
===============================

Vérifie que le preprocessing Colab == preprocessing Backend

Usage:
    python test_pipeline.py <image_path>

Exemple:
    python test_pipeline.py uploads/raw_bill.jpg
"""

import os
import sys
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

# =========================
# FONCTION PREPROCESSING
# =========================

def preprocess_image(image_path):
    """
    IDENTIQUE à app.py
    """
    img = Image.open(image_path)
    
    # Corriger l'orientation EXIF
    img = ImageOps.exif_transpose(img)
    
    # Convertir en RGB
    img = img.convert('RGB')
    
    # Redimensionner
    img = img.resize((224, 224), Image.Resampling.BICUBIC)
    
    # Convertir en array
    img_array = np.array(img, dtype=np.float32)
    
    # preprocess_input MobileNetV2
    img_array = preprocess_input(img_array)
    
    # Ajouter dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# =========================
# CHARGER LE MODÈLE
# =========================

def load_model():
    """Charger le modèle Keras"""
    model_paths = ["model (1).h5", "best_model.h5", "model.h5"]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Modèle trouvé: {path}")
            return tf.keras.models.load_model(path)
    
    raise FileNotFoundError("Aucun modèle trouvé")

# =========================
# TEST
# =========================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python test_pipeline.py <image_path>")
        print("Exemple: python test_pipeline.py uploads/raw_bill.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Fichier non trouvé: {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print(f"TEST PIPELINE MOBILENET V2")
    print(f"{'='*50}\n")
    
    print(f"📷 Image: {image_path}")
    print(f"   Taille: {os.path.getsize(image_path)} bytes\n")
    
    # Charger modèle
    print("Loading model...")
    model = load_model()
    print(f"✅ Modèle chargé\n")
    
    # Prétraiter
    print("Preprocessing...")
    img_array = preprocess_image(image_path)
    print(f"✅ Prétraitement OK")
    print(f"   Shape: {img_array.shape}")
    print(f"   Min value: {img_array.min():.4f}")
    print(f"   Max value: {img_array.max():.4f}")
    print(f"   Mean: {img_array.mean():.4f}\n")
    
    # Prédire
    print("Predicting...")
    preds = model.predict(img_array, verbose=0)[0]
    
    # Labels
    labels = {
        0: "100 CDF", 1: "50 CDF", 2: "200 CDF", 3: "500 CDF",
        4: "1000 CDF", 5: "5000 CDF", 6: "10000 CDF", 7: "20000 CDF",
        8: "100 USD", 9: "5 USD", 10: "10 USD", 11: "50 USD",
        12: "20 USD", 13: "1 USD"
    }
    
    print(f"✅ Prédictions:\n")
    
    # Afficher top 5
    top_indices = np.argsort(preds)[::-1][:5]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {labels[idx]:12} → {preds[idx]:6.2%}")
    
    # Meilleure prédiction
    best_idx = int(np.argmax(preds))
    best_label = labels[best_idx]
    confidence = float(preds[best_idx])
    
    print(f"\n{'='*50}")
    print(f"🎯 RÉSULTAT FINAL")
    print(f"{'='*50}")
    print(f"Classe: {best_label}")
    print(f"Confiance: {confidence:.2%}")
    print(f"{'='*50}\n")
    
    # Instructions
    print("📝 INSTRUCTIONS POUR VÉRIFIER:")
    print("1. Envoie cette image au backend: POST /predict")
    print("2. Compare le résultat avec celui-ci")
    print("3. ✅ Si identique → pipeline correct")
    print("4. ❌ Si différent → problème dans l'app/caméra\n")
