"""
VALIDATION COMPLÈTE - COLAB VS BACKEND VS APP
==============================================

Vérifies que le pipeline est IDENTIQUE partout.

Le modèle utilise rescale=1./255 (normalisation [0, 1])

Usage:
    python validate_pipeline.py <image_path>

Cet script fait :
1. ✅ Prétraite l'image comme Colab (rescale=1./255)
2. ✅ Prétraite l'image comme Backend (rescale=1./255)
3. ✅ Montre les différences (doit être 0%)
4. ✅ Compare les prédictions
"""

import os
import sys
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# =========================
# PREPROCESSING COLAB
# =========================

def preprocess_colab_style(image_path):
    """
    Style Colab - rescale=1./255
    Redimensionnement: LANCZOS (comme ImageDataGenerator)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalisation rescale=1./255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# PREPROCESSING BACKEND
# =========================

def preprocess_backend_style(image_path):
    """
    Style Backend - avec EXIF transpose
    Normalisation: rescale=1./255
    Redimensionnement: LANCZOS (comme ImageDataGenerator)
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalisation rescale=1./255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# CHARGER MODÈLE
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
# LABELS
# =========================

LABELS = {
    0: "100 CDF", 1: "50 CDF", 2: "200 CDF", 3: "500 CDF",
    4: "1000 CDF", 5: "5000 CDF", 6: "10000 CDF", 7: "20000 CDF",
    8: "100 USD", 9: "5 USD", 10: "10 USD", 11: "50 USD",
    12: "20 USD", 13: "1 USD"
}

# =========================
# TEST
# =========================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python validate_pipeline.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Fichier non trouvé: {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION PIPELINE - COLAB vs BACKEND")
    print(f"{'='*60}\n")
    
    print(f"📷 Image: {image_path}\n")
    
    # Charger modèle
    print("Loading model...")
    model = load_model()
    print(f"✅ Modèle chargé\n")
    
    # Prétraiter - COLAB STYLE
    print("1️⃣ Preprocessing COLAB style...")
    img_colab = preprocess_colab_style(image_path)
    print(f"   ✅ Shape: {img_colab.shape}")
    print(f"   Min: {img_colab.min():.4f}, Max: {img_colab.max():.4f}")
    print(f"   Mean: {img_colab.mean():.4f}\n")
    
    # Prétraiter - BACKEND STYLE
    print("2️⃣ Preprocessing BACKEND style...")
    img_backend = preprocess_backend_style(image_path)
    print(f"   ✅ Shape: {img_backend.shape}")
    print(f"   Min: {img_backend.min():.4f}, Max: {img_backend.max():.4f}")
    print(f"   Mean: {img_backend.mean():.4f}\n")
    
    # Comparer les différences
    print("3️⃣ Vérification d'alignement...")
    diff = np.abs(img_colab - img_backend).mean()
    print(f"   Différence moyenne: {diff:.6f}")
    
    if diff < 0.001:
        print(f"   ✅ PARFAIT - Pipeline aligné à 99.9%\n")
    elif diff < 0.01:
        print(f"   ⚠️  OK - Petit écart (probablement EXIF)\n")
    else:
        print(f"   ❌ ERREUR - Écart trop grand!\n")
    
    # Prédictions
    print("4️⃣ Prédictions...\n")
    preds_colab = model.predict(img_colab, verbose=0)[0]
    preds_backend = model.predict(img_backend, verbose=0)[0]
    
    # Top 3 Colab
    print("COLAB predictions:")
    top_indices = np.argsort(preds_colab)[::-1][:3]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {LABELS[idx]:12} → {preds_colab[idx]:6.2%}")
    
    print()
    
    # Top 3 Backend
    print("BACKEND predictions:")
    top_indices = np.argsort(preds_backend)[::-1][:3]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {LABELS[idx]:12} → {preds_backend[idx]:6.2%}")
    
    # Meilleure classe
    best_colab = np.argmax(preds_colab)
    best_backend = np.argmax(preds_backend)
    
    print(f"\n{'='*60}")
    print(f"RÉSULTAT FINAL")
    print(f"{'='*60}")
    
    if best_colab == best_backend:
        print(f"✅ SUCCÈS - Classes identiques:")
        print(f"   → {LABELS[best_colab]}")
        print(f"   Colab: {preds_colab[best_colab]:.2%}")
        print(f"   Backend: {preds_backend[best_backend]:.2%}\n")
    else:
        print(f"❌ ERREUR - Classes différentes!")
        print(f"   Colab: {LABELS[best_colab]} ({preds_colab[best_colab]:.2%})")
        print(f"   Backend: {LABELS[best_backend]} ({preds_backend[best_backend]:.2%})\n")
        print(f"   👉 Vérifies EXIF transpose dans l'app\n")
    
    print("📝 ÉTAPES SUIVANTES:")
    print("   1. Upload cette image au backend: POST /predict")
    print("   2. Compare avec le résultat COLAB ci-dessus")
    print("   3. Si différent → problème caméra/app\n")
