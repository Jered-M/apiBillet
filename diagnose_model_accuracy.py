#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic du modèle - Compare TFLite vs H5 et teste la reproductibilité
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import json

# Fix pour Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
IMG_SIZE = (224, 224)
BILL_LABELS = {
    0: "100 CDF", 1: "50 CDF", 2: "200 CDF", 3: "500 CDF",
    4: "1000 CDF", 5: "5000 CDF", 6: "10000 CDF", 7: "20000 CDF",
    8: "100 USD", 9: "5 USD", 10: "10 USD", 11: "50 USD",
    12: "20 USD", 13: "1 USD"
}

def preprocess_image(image_path):
    """Preprocessing IDENTIQUE"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_tflite():
    """Charge TFLite"""
    try:
        interp = tf.lite.Interpreter(model_path="model (1).tflite")
        interp.allocate_tensors()
        return interp
    except Exception as e:
        print(f"❌ TFLite: {e}")
        return None

def load_h5():
    """Charge H5"""
    try:
        if os.path.exists("model.h5"):
            model = tf.keras.models.load_model("model.h5")
            return model
        else:
            print("⚠️  model.h5 non trouvé")
            return None
    except Exception as e:
        print(f"❌ H5: {e}")
        return None

def predict_tflite(interp, img_array):
    """Prédit avec TFLite"""
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    
    interp.set_tensor(input_details[0]['index'], img_array)
    interp.invoke()
    
    predictions = interp.get_tensor(output_details[0]['index'])
    return predictions[0]

def predict_h5(model, img_array):
    """Prédit avec H5"""
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

def show_top_predictions(predictions, title="Prédictions"):
    """Affiche le top 5 des prédictions"""
    print(f"\n{title}:")
    print(f"{'─'*60}")
    
    top_indices = np.argsort(predictions)[-5:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        conf = predictions[idx]
        label = BILL_LABELS.get(idx, f"Unknown {idx}")
        bar = "█" * int(conf * 40)
        print(f"{rank}. {label:15} {conf:7.2%} {bar}")

def test_reproducibility(image_path, num_tests=5):
    """Teste si TFLite donne les mêmes résultats à chaque appel"""
    print(f"\n{'='*60}")
    print(f"🔄 TEST REPRODUCTIBILITÉ ({num_tests} appels)")
    print(f"{'='*60}")
    
    interp = load_tflite()
    if interp is None:
        return False
    
    img_array = preprocess_image(image_path)
    results = []
    
    for i in range(num_tests):
        pred = predict_tflite(interp, img_array)
        top_idx = int(np.argmax(pred))
        conf = float(pred[top_idx])
        label = BILL_LABELS.get(top_idx, f"Unknown {top_idx}")
        results.append((label, conf))
        print(f"  Appel {i+1}: {label:15} {conf:7.2%}")
    
    # Vérifier la reproductibilité
    if len(set(r[0] for r in results)) == 1:
        print(f"\n✅ REPRODUCTIBLE: Tous les résultats sont identiques")
        return True
    else:
        print(f"\n❌ NON REPRODUCTIBLE: Les résultats varient!")
        print(f"   Cela indique un problème avec le modèle TFLite")
        return False

def compare_tflite_vs_h5(image_path):
    """Compare TFLite vs H5 sur la même image"""
    print(f"\n{'='*60}")
    print(f"⚖️  COMPARAISON TFLite vs H5")
    print(f"{'='*60}")
    
    img_array = preprocess_image(image_path)
    
    # TFLite
    print(f"\n📱 TFLITE (optimisé):")
    interp = load_tflite()
    if interp is not None:
        pred_tflite = predict_tflite(interp, img_array)
        show_top_predictions(pred_tflite, "TFLite - Top 5")
    else:
        print("❌ TFLite non disponible")
        return False
    
    # H5
    print(f"\n🔧 KERAS H5 (original):")
    model = load_h5()
    if model is not None:
        pred_h5 = predict_h5(model, img_array)
        show_top_predictions(pred_h5, "H5 - Top 5")
        
        # Comparer les sorties
        print(f"\n📊 Différence entre TFLite et H5:")
        diff = np.abs(pred_tflite - pred_h5)
        print(f"   Max difference: {diff.max():.4f}")
        print(f"   Mean difference: {diff.mean():.4f}")
        print(f"   L2 distance: {np.linalg.norm(diff):.4f}")
        
        if diff.max() > 0.1:
            print(f"   ⚠️  ATTENTION: Grandes différences détectées!")
            print(f"      → Le TFLite peut avoir des poids incorrects")
            print(f"      → Vérifier la conversion H5 → TFLite")
        else:
            print(f"   ✅ Différences acceptables")
    else:
        print("⚠️  H5 non disponible (skipping comparison)")
    
    return True

def verify_preprocessing(image_path):
    """Vérifie que le preprocessing produit une image valide"""
    print(f"\n{'='*60}")
    print(f"✓ VÉRIFICATION PREPROCESSING")
    print(f"{'='*60}")
    
    img_array = preprocess_image(image_path)
    
    print(f"  Shape: {img_array.shape}")
    print(f"  Dtype: {img_array.dtype}")
    print(f"  Min: {img_array.min():.4f}")
    print(f"  Max: {img_array.max():.4f}")
    print(f"  Mean: {img_array.mean():.4f}")
    print(f"  Std: {img_array.std():.4f}")
    
    # Vérifications
    checks = [
        (img_array.shape == (1, 224, 224, 3), "Shape correct"),
        (img_array.dtype == np.float32, "Dtype float32"),
        (img_array.min() >= 0.0, "Min >= 0.0"),
        (img_array.max() <= 1.0, "Max <= 1.0"),
    ]
    
    all_ok = True
    for check, desc in checks:
        status = "✅" if check else "❌"
        print(f"  {status} {desc}")
        all_ok = all_ok and check
    
    return all_ok

def main():
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC MODÈLE - ACCURACY ISSUES")
    print("="*60)
    
    # Trouver une image de test
    image_path = None
    for path in ["test_image.jpg", "uploads/bill.jpg", "bill.jpg"]:
        if os.path.exists(path):
            image_path = path
            print(f"\n📸 Image de test: {image_path}")
            break
    
    if image_path is None:
        print("❌ Pas d'image de test trouvée")
        sys.exit(1)
    
    # Tests
    verify_preprocessing(image_path)
    compare_tflite_vs_h5(image_path)
    test_reproducibility(image_path)
    
    print("\n" + "="*60)
    print("📋 RECOMMANDATIONS")
    print("="*60)
    print("""
Si les résultats sont inexacts:

1. ❌ Non reproductible (différents résultats chaque fois)
   → Problème: TFLite a peut-être des poids corrompus
   → Solution: Reconvertir le modèle H5 en TFLite

2. ❌ TFLite ≠ H5 (grande différence)
   → Problème: Conversion H5→TFLite échouée
   → Solution: Vérifier la conversion avec convert_to_tflite.py

3. ❌ Tous les deux mauvais (mauvaises prédictions)
   → Problème: Le modèle n'a pas bien appris
   → Solution: Réentraîner le modèle avec plus de données

4. ✅ Tous reproductibles et corrects
   → Le modèle fonctionne correctement
   → Problème peut être ailleurs (upload image, preprocessing côté client)
    """)

if __name__ == "__main__":
    main()
