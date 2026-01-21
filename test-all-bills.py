#!/usr/bin/env python3
"""
Test tous les billets du dossier test_bills/
Montre un résumé de ce que le modèle reconnaît bien/mal
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# Configuration
IMG_SIZE = (224, 224)
MODEL_PATH = "model.h5"
TEST_DIR = "test_bills"

# Mapping label → index
LABELS = {
    0: "100 CDF", 1: "50 CDF", 2: "200 CDF", 3: "500 CDF",
    4: "1000 CDF", 5: "5000 CDF", 6: "10000 CDF", 7: "20000 CDF",
    8: "100 USD", 9: "5 USD", 10: "10 USD", 11: "50 USD",
    12: "20 USD", 13: "1 USD",
}

def get_label_from_filename(filename):
    """Extraire le label du nom du fichier"""
    # bill_500_CDF_0.jpg → "500 CDF"
    if filename.startswith("bill_"):
        parts = filename.replace("bill_", "").replace(".jpg", "").rsplit("_", 1)
        if len(parts) == 2:
            denomination = parts[0]  # "500", "20 CDF", etc
            # Convertir "500" → "500 CDF" ou "20" → "20 USD", etc
            # Vérifier format dans le nom
            if " " in denomination:
                return denomination  # "500 CDF" ou "20 USD"
            # Si juste le nombre, il faut deviner la devise
            for label in LABELS.values():
                if denomination in label:
                    return label
    return None

def predict_image(model, image_path):
    """Prédire pour une image"""
    try:
        img = Image.open(image_path)
        img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_batch, verbose=0)
        top_idx = np.argmax(predictions[0])
        top_score = predictions[0][top_idx]
        top_label = LABELS.get(top_idx, f"Unknown {top_idx}")
        
        return top_label, float(top_score), predictions[0]
    except Exception as e:
        print(f"❌ Erreur {image_path}: {e}")
        return None, 0, None

def main():
    print("=" * 70)
    print("🧪 Test Complet: Tous les Billets")
    print("=" * 70)
    print()
    
    # Charger modèle
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modèle non trouvé: {MODEL_PATH}")
        return
    
    print(f"📦 Chargement modèle: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"   ✅ Modèle chargé (input: {model.input_shape}, output: {model.output_shape})\n")
    
    # Récupérer tous les fichiers jpg
    jpg_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')])
    
    if not jpg_files:
        print(f"❌ Aucun fichier .jpg trouvé dans {TEST_DIR}/")
        return
    
    print(f"📸 Trouvé {len(jpg_files)} images\n")
    
    # Résultats par classe
    results_by_class = {}
    correct = 0
    total = 0
    confusions = []
    
    # Tester chaque image
    for filename in jpg_files:
        filepath = os.path.join(TEST_DIR, filename)
        expected_label = get_label_from_filename(filename)
        predicted_label, confidence, all_scores = predict_image(model, filepath)
        
        if predicted_label is None:
            continue
        
        total += 1
        
        # Initialiser classe si besoin
        if expected_label not in results_by_class:
            results_by_class[expected_label] = {
                'correct': 0,
                'total': 0,
                'predictions': []
            }
        
        results_by_class[expected_label]['total'] += 1
        results_by_class[expected_label]['predictions'].append({
            'file': filename,
            'predicted': predicted_label,
            'confidence': confidence
        })
        
        # Vérifier si correct
        is_correct = (predicted_label == expected_label)
        if is_correct:
            correct += 1
            results_by_class[expected_label]['correct'] += 1
            status = "✅"
        else:
            status = "❌"
            confusions.append({
                'file': filename,
                'expected': expected_label,
                'predicted': predicted_label,
                'confidence': confidence
            })
        
        print(f"{status} {filename.ljust(30)} → {predicted_label.ljust(12)} ({confidence*100:5.1f}%)")
    
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    print()
    
    # Accuracy par classe
    print("📋 Accuracy par Classe:")
    for label in sorted(results_by_class.keys()):
        data = results_by_class[label]
        acc = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        status = "✅" if acc >= 80 else "⚠️ " if acc >= 50 else "❌"
        print(f"   {status} {label.ljust(12)} : {data['correct']}/{data['total']} ({acc:5.1f}%)")
    
    # Accuracy totale
    print()
    total_acc = (correct / total * 100) if total > 0 else 0
    print(f"🎯 Accuracy Totale: {correct}/{total} ({total_acc:.1f}%)")
    
    # Top confusions
    if confusions:
        print()
        print("🔴 Confusions Détectées:")
        for i, confusion in enumerate(confusions[:5], 1):
            print(f"   {i}. {confusion['expected'].ljust(12)} → {confusion['predicted'].ljust(12)} ({confusion['confidence']*100:.1f}%)")
            print(f"      {confusion['file']}")
    
    # Résumé
    print()
    print("=" * 70)
    if total_acc >= 90:
        print("✅ MODÈLE BON - Accuracy ≥ 90%")
    elif total_acc >= 75:
        print("⚠️  MODÈLE OK - Accuracy 75-90% (à améliorer)")
    else:
        print(f"❌ MODÈLE CASSÉ - Accuracy < 75% ({total_acc:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
