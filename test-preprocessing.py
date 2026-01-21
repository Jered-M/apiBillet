#!/usr/bin/env python3
"""
Diagnostic: Comparer preprocessing Colab vs Backend
Aide à identifier les différences qui causent la confusion 500CDF vs 20USD
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# Configuration
IMG_SIZE = (224, 224)
MODEL_PATH = "model.h5"

def load_and_preprocess_image(image_path):
    """Preprocessing identique à Colab/Backend"""
    try:
        # Charger image
        img = Image.open(image_path)
        print(f"✅ Image chargée: {image_path}")
        print(f"   Size originale: {img.size}")
        print(f"   Mode: {img.mode}")
        
        # Redimensionner à 224x224
        img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        print(f"   ✓ Redimensionné à: {IMG_SIZE}")
        
        # Convertir en array
        img_array = np.array(img_resized, dtype=np.float32)
        print(f"   ✓ Array shape: {img_array.shape}")
        print(f"   ✓ Array dtype: {img_array.dtype}")
        print(f"   ✓ Min/Max avant norm: {img_array.min():.2f} / {img_array.max():.2f}")
        
        # Normaliser /255
        img_normalized = img_array / 255.0
        print(f"   ✓ Normalisé: /255.0")
        print(f"   ✓ Min/Max après norm: {img_normalized.min():.4f} / {img_normalized.max():.4f}")
        
        # Ajouter batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        print(f"   ✓ Batch shape: {img_batch.shape}")
        
        return img_batch, img_normalized
        
    except Exception as e:
        print(f"❌ Erreur chargement image: {e}")
        return None, None

def test_model_prediction(model, img_batch, img_path):
    """Tester prédiction avec modèle"""
    try:
        print(f"\n🤖 Prédiction avec model.h5...")
        predictions = model.predict(img_batch, verbose=0)
        print(f"   Output shape: {predictions.shape}")
        print(f"   Output dtype: {predictions.dtype}")
        
        # Top 5
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        
        labels = {
            0: "100 CDF", 1: "50 CDF", 2: "200 CDF", 3: "500 CDF",
            4: "1000 CDF", 5: "5000 CDF", 6: "10000 CDF", 7: "20000 CDF",
            8: "100 USD", 9: "5 USD", 10: "10 USD", 11: "50 USD",
            12: "20 USD", 13: "1 USD",
        }
        
        print(f"\n📊 Top 5 prédictions:")
        for idx, pred_idx in enumerate(top_5_idx):
            score = predictions[0][pred_idx]
            label = labels.get(pred_idx, f"Unknown {pred_idx}")
            bar = "█" * int(score * 20)
            print(f"   {idx+1}. {label.ljust(12)} {bar.ljust(20)} {score*100:.2f}%")
        
        # Détection confusion
        print(f"\n🔍 Analyse confusion:")
        pred_500_cdf = predictions[0][3]  # Index 3 = 500 CDF
        pred_20_usd = predictions[0][12]  # Index 12 = 20 USD
        
        diff = abs(pred_500_cdf - pred_20_usd)
        print(f"   Score 500 CDF: {pred_500_cdf*100:.2f}%")
        print(f"   Score 20 USD:  {pred_20_usd*100:.2f}%")
        print(f"   Différence:    {diff*100:.2f}%")
        
        if diff < 0.1:  # Moins de 10% de différence
            print(f"   ⚠️  CONFUSION DÉTECTÉE: Scores très proches!")
            return False
        else:
            print(f"   ✅ Pas de confusion: Différence suffisante")
            return True
            
    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test-preprocessing.py <image_path>")
        print("Example: python test-preprocessing.py test_bills/500-cdf.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Fichier non trouvé: {image_path}")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modèle non trouvé: {MODEL_PATH}")
        sys.exit(1)
    
    print("=" * 60)
    print("🔍 Diagnostic: Preprocessing 500CDF vs 20USD")
    print("=" * 60)
    print()
    
    # 1. Charger et prétraiter
    print(f"📸 Image test: {image_path}\n")
    img_batch, img_normalized = load_and_preprocess_image(image_path)
    
    if img_batch is None:
        sys.exit(1)
    
    # 2. Charger modèle
    print(f"\n📦 Chargement modèle: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"   ✅ Modèle chargé")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        sys.exit(1)
    
    # 3. Tester prédiction
    result = test_model_prediction(model, img_batch, image_path)
    
    # 4. Recommandations
    print(f"\n💡 Recommandations:")
    if result is False:
        print("   - Confusion détectée entre 500 CDF et 20 USD")
        print("   - Options:")
        print("     1. Augmenter données d'entraînement pour 20 USD")
        print("     2. Utiliser data augmentation")
        print("     3. Ajuster les poids de la loss function")
        print("     4. Augmenter la confiance minimale requise")
    elif result is True:
        print("   ✅ Pas de confusion - Modèle fonctionne bien!")
    else:
        print("   Erreur lors du test - vérifier les logs")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
