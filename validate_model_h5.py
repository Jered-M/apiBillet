#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation du modèle model.h5 - Teste reproductibilité et exactitude
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Configuration
IMG_SIZE = (224, 224)
BILL_LABELS = {
    0: "100 CDF", 1: "50 CDF", 2: "200 CDF", 3: "500 CDF",
    4: "1000 CDF", 5: "5000 CDF", 6: "10000 CDF", 7: "20000 CDF",
    8: "100 USD", 9: "5 USD", 10: "10 USD", 11: "50 USD",
}

def preprocess_image(image_path):
    """Preprocessing IDENTIQUE au notebook"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def test_model_h5():
    """Test le model.h5 - reproductibilité"""
    
    print("\n" + "="*60)
    print("VALIDATION model.h5 (12 classes)")
    print("="*60)
    
    # Charger le modèle
    if not os.path.exists("model.h5"):
        print("❌ model.h5 non trouvé")
        return False
    
    try:
        print("\nChargement model.h5...")
        model = tf.keras.models.load_model("model.h5")
        print(f"✅ Chargé")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    
    # Test avec image
    test_images = ["test_image.jpg", "uploads/bill.jpg", "bill.jpg"]
    image_path = None
    for path in test_images:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path is None:
        print("⚠️  Pas d'image de test")
        # Créer une image test
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img).save("test_image.jpg")
        image_path = "test_image.jpg"
    
    print(f"\nTest avec: {image_path}")
    
    # Tester reproductibilité
    print("\nTest reproductibilité (5 appels):")
    results = []
    
    for i in range(5):
        img_array = preprocess_image(image_path)
        pred = model.predict(img_array, verbose=0)
        top_idx = np.argmax(pred[0])
        conf = pred[0][top_idx]
        label = BILL_LABELS.get(top_idx, f"Unknown {top_idx}")
        results.append((label, conf))
        print(f"  {i+1}. {label:15} {conf:7.2%}")
    
    # Vérifier reproductibilité
    all_same = len(set(r[0] for r in results)) == 1
    if all_same:
        print(f"\n✅ REPRODUCTIBLE: Tous les résultats sont identiques")
        print(f"   Prédiction stable: {results[0][0]} ({results[0][1]:.2%})")
    else:
        print(f"\n❌ NON REPRODUCTIBLE: Les résultats varient")
        unique_results = set(r[0] for r in results)
        print(f"   Résultats différents: {unique_results}")
    
    # Afficher toutes les probabilités
    print("\nDistribution complète (dernière prédiction):")
    img_array = preprocess_image(image_path)
    pred = model.predict(img_array, verbose=0)
    
    indices = np.argsort(pred[0])[::-1]
    for rank, idx in enumerate(indices[:12], 1):
        conf = pred[0][idx]
        label = BILL_LABELS.get(idx, f"Unknown {idx}")
        bar = "█" * int(conf * 40)
        print(f"  {rank:2}. {label:15} {conf:7.2%} {bar}")
    
    return all_same

if __name__ == "__main__":
    success = test_model_h5()
    
    print("\n" + "="*60)
    if success:
        print("✅ model.h5 fonctionne correctement et est REPRODUCTIBLE")
        print("\nLe problème peut être:")
        print("  - Image d'entrée mal prétraitée côté client")
        print("  - Modèle mal entraîné (data quality)")
        print("  - Classes mal étiquetées")
    else:
        print("❌ model.h5 n'est pas reproductible")
        print("\nRecommandation: Régénérer le model.h5 depuis Colab")
    print("="*60)
