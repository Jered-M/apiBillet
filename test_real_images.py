#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test d'accuracy sur VRAIES images de bills
À utiliser quand vous avez des images réelles de bills
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json

IMG_SIZE = (224, 224)

BILL_LABELS = {
    0: "100 CDF", 1: "50 CDF", 2: "200 CDF", 3: "500 CDF",
    4: "1000 CDF", 5: "5000 CDF", 6: "10000 CDF", 7: "20000 CDF",
    8: "100 USD", 9: "5 USD", 10: "10 USD", 11: "50 USD",
    12: "20 USD", 13: "1 USD"
}

def preprocess_image(image_path):
    """Preprocessing identique"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def test_on_real_images(model_path, test_images_dir, labels_file=None):
    """
    Test le modèle sur vraies images
    
    Attendu structure:
    test_images_dir/
    ├── 100_CDF/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── 50_CDF/
    ├── ...
    
    Ou avec labels.json:
    {
        "img1.jpg": "100 CDF",
        "img2.jpg": "50 CDF",
        ...
    }
    """
    
    print("\n" + "="*60)
    print("🧪 TEST ACCURACY - VRAIES IMAGES")
    print("="*60)
    
    # Charger le modèle
    try:
        print(f"\n📍 Chargement: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modèle chargé - {model.output_shape[-1]} classes")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    
    # Charger images et labels
    if labels_file and os.path.exists(labels_file):
        # Depuis JSON
        print(f"\n📁 Chargement depuis: {labels_file}")
        with open(labels_file, 'r') as f:
            labels_dict = json.load(f)
        
        images = []
        labels = []
        
        for img_name, label_str in labels_dict.items():
            img_path = os.path.join(test_images_dir, img_name)
            if os.path.exists(img_path):
                images.append(img_path)
                # Convertir label string en index
                label_idx = [i for i, v in BILL_LABELS.items() if v == label_str]
                if label_idx:
                    labels.append(label_idx[0])
                else:
                    print(f"⚠️  Label inconnu: {label_str}")
                    continue
        
        print(f"✅ Loaded {len(images)} images")
    
    elif os.path.isdir(test_images_dir):
        # Depuis dossiers
        print(f"\n📁 Chargement depuis: {test_images_dir}")
        images = []
        labels = []
        
        for class_idx, class_name in BILL_LABELS.items():
            class_dir = os.path.join(test_images_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_file)
                        images.append(img_path)
                        labels.append(class_idx)
        
        print(f"✅ Loaded {len(images)} images")
    
    else:
        print(f"❌ Pas d'images trouvées")
        return False
    
    if not images:
        print(f"❌ Aucune image chargée")
        return False
    
    # Faire les prédictions
    print(f"\n🔮 Prédictions en cours...")
    predictions = []
    
    for i, img_path in enumerate(images):
        try:
            img_array = preprocess_image(img_path)
            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])
            pred_conf = float(np.max(pred[0]))
            predictions.append((pred_class, pred_conf))
            
            if (i+1) % max(1, len(images)//10) == 0:
                print(f"  {i+1}/{len(images)} ...")
        except Exception as e:
            print(f"  ❌ Erreur sur {img_path}: {e}")
            predictions.append((None, None))
    
    predictions = np.array(predictions)
    pred_classes = predictions[:, 0].astype(int)
    pred_confs = predictions[:, 1]
    
    # Métriques
    print("\n" + "-"*60)
    print("📊 MÉTRIQUES")
    print("-"*60)
    
    # Accuracy globale
    accuracy = accuracy_score(labels, pred_classes)
    print(f"\n✅ Accuracy global: {accuracy:.2%}")
    
    # Confiance
    print(f"\n📈 Confiance des prédictions:")
    print(f"   Min:  {pred_confs.min():.2%}")
    print(f"   Moy:  {pred_confs.mean():.2%}")
    print(f"   Max:  {pred_confs.max():.2%}")
    
    # Classification report
    print(f"\n📋 Report par classe:")
    report = classification_report(
        labels, pred_classes,
        target_names=[BILL_LABELS[i] for i in range(14)],
        zero_division=0
    )
    print(report)
    
    # Confusion matrix
    print(f"\n🎯 Confusion Matrix:")
    cm = confusion_matrix(labels, pred_classes, labels=range(14))
    
    # Afficher les principales erreurs
    errors = []
    for true_idx in range(14):
        for pred_idx in range(14):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                errors.append((cm[true_idx, pred_idx], true_idx, pred_idx))
    
    errors.sort(reverse=True)
    print("\nTop erreurs de confusion:")
    for count, true_idx, pred_idx in errors[:5]:
        print(f"  {BILL_LABELS[true_idx]} → {BILL_LABELS[pred_idx]}: {count}x")
    
    print("\n" + "="*60)
    print(f"✅ TEST COMPLÉTÉ - Accuracy: {accuracy:.2%}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    # Configuration
    model_path = r"c:\Users\HP\Downloads\model (1).h5"
    test_images_dir = r"test_bills"  # Images créées ci-dessus
    labels_file = r"test_bills\labels.json"  # Labels JSON
    
    # Exécuter le test
    test_on_real_images(model_path, test_images_dir, labels_file)
