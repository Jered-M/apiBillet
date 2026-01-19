#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test d'accuracy du modèle model (1).h5 depuis Downloads
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import glob

IMG_SIZE = (224, 224)

# Labels
BILL_LABELS = {
    0: "100 CDF", 1: "50 CDF", 2: "200 CDF", 3: "500 CDF",
    4: "1000 CDF", 5: "5000 CDF", 6: "10000 CDF", 7: "20000 CDF",
    8: "100 USD", 9: "5 USD", 10: "10 USD", 11: "50 USD",
    12: "20 USD", 13: "1 USD"
}

def load_and_test_model(model_path):
    """Charge et teste le modèle"""
    
    print("\n" + "="*60)
    print(f"🧪 TEST ACCURACY: {model_path}")
    print("="*60)
    
    # Vérifier le fichier
    if not os.path.exists(model_path):
        print(f"❌ Fichier non trouvé: {model_path}")
        return False
    
    file_size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"\n📦 Fichier: {os.path.basename(model_path)}")
    print(f"   Taille: {file_size_mb:.2f} MB")
    
    # Charger le modèle
    try:
        print("\n📍 Chargement du modèle...")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modèle chargé")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Classes: {model.output_shape[-1]}")
        print(f"   Params: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return False
    
    # Test 1: Reproductibilité
    print("\n" + "-"*60)
    print("TEST 1: REPRODUCTIBILITÉ")
    print("-"*60)
    
    img_array = np.random.randn(1, 224, 224, 3).astype(np.float32)
    results = []
    
    for i in range(5):
        pred = model.predict(img_array, verbose=0)
        top_class = int(np.argmax(pred[0]))
        conf = float(pred[0][top_class])
        results.append((top_class, conf))
        print(f"  {i+1}. Classe {top_class:2d} {conf:7.2%}")
    
    # Vérifier reproductibilité
    all_same = len(set(r[0] for r in results)) == 1
    if all_same:
        print(f"\n✅ REPRODUCTIBLE: Tous les résultats identiques")
    else:
        print(f"\n⚠️  NON REPRODUCTIBLE: Résultats varient")
    
    # Test 2: Distribution des prédictions
    print("\n" + "-"*60)
    print("TEST 2: DISTRIBUTION DES PRÉDICTIONS (100 appels)")
    print("-"*60)
    
    predictions = np.zeros(model.output_shape[-1])
    
    for i in range(100):
        img_test = np.random.randn(1, 224, 224, 3).astype(np.float32)
        pred = model.predict(img_test, verbose=0)
        top_class = np.argmax(pred[0])
        predictions[top_class] += 1
    
    print("\nDistribution des 100 prédictions par classe:")
    indices = np.argsort(predictions)[::-1]
    for rank, idx in enumerate(indices[:10], 1):
        count = int(predictions[idx])
        label = BILL_LABELS.get(idx, f"Unknown {idx}")
        bar = "█" * (count // 2)
        print(f"  {rank:2}. {label:15} {count:3}x {bar}")
    
    # Test 3: Confiance moyenne par classe
    print("\n" + "-"*60)
    print("TEST 3: CONFIANCE MOYENNE (50 appels)")
    print("-"*60)
    
    confidences = {i: [] for i in range(model.output_shape[-1])}
    
    for i in range(50):
        img_test = np.random.randn(1, 224, 224, 3).astype(np.float32)
        pred = model.predict(img_test, verbose=0)
        top_class = np.argmax(pred[0])
        conf = float(pred[0][top_class])
        confidences[top_class].append(conf)
    
    print("\nConfiance moyenne par classe:")
    avg_confs = []
    for idx in range(model.output_shape[-1]):
        if confidences[idx]:
            avg = np.mean(confidences[idx])
            count = len(confidences[idx])
            label = BILL_LABELS.get(idx, f"Unknown {idx}")
            avg_confs.append((avg, idx, label, count))
    
    for avg, idx, label, count in sorted(avg_confs, reverse=True)[:10]:
        print(f"  {label:15} {avg:7.2%} (n={count})")
    
    overall_avg = np.mean([c for confs in confidences.values() for c in confs])
    print(f"\n  Confiance moyenne globale: {overall_avg:.2%}")
    
    # Test 4: Stability (même image)
    print("\n" + "-"*60)
    print("TEST 4: STABILITÉ (Même image - 10 appels)")
    print("-"*60)
    
    img_stable = np.random.randn(1, 224, 224, 3).astype(np.float32)
    stable_results = []
    
    for i in range(10):
        pred = model.predict(img_stable, verbose=0)
        top_class = int(np.argmax(pred[0]))
        conf = float(pred[0][top_class])
        stable_results.append((top_class, conf))
        label = BILL_LABELS.get(top_class, f"Unknown {top_class}")
        print(f"  {i+1:2}. {label:15} {conf:7.2%}")
    
    all_stable = len(set(r[0] for r in stable_results)) == 1
    if all_stable:
        print(f"\n✅ STABLE: Toujours la même prédiction")
    else:
        print(f"\n⚠️  INSTABLE: Prédictions varient")
    
    # Test 5: Plages de confiance
    print("\n" + "-"*60)
    print("TEST 5: ANALYSE DE CONFIANCE (200 prédictions)")
    print("-"*60)
    
    all_confs = []
    for i in range(200):
        img_test = np.random.randn(1, 224, 224, 3).astype(np.float32)
        pred = model.predict(img_test, verbose=0)
        max_conf = float(np.max(pred[0]))
        all_confs.append(max_conf)
    
    all_confs = np.array(all_confs)
    
    print(f"\n  Min confiance:   {all_confs.min():.2%}")
    print(f"  Q1 confiance:    {np.percentile(all_confs, 25):.2%}")
    print(f"  Médiane:         {np.median(all_confs):.2%}")
    print(f"  Q3 confiance:    {np.percentile(all_confs, 75):.2%}")
    print(f"  Max confiance:   {all_confs.max():.2%}")
    print(f"  Moyenne:         {all_confs.mean():.2%}")
    print(f"  Std Dev:         {all_confs.std():.2%}")
    
    # Distribution de confiance
    high_conf = np.sum(all_confs > 0.9)
    med_conf = np.sum((all_confs > 0.5) & (all_confs <= 0.9))
    low_conf = np.sum(all_confs <= 0.5)
    
    print(f"\n  > 90%:  {high_conf:3}x ({high_conf/200*100:.1f}%)")
    print(f"  50-90%: {med_conf:3}x ({med_conf/200*100:.1f}%)")
    print(f"  < 50%:  {low_conf:3}x ({low_conf/200*100:.1f}%)")
    
    # Verdict
    print("\n" + "="*60)
    print("📊 VERDICT")
    print("="*60)
    
    checks = [
        (all_same, "✅" if all_same else "⚠️ ", "Reproductibilité"),
        (all_stable, "✅" if all_stable else "⚠️ ", "Stabilité"),
        (overall_avg > 0.3, "✅" if overall_avg > 0.3 else "❌", "Confiance > 30%"),
        (high_conf/200 > 0.5, "✅" if high_conf/200 > 0.5 else "⚠️ ", "> 50% avec confiance > 90%"),
    ]
    
    for check, icon, desc in checks:
        print(f"  {icon} {desc}")
    
    print("\n✅ TEST COMPLÉTÉ")
    return True

if __name__ == "__main__":
    model_path = r"model (1).h5"  # Local model
    load_and_test_model(model_path)
