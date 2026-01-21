#!/usr/bin/env python3
"""
Tester tous les billets via l'API locale
"""

import os
import requests
import json

TEST_DIR = "test_bills"
API_URL = "http://127.0.0.1:5000/predict"

def get_label_from_filename(filename):
    """Extraire le label du nom du fichier"""
    if filename.startswith("bill_"):
        parts = filename.replace("bill_", "").replace(".jpg", "").rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
    return None

def test_image_via_api(filepath):
    """Tester une image via l'API"""
    try:
        with open(filepath, 'rb') as f:
            files = {'file': f}
            response = requests.post(API_URL, files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('result'), data.get('confidence', 0), True
        else:
            return None, 0, False
    except Exception as e:
        print(f"❌ Erreur {filepath}: {e}")
        return None, 0, False

def main():
    print("=" * 70)
    print("🧪 Test Complet: Tous les Billets via API")
    print(f"🌐 API: {API_URL}")
    print("=" * 70)
    print()
    
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
    for i, filename in enumerate(jpg_files, 1):
        filepath = os.path.join(TEST_DIR, filename)
        expected_label = get_label_from_filename(filename)
        predicted_label, confidence, success = test_image_via_api(filepath)
        
        if not success or predicted_label is None:
            print(f"❌ [{i:2d}/{len(jpg_files)}] {filename.ljust(30)} → API ERROR")
            continue
        
        total += 1
        
        # Initialiser classe si besoin
        if expected_label not in results_by_class:
            results_by_class[expected_label] = {
                'correct': 0,
                'total': 0,
            }
        
        results_by_class[expected_label]['total'] += 1
        
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
        
        conf_pct = confidence * 100 if isinstance(confidence, float) else 0
        print(f"{status} [{i:2d}/{len(jpg_files)}] {filename.ljust(30)} → {predicted_label.ljust(12)} ({conf_pct:5.1f}%)")
    
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
        for i, confusion in enumerate(confusions[:10], 1):
            print(f"   {i}. {confusion['expected'].ljust(12)} → {confusion['predicted'].ljust(12)} ({confusion['confidence']*100:.1f}%)")
    
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
