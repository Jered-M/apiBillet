#!/usr/bin/env python3
"""
Convertir le modèle Keras en format TensorFlow.js
Compatible avec Expo et React Native
"""

import os
import sys
import json
import tensorflow as tf
import numpy as np

# Chercher le modèle H5
MODEL_PATHS = [
    "best_model.h5",
    "model.h5",
    "model (1).h5",
]

h5_model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        h5_model = path
        print(f"✓ Trouvé modèle H5: {path}")
        break

if not h5_model:
    print("❌ Aucun modèle H5 trouvé!")
    sys.exit(1)

# Charger le modèle Keras
print(f"📦 Chargement du modèle: {h5_model}")
model = tf.keras.models.load_model(h5_model)
print(f"✅ Modèle chargé - Input: {model.input_shape}, Output: {model.output_shape}")

# Créer le répertoire de sortie
output_dir = "model_web"
os.makedirs(output_dir, exist_ok=True)

# Sauvegarder en format SavedModel (intermédiaire)
saved_model_dir = "temp_saved_model"
model.save(saved_model_dir, save_format='tf')
print(f"✅ SavedModel créé: {saved_model_dir}")

# Convertir en TensorFlow.js format
print("🔄 Conversion en TensorFlow.js format...")
os.system(f"tensorflowjs_converter --input_format tf_saved_model --output_format tfjs_graph_model {saved_model_dir} {output_dir}")

print(f"✅ Conversion réussie!")
print(f"📁 Fichiers générés dans: {output_dir}")
print()
print("📋 PROCHAINE ÉTAPE:")
print(f"1. Copie le contenu de '{output_dir}' vers: BillRecognition/assets/models/")
print(f"2. Installe les dépendances:")
print(f"   npm install @tensorflow/tfjs @tensorflow/tfjs-react-native")
print()

# Nettoyer le répertoire temporaire
import shutil
shutil.rmtree(saved_model_dir)
print(f"✅ Nettoyé: {saved_model_dir}")
