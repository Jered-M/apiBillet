#!/usr/bin/env python3
"""
Script pour convertir le modèle H5 en TFLite
TFLite est ~4x plus petit et plus rapide
"""

import os
import sys
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"✓ Trouvé modèle H5: {path}")
        break

if not h5_model:
    logger.error("❌ Aucun modèle H5 trouvé!")
    sys.exit(1)

# Charger le modèle Keras
logger.info(f"📦 Chargement du modèle: {h5_model}")
model = tf.keras.models.load_model(h5_model)
logger.info(f"✅ Modèle chargé - Input: {model.input_shape}, Output: {model.output_shape}")

# Convertir en TFLite
logger.info("🔄 Conversion en TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

# Sauvegarder
output_path = "model.tflite"
with open(output_path, 'wb') as f:
    f.write(tflite_model)

h5_size = os.path.getsize(h5_model) / 1024 / 1024
tflite_size = os.path.getsize(output_path) / 1024 / 1024

logger.info(f"✅ Conversion réussie!")
logger.info(f"📊 Tailles:")
logger.info(f"   H5:     {h5_size:.1f} MB")
logger.info(f"   TFLite: {tflite_size:.1f} MB ({100*tflite_size/h5_size:.0f}%)")
logger.info(f"✨ Fichier sauvegardé: {output_path}")
