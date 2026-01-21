#!/usr/bin/env python3
"""
Convertir best_model.h5 en format compatible
"""

import tensorflow as tf
import sys

print("Chargement du modèle avec paramètres compatibilité...")
try:
    # Désactiver les validations strictes
    model = tf.keras.models.load_model(
        'best_model.h5',
        safe_mode=False,
        compile=False,
    )
    print("✅ Modèle chargé")
    print(f"   Input: {model.input_shape}")
    print(f"   Output: {model.output_shape}")
    
    # Compiler
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✅ Modèle compilé")
    
    # Sauvegarder
    model.save('model_compatible.h5')
    print("✅ Sauvegardé en: model_compatible.h5")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)
