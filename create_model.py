"""
Script pour créer un modèle MobileNetV2 pour la reconnaissance de billets
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np

# Nombre de classes (14 billets)
NUM_CLASSES = 14

print("🔨 Création du modèle MobileNetV2 avec 14 classes...")

# Charger MobileNetV2 pré-entraîné (ImageNet)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Geler les poids du modèle de base
base_model.trainable = False
print(f"✅ Base MobileNetV2 chargée: {base_model.count_params()} paramètres")

# Ajouter des couches custom pour la classification
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

# Compiler le modèle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Modèle créé avec {model.count_params()} paramètres")
print(f"📊 Input: {model.input_shape}")
print(f"📊 Output: {model.output_shape}")

# Sauvegarder le modèle
model_path = 'model.h5'
model.save(model_path)
print(f"💾 Modèle sauvegardé: {model_path}")

# Test rapide
print("\n🧪 Test du nouveau modèle:")
test_img = np.random.rand(1, 224, 224, 3).astype('float32')
pred = model.predict(test_img, verbose=0)
print(f"Confiance max: {np.max(pred):.2%}")
print(f"Classe prédite: {np.argmax(pred)}")
print(f"Distribution: {pred[0]}")

print("\n✅ Modèle prêt ! À entraîner avec vos données de billets (14 classes).")
