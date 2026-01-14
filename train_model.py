"""
Script d'entraînement du modèle Bill Recognition
Adapté du code Colab pour fonctionner localement
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

import logging

# =========================
# CONFIGURATION
# =========================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = "./dataset"  # Adapter le chemin selon votre structure
MODEL_SAVE_PATH = "model (1).h5"

# Autoriser les images tronquées
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration TensorFlow
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelTraining")

# =========================
# ÉTAPE 1: SCANNER LE DATASET
# =========================

def get_dataset_structure(dataset_path):
    """Vérifie et liste la structure du dataset"""
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset non trouvé: {dataset_path}")
        return None
    
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    if not classes:
        logger.error(f"❌ Aucune classe trouvée dans {dataset_path}")
        return None
    
    logger.info(f"✅ {len(classes)} classes trouvées:")
    for i, cls in enumerate(classes, 1):
        class_path = os.path.join(dataset_path, cls)
        num_images = len([f for f in os.listdir(class_path) 
                         if os.path.isfile(os.path.join(class_path, f))])
        logger.info(f"  {i}. {cls}: {num_images} images")
    
    return classes

# =========================
# ÉTAPE 2: SCANNER LES IMAGES VALIDES
# =========================

def scan_valid_images(dataset_path, classes):
    """Scanne et recense les images valides"""
    valid_images = []
    corrupted_images = []
    
    logger.info("\n📸 Scanning des images...")
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            if not os.path.isfile(img_path):
                continue
            
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_images.append({'path': img_path, 'class': class_name})
            except Exception as e:
                corrupted_images.append({'path': img_path, 'error': str(e)})
    
    logger.info(f"✅ {len(valid_images)} images valides trouvées")
    if corrupted_images:
        logger.warning(f"⚠️ {len(corrupted_images)} images corrompues détectées")
    
    return valid_images

# =========================
# ÉTAPE 3: CRÉER LES DATA GENERATORS
# =========================

def create_data_generators(valid_images):
    """Crée les générateurs de données train/validation"""
    logger.info("\n🔄 Création des data generators...")
    
    # Créer un DataFrame
    valid_df = pd.DataFrame(valid_images)
    
    # Split 80/20
    train_df, validation_df = train_test_split(
        valid_df,
        test_size=0.2,
        random_state=42,
        stratify=valid_df['class']
    )
    
    logger.info(f"✅ Training: {len(train_df)} images")
    logger.info(f"✅ Validation: {len(validation_df)} images")
    
    # Data augmentation pour training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Pas d'augmentation pour validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Créer les générateurs
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col='path',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

# =========================
# ÉTAPE 4: CONSTRUIRE LE MODÈLE
# =========================

def build_model(num_classes):
    """Construit le modèle avec transfer learning MobileNetV2"""
    logger.info("\n🏗️ Construction du modèle...")
    
    # Charger MobileNetV2 pré-entraîné
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Geler les poids du base model
    base_model.trainable = False
    logger.info("✅ MobileNetV2 chargé et gelé")
    
    # Construire le modèle personnalisé
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Dégeler les dernières couches pour fine-tuning
    fine_tune_at = -30
    for layer in base_model.layers[fine_tune_at:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    logger.info(f"✅ Modèle construit ({len(model.layers)} couches)")
    
    return model

# =========================
# ÉTAPE 5: COMPILER ET ENTRAÎNER
# =========================

def train_model(model, train_generator, validation_generator):
    """Entraîne le modèle"""
    logger.info("\n⚙️ Compilation du modèle...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("✅ Modèle compilé")
    logger.info(f"\n🚀 Démarrage de l'entraînement ({EPOCHS} epochs)...\n")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    
    logger.info("\n✅ Entraînement terminé")
    return history

# =========================
# ÉTAPE 6: ÉVALUER ET SAUVEGARDER
# =========================

def evaluate_model(model, validation_generator):
    """Évalue le modèle"""
    logger.info("\n📊 Évaluation du modèle...")
    
    validation_generator.reset()
    Y_true = validation_generator.classes
    
    Y_pred_probs = model.predict(validation_generator, verbose=1)
    Y_pred = np.argmax(Y_pred_probs, axis=1)
    
    class_labels = list(validation_generator.class_indices.keys())
    
    # Confusion matrix
    cm = confusion_matrix(Y_true, Y_pred)
    
    # Classification report
    report = classification_report(Y_true, Y_pred, target_names=class_labels)
    
    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*60)
    logger.info(report)
    
    return cm, report

def save_model(model, save_path):
    """Sauvegarde le modèle"""
    logger.info(f"\n💾 Sauvegarde du modèle à {save_path}...")
    model.save(save_path)
    logger.info("✅ Modèle sauvegardé")

# =========================
# MAIN
# =========================

def main():
    logger.info("="*60)
    logger.info("BILL RECOGNITION - ENTRAÎNEMENT DU MODÈLE")
    logger.info("="*60)
    
    # Étape 1: Vérifier le dataset
    logger.info(f"\n📁 Recherche du dataset: {DATASET_PATH}")
    classes = get_dataset_structure(DATASET_PATH)
    if not classes:
        return
    
    # Étape 2: Scanner les images
    valid_images = scan_valid_images(DATASET_PATH, classes)
    if not valid_images:
        logger.error("❌ Aucune image valide trouvée")
        return
    
    # Étape 3: Créer les generators
    train_gen, val_gen = create_data_generators(valid_images)
    
    # Étape 4: Construire le modèle
    model = build_model(len(classes))
    
    # Étape 5: Entraîner
    history = train_model(model, train_gen, val_gen)
    
    # Étape 6: Évaluer et sauvegarder
    evaluate_model(model, val_gen)
    save_model(model, MODEL_SAVE_PATH)
    
    logger.info("\n" + "="*60)
    logger.info("✅ PROCESSUS TERMINÉ AVEC SUCCÈS")
    logger.info("="*60)
    logger.info(f"📦 Modèle disponible à: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
