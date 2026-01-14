#!/usr/bin/env python3
"""
Script pour convertir le modèle H5 en format SavedModel (plus stable et compatible)
"""

import os
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelConverter")

def convert_h5_to_saved_model():
    """Convertir les modèles H5 disponibles en format SavedModel"""
    
    h5_models = [
        "best_model.h5",
        "model.h5",
        "model (1).h5",
    ]
    
    for h5_path in h5_models:
        if not os.path.exists(h5_path):
            logger.info(f"⊘ {h5_path} non trouvé")
            continue
        
        logger.info(f"📦 Chargement: {h5_path}")
        try:
            # Charger le modèle H5
            model = tf.keras.models.load_model(h5_path)
            logger.info(f"✓ Modèle chargé: {model.input_shape} -> {model.output_shape}")
            
            # Sauvegarder en format SavedModel
            output_dir = "model_saved"
            logger.info(f"💾 Sauvegarde en SavedModel: {output_dir}")
            
            # Utiliser export() pour SavedModel format (compatible TFServing)
            model.export(output_dir)
            
            logger.info(f"✅ Conversion réussie: {h5_path} → {output_dir}")
            logger.info(f"   Paramètres du modèle: {model.count_params():,}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la conversion de {h5_path}: {type(e).__name__}: {str(e)}")
            continue
    
    logger.error("❌ Aucun modèle H5 n'a pu être convertir")
    return False

if __name__ == "__main__":
    logger.info("🔄 Démarrage de la conversion des modèles...")
    success = convert_h5_to_saved_model()
    
    if success:
        logger.info("✅ Conversion complétée avec succès!")
        logger.info("📝 Commandes Git pour pousser le changement:")
        logger.info("   git add model_saved/")
        logger.info("   git commit -m 'Convert model to SavedModel format'")
        logger.info("   git push origin main")
    else:
        logger.error("⚠️  La conversion a échoué")
