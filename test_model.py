"""
Script de test du modèle Bill Recognition
Utilise exactement le même code que le test Colab
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys

# 1. Configuration des paramètres
IMG_HEIGHT = 224
IMG_WIDTH = 224

def get_script_dir():
    """Obtient le répertoire du script"""
    return os.path.dirname(os.path.abspath(__file__))

def get_default_paths():
    """Retourne les chemins par défaut locaux"""
    script_dir = get_script_dir()
    return {
        'model_path': os.path.join(script_dir, 'model (1).h5'),
        'dataset_path': None  # Sera défini dynamiquement
    }

def get_class_labels(dataset_path):
    """Récupère les labels des classes du dataset (trié alphabétiquement)"""
    if os.path.exists(dataset_path):
        class_labels = sorted([d for d in os.listdir(dataset_path) 
                              if os.path.isdir(os.path.join(dataset_path, d))])
        print(f"✅ {len(class_labels)} classes trouvées dans le dataset")
        return class_labels
    else:
        print("⚠️ Dossier dataset non trouvé.")
        return []

def load_trained_model(model_path=None):
    """Charge le modèle entraîné"""
    if model_path is None:
        model_path = get_default_paths()['model_path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modèle introuvable: {model_path}")
    
    print(f"📂 Chargement du modèle depuis: {model_path}")
    model = load_model(model_path)
    print("✅ Modèle chargé avec succès!")
    return model

def preprocess_image_for_prediction(image_path):
    """
    Prétraite l'image pour la prédiction
    Utilise la même normalisation que le script Colab: division par 255.0
    Convertit les pixels de [0, 255] à [0, 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image introuvable: {image_path}")
    
    # Charger l'image
    img = Image.open(image_path).convert('RGB')
    print(f"📖 Image ouverte: {img.size}")
    
    # Redimensionner
    img_resized = img.resize((IMG_HEIGHT, IMG_WIDTH))
    print(f"✅ Image redimensionnée: {(IMG_HEIGHT, IMG_WIDTH)}")
    
    # Convertir en array et normaliser par 255.0 (comme Colab)
    img_array = np.array(img_resiz, class_labels):
    """
    Prédit la classe du billet à partir d'une image
    Utilise exactement la logique du code Colab
    """
    print("\n" + "=" * 60)
    print("--- Démarrage de la prédiction ---")
    print("=" * 60)
    
    try:
        # Chargement et Prétraitement de l'image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"L'image est introuvable à : {image_path}")

        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img_resized) / 255.0  # Normalisation (comme Colab)
        img_array = np.expand_dims(img_array, axis=0)  # Ajout de la dimension batch

        print("✅ Image préparée avec succès.")

        # 4. Prédiction
        print("🤖 Exécution de la prédiction...")
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class_idx] * 100

        # 5. Get predicted class label using the obtained class_labels
        if predicted_class_idx < len(class_labels):
            predicted_class_label = class_labels[predicted_class_idx]
        else:
            predicted_class_label = f"Unknown Class (Index: {predicted_class_idx})"
            print(f"⚠️ Warning: Predicted class index {predicted_class_idx} is out of bounds for class_labels length {len(class_labels)}.")

        # 6. Affichage des résultats
        print("\n" + "=" * 60)
        print("✅ RÉSULTAT FINAL")
        print("=" * 60)
        print(f"Résultat : {predicted_class_label}")
        print(f"Confiance : {confidence:.2f}%")
        print("=" * 60)

        # Afficher l'image avec le résultat
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Prédiction: {predicted_class_label}\nConfiance: {confidence:.2f}%")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        return {
            'predicted_class': predicted_class_label,
            'confidence': confidence / 100.0
        plt.title(f"Prédiction: {predicted_label}\nConfiance: {confidence * 100:.2f}%", 
                  fontsize=12, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return {
            'predicted_class': predicted_label,
            'confidence': confidence,
            'class_idx': int(predicted_class_idx)
        }
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la prédiction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_model_with_sample():
    """Teste le modèle avec une image d'exemple du dossier uploads"""
    script_dir = get_script_dir()
    uploads_dir = os.path.join(script_dir, 'uploads')
    
    if not os.path.exists(uploads_dir):
        print(f"⚠️ Dossier uploads non trouvé: {uploads_dir}")
        return
    
    # Chercher les images
    image_files = [f for f in os.listdir(uploads_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    if not image_files:
        print(f"⚠️ Aucune image trouvée dans {uploads_dir}")
        return
     et les classes
    try:
        model = load_trained_model()
        class_labels = get_class_labels(script_dir)  # Essayer de trouver les classes localement
        
        # Tester avec la première image
        test_image = os.path.join(uploads_dir, image_files[0])
        print(f"\n📸 Test avec l'image: {image_files[0]}")
        predict_bill(model, test_image, class_labels)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Fonction principale"""
    print("\n" + "🎯 " * 20)
    print("BILL RECOGNITION - TEST LOCAL")
    print("🎯 " * 20 + "\n")
    
    # Vérifications initiales
    script_dir = get_script_dir()
    print(f"📁 Répertoire de travail: {script_dir}\n")
    
    # Si des arguments sont passés, utiliser le premier comme chemin d'image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"--- Démarrage du processus ---\n")
        
        try:
            model = load_trained_model()
            print()
            
            # Essayer de charger les classes du dataset
            class_labels = get_class_labels(script_dir)
            if not class_labels:
                # Si pas de dataset local, utiliser les labels par défaut
                class_labels = sorted(['100FC', '50FC', '200FC', '500FC', '1000FC', 
                                      '5000FC', '10000FC', '20000FC', '100$', '5$', 
                                      '10$', '50$', '20$', '1$'])
                print(f"✅ {len(class_labels)} classes par défaut utilisées\n")
            
            predict_bill(model, image_path, class_labels)
        except Exception as e:
            print(f"\n❌ Erreur critique : {e}")
            import traceback
            traceback.print_exc()
    else:
        # Mode interactif
        print("Mode: Test avec image du dossier uploads")
        print("Usage: python test_model.py <chemin_image>
        print("(Vous pouvez aussi lancer: python test_model.py <chemin_image>)\n")
        test_model_with_sample()

if __name__ == '__main__':
    main()
