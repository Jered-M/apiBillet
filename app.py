"""
API FastAPI pour la reconnaissance de billets
Utilise le modèle entraîné et applique la même préparation de données que le notebook
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import io
from datetime import datetime
import json

# ============ CONFIGURATION ============
IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_SAVE_DIR = r"C:\Users\HP\Music\MLBillet"
DATASET_PATH = r"C:\Users\HP\Pictures\ML\BillRecognition-API"

# ============ INITIALISATION DE L'APP ============
app = FastAPI(
    title="API Reconnaissance de Billets",
    description="Prédiction de billets avec le modèle EfficientNet/MobileNetV2",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ VARIABLES GLOBALES ============
model = None
class_labels = []

# ============ INITIALISATION ============

def load_model_and_labels():
    """Charge le modèle et les labels au démarrage"""
    global model, class_labels
    
    try:
        # Charger les labels à partir du dataset
        if os.path.exists(DATASET_PATH):
            class_labels = sorted([
                d for d in os.listdir(DATASET_PATH) 
                if os.path.isdir(os.path.join(DATASET_PATH, d))
            ])
            print(f"[OK] Classes détectées : {class_labels}")
        else:
            raise FileNotFoundError(f"Dataset path non trouvé: {DATASET_PATH}")
        
        # Chercher le modèle
        model_names = [
            'best_efficientnet_model.h5',
            'modele_reconnaissance_billets.h5',
            'best_model.h5',
            'model.h5'
        ]
        
        model_path = None
        for model_name in model_names:
            potential_path = os.path.join(MODEL_SAVE_DIR, model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"Aucun modèle trouvé dans {MODEL_SAVE_DIR}. "
                f"Fichiers attendus: {model_names}"
            )
        
        # Charger le modèle
        model = load_model(model_path)
        print(f"[OK] Modèle chargé avec succès: {os.path.basename(model_path)}")
        return True
        
    except Exception as e:
        print(f"[ERREUR] Erreur lors du chargement du modèle: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Exécuté au démarrage de l'API"""
    print("[INFO] Démarrage de l'API...")
    if not load_model_and_labels():
        print("[ATTENTION] Le modèle n'a pas pu être chargé")

# ============ FONCTIONS UTILITAIRES ============

def prepare_image(image_bytes) -> np.ndarray:
    """
    Prépare l'image comme dans le notebook
    
    Args:
        image_bytes: Bytes de l'image
        
    Returns:
        Array normalisé prêt pour la prédiction
    """
    # Ouvrir l'image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Redimensionner
    img_resized = img.resize((IMG_HEIGHT, IMG_WIDTH))
    
    # Normaliser les pixels entre 0 et 1
    img_array = np.array(img_resized) / 255.0
    
    # Ajouter batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(image_array: np.ndarray) -> dict:
    """
    Effectue la prédiction
    
    Args:
        image_array: Array d'image préparé
        
    Returns:
        Dict avec prédiction et confiance
    """
    if model is None:
        raise RuntimeError("Modèle non chargé")
    
    # Prédiction
    predictions = model.predict(image_array, verbose=0)
    
    # Récupérer la classe prédite et la confiance
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx] * 100)
    
    # Récupérer le label
    label = (
        class_labels[predicted_class_idx] 
        if predicted_class_idx < len(class_labels) 
        else "Inconnu"
    )
    
    # Toutes les prédictions avec scores
    all_predictions = {
        class_labels[i]: float(predictions[0][i] * 100)
        for i in range(len(class_labels))
    }
    
    return {
        "classe_predite": label,
        "confiance": round(confidence, 2),
        "index": int(predicted_class_idx),
        "toutes_predictions": all_predictions,
        "timestamp": datetime.now().isoformat()
    }

# ============ ENDPOINTS ============

@app.get("/")
async def root():
    """Endpoint racine - Info sur l'API"""
    return {
        "nom": "API Reconnaissance de Billets",
        "version": "1.0.0",
        "classes": class_labels,
        "nb_classes": len(class_labels),
        "endpoints": [
            "/predict - POST - Prédire une image",
            "/info - GET - Infos sur le modèle",
            "/health - GET - Vérifier la santé de l'API"
        ]
    }

@app.get("/info")
async def info():
    """Infos sur le modèle et les classes"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "statut": "✅ Prêt",
        "modele_charge": True,
        "classes": class_labels,
        "nb_classes": len(class_labels),
        "img_hauteur": IMG_HEIGHT,
        "img_largeur": IMG_WIDTH,
        "model_save_dir": MODEL_SAVE_DIR,
        "dataset_path": DATASET_PATH
    }

@app.get("/health")
async def health():
    """Vérifier la santé de l'API"""
    return {
        "statut": "✅ Opérationnel",
        "modele_charge": model is not None,
        "classes_disponibles": len(class_labels) > 0
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prédire la classe d'une image
    
    - **file**: Image à analyser (jpg, png, jpeg)
    
    Retourne:
    - classe_predite: Label de la classe
    - confiance: Confiance en %
    - toutes_predictions: Scores pour chaque classe
    """
    
    # Vérifier que le modèle est chargé
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Vérifier le type de fichier
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400,
            detail="Format fichier non supporté. Utilisez JPG, JPEG ou PNG"
        )
    
    try:
        # Lire le fichier
        image_bytes = await file.read()
        
        # Préparer l'image (même traitement que le notebook)
        img_array = prepare_image(image_bytes)
        
        # Prédiction
        result = predict_image(img_array)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur lors du traitement de l'image: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Prédire plusieurs images à la fois
    
    - **files**: Listes d'images à analyser
    
    Retourne:
    - resultats: Liste des prédictions pour chaque image
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    resultats = []
    
    for file in files:
        try:
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                resultats.append({
                    "filename": file.filename,
                    "erreur": "Format non supporté"
                })
                continue
            
            # Lire et préparer
            image_bytes = await file.read()
            img_array = prepare_image(image_bytes)
            
            # Prédiction
            result = predict_image(img_array)
            result["filename"] = file.filename
            
            resultats.append(result)
            
        except Exception as e:
            resultats.append({
                "filename": file.filename,
                "erreur": str(e)
            })
    
    return {"nombre_images": len(files), "resultats": resultats}

# ============ GESTION DES ERREURS ============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire global des erreurs"""
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur", "error": str(exc)}
    )

# ============ LANCER L'APP ============

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("[INFO] Lancement de l'API")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
