# 🔍 Diagnostic: Confusion 500 CDF vs 20 USD

## Problème Décrit
- ✅ Sur Colab: Bon résultats, reconnaît 500 CDF vs 20 USD
- ❌ Sur app: Confusion entre 500 CDF et 20 USD

## Causes Possibles

### 1. **Modèle Différent** (Probabilité: HAUTE)
```
Colab:         BillRecognition-API/
├── model.h5        ← Modèle entraîné (14 classes)
└── best_model.h5   ← Ancien modèle?

App:           bill-recognition-v2/
├── assets/model.tflite  ← Version convertie?
└── Différent du Colab?
```

**Action:**
1. Vérifier le checksum du `model.h5` utilisé dans l'API
2. Vérifier quel modèle a été converti en TFLite
3. Comparer les outputs: `model.h5` vs `model.tflite`

### 2. **Preprocessing Différent** (Probabilité: MOYENNE)

**Colab (ML1.ipynb):**
```python
# Normalement:
img = load_image(path)
img = img.resize((224, 224))
img = img / 255.0  # Normalisation
img = np.expand_dims(img, axis=0)  # Batch
predictions = model.predict(img)
```

**App:**
```javascript
// services/imagePreprocessing.js
// ❓ Fait quoi exactement?
// Image brute → Backend fait le preprocessing
```

**Action:**
Vérifier que [imagePreprocessing.js](../../bill-recognition-v2/services/imagePreprocessing.js) n'ajoute/retire rien

### 3. **Ordre des Classes Différent** (Probabilité: BASSE)

Les labels sont identiques:
```
0: "100 CDF", ..., 3: "500 CDF", ..., 12: "20 USD", ...
```

Vérifier que `BILL_LABELS` est identique partout:
- [app.py](app.py) ligne ~37
- [tfliteLocal.js](../../bill-recognition-v2/services/tfliteLocal.js) ligne ~11

### 4. **Données d'Entraînement Déséquilibrées** (Probabilité: MOYENNE)

Si le modèle Colab a:
- ✅ 500 CDF: 1000 images
- ❌ 20 USD: 100 images

→ Peut confondre 20 USD (peu de données)

---

## 🧪 Plan de Test

### Étape 1: Tester Image Spécifique

```bash
# Prendre une photo de 500 CDF dans l'app
# Sauvegarder comme test_bills/500-cdf.jpg

# Puis tester sur API
cd BillRecognition-API
node test-model-confusion.js test_bills/500-cdf.jpg
```

**Résultat attendu:**
```
✅ Résultat: 500 CDF
   Confiance: 0.95

📊 Top 5 prédictions:
   1. 500 CDF      ████████████████████ 95.23%
   2. 20000 CDF    ███                  2.15%
   3. 20 USD       ██                   1.45%
```

**Résultat problème:**
```
❌ Résultat: 20 USD
   Confiance: 0.52

📊 Top 5 prédictions:
   1. 20 USD       ███████████          52.15%
   2. 500 CDF      ███████████          47.85%  ← Très proche!
   3. ...
```

### Étape 2: Comparer Modèles

```bash
# 1. Vérifier quel modèle l'API utilise
curl https://apibillet-1.onrender.com/health

# Réponse doit montrer: "file": "model.h5"

# 2. Vérifier le modèle local
ls -lh BillRecognition-API/model.h5
ls -lh BillRecognition-API/best_model.h5
ls -lh bill-recognition-v2/assets/model.tflite

# 3. Comparer tailles:
# Si model.h5 ≠ best_model.h5 → Mauvais modèle chargé!
```

### Étape 3: Tester Preprocessing

Créer script `test-preprocessing.py`:
```python
import numpy as np
from PIL import Image

# Charger même image
img = Image.open('test_bills/500-cdf.jpg')

# Preprocessing Colab
img_224 = img.resize((224, 224))
img_array = np.array(img_224) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Vérifier shapes
print(f"Image shape: {img_array.shape}")      # (224, 224, 3)
print(f"Batch shape: {img_batch.shape}")      # (1, 224, 224, 3)
print(f"Min/Max values: {img_array.min():.4f} / {img_array.max():.4f}")  # 0.0 / 1.0

# Tester prédiction
model = tf.keras.models.load_model('model.h5')
predictions = model.predict(img_batch)
print(f"Predictions: {predictions[0]}")
```

---

## 🔧 Solutions Potentielles

### Solution 1: Utiliser le Bon Modèle
```bash
# Si model.h5 n'est pas le bon
cp best_model.h5 model.h5
# Puis redémarrer API
```

### Solution 2: Réentraîner avec Plus de Données
```python
# Colab: Ajouter plus de 20 USD images
# Vérifier balance dataset:
print(f"500 CDF: {count_500_cdf} images")
print(f"20 USD: {count_20_usd} images")

# Augmentation données (augmentation d'images):
# - Rotation ±15°
# - Zoom 0.8-1.2
# - Brightness ±0.2
```

### Solution 3: Augmenter Confiance Seuil

Si 500 CDF vs 20 USD sont toujours proches, accepter seulement si confiance > 0.75:
```javascript
if (data.confidence < 0.75) {
  // Résultat ambigu - demander à l'utilisateur
  return { ambiguous: true, top2: [top_class, second_class] };
}
```

### Solution 4: Ajouter Vérification Visuelle

Si confiance trop basse, demander confirmation utilisateur

---

## 📊 Checklist Diagnostic

- [ ] Vérifier `model.h5` utilisé sur API (curl /health)
- [ ] Comparer avec Colab: même modèle utilisé?
- [ ] Tester avec image 500 CDF réelle
- [ ] Tester avec image 20 USD réelle
- [ ] Vérifier preprocessing identique
- [ ] Vérifier labels identiques
- [ ] Vérifier balance dataset
- [ ] Check si confiance score faible (< 0.75)

---

## 🚀 Test Immédiat

### Sur Linux/macOS:
```bash
cd BillRecognition-API

# Tester API
curl -X POST -F "file=@test_bills/500-cdf.jpg" \
  https://apibillet-1.onrender.com/predict | jq .
```

### Sur Windows (PowerShell):
```powershell
$file = Get-Item "test_bills\500-cdf.jpg"
$fileContent = [System.IO.File]::ReadAllBytes($file.FullName)
$fileEnc = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetString($fileContent)

$uri = "https://apibillet-1.onrender.com/predict"
$body = @{file = $fileEnc}

Invoke-RestMethod -Uri $uri -Method Post -Body $body
```

---

## 📚 Ressources

- [ML1.ipynb](../ML1.ipynb) - Notebook d'entraînement original
- [app.py](app.py) - API backend
- [model.h5](model.h5) - Modèle Keras

---

**Status:** 🔍 Investigation Needed  
**Priority:** 🔴 High (Feature Breaking)  
**Date:** 20 Janvier 2026
