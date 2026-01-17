# 🚨 CORRECTION MOBILENETV2 - 18 JAN 2026

## ❌ LE BUG

Le modèle MobileNetV2 utilise une normalisation **SPÉCIALE**:

```
pixel_value = (pixel / 127.5) - 1
```

Cela convertit [0, 255] → [-1, 1], pas [0, 1] comme un rescale normal.

### Ce qui était faux :

```python
# ❌ INCORRECT - donne [0, 1]
img_array = np.array(img) / 255.0

# ❌ INCORRECT - donne [0, 1]
train_datagen = ImageDataGenerator(rescale=1./255)
```

### Ce qui est correct :

```python
# ✅ CORRECT - donne [-1, 1]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
img_array = preprocess_input(img_array)

# ✅ CORRECT - dans ImageDataGenerator
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: preprocess_input(x)
)
```

---

## ✅ FICHIERS CORRIGÉS

### 1. ML1.ipynb (Colab)

- ✅ ImageDataGenerator avec `preprocessing_function=preprocess_input`
- ✅ Pas de `rescale=1./255` au training
- ✅ Au test: utilise `preprocess_input`

### 2. app.py (Backend Flask)

- ✅ Déjà correct - utilise `preprocess_input`
- ✅ Ajoute ExifTranspose pour iPhone
- ✅ Endpoint `/debug/save-raw` pour tests

### 3. React Native (App)

- ⏳ À faire: skipProcessing + ExifTranspose backend

---

## 🧪 SCRIPTS DE TEST

### 1. test_pipeline.py

```bash
python test_pipeline.py uploads/raw_bill.jpg
```

Teste le preprocessing comme Colab.

### 2. validate_pipeline.py (NOUVEAU)

```bash
python validate_pipeline.py uploads/raw_bill.jpg
```

Compare Colab vs Backend - doit être **100% identique**.

### 3. test_api.bat

```bash
test_api.bat uploads/raw_bill.jpg
```

Teste le backend Flask.

---

## 🎯 PROCÉDURE DE VÉRIFICATION

1. **Colab** : Télécharge ML1.ipynb corrigé

   ```python
   img_array = preprocess_input(img_array)
   ```

2. **Backend** : Lance `python app.py` (déjà correct)

3. **Test** :

   ```bash
   python validate_pipeline.py uploads/test.jpg
   ```

   Doit afficher **✅ Classes identiques**

4. **App** :
   - Prends une photo
   - Envoie au backend
   - Compare avec Colab

---

## 🔍 POURQUOI C'EST CRITIQUE

Une **image différente** → prédiction différente → **projet échoué**

```
Same Image
    ↓
Colab: Correct (preprocess_input)
Backend: Correct (preprocess_input)
App: ?
    ↓
Si tout pareil → SUCCESS ✅
Si un différent → FAIL ❌
```

---

## 📋 CHECKLIST

- [ ] ML1.ipynb corrigé (preprocess_input au training)
- [ ] Backend OK (app.py déjà bon)
- [ ] validate_pipeline.py montre 100% alignment
- [ ] App envoie photo brute
- [ ] Backend fait preprocess
- [ ] Même image = même résultat partout

---

**Status**: CRITÈRE ACCEPTATION = Même image → Même résultat (Colab/Backend/App)
