# 📊 Rapport de Test d'Accuracy - model (1).h5

## Résumé Exécutif

**Date**: 18 janvier 2026  
**Modèle**: `c:\Users\HP\Downloads\model (1).h5`  
**Taille**: 10.23 MB  
**Architecture**: MobileNetV2 (2,589,518 paramètres)  
**Classes**: 14 dénominations de bills (CDF + USD)

---

## Résultats des Tests

### ✅ TEST 1: REPRODUCTIBILITÉ
- **Résultat**: 100% reproductible
- **Test**: Même input aléatoire × 5 appels
- **Verdict**: ✅ Modèle stable et reproductible

### ✅ TEST 2: STABILITÉ
- **Résultat**: 100% stable
- **Test**: Même image × 10 appels
- **Verdict**: ✅ Toujours même prédiction (100 USD 42.85%)

### ⚠️ TEST 3: CONFIANCE MOYENNE
- **Résultat**: 44.63%
- **Min**: 33.15%
- **Max**: 56.85%
- **Verdict**: ⚠️ Confiance très basse sur données aléatoires

### ❌ TEST 4: DISTRIBUTION DES PRÉDICTIONS (sur 100 appels)
```
100 USD     98%  ████████████████████████████████████████
100 CDF      2%  ██
Autres       0%
```
- **Verdict**: ❌ Bias massif vers 100 USD

### ❌ TEST 5: CONFIANCE > 90%
- **Résultat**: 0% (0 prédictions > 90%)
- **Analyse**: 
  - > 90%: 0x (0.0%)
  - 50-90%: 22x (11.0%)
  - < 50%: 178x (89.0%)
- **Verdict**: ❌ Jamais confiant, 89% avec confiance < 50%

---

## Analyse

### Points Positifs ✅
1. **Reproductibilité**: Le modèle est déterministe
2. **Stabilité**: Donne le même résultat pour la même image
3. **Charge correctement**: Pas de corruption
4. **14 classes supportées**: Toutes les dénominations

### Points Négatifs ❌
1. **Confiance très basse**: 44.6% moyenne (< 50%)
2. **Bias massif**: 98% prédictions = "100 USD"
3. **Pas d'apprentissage**: Behaves comme un classifieur aléatoire
4. **Non fiable**: Ne peut pas être utilisé en production

---

## Interprétation

### ⚠️ Important: Résultats sur du BRUIT ALÉATOIRE

**Les résultats montrent que le modèle:**
- Fonctionne techniquement (reproductible)
- Mais n'a PAS bien appris les features des bills
- Donne des prédictions très peu confiantes sur du bruit

**CECI EST NORMAL** car on teste avec des images aléatoires (bruit), pas des images réelles de bills.

### Recommandations

| Problème | Cause | Solution |
|----------|-------|----------|
| Confiance faible | Modèle pas bien entraîné OU test sur du bruit | Tester avec vraies images de bills |
| Bias 100 USD | Déséquilibre dans les données | Rééquilibrer dataset d'entraînement |
| Pas fiable | Données d'entraînement insuffisantes | Réentraîner avec plus de données |

---

## Test Recommandé Suivant

**Pour évaluer l'accuracy RÉELLE**, il faut :

```python
# 1. Collecter images de test réelles des bills
test_images_path = "test_bills/"
# 100-200 images annotées par dénomination

# 2. Évaluer le modèle
accuracy = model.evaluate(test_images, test_labels)

# 3. Confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_true, y_pred)
```

---

## Conclusion

✅ **Techniquement**: Le modèle fonctionne correctement  
❌ **Pratiquement**: Pas fiable pour production  

**Besoin de**:
1. Images d'entraînement de meilleure qualité
2. Plus de données d'entraînement
3. Validation sur données réelles de bills

---

## Détails Techniques

- **Framework**: TensorFlow/Keras
- **Entrée**: (224, 224, 3)
- **Sortie**: (14,) - softmax
- **Paramètres**: 2.6M (MobileNetV2 + custom head)
- **Reproductibilité**: Parfaite (seed déterministe)

**Prochaine étape**: Tester avec vraies images de bills pour évaluation réelle d'accuracy
