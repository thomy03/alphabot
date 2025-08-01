# 📚 Documentation AlphaBot ML/DL Training - Index Principal

## 🎯 Vue d'ensemble

Cette documentation couvre le workflow complet d'entraînement ML/DL pour AlphaBot sur Google Colab, avec gestion des données multi-formats yfinance.

## 📋 Étapes du Workflow

### 1. 🚀 Ouverture Google Colab
- Accéder à [Google Colab](https://colab.research.google.com)
- Uploader `ALPHABOT_ML_TRAINING_COLAB.ipynb`
- Vérifier le runtime (GPU L4 recommandé)

### 2. ⚙️ Configuration GPU/TPU
- Runtime → Change runtime type → GPU L4
- Exécuter cellule 1 pour vérifier la configuration
- Si GPU non disponible, le système bascule automatiquement sur CPU

### 3. 🏃 Entraînement des modèles

#### Problème détecté : Format MultiIndex yfinance
Les données téléchargées ont un format MultiIndex avec structure :
```
Price  | Close | High | Low | Open | Volume
Ticker | AAPL  | AAPL | AAPL| AAPL | AAPL
```

**Solution appliquée** :
- Détection automatique du format MultiIndex
- Extraction correcte des colonnes (Close, High, Low, Volume)
- Gestion des cas où Volume est manquant (création d'un proxy)
- Seuils de prédiction ajustés à ±0.2% pour plus de données

#### Séquence d'exécution :
1. **Cellule 0** : Suivi de progression
2. **Cellule 1** : Setup GPU/TPU
3. **Cellule 2** : Montage Google Drive
4. **Cellule 4** : Téléchargement données (29 tickers)
5. **Cellule 5** : Pattern Detector (LSTM ou Dense fallback)
6. **Cellule 6** : Sentiment Analyzer (FinBERT)
7. **Cellule 7** : RAG Integrator (FAISS)
8. **Cellule 8** : Intégration finale

### 4. 🔄 Push vers GitHub
```bash
# Dans Colab
!cd /content && git add -A
!cd /content && git commit -m "Training update: models trained"
!cd /content && git push origin main
```

### 5. 🧪 Tests locaux
```bash
# En local
git pull origin main
python test_hybrid_orchestrator.py
```

### 6. 🚀 Déploiement
- Vérifier les modèles dans `/content/drive/MyDrive/AlphaBot_ML_Training/models/`
- Transférer vers serveur de production si nécessaire

## 📊 Monitoring et Debug

### Logs disponibles :
- `/content/drive/MyDrive/AlphaBot_ML_Training/logs/pattern_debug.txt` : Debug détaillé de la préparation des données
- `/content/drive/MyDrive/AlphaBot_ML_Training/logs/windows_csv/` : Dumps CSV des fenêtres créées
- `/content/market_data_csv/` : Données brutes téléchargées

### Vérifications importantes :
1. **Après téléchargement (cellule 4)** :
   - Vérifier "Symbols téléchargés: [...]" non vide
   - Vérifier les CSV dans `/content/market_data_csv/`

2. **Après préparation Pattern (cellule 5)** :
   - Si X=(0,), vérifier `pattern_debug.txt`
   - Chercher les lignes avec "columns=" pour voir le format détecté
   - Vérifier que close/high/low/volume sont bien trouvées

## 🔧 Troubleshooting

### Problème : "Dataset vide" malgré données téléchargées
**Cause** : Format MultiIndex non géré correctement
**Solution** : Le notebook v2 gère maintenant ce format automatiquement

### Problème : GPU non détecté
**Solution** : Le système bascule automatiquement sur modèle Dense (sans LSTM)

### Problème : Montage Drive échoue
**Solution** : Relancer la cellule 2, elle nettoie et remonte automatiquement

## 📈 Résultats attendus

- **Pattern Detector** : Accuracy ~35-40% (3 classes : hausse/stable/baisse)
- **Sentiment Analyzer** : Fine-tuning minimal de FinBERT
- **RAG** : Index FAISS avec 5 documents de base

## 🔄 Workflow de reprise

Si interruption :
1. Relancer notebook
2. Exécuter cellule 0 (suivi progression)
3. Le système indique où reprendre
4. Continuer depuis la cellule suggérée

## 📝 Notes importantes

- Les données sont sauvegardées sur Google Drive pour persistance
- Le notebook supporte les interruptions/reprises
- Tous les modèles sont sauvegardés en `.keras` et `.pkl`
- Les logs détaillés permettent de diagnostiquer tout problème

---
*Dernière mise à jour : 1er Août 2025*
*Version : 2.0 - Support MultiIndex yfinance*
