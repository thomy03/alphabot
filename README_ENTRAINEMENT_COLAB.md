# 🚀 AlphaBot ML/DL Training - Guide Google Colab

## 📋 Vue d'ensemble

Ce guide explique comment entraîner les modèles Machine Learning et Deep Learning d'AlphaBot sur Google Colab en utilisant les ressources GPU/TPU optimisées.

## 🎯 Objectifs

- **Entraîner 3 composants ML/DL** :
  - Pattern Detector (LSTM + CNN)
  - Sentiment Analyzer (FinBERT + RoBERTa)
  - RAG Integrator (Embeddings + FAISS)

- **Optimisations Colab** :
  - GPU/TPU acceleration
  - Mixed precision training
  - Memory management
  - Automatic checkpoints
  - Timeout protection

## 📁 Structure des fichiers

```
Tradingbot_V2/
├── ALPHABOT_ML_TRAINING_COLAB.ipynb    # Notebook principal Colab
├── colab_utils.py                      # Utilitaires Colab optimisés
├── drive_manager.py                    # Gestion Google Drive
├── alphabot/ml/                        # Composants ML/DL
│   ├── pattern_detector.py             # Détecteur de patterns
│   ├── sentiment_analyzer.py           # Analyseur de sentiment
│   └── rag_integrator.py               # Intégrateur RAG
└── README_ENTRAINEMENT_COLAB.md        # Ce guide
```

## 🛠️ Prérequis

### 1. Compte Google avec Drive
- Compte Google valide
- Google Drive activé
- 15GB+ d'espace disponible

### 2. Accès Colab
- Accès à Google Colab (gratuit ou Pro)
- Pour GPU : Colab Pro recommandé pour sessions plus longues

### 3. Dépendances locales (optionnel)
Pour tester localement avant Colab :
```bash
pip install tensorflow torch transformers
pip install yfinance pandas numpy scikit-learn
```

## 🚀 Démarrage rapide

### Étape 1: Ouvrir Google Colab
1. Aller sur [colab.research.google.com](https://colab.research.google.com)
2. Créer un nouveau notebook
3. Renommer-le "AlphaBot_ML_Training"

### Étape 2: Configuration GPU
1. Dans le menu : `Runtime` → `Change runtime type`
2. Sélectionner `GPU` ou `TPU`
3. Cliquer `Save`

### Étape 3: Upload des fichiers
Méthode 1: Direct upload
```python
from google.colab import files
uploaded = files.upload()
```

Méthode 2: Clone depuis GitHub (recommandé)
```python
!git clone https://github.com/votre-username/AlphaBot.git
%cd AlphaBot
```

### Étape 4: Exécution du notebook
1. Exécuter les cellules dans l'ordre
2. Attendre la fin de chaque cellule avant de passer à la suivante
3. Surveiller les logs pour les erreurs

## 📊 Détail du notebook Colab

### CELLULE 1: Setup GPU/TPU optimisé
- Installation des dépendances
- Détection GPU/TPU
- Configuration mixed precision
- Memory management

### CELLULE 2: Google Drive setup
- Montage de Google Drive
- Création de la structure de dossiers
- Vérification des permissions

### CELLULE 3: Code AlphaBot setup
- Upload/configuration du code AlphaBot
- Import des modules ML
- Vérification de l'intégrité

### CELLULE 4: Téléchargement des données
- Téléchargement des données de marché (Yahoo Finance)
- Préparation des datasets
- Visualisation des distributions

### CELLULE 5: Entraînement Pattern Detector
- Configuration LSTM/CNN
- Entraînement avec callbacks optimisés
- Sauvegarde des modèles

### CELLULE 6: Entraînement Sentiment Analyzer
- Fine-tuning FinBERT/RoBERTa
- Préparation des données texte
- Entraînement avec monitoring

### CELLULE 7: Entraînement RAG Integrator
- Création des embeddings
- Indexation FAISS
- Tests de recherche sémantique

### CELLULE 8: Intégration et tests
- Intégration des modèles entraînés
- Tests de performance
- Export des résultats

## ⚡ Optimisations Colab

### GPU Memory Management
```python
# Configuration mémoire GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Mixed Precision Training
```python
# Activer la précision mixte
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### Batch Size Optimization
```python
# Optimiser le batch size selon la mémoire disponible
batch_size = optimize_batch_size(
    dataset_size=len(X_train),
    available_memory_gb=12.0,  # GPU RAM
    model_complexity='medium'
)
```

### Checkpoint Management
```python
# Callbacks optimisés
callbacks = create_colab_callbacks(
    model_name="pattern_detector",
    save_path="/content/drive/MyDrive/AlphaBot_ML_Training",
    patience=10
)
```

## 📈 Monitoring et Performance

### Métriques suivies
- **Accuracy** : Précision du modèle
- **Loss** : Fonction de perte
- **Val Accuracy** : Précision sur validation
- **Memory Usage** : Utilisation RAM/GPU
- **Training Time** : Temps d'entraînement

### Visualisation
```python
# Courbes d'apprentissage
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()
```

## 💾 Sauvegarde et Gestion

### Google Drive Structure
```
/AlphaBot_ML_Training/
├── models/                    # Modèles entraînés
│   ├── pattern_detector/
│   ├── sentiment_analyzer/
│   └── rag_integrator/
├── data/                     # Données d'entraînement
├── logs/                     # Logs et métriques
├── checkpoints/              # Checkpoints d'entraînement
├── exports/                  # Exports et backups
└── configs/                  # Fichiers de configuration
```

### Sauvegarde automatique
```python
# Sauvegarder un modèle
drive_manager.save_model(
    model=trained_model,
    model_name="lstm_pattern_v1",
    model_type="pattern_detector",
    metadata={
        "accuracy": 0.89,
        "epochs": 50,
        "batch_size": 32
    }
)
```

## 🔧 Dépannage

### Problèmes courants

1. **GPU non disponible**
   ```
   Solution: Runtime → Change runtime type → GPU
   ```

2. **Mémoire insuffisante**
   ```
   Solution: Réduire batch_size ou activer memory growth
   ```

3. **Timeout Colab**
   ```
   Solution: Activer timeout protection dans les callbacks
   ```

4. **Erreur d'import**
   ```
   Solution: Vérifier l'installation des dépendances
   ```

### Logs utiles
```python
# Vérifier l'environnement
env_info = get_colab_environment()
print(json.dumps(env_info, indent=2))

# Vérifier la mémoire
memory_usage = ColabMemoryMonitor().get_memory_usage()
print(f"RAM utilisée: {memory_usage['percent_used']:.1f}%")
```

## 🎯 Résultats attendus

### Pattern Detector
- **Accuracy cible** : > 85%
- **Latence** : < 50ms
- **Mémoire** : < 100MB

### Sentiment Analyzer
- **Accuracy cible** : > 90%
- **F1-score** : > 0.85
- **Temps d'inférence** : < 100ms

### RAG Integrator
- **Recall@10** : > 0.8
- **Indexation** : < 1s pour 1000 documents
- **Mémoire** : < 500MB

## 📚 Prochaines étapes

### Après l'entraînement
1. **Télécharger les modèles** depuis Google Drive
2. **Intégrer dans AlphaBot** local
3. **Tester en paper trading**
4. **Déployer en production**

### Améliorations possibles
- **Data augmentation** pour plus de données
- **Hyperparameter tuning** avec Optuna
- **Model ensemble** pour meilleure performance
- **Real-time training** avec streaming data

## 📞 Support

Pour toute question ou problème :
1. Vérifier les logs dans Colab
2. Consulter la documentation dans `docs/`
3. Vérifier les issues GitHub
4. Contacter le support technique

---

**Note** : Ce guide est optimisé pour Google Colab Pro. Pour la version gratuite, réduire les batch sizes et la durée d'entraînement.
