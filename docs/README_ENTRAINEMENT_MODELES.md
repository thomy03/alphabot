# Guide d'Entraînement des Modèles ML/DL - AlphaBot

## 📋 Vue d'Ensemble

Ce guide explique comment entraîner les modèles Machine Learning et Deep Learning d'AlphaBot pour le système hybride de trading.

## 🏗️ Architecture des Composants ML

### 1. **ML Pattern Detector** (`alphabot/ml/pattern_detector.py`)
- **Objectif**: Détecter des patterns de prix et volumes complexes
- **Modèles**: LSTM, CNN, Random Forest, Gradient Boosting
- **Données**: Séquences temporelles de prix (OHLC) et volumes

### 2. **Sentiment DL Analyzer** (`alphabot/ml/sentiment_analyzer.py`)
- **Objectif**: Analyser le sentiment des marchés financiers
- **Modèles**: FinBERT, RoBERTa, VADER
- **Données**: News, tweets, rapports financiers

### 3. **RAG Integrator** (`alphabot/ml/rag_integrator.py`)
- **Objectif**: Fournir du contexte et des explications
- **Technologies**: Sentence Transformers, FAISS, TF-IDF
- **Données**: Documents financiers, Q&A, connaissances de trading

## 📁 Structure des Dossiers

```
Tradingbot_V2/
├── alphabot/
│   ├── ml/
│   │   ├── __init__.py                 # Package ML unifié
│   │   ├── pattern_detector.py         # Détection de patterns
│   │   ├── sentiment_analyzer.py       # Analyse de sentiment
│   │   └── rag_integrator.py          # Intégration RAG
│   └── core/
│       └── hybrid_orchestrator.py      # Orchestrateur hybride
├── data/                              # Données d'entraînement
├── old_scripts/                       # Anciens scripts déplacés
├── train_ml_models.py                 # Script d'entraînement
├── test_hybrid_orchestrator.py        # Tests d'intégration
└── README_ENTRAINEMENT_MODELES.md     # Ce guide
```

## 🚀 Processus d'Entraînement

### Étape 1: Prérequis

Assurez-vous d'avoir installé toutes les dépendances :

```bash
pip install numpy pandas yfinance tensorflow scikit-learn sentence-transformers faiss-cpu requests
```

### Étape 2: Lancer l'Entraînement

Exécutez le script d'entraînement :

```bash
python train_ml_models.py
```

Le script va automatiquement :

1. **Télécharger les données de marché** (5 ans d'historique pour 10 symboles)
2. **Préparer les datasets** pour chaque modèle
3. **Entraîner les modèles** ML/DL
4. **Sauvegarder les résultats** dans le dossier `./data/`

### Étape 3: Vérifier les Résultats

Après l'entraînement, vérifiez :

```bash
ls -la data/
```

Vous devriez voir :
- Données de marché : `*_market_data.csv`
- Données Pattern : `pattern_*.npy`
- Données Sentiment : `sentiment_texts.json`, `sentiment_labels.npy`
- Données RAG : `rag_documents.json`, `rag_qa_pairs.json`
- Configuration : `training_config.json`

## 📊 Détails des Données d'Entraînement

### Pattern Detector
- **Symboles**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, JNJ, V
- **Période**: 5 ans de données quotidiennes
- **Features**: Prix (Open, High, Low, Close), Volume
- **Séquences**: 30 jours
- **Labels**: UP (+2%), DOWN (-2%), SIDEWAYS
- **Split**: 70% train, 20% validation, 10% test

### Sentiment Analyzer
- **Sources**: News financières simulées
- **Categories**: Positive, Negative, Neutral
- **Augmentation**: Variations textuelles pour augmenter le dataset
- **Modèles**: FinBERT, RoBERTa, VADER
- **Labels**: 0 (Neutral), 1 (Negative), 2 (Positive)

### RAG Integrator
- **Documents**: 5 documents financiers thématiques
- **Q&A**: 5 paires question-réponse
- **Indexation**: FAISS pour recherche sémantique
- **Embeddings**: Sentence Transformers

## 🧪 Tester les Modèles Entraînés

Après l'entraînement, testez l'intégration complète :

```bash
python test_hybrid_orchestrator.py
```

Ce script va :
1. Initialiser l'orchestrateur hybride
2. Tester les composants ML individuellement
3. Valider l'analyse Core System
4. Tester l'analyse ML Enhanced
5. Vérifier les métriques de performance

## 🔧 Configuration Personnalisée

### Modifier les Symboles

Dans `train_ml_models.py`, modifiez la liste des symboles :

```python
self.training_config = {
    'symbols': ['VOTRE_SYMBOLE1', 'VOTRE_SYMBOLE2'],  # Personnalisez ici
    'start_date': (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'),
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'validation_split': 0.2,
    'test_split': 0.1
}
```

### Modifier la Période

```python
# Pour 10 ans de données
'start_date': (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d'),

# Pour 1 an de données
'start_date': (datetime.now() - timedelta(days=1*365)).strftime('%Y-%m-%d'),
```

### Ajuster les Splits

```python
'validation_split': 0.15,  # 15% validation
'test_split': 0.15,       # 15% test
# Train = 70% automatiquement
```

## 📈 Monitoring et Logs

### Logs d'Entraînement

Le script génère un fichier `ml_training.log` avec :
- Progression du téléchargement des données
- Détails de la préparation des datasets
- Statuts d'entraînement des modèles
- Erreurs éventuelles

### Métriques de Performance

Après l'entraînement, consultez `training_config.json` pour :
- Date d'entraînement
- Symboles utilisés
- Période des données
- Statuts des modèles entraînés

## 🚨 Dépannage

### Problèmes Courants

1. **ModuleNotFoundError**
   ```bash
   pip install numpy pandas yfinance tensorflow scikit-learn sentence-transformers faiss-cpu requests
   ```

2. **Données non téléchargées**
   - Vérifiez votre connexion internet
   - Essayez avec moins de symboles

3. **Mémoire insuffisante**
   - Réduisez le nombre de symboles
   - Diminuez la période d'historique

4. **Modèles non sauvegardés**
   - Vérifiez les permissions du dossier `data/`
   - Assurez-vous d'avoir assez d'espace disque

### Vérifier l'Intégrité

```bash
# Vérifier que les fichiers existent
ls -la data/*.npy data/*.json data/*.csv

# Tester l'import des modèles
python -c "from alphabot.ml.pattern_detector import MLPatternDetector; print('✅ Pattern Detector OK')"
python -c "from alphabot.ml.sentiment_analyzer import SentimentDLAnalyzer; print('✅ Sentiment Analyzer OK')"
python -c "from alphabot.ml.rag_integrator import RAGIntegrator; print('✅ RAG Integrator OK')"
```

## 🔄 Mise à Jour des Modèles

Pour réentraîner les modèles avec de nouvelles données :

1. **Supprimer les anciennes données** (optionnel) :
   ```bash
   rm -rf data/*.npy data/*.json
   ```

2. **Relancer l'entraînement** :
   ```bash
   python train_ml_models.py
   ```

3. **Tester la nouvelle version** :
   ```bash
   python test_hybrid_orchestrator.py
   ```

## 📚 Prochaines Étapes

Après l'entraînement réussi :

1. **Backtesting Complet**
   - Utilisez les modèles entraînés dans des backtests historiques
   - Comparez les performances avec et sans ML/DL

2. **Paper Trading**
   - Déployez le système en paper trading
   - Validez les performances en conditions réelles

3. **Optimisation**
   - Ajustez les seuils de confiance
   - Optimisez la pondération des signaux ML

4. **Production**
   - Mettez en place le monitoring continu
   - Configurez le réentraînement périodique

## 📞 Support

En cas de problèmes :
1. Consultez les logs dans `ml_training.log`
2. Vérifiez ce guide de dépannage
3. Testez l'import des composants individuellement
4. Exécutez la suite de tests complète

---

**Note**: Les modèles actuels utilisent des données simulées pour le sentiment et le RAG. Pour un environnement de production, vous devrez intégrer des sources de données réelles (NewsAPI, Twitter, etc.).
