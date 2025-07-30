# 🚀 Guide Complet - Test du Workflow ML/DL sur Google Colab (Alphabot)

## 📋 Instructions Pas à Pas

### **Étape 1: Accès à Google Colab**
1. Allez sur https://colab.research.google.com/
2. Connectez-vous avec votre compte Google
3. Créez un nouveau notebook ou ouvrez `ALPHABOT_ML_TRAINING_COLAB.ipynb`

### **Étape 2: Configuration de Base**
Copiez et exécutez ce code dans la première cellule:

```python
# Cellule 1: Configuration de l'environnement
!pip install -r requirements_colab.txt
!pip install tensorflow-gpu torch transformers scikit-learn pandas numpy matplotlib seaborn plotly

# Vérification GPU/TPU
import tensorflow as tf
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
print("TPU disponible:", tf.config.list_physical_devices('TPU'))
```

### **Étape 3: Montage Google Drive**
```python
# Cellule 2: Montage du Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Créer les dossiers nécessaires
!mkdir -p /content/drive/MyDrive/Alphabot/models
!mkdir -p /content/drive/MyDrive/Alphabot/data
!mkdir -p /content/drive/MyDrive/Alphabot/logs
```

### **Étape 4: Clone du Repository**
```python
# Cellule 3: Clone du repository Alphabot
!git clone https://github.com/thomy03/alphabot.git
%cd alphabot
!ls -la
```

### **Étape 5: Test des Modèles Individuels**

#### **5.1 Pattern Detector**
```python
# Cellule 4: Test Pattern Detector
from alphabot.ml.pattern_detector import PatternDetector
import pandas as pd
import numpy as np

# Données de test simulées
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100) * 2)
volumes = np.random.randint(1000, 10000, 100)

test_data = pd.DataFrame({
    'date': dates,
    'close': prices,
    'volume': volumes
})

# Initialiser et tester
detector = PatternDetector()
patterns = detector.detect_patterns(test_data)
print("Patterns détectés:", len(patterns))
print("Premiers patterns:", patterns[:3] if patterns else "Aucun pattern trouvé")
```

#### **5.2 Sentiment Analyzer**
```python
# Cellule 5: Test Sentiment Analyzer
from alphabot.ml.sentiment_analyzer import SentimentAnalyzer

# Textes de test financiers
test_texts = [
    "Le marché monte fortement aujourd'hui, excellente performance",
    "Catastrophe financière, tout s'effondre rapidement",
    "Le marché reste stable avec peu de volatilité",
    "Bullish sur les tech stocks, croissance continue",
    "Bear market imminent, vendez maintenant"
]

analyzer = SentimentAnalyzer()
results = []
for text in test_texts:
    sentiment = analyzer.analyze(text)
    results.append({
        'text': text,
        'sentiment': sentiment['label'],
        'confidence': sentiment['confidence']
    })

import pandas as pd
df_results = pd.DataFrame(results)
print(df_results)
```

#### **5.3 RAG Integrator**
```python
# Cellule 6: Test RAG Integrator
from alphabot.ml.rag_integrator import RAGIntegrator

# Contexte et questions de test
context = """
Le marché boursier a augmenté de 5% cette semaine. 
Les actions technologiques ont particulièrement bien performé avec +8%.
Les secteurs défensifs ont stagné.
Le volume de trading a augmenté de 20%.
"""

questions = [
    "Quelle est la performance du marché ?",
    "Quels secteurs ont le mieux performé ?",
    "Comment évolue le volume de trading ?"
]

integrator = RAGIntegrator()
for question in questions:
    response = integrator.query(context, question)
    print(f"Q: {question}")
    print(f"R: {response}")
    print("-" * 50)
```

### **Étape 6: Entraînement Complet**
```python
# Cellule 7: Lancement de l'entraînement
import subprocess
import sys

# Exécuter l'entraînement avec monitoring
print("🚀 Lancement de l'entraînement des modèles...")
result = subprocess.run([sys.executable, 'train_ml_models.py', '--mode=colab', '--gpu=True'], 
                       capture_output=True, text=True)

print("=== Sortie Standard ===")
print(result.stdout)
print("=== Erreurs ===")
print(result.stderr)

# Vérifier les fichiers créés
!ls -la models/
!ls -la training_logs/
```

### **Étape 7: Monitoring en Temps Réel**
```python
# Cellule 8: Dashboard de monitoring
import matplotlib.pyplot as plt
import json
import os

def plot_training_metrics():
    """Affiche les métriques d'entraînement"""
    try:
        # Charger les logs d'entraînement
        with open('training_logs/latest.json', 'r') as f:
            logs = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0,0].plot(logs['loss'])
        axes[0,0].set_title('Loss')
        axes[0,0].set_xlabel('Epoch')
        
        # Accuracy
        axes[0,1].plot(logs['accuracy'])
        axes[0,1].set_title('Accuracy')
        axes[0,1].set_xlabel('Epoch')
        
        # Validation Loss
        axes[1,0].plot(logs['val_loss'])
        axes[1,0].set_title('Validation Loss')
        axes[1,0].set_xlabel('Epoch')
        
        # Validation Accuracy
        axes[1,1].plot(logs['val_accuracy'])
        axes[1,1].set_title('Validation Accuracy')
        axes[1,1].set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erreur lors du chargement des métriques: {e}")

# Exécuter le monitoring
plot_training_metrics()
```

### **Étape 8: Tests d'Intégration**
```python
# Cellule 9: Tests d'intégration complète
print("🧪 Lancement des tests d'intégration...")
result = subprocess.run([sys.executable, 'test_hybrid_orchestrator.py', '--test-mode=colab'], 
                       capture_output=True, text=True)

print("=== Résultats des Tests ===")
print(result.stdout)

# Analyse des résultats
try:
    with open('test_results.json', 'r') as f:
        results = json.load(f)
    
    print("\n📊 Résumé des Tests:")
    print(f"✅ Pattern Detector: {results.get('pattern_detector', {}).get('status', 'N/A')}")
    print(f"✅ Sentiment Analyzer: {results.get('sentiment_analyzer', {}).get('status', 'N/A')}")
    print(f"✅ RAG Integrator: {results.get('rag_integrator', {}).get('status', 'N/A')}")
    print(f"✅ Intégration Globale: {results.get('integration', {}).get('status', 'N/A')}")
    
except Exception as e:
    print(f"⚠️ Impossible de charger les résultats: {e}")
```

### **Étape 9: Sauvegarde vers Google Drive**
```python
# Cellule 10: Sauvegarde automatique
import shutil
import datetime

# Configuration des chemins
models_dir = "/content/drive/MyDrive/Alphabot/models"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"{models_dir}/{timestamp}"

# Créer le dossier de sauvegarde
!mkdir -p {save_path}

# Copier les modèles et logs
!cp -r models/* {save_path}/
!cp -r training_logs/* {save_path}/
!cp test_results.json {save_path}/

print(f"✅ Sauvegarde complète dans: {save_path}")
print("📁 Contenu sauvegardé:")
!ls -la {save_path}
```

### **Étape 10: Push vers GitHub**
```python
# Cellule 11: Configuration Git et push
# (Optionnel - nécessite vos credentials GitHub)
import getpass

email = getpass.getpass("Entrez votre email GitHub: ")
name = getpass.getpass("Entrez votre nom GitHub: ")

!git config --global user.email "{email}"
!git config --global user.name "{name}"

# Ajouter et committer
!git add models/
!git add training_logs/
!git add test_results.json
!git commit -m "Ajout des modèles entraînés - $(date)"

print("✅ Prêt pour le push (utilisez !git push si configuré)")
```

## 🎯 **Commandes Rapides pour Copier-Coller**

### **Lancement Complet en Une Cellule**
```python
# Cellule Bonus: Workflow Complet
%%bash
echo "🚀 Démarrage du workflow ML/DL complet..."
python train_ml_models.py --mode=colab --gpu=True --epochs=5
python test_hybrid_orchestrator.py --test-mode=colab
echo "✅ Workflow terminé!"
```

## 📊 **Monitoring en Temps Réel**

### **Dashboard Simple**
```python
# Cellule Monitoring: Affichage en temps réel
import time
from IPython.display import clear_output, HTML

def monitor_training():
    while True:
        clear_output(wait=True)
        try:
            # Afficher les dernières métriques
            !tail -20 training_logs/latest.log
            print("\n" + "="*50)
            print("🔄 Monitoring actif... (Ctrl+C pour arrêter)")
        except:
            print("📊 En attente de données...")
        time.sleep(30)

# monitor_training()  # Décommentez pour activer
```

## ⚡ **Prochaines Étapes**

1. **Exécutez les cellules 1-3** pour configurer l'environnement
2. **Testez les modèles individuellement** avec les cellules 4-6
3. **Lancez l'entraînement complet** avec la cellule 7
4. **Vérifiez les résultats** avec les cellules 8-9
5. **Sauvegardez vers Drive** avec la cellule 10

**Commencez maintenant** en copiant ces cellules dans votre notebook Colab et exécutez-les dans l'ordre !
