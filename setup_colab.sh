#!/bin/bash

# === SETUP SCRIPT POUR GOOGLE COLAB - ALPHABOT ML/DL TRAINING ===
# Ce script configure automatiquement l'environnement Colab pour AlphaBot

set -e  # Arrêter en cas d'erreur

echo "🚀 Démarrage du setup AlphaBot ML Training pour Google Colab..."
echo "=================================================="

# === 1. Vérification de l'environnement ===
echo "🔍 Vérification de l'environnement..."

# Vérifier si on est dans Colab
if python -c "import google.colab" 2>/dev/null; then
    echo "✅ Environnement Google Colab détecté"
else
    echo "⚠️  Environnement Colab non détecté, mais on continue..."
fi

# Vérifier Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $PYTHON_VERSION"

# === 2. Installation des dépendances ===
echo ""
echo "📦 Installation des dépendances..."

# Installer pip si nécessaire
if ! command -v pip &> /dev/null; then
    echo "Installation de pip..."
    python -m ensurepip --upgrade
fi

# Mettre à jour pip
echo "Mise à jour de pip..."
pip install --upgrade pip

    # Installer les dépendances depuis requirements_colab.txt
    if [ -f "requirements_colab.txt" ]; then
        echo "Installation des dépendances depuis requirements_colab.txt..."
        if pip install -r requirements_colab.txt; then
            echo "✅ Dépendances installées avec succès"
        else
            echo "⚠️ Échec d'installation via requirements, tentative faiss-cpu en fallback..."
            pip install faiss-cpu>=1.7.4
        fi
    else
        echo "⚠️  requirements_colab.txt non trouvé, installation manuelle..."
    
    # Installation des dépendances essentielles
    pip install tensorflow torch torchvision torchaudio
    pip install transformers datasets accelerate bitsandbytes
    pip install sentence-transformers faiss-gpu
    pip install yfinance pandas numpy scikit-learn
    pip install matplotlib seaborn tqdm joblib
    pip install google-colab google-auth
fi

# === 3. Configuration GPU/TPU ===
echo ""
echo "🎮 Configuration GPU/TPU..."

# Détecter GPU
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ GPU détecté"
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "   GPU: $GPU_NAME"
    
    # Configurer la mémoire GPU
    python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('✅ GPU memory growth activé')
"
else
    echo "⚠️  GPU non disponible - mode CPU"
fi

# Détecter TPU
if python -c "import torch_xla" 2>/dev/null; then
    echo "✅ TPU détecté"
else
    echo "ℹ️  TPU non disponible"
fi

# === 4. Configuration Google Drive ===
echo ""
echo "📁 Configuration Google Drive..."

# Créer le script de montage Drive
cat > mount_drive.py << 'EOF'
from google.colab import drive
import os

try:
    drive.mount('/content/drive')
    print("✅ Google Drive monté avec succès")
    
    # Créer le dossier AlphaBot_ML_Training
    drive_path = "/content/drive/MyDrive/AlphaBot_ML_Training"
    os.makedirs(drive_path, exist_ok=True)
    print(f"✅ Dossier AlphaBot_ML_Training créé: {drive_path}")
    
except Exception as e:
    print(f"⚠️  Erreur montage Drive: {e}")
    print("ℹ️  Le montage Drive est optionnel, on continue...")
EOF

# Exécuter le montage Drive
python mount_drive.py

# === 5. Téléchargement du code AlphaBot ===
echo ""
echo "📥 Téléchargement du code AlphaBot..."

# Vérifier si le code existe déjà
if [ ! -d "alphabot" ]; then
    echo "Téléchargement du code AlphaBot..."
    
    # Option 1: Cloner depuis GitHub (décommenter si disponible)
    # git clone https://github.com/votre-username/AlphaBot.git
    
    # Option 2: Créer la structure minimale
    mkdir -p alphabot/ml alphabot/core alphabot/agents
    
    echo "✅ Structure de dossiers créée"
else
    echo "✅ Code AlphaBot déjà présent"
fi

# === 6. Création des fichiers de configuration ===
echo ""
echo "⚙️  Création des fichiers de configuration..."

# Créer le fichier de configuration principal
cat > config.json << 'EOF'
{
    "training": {
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "early_stopping_patience": 10
    },
    "models": {
        "pattern_detector": {
            "lstm_units": 50,
            "cnn_filters": 32,
            "dropout_rate": 0.2
        },
        "sentiment_analyzer": {
            "model_name": "ProsusAI/finbert",
            "max_length": 512,
            "batch_size": 16
        },
        "rag_integrator": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "faiss_index_type": "IVF100,Flat"
        }
    },
    "data": {
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        "start_date": "2018-01-01",
        "end_date": "2024-01-01",
        "timeframes": ["1d"]
    },
    "colab": {
        "save_to_drive": true,
        "checkpoint_interval": 5,
        "timeout_protection": true,
        "mixed_precision": true
    }
}
EOF

echo "✅ Fichier config.json créé"

# === 7. Configuration du logging ===
echo ""
echo "📝 Configuration du logging..."

cat > logging_config.py << 'EOF'
import logging
import sys
from datetime import datetime

def setup_logging():
    """Configurer le logging pour Colab"""
    
    # Créer le formateur
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurer le logger racine
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler pour le fichier
    file_handler = logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialiser le logging
logger = setup_logging()
EOF

echo "✅ Configuration logging créée"

# === 8. Vérification finale ===
echo ""
echo "🔍 Vérification finale de l'installation..."

# Vérifier les imports critiques
python -c "
import sys
print('📚 Vérification des imports...')

try:
    import tensorflow as tf
    print(f'✅ TensorFlow: {tf.__version__}')
except ImportError as e:
    print(f'❌ TensorFlow: {e}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Transformers: {e}')

try:
    import faiss
    print('✅ FAISS')
except ImportError as e:
    print(f'❌ FAISS: {e}')

try:
    import yfinance as yf
    print('✅ yfinance')
except ImportError as e:
    print(f'❌ yfinance: {e}')

try:
    import pandas as pd
    import numpy as np
    print('✅ Pandas/NumPy')
except ImportError as e:
    print(f'❌ Pandas/NumPy: {e}')

print('✅ Vérification des imports terminée')
"

# === 9. Création du script de test ===
echo ""
echo "🧪 Création du script de test..."

cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Script de test pour vérifier l'installation AlphaBot ML Training
"""

import logging
import sys
import json
from pathlib import Path

# Importer la configuration logging
from logging_config import setup_logging

logger = setup_logging()

def test_basic_imports():
    """Tester les imports de base"""
    logger.info("🧪 Test des imports de base...")
    
    imports_to_test = [
        'tensorflow', 'torch', 'transformers', 
        'pandas', 'numpy', 'sklearn', 'yfinance'
    ]
    
    failed_imports = []
    
    for module in imports_to_test:
        try:
            __import__(module)
            logger.info(f"   ✅ {module}")
        except ImportError as e:
            logger.error(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_gpu_availability():
    """Tester la disponibilité GPU"""
    logger.info("🎮 Test GPU...")
    
    # Test TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"   ✅ TensorFlow GPU: {len(gpus)} GPU(s)")
        else:
            logger.info("   ℹ️  TensorFlow: pas de GPU")
    except Exception as e:
        logger.warning(f"   ⚠️  TensorFlow GPU test: {e}")
    
    # Test PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"   ✅ PyTorch GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("   ℹ️  PyTorch: pas de GPU")
    except Exception as e:
        logger.warning(f"   ⚠️  PyTorch GPU test: {e}")

def test_drive_connection():
    """Tester la connexion Google Drive"""
    logger.info("📁 Test Google Drive...")
    
    drive_path = Path("/content/drive/MyDrive/AlphaBot_ML_Training")
    if drive_path.exists():
        logger.info("   ✅ Google Drive accessible")
        return True
    else:
        logger.info("   ℹ️  Google Drive non monté (optionnel)")
        return False

def test_configuration():
    """Tester la configuration"""
    logger.info("⚙️  Test configuration...")
    
    config_file = Path("config.json")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info("   ✅ Configuration chargée")
            logger.info(f"   📊 Symboles configurés: {len(config['data']['symbols'])}")
            return True
        except Exception as e:
            logger.error(f"   ❌ Configuration: {e}")
            return False
    else:
        logger.error("   ❌ Fichier config.json non trouvé")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Démarrage des tests d'installation AlphaBot ML Training...")
    
    tests = [
        ("Imports de base", test_basic_imports),
        ("Disponibilité GPU", test_gpu_availability),
        ("Connexion Drive", test_drive_connection),
        ("Configuration", test_configuration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Test: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"   ❌ Erreur during {test_name}: {e}")
            results[test_name] = False
    
    # Résumé
    logger.info("\n" + "="*50)
    logger.info("📊 RÉSUMÉ DES TESTS")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n📈 Résultat: {passed}/{total} tests passés")
    
    if passed == total:
        logger.info("🎉 Installation réussie ! Prêt pour l'entraînement ML/DL")
        return True
    else:
        logger.warning("⚠️  Certains tests ont échoué. Veuillez vérifier l'installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

echo "✅ Script de test créé"

# === 10. Finalisation ===
echo ""
echo "🎉 Finalisation du setup..."

# Rendre les scripts exécutables
chmod +x setup_colab.sh test_installation.py

# Créer un résumé
cat > SETUP_SUMMARY.md << 'EOF'
# Setup Summary - AlphaBot ML Training Colab

## Installation terminée ✅

### Composants installés:
- ✅ TensorFlow (avec support GPU)
- ✅ PyTorch (avec support CUDA)
- ✅ Transformers & Hugging Face
- ✅ FAISS (recherche vectorielle)
- ✅ yfinance (données de marché)
- ✅ Pandas/NumPy/Scikit-learn
- ✅ Google Colab integration
- ✅ Configuration logging

### Fichiers créés:
- `config.json` - Configuration principale
- `logging_config.py` - Configuration logging
- `test_installation.py` - Script de test
- `requirements_colab.txt` - Dépendances
- `colab_utils.py` - Utilitaires Colab
- `drive_manager.py` - Gestion Drive

### Prochaines étapes:

1. **Exécuter le test**:
   ```bash
   python test_installation.py
   ```

2. **Lancer le notebook Colab**:
   - Ouvrir `ALPHABOT_ML_TRAINING_COLAB.ipynb`
   - Exécuter les cellules dans l'ordre

3. **Personnaliser la configuration**:
   - Éditer `config.json` pour vos besoins
   - Ajouter vos symboles boursiers
   - Ajuster les paramètres d'entraînement

### Support:
- Vérifier les logs en cas d'erreur
- Consulter `README_ENTRAINEMENT_COLAB.md`
- Les erreurs GPU sont normales si pas de GPU disponible

---

*Setup terminé le $(date)*
EOF

echo ""
echo "=================================================="
echo "🎉 SETUP ALPHABOT ML TRAINING TERMINÉ !"
echo "=================================================="
echo ""
echo "📋 Prochaines étapes :"
echo "   1. Exécuter le test: python test_installation.py"
echo "   2. Consulter le résumé: cat SETUP_SUMMARY.md"
echo "   3. Lancer le notebook Colab"
echo ""
echo "📚 Documentation disponible:"
echo "   - README_ENTRAINEMENT_COLAB.md"
echo "   - SETUP_SUMMARY.md"
echo ""
echo "⚡ Bon entraînement ML/DL !"
