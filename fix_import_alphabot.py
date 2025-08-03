#!/usr/bin/env python3
# Script pour corriger l'import du module alphabot dans la cellule de test

import json
import shutil
from datetime import datetime

def fix_alphabot_import():
    # Créer une sauvegarde
    backup_name = f'ALPHABOT_ML_TRAINING_COLAB_v2_backup_import_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb'
    shutil.copy('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', backup_name)
    print(f"Sauvegarde créée: {backup_name}")
    
    # Lire le fichier
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    
    # Trouver et corriger la dernière cellule de test
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'from alphabot.core.hybrid_orchestrator import HybridOrchestrator' in source:
                print(f"Cellule de test d'import trouvée à l'index {i}")
                
                # Nouvelle cellule avec diagnostic et correction
                new_source = [
                    "import os, sys, asyncio\n",
                    "# === Test V2: ModelSelectorV2 et HybridOrchestrator ===\n",
                    "# Prérequis: repo monté (Drive), chemin racine correct, dépendances installées\n",
                    "\n",
                    "# === DIAGNOSTIC ET CORRECTION PYTHONPATH ===\n",
                    "print('=== Diagnostic du système ===')\n",
                    "print(f'Working directory: {os.getcwd()}')\n",
                    "print(f'Python path: {sys.path[:5]}...')  # Premiers éléments\n",
                    "\n",
                    "# Vérification repo_path depuis la cellule précédente\n",
                    "try:\n",
                    "    print(f'repo_path variable: {repo_path}')\n",
                    "    print(f'repo_path exists: {os.path.exists(repo_path)}')\n",
                    "    if os.path.exists(repo_path):\n",
                    "        print(f'Contenu repo_path: {os.listdir(repo_path)[:10]}')\n",
                    "        alphabot_path = os.path.join(repo_path, 'alphabot')\n",
                    "        print(f'alphabot folder exists: {os.path.exists(alphabot_path)}')\n",
                    "        if os.path.exists(alphabot_path):\n",
                    "            print(f'Contenu alphabot/: {os.listdir(alphabot_path)}')\n",
                    "except NameError:\n",
                    "    print('❌ Variable repo_path non définie - Exécutez d\\'abord la cellule de montage!')\n",
                    "    repo_path = '/content/drive/MyDrive/Alphabot'  # Fallback\n",
                    "\n",
                    "# Force l'ajout du chemin si pas déjà fait\n",
                    "if repo_path not in sys.path:\n",
                    "    sys.path.insert(0, repo_path)\n",
                    "    print(f'✅ Ajouté {repo_path} au sys.path')\n",
                    "else:\n",
                    "    print(f'✅ {repo_path} déjà dans sys.path')\n",
                    "\n",
                    "# Vérification que les modules sont accessibles\n",
                    "print('\\n=== Test d\\'accessibilité des modules ===')\n",
                    "try:\n",
                    "    import alphabot\n",
                    "    print(f'✅ Module alphabot trouvé: {alphabot.__file__ if hasattr(alphabot, \"__file__\") else \"package\"}')\n",
                    "except ImportError as e:\n",
                    "    print(f'❌ Import alphabot échoué: {e}')\n",
                    "    # Diagnostic des chemins possibles\n",
                    "    possible_paths = [\n",
                    "        '/content/drive/MyDrive/Alphabot',\n",
                    "        '/content/drive/MyDrive/AlphaBot_ML_Training',\n",
                    "        '/content/drive/MyDrive/Tradingbot_V2'\n",
                    "    ]\n",
                    "    for path in possible_paths:\n",
                    "        alphabot_subdir = os.path.join(path, 'alphabot')\n",
                    "        if os.path.exists(alphabot_subdir):\n",
                    "            print(f'💡 Candidat trouvé: {alphabot_subdir}')\n",
                    "            if path not in sys.path:\n",
                    "                sys.path.insert(0, path)\n",
                    "                print(f'✅ Ajouté {path} au sys.path')\n",
                    "            break\n",
                    "\n",
                    "# Test d'import spécifique\n",
                    "try:\n",
                    "    from alphabot.core.hybrid_orchestrator import HybridOrchestrator, HybridWorkflowType\n",
                    "    print('✅ Import HybridOrchestrator réussi!')\n",
                    "    \n",
                    "    print('\\n=== Test 1: ModelSelectorV2 thresholds ===')\n",
                    "    \n",
                    "    # Test basique de l'orchestrateur\n",
                    "    try:\n",
                    "        orchestrator = HybridOrchestrator(\n",
                    "            workflow_type=HybridWorkflowType.BACKTESTING,\n",
                    "            config={'enable_model_selection': True}\n",
                    "        )\n",
                    "        print('✅ HybridOrchestrator instancié avec succès')\n",
                    "        \n",
                    "        # Test de la configuration\n",
                    "        print(f'Workflow type: {orchestrator.workflow_type}')\n",
                    "        print(f'Config: {orchestrator.config}')\n",
                    "        \n",
                    "    except Exception as e:\n",
                    "        print(f'❌ Erreur instanciation HybridOrchestrator: {e}')\n",
                    "        import traceback\n",
                    "        traceback.print_exc()\n",
                    "        \n",
                    "except ImportError as e:\n",
                    "    print(f'❌ Import HybridOrchestrator échoué: {e}')\n",
                    "    print('\\n🔧 SOLUTIONS POSSIBLES:')\n",
                    "    print('1. Vérifiez que vous avez exécuté la cellule de montage Drive')\n",
                    "    print('2. Vérifiez que le dossier Alphabot contient bien un sous-dossier alphabot/')\n",
                    "    print('3. Si nécessaire, uploadez le bon dossier sur Google Drive')\n",
                    "    print('4. Ou modifiez le chemin repo_path dans la cellule précédente')\n",
                    "    \n",
                    "    # Diagnostic final\n",
                    "    print('\\n=== Diagnostic complet ===')\n",
                    "    print(f'sys.path complet: {sys.path}')\n",
                    "    print('\\nContenu du répertoire de travail:')\n",
                    "    try:\n",
                    "        for item in os.listdir('.'):\n",
                    "            print(f'  - {item}')\n",
                    "    except Exception as e:\n",
                    "        print(f'Erreur listage: {e}')\n"
                ]
                
                cell['source'] = new_source
                print("✅ Cellule de test mise à jour avec diagnostic complet")
                break
    
    # Sauvegarder
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print("✅ Notebook mis à jour avec diagnostic d'import")
    return True

if __name__ == "__main__":
    try:
        fix_alphabot_import()
        print("\n🔧 Cellule de test corrigée!")
        print("La nouvelle cellule va:")
        print("- Diagnostiquer le problème d'import")
        print("- Vérifier et corriger le PYTHONPATH")
        print("- Trouver automatiquement le bon dossier alphabot")
        print("- Donner des solutions si le problème persiste")
    except Exception as e:
        print(f"❌ Erreur: {e}")
