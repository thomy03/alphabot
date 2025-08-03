#!/usr/bin/env python3
# Script pour ajouter la cellule de montage et corriger base_path dans le notebook Colab

import json
import shutil
from datetime import datetime

def fix_notebook():
    # Créer une sauvegarde
    backup_name = f'ALPHABOT_ML_TRAINING_COLAB_v2_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb'
    shutil.copy('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', backup_name)
    print(f"Sauvegarde créée: {backup_name}")
    
    # Lire le fichier
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    print(f"Cellules originales: {len(cells)}")
    
    # Créer la cellule de montage
    mount_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from google.colab import drive\n",
            "import sys, os\n",
            "\n",
            "# === Montage Drive et ajout du repo au PYTHONPATH ===\n",
            "drive.mount('/content/drive', force_remount=True)\n",
            "repo_path = '/content/drive/MyDrive/Tradingbot_V2'  # Adapter si besoin\n",
            "assert os.path.isdir(repo_path), f'Repo introuvable: {repo_path}. Vérifie le chemin ou déplace le dossier.'\n",
            "if repo_path not in sys.path:\n",
            "    sys.path.insert(0, repo_path)\n",
            "print('sys.path OK, repo_path =', repo_path)\n",
            "print('Contenu repo_path:', os.listdir(repo_path)[:20])\n"
        ]
    }
    
    # Insérer au début
    new_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["## 9) Préparation tests V2 (montage et chemins)\n"]},
        mount_cell
    ]
    new_cells.extend(cells)
    
    # Corriger base_path dans les cellules de test
    for cell in new_cells:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                # Chercher et remplacer base_path
                for i, line in enumerate(source):
                    if "base_path = '/content/drive/MyDrive/Tradingbot_V2'" in line:
                        source[i] = line.replace("base_path = '/content/drive/MyDrive/Tradingbot_V2'", "base_path = repo_path")
                        print(f"Ligne corrigée: {line.strip()} -> base_path = repo_path")
    
    # Mettre à jour le notebook
    nb['cells'] = new_cells
    
    # Sauvegarder
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print(f"Cellules après modification: {len(new_cells)}")
    
    # Vérification
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'r', encoding='utf-8') as f:
        data = f.read()
    
    print("\nVérification:")
    print(f"Montage cell présent: {'Montage Drive et ajout du repo au PYTHONPATH' in data}")
    print(f"Champions test présent: {'Génération champions.json de test' in data}")
    print(f"Test V2 présent: {'Test V2: ModelSelectorV2' in data}")
    print(f"base_path = repo_path présent: {'base_path = repo_path' in data}")
    
    return True

if __name__ == "__main__":
    try:
        fix_notebook()
        print("\n✅ Notebook corrigé avec succès!")
        print("Ouvre/recharge le fichier dans VSCode pour voir les modifications.")
    except Exception as e:
        print(f"❌ Erreur: {e}")
