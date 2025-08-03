#!/usr/bin/env python3
# Script pour corriger le chemin Drive dans la cellule de montage

import json
import shutil
from datetime import datetime

def fix_drive_path():
    # Créer une sauvegarde
    backup_name = f'ALPHABOT_ML_TRAINING_COLAB_v2_backup_path_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb'
    shutil.copy('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', backup_name)
    print(f"Sauvegarde créée: {backup_name}")
    
    # Lire le fichier
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    
    # Trouver et corriger la cellule de montage
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'Montage Drive et ajout du repo au PYTHONPATH' in source:
                print(f"Cellule de montage trouvée à l'index {i}")
                
                # Nouvelle cellule de montage avec diagnostic
                new_source = [
                    "from google.colab import drive\n",
                    "import sys, os\n",
                    "\n",
                    "# === Montage Drive et ajout du repo au PYTHONPATH ===\n",
                    "drive.mount('/content/drive', force_remount=True)\n",
                    "\n",
                    "# Diagnostic du contenu de Drive\n",
                    "print('Contenu de /content/drive/MyDrive:')\n",
                    "try:\n",
                    "    mydrive_content = os.listdir('/content/drive/MyDrive')\n",
                    "    for item in sorted(mydrive_content):\n",
                    "        print(f'  - {item}')\n",
                    "except Exception as e:\n",
                    "    print(f'Erreur lecture MyDrive: {e}')\n",
                    "\n",
                    "# Chemins possibles pour le repo\n",
                    "possible_paths = [\n",
                    "    '/content/drive/MyDrive/Tradingbot_V2',\n",
                    "    '/content/drive/MyDrive/Colab Notebooks/Tradingbot_V2',\n",
                    "    '/content/drive/MyDrive/tradingbot_v2',\n",
                    "    '/content/drive/MyDrive/alphabot',\n",
                    "    '/content/drive/MyDrive/Tradingbot_v2',\n",
                    "    '/content/drive/MyDrive/TradingBot_V2'\n",
                    "]\n",
                    "\n",
                    "repo_path = None\n",
                    "print('\\nRecherche du dossier repo...')\n",
                    "for path in possible_paths:\n",
                    "    if os.path.isdir(path):\n",
                    "        repo_path = path\n",
                    "        print(f'✅ Repo trouvé: {path}')\n",
                    "        break\n",
                    "    else:\n",
                    "        print(f'❌ Pas trouvé: {path}')\n",
                    "\n",
                    "if repo_path is None:\n",
                    "    print('\\n🔍 Recherche de dossiers contenant \"trading\" ou \"alpha\"...')\n",
                    "    try:\n",
                    "        for item in os.listdir('/content/drive/MyDrive'):\n",
                    "            item_path = f'/content/drive/MyDrive/{item}'\n",
                    "            if os.path.isdir(item_path) and ('trading' in item.lower() or 'alpha' in item.lower()):\n",
                    "                print(f'  Candidat trouvé: {item_path}')\n",
                    "                if os.path.exists(f'{item_path}/alphabot') or os.path.exists(f'{item_path}/requirements.txt'):\n",
                    "                    repo_path = item_path\n",
                    "                    print(f'✅ Repo détecté: {repo_path}')\n",
                    "                    break\n",
                    "    except Exception as e:\n",
                    "        print(f'Erreur recherche: {e}')\n",
                    "\n",
                    "# Si toujours pas trouvé, demander à l'utilisateur\n",
                    "if repo_path is None:\n",
                    "    print('\\n⚠️  REPO NON TROUVÉ AUTOMATIQUEMENT')\n",
                    "    print('Please upload your Tradingbot_V2 folder to Google Drive:')\n",
                    "    print('1. Go to drive.google.com')\n",
                    "    print('2. Upload the Tradingbot_V2 folder to MyDrive')\n",
                    "    print('3. Or specify the correct path below:')\n",
                    "    print()\n",
                    "    repo_path = input('Enter the correct path (or press Enter to use default): ').strip()\n",
                    "    if not repo_path:\n",
                    "        repo_path = '/content/drive/MyDrive/Tradingbot_V2'  # Default\n",
                    "\n",
                    "# Vérification finale et ajout au PYTHONPATH\n",
                    "print(f'\\nUtilisation du chemin: {repo_path}')\n",
                    "if os.path.isdir(repo_path):\n",
                    "    if repo_path not in sys.path:\n",
                    "        sys.path.insert(0, repo_path)\n",
                    "    print('✅ sys.path OK, repo_path =', repo_path)\n",
                    "    print('Contenu repo_path:', os.listdir(repo_path)[:20])\n",
                    "else:\n",
                    "    print(f'❌ ERREUR: Le chemin {repo_path} n\\'existe toujours pas!')\n",
                    "    print('Vérifiez que le dossier Tradingbot_V2 est bien uploadé sur Google Drive.')\n"
                ]
                
                cell['source'] = new_source
                print("✅ Cellule de montage mise à jour avec diagnostic automatique")
                break
    
    # Sauvegarder
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print("✅ Notebook mis à jour avec diagnostic de chemin Drive")
    return True

if __name__ == "__main__":
    try:
        fix_drive_path()
        print("\n🔧 Cellule de montage améliorée!")
        print("La nouvelle cellule va:")
        print("- Lister le contenu de Google Drive")
        print("- Chercher automatiquement le dossier Tradingbot_V2")
        print("- Proposer des chemins alternatifs")
        print("- Permettre la saisie manuelle si nécessaire")
    except Exception as e:
        print(f"❌ Erreur: {e}")
