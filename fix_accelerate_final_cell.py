#!/usr/bin/env python3
"""
Script pour ajouter le patch accelerate en tête de la dernière cellule du notebook Colab
"""

import json
import os

def fix_accelerate_final_cell():
    notebook_path = 'ALPHABOT_ML_TRAINING_COLAB_v2.ipynb'
    
    # Lire le notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    
    # Trouver la dernière cellule de code
    last_code_idx = None
    for i in range(len(cells)-1, -1, -1):
        if cells[i].get('cell_type') == 'code':
            last_code_idx = i
            break
    
    if last_code_idx is None:
        print("❌ Aucune cellule de code trouvée")
        return False
    
    src = cells[last_code_idx].get('source', [])
    current_content = ''.join(src)
    
    # Vérifier si le patch accelerate spécifique est déjà présent
    if "accelerate déjà présent" in current_content:
        print("ℹ️ Patch accelerate déjà présent, aucune modification")
        return True
    
    # Patch accelerate à insérer en tête
    accelerate_patch = [
        "# 0) Dépendances minimales pour éviter l'erreur No module named 'accelerate'\n",
        "try:\n",
        "    import accelerate\n",
        "    print('✅ accelerate déjà présent')\n",
        "except Exception:\n",
        "    print('📦 Installation de accelerate (minimal)...')\n",
        "    !pip install -q \"accelerate>=0.21.0,<1.0\"\n",
        "    import importlib, sys\n",
        "    importlib.invalidate_caches()\n",
        "    sys.modules.pop('accelerate', None)\n",
        "    import accelerate\n",
        "    print('✅ accelerate installé')\n",
        "\n",
        "# Désactive l'utilisation d'Accelerate par Transformers (sécurité)\n",
        "import os\n",
        "os.environ['TRANSFORMERS_NO_ACCELERATE'] = '1'\n",
        "\n",
        "# Purge modules déjà chargés pouvant provoquer des effets de bord\n",
        "import sys, importlib\n",
        "for m in list(sys.modules):\n",
        "    if m.startswith(('accelerate.logging', 'peft')):\n",
        "        sys.modules.pop(m, None)\n",
        "importlib.invalidate_caches()\n",
        "\n"
    ]
    
    # Insérer le patch en tête du contenu existant
    cells[last_code_idx]['source'] = accelerate_patch + src
    
    # Sauvegarder
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Patch accelerate ajouté en tête de la cellule {last_code_idx}")
    print("📝 Notebook sauvegardé avec le correctif accelerate")
    print("\n💡 Instructions pour Colab:")
    print("1. Runtime > Restart runtime")
    print("2. Exécuter la cellule 2 (montage Drive)")
    print("3. Exécuter directement la dernière cellule (avec patch accelerate)")
    print("\nCe patch résout l'erreur 'No module named accelerate' de transformers.trainer")
    
    return True

if __name__ == "__main__":
    fix_accelerate_final_cell()
