#!/usr/bin/env python3
"""
Script pour nettoyer les imports redondants qui causent des conflits dans Colab
"""

import json
import os

def fix_imports_conflicts():
    notebook_path = 'ALPHABOT_ML_TRAINING_COLAB_v2.ipynb'
    
    # Lire le notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    
    # Cellule 1: Setup - garder les imports de base
    # Cellule finale: éviter les réimports, utiliser try/except
    
    # Trouver la dernière cellule de code
    last_code_idx = None
    for i in range(len(cells)-1, -1, -1):
        if cells[i].get('cell_type') == 'code':
            last_code_idx = i
            break
    
    if last_code_idx is None:
        print("❌ Aucune cellule de code trouvée")
        return False
    
    # Nouvelle cellule finale qui évite les conflits d'imports
    new_cell_source = [
        "# =================================================================\n",
        "# CELLULE FINALE: Test d'intégration V2 (évite les conflits imports)\n",
        "# =================================================================\n",
        "\n",
        "# 1) Installation des dépendances (si pas déjà fait)\n",
        "try:\n",
        "    import redis.asyncio\n",
        "    import pydantic_settings\n",
        "    print('✅ Dépendances déjà installées')\n",
        "except ImportError:\n",
        "    print('📦 Installation des dépendances manquantes...')\n",
        "    !pip install -q \"redis>=4.5\" \"pydantic-settings>=2.0\" \"pydantic>=2.0\" \"pyyaml\"\n",
        "    print('✅ Installation terminée')\n",
        "\n",
        "# 2) Clonage du repo GitHub (si pas déjà fait)\n",
        "import os, sys\n",
        "if not os.path.exists('/content/alphabot_repo'):\n",
        "    !git clone https://github.com/thomy03/alphabot.git /content/alphabot_repo\n",
        "    print('✅ Repo cloné')\n",
        "else:\n",
        "    print('✅ Repo déjà présent')\n",
        "\n",
        "# 3) Configuration des chemins\n",
        "code_path = \"/content/alphabot_repo\"\n",
        "if code_path not in sys.path:\n",
        "    sys.path.insert(0, code_path)\n",
        "    print(f\"✅ Ajouté {code_path} au PYTHONPATH\")\n",
        "\n",
        "base_path = \"/content/drive/MyDrive/Alphabot\"\n",
        "assert os.path.isdir(base_path), f\"Base path invalide: {base_path}\"\n",
        "assert os.path.exists(os.path.join(base_path, \"champions.json\")), \"champions.json introuvable\"\n",
        "print(f\"✅ Base path validé: {base_path}\")\n",
        "\n",
        "# 4) Import des modules alphabot (éviter les conflits)\n",
        "try:\n",
        "    # Import avec gestion des modules déjà chargés\n",
        "    import importlib\n",
        "    \n",
        "    # Si le module est déjà chargé, le recharger proprement\n",
        "    if 'alphabot.core.hybrid_orchestrator' in sys.modules:\n",
        "        import alphabot.core.hybrid_orchestrator\n",
        "        importlib.reload(alphabot.core.hybrid_orchestrator)\n",
        "    \n",
        "    from alphabot.core.hybrid_orchestrator import HybridOrchestrator, HybridWorkflowType\n",
        "    print(\"✅ Import alphabot OK\")\n",
        "    \n",
        "except Exception as e:\n",
        "    print(f\"❌ Import échoué: {e}\")\n",
        "    print(\"💡 Solution: Runtime > Restart runtime, puis exécuter seulement cette cellule\")\n",
        "    raise\n",
        "\n",
        "# 5) Chargement de la configuration\n",
        "import json\n",
        "with open(os.path.join(base_path, \"champions.json\"), \"r\") as f:\n",
        "    champs = json.load(f)\n",
        "print(\"Champions chargés:\", list(champs.keys()))\n",
        "\n",
        "# 6) Test de l'orchestrateur\n",
        "try:\n",
        "    orchestrator = HybridOrchestrator(\n",
        "        workflow_type=HybridWorkflowType.BACKTESTING,\n",
        "        config={\"enable_model_selection\": True, \"base_path\": base_path}\n",
        "    )\n",
        "    print(\"✅ HybridOrchestrator instancié avec succès\")\n",
        "    print(f\"   Workflow type: {orchestrator.workflow_type}\")\n",
        "    print(f\"   Base path: {base_path}\")\n",
        "    \n",
        "except Exception as e:\n",
        "    print(f\"❌ Erreur orchestrateur: {e}\")\n",
        "    raise\n",
        "\n",
        "print(\"\\n🎯 Test d'intégration V2 réussi!\")\n",
        "print(\"\\n📋 Prochaines étapes:\")\n",
        "print(\"   - L'orchestrateur est prêt pour les tests\")\n",
        "print(\"   - Vous pouvez maintenant exécuter des backtests\")\n",
        "print(\"   - Les modèles champions sont configurés\")\n"
    ]
    
    # Remplacer la cellule finale
    cells[last_code_idx]['source'] = new_cell_source
    
    # Sauvegarder
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Cellule {last_code_idx} remplacée avec gestion des conflits d'imports")
    print("📝 Notebook sauvegardé")
    
    print("\n💡 Instructions pour Colab:")
    print("1. Runtime > Restart runtime")
    print("2. Exécuter la cellule 2 (montage Drive)")
    print("3. Exécuter directement la cellule finale (évite les conflits)")
    
    return True

if __name__ == "__main__":
    fix_imports_conflicts()
