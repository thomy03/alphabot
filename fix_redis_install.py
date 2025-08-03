#!/usr/bin/env python3
"""
Script pour ajouter l'installation de redis au début de la cellule finale
"""

import json
import os

def fix_redis_install():
    notebook_path = 'ALPHABOT_ML_TRAINING_COLAB_v2.ipynb'
    
    # Lire le notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Trouver la dernière cellule de code
    cells = nb.get('cells', [])
    last_code_idx = None
    
    for i in range(len(cells)-1, -1, -1):
        if cells[i].get('cell_type') == 'code':
            last_code_idx = i
            break
    
    if last_code_idx is None:
        print("❌ Aucune cellule de code trouvée")
        return False
    
    # Vérifier le contenu actuel de la cellule
    current_source = cells[last_code_idx].get('source', [])
    print(f"Contenu actuel cellule {last_code_idx} (premières lignes):")
    for i, line in enumerate(current_source[:5]):
        print(f"  {i}: {line.strip()}")
    
    # Nouvelle cellule complète avec installation de redis en premier
    new_cell_source = [
        "# Installation des dépendances requises (redis pour signal_hub + pydantic_settings)\n",
        "!pip install -q \"redis>=4.5\" \"pydantic-settings>=2.0\" \"pydantic>=2.0\" \"pyyaml\" \"numpy>=1.24,<1.27\" \"pandas\" \"matplotlib\" \"scikit-learn\" \"torch\" \"transformers>=4.41,<4.47\" \"huggingface-hub>=0.28\" \"safetensors>=0.4.3\"\n",
        "\n",
        "# Clonage du repo GitHub\n",
        "!git clone https://github.com/thomy03/alphabot.git /content/alphabot_repo || echo 'Repo déjà cloné'\n",
        "\n",
        "import os, sys, json\n",
        "\n",
        "# 1) Ajouter le code au PYTHONPATH (repo GitHub cloné)\n",
        "code_path = \"/content/alphabot_repo\"\n",
        "if code_path not in sys.path:\n",
        "    sys.path.insert(0, code_path)\n",
        "    print(f\"✅ Ajouté {code_path} au PYTHONPATH\")\n",
        "\n",
        "# 2) Définir base_path vers tes artefacts (modèles) sur Drive\n",
        "base_path = \"/content/drive/MyDrive/Alphabot\"  # contient models/ et champions.json\n",
        "assert os.path.isdir(base_path), f\"Base path invalide: {base_path}\"\n",
        "assert os.path.exists(os.path.join(base_path, \"champions.json\")), \"champions.json introuvable\"\n",
        "print(f\"✅ Base path validé: {base_path}\")\n",
        "\n",
        "# 3) Vérifier l'import et afficher les champions (APRÈS installation redis)\n",
        "try:\n",
        "    from alphabot.core.hybrid_orchestrator import HybridOrchestrator, HybridWorkflowType\n",
        "    print(\"✅ Import alphabot OK\")\n",
        "except ImportError as e:\n",
        "    print(f\"❌ Import échoué: {e}\")\n",
        "    raise\n",
        "\n",
        "with open(os.path.join(base_path, \"champions.json\"), \"r\") as f:\n",
        "    champs = json.load(f)\n",
        "print(\"Champions chargés:\", list(champs.keys()))\n",
        "\n",
        "# 4) Instancier l'orchestrateur avec base_path\n",
        "try:\n",
        "    orchestrator = HybridOrchestrator(\n",
        "        workflow_type=HybridWorkflowType.BACKTESTING,\n",
        "        config={\"enable_model_selection\": True, \"base_path\": base_path}\n",
        "    )\n",
        "    print(\"✅ HybridOrchestrator prêt\")\n",
        "    print(f\"Workflow type: {orchestrator.workflow_type}\")\n",
        "    print(f\"Config: {orchestrator.config}\")\n",
        "except Exception as e:\n",
        "    print(f\"❌ Erreur orchestrateur: {e}\")\n",
        "    raise\n",
        "\n",
        "print(\"\\n🎯 Test d'intégration V2 réussi!\")\n"
    ]
    
    # Remplacer complètement la cellule
    cells[last_code_idx]['source'] = new_cell_source
    
    # Sauvegarder
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Cellule {last_code_idx} remplacée avec installation redis en tête")
    print("📝 Nouvelle cellule sauvegardée")
    return True

if __name__ == "__main__":
    fix_redis_install()
