#!/usr/bin/env python3
"""
Script pour corriger la dernière cellule du notebook Colab v2
Remplace le système de diagnostic Drive par le clonage GitHub
"""

import json
import os

def fix_final_cell():
    notebook_path = 'ALPHABOT_ML_TRAINING_COLAB_v2.ipynb'
    
    # Lire le notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Code de la nouvelle cellule (clonage GitHub)
    new_cell_code = [
        "!git clone https://github.com/thomy03/alphabot.git /content/alphabot_repo\n",
        "import os, sys, json\n",
        "\n",
        "# 1) Ajouter le code au PYTHONPATH (repo GitHub cloné)\n",
        "code_path = \"/content/alphabot_repo\"\n",
        "if code_path not in sys.path:\n",
        "    sys.path.insert(0, code_path)\n",
        "\n",
        "# 2) Définir base_path vers tes artefacts (modèles) sur Drive\n",
        "base_path = \"/content/drive/MyDrive/Alphabot\"  # contient models/ et champions.json\n",
        "assert os.path.isdir(base_path), f\"Base path invalide: {base_path}\"\n",
        "assert os.path.exists(os.path.join(base_path, \"champions.json\")), \"champions.json introuvable\"\n",
        "\n",
        "# 3) Vérifier l'import et afficher les champions\n",
        "from alphabot.core.hybrid_orchestrator import HybridOrchestrator, HybridWorkflowType\n",
        "print(\"✅ Import alphabot OK\")\n",
        "with open(os.path.join(base_path, \"champions.json\"), \"r\") as f:\n",
        "    champs = json.load(f)\n",
        "print(\"Champions:\", json.dumps(champs, indent=2)[:400], \"...\")\n",
        "\n",
        "# 4) Instancier l'orchestrateur avec base_path\n",
        "orchestrator = HybridOrchestrator(\n",
        "    workflow_type=HybridWorkflowType.BACKTESTING,\n",
        "    config={\"enable_model_selection\": True, \"base_path\": base_path}\n",
        ")\n",
        "print(\"✅ HybridOrchestrator prêt\")\n",
        "\n",
        "# 5) Tests rapides\n",
        "print(\"\\n=== Tests de validation ===\")\n",
        "print(f\"Workflow type: {orchestrator.workflow_type}\")\n",
        "print(f\"Config: {orchestrator.config}\")\n",
        "\n",
        "# 6) Afficher champions chargés\n",
        "for regime, info in champs.items():\n",
        "    model_path = os.path.join(base_path, info.get('path', 'N/A'))\n",
        "    exists = os.path.exists(model_path) if info.get('path') else False\n",
        "    print(f\"  {regime}: {info.get('model_name', 'N/A')} -> {exists}\")\n",
        "\n",
        "print(\"\\n🎯 Prêt pour l'intégration V2!\")\n"
    ]
    
    # Trouver la dernière cellule de code (celle avec le diagnostic)
    cells = nb.get('cells', [])
    last_code_cell_idx = None
    
    for i in range(len(cells)-1, -1, -1):
        if cells[i].get('cell_type') == 'code':
            last_code_cell_idx = i
            break
    
    if last_code_cell_idx is not None:
        # Remplacer le contenu de la dernière cellule de code
        cells[last_code_cell_idx]['source'] = new_cell_code
        print(f"✅ Cellule {last_code_cell_idx} (dernière cellule de code) modifiée")
    else:
        # Ajouter une nouvelle cellule de code
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": new_cell_code
        }
        cells.append(new_cell)
        print(f"✅ Nouvelle cellule ajoutée à la fin")
    
    # Sauvegarder le notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    
    print(f"📝 Notebook mis à jour: {notebook_path}")
    return True

if __name__ == "__main__":
    fix_final_cell()
