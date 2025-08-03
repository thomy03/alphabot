#!/usr/bin/env python3
"""
Script pour remplacer la dernière cellule par une version robuste qui gère tous les conflits Colab
"""

import json
import os

def fix_final_cell_robust():
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
    
    # Nouvelle cellule finale robuste qui gère tous les conflits
    new_cell_source = [
        "# ================================================================\n",
        "# CELLULE FINALE ROBUSTE (intégration V2, Colab-safe)\n",
        "# ================================================================\n",
        "\n",
        "# 0) Préparer environnement minimal: redis + settings (install seulement si manquants)\n",
        "try:\n",
        "    import redis.asyncio\n",
        "    import pydantic_settings\n",
        "    print('✅ Dépendances redis/pydantic-settings déjà installées')\n",
        "except Exception:\n",
        "    print('📦 Installation dépendances minimales...')\n",
        "    !pip install -q \"redis>=4.5\" \"pydantic-settings>=2.0\" \"pydantic>=2.0\" \"pyyaml\"\n",
        "    print('✅ Installation terminée')\n",
        "\n",
        "# 1) Conflits Colab courants: ignorer accelerate/peft si présents (nous ne les utilisons pas ici)\n",
        "import os, sys, importlib\n",
        "os.environ['TRANSFORMERS_NO_ACCELERATE']='1'\n",
        "for m in list(sys.modules):\n",
        "    if m.startswith(('accelerate', 'peft')):\n",
        "        sys.modules.pop(m, None)\n",
        "importlib.invalidate_caches()\n",
        "\n",
        "# 2) Clonage du repo GitHub (idempotent)\n",
        "if not os.path.exists('/content/alphabot_repo'):\n",
        "    !git clone https://github.com/thomy03/alphabot.git /content/alphabot_repo\n",
        "    print('✅ Repo cloné')\n",
        "else:\n",
        "    print('✅ Repo déjà présent')\n",
        "\n",
        "# 3) PYTHONPATH + base_path\n",
        "code_path = \"/content/alphabot_repo\"\n",
        "if code_path not in sys.path:\n",
        "    sys.path.insert(0, code_path)\n",
        "    print(f\"✅ Ajouté {code_path} au PYTHONPATH\")\n",
        "base_path = \"/content/drive/MyDrive/Alphabot\"\n",
        "assert os.path.isdir(base_path), f\"Base path invalide: {base_path}\"\n",
        "assert os.path.exists(os.path.join(base_path, \"champions.json\")), \"champions.json introuvable\"\n",
        "print(f\"✅ Base path validé: {base_path}\")\n",
        "\n",
        "# 4) Correctif dataclass: assurer ordre des champs dans ModelSelectionResult (execution_timestamp avant defaults)\n",
        "try:\n",
        "    import re\n",
        "    p = os.path.join(code_path, \"alphabot/core/hybrid_orchestrator.py\")\n",
        "    s = open(p, \"r\", encoding=\"utf-8\").read()\n",
        "    m = re.search(r\"@dataclass\\s*\\nclass\\s+ModelSelectionResult\\s*:\\s*\\n(?P<body>(?:[ \\t].*\\n)+)\", s)\n",
        "    if m:\n",
        "        body = m.group('body')\n",
        "        lines = [ln for ln in body.splitlines() if (\":\" in ln and not ln.lstrip().startswith('#'))] \n",
        "        ex = [ln for ln in lines if \"execution_timestamp\" in ln]\n",
        "        if ex:\n",
        "            ex_line = \"    \" + ex[0].strip().split(\"=\")[0].rstrip()\n",
        "            rest = [\"    \"+ln.strip() for ln in lines if \"execution_timestamp\" not in ln]\n",
        "            no_def = [ln for ln in rest if \"=\" not in ln]\n",
        "            with_def = [ln for ln in rest if \"=\" in ln]\n",
        "            new_fields = [ex_line] + no_def + ([\"\"] if no_def else []) + with_def\n",
        "            start = body.find(lines[0]); end = body.rfind(lines[-1]) + len(lines[-1])\n",
        "            new_body = body[:start] + \"\\n\".join(new_fields) + body[end:]\n",
        "            s = s.replace(body, new_body, 1)\n",
        "            open(p, \"w\", encoding=\"utf-8\").write(s)\n",
        "            print(\"✅ Correctif dataclass appliqué\")\n",
        "except Exception as e:\n",
        "    print(\"⚠️ Correctif dataclass non appliqué:\", e)\n",
        "\n",
        "# 5) Import alphabot en évitant les conflits de cache\n",
        "try:\n",
        "    if \"alphabot\" in sys.modules:\n",
        "        importlib.reload(sys.modules[\"alphabot\"])\n",
        "    if \"alphabot.core.hybrid_orchestrator\" in sys.modules:\n",
        "        importlib.reload(sys.modules[\"alphabot.core.hybrid_orchestrator\"])\n",
        "    from alphabot.core.hybrid_orchestrator import HybridOrchestrator, HybridWorkflowType\n",
        "    print(\"✅ Import alphabot OK\")\n",
        "except Exception as e:\n",
        "    print(f\"❌ Import échoué: {e}\")\n",
        "    print(\"💡 Si l'erreur concerne huggingface_hub GatedRepoError: ignorable si aucun snapshot_download n'est utilisé ici.\")\n",
        "    raise\n",
        "\n",
        "# 6) Charger champions + instancier orchestrateur\n",
        "import json\n",
        "with open(os.path.join(base_path, \"champions.json\"), \"r\") as f:\n",
        "    champs = json.load(f)\n",
        "print(\"Champions chargés:\", list(champs.keys()))\n",
        "orchestrator = HybridOrchestrator(\n",
        "    workflow_type=HybridWorkflowType.BACKTESTING,\n",
        "    config={\"enable_model_selection\": True, \"base_path\": base_path}\n",
        ")\n",
        "print(\"✅ HybridOrchestrator instancié\")\n",
        "print(f\"   Workflow type: {orchestrator.workflow_type}\")\n",
        "print(f\"   Base path: {base_path}\")\n",
        "\n",
        "print(\"\\n🎯 Test d'intégration V2 réussi!\")\n"
    ]
    
    # Remplacer complètement la cellule
    cells[last_code_idx]['source'] = new_cell_source
    
    # Sauvegarder
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Cellule {last_code_idx} remplacée par version robuste")
    print("📝 Notebook sauvegardé localement")
    print("\n💡 Instructions pour Colab:")
    print("1. Runtime > Restart runtime")
    print("2. Exécuter la cellule 2 (montage Drive)")
    print("3. Exécuter directement la dernière cellule (robuste)")
    
    return True

if __name__ == "__main__":
    fix_final_cell_robust()
