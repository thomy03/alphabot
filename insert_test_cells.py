#!/usr/bin/env python3
# Script pour insérer les cellules de test V2 dans ALPHABOT_ML_TRAINING_COLAB_v2.ipynb
import json
import sys

def insert_test_cells():
    # Lire le notebook actuel
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Cellule 1: Génération champions.json de test
    cell_champions = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json, os\n",
            "\n",
            "# === Génération champions.json de test ===\n",
            "base_path = '/content/drive/MyDrive/Tradingbot_V2'  # Adapter si besoin selon votre montage Drive\n",
            "os.makedirs(base_path, exist_ok=True)\n",
            "champions = {\n",
            "  'bull': {'model_name': 'lstm_conv', 'path': 'models/lstm_conv_bull.onnx', 'thresholds': {'min_confidence_prob': 0.55, 'min_expected_edge': 0.0}},\n",
            "  'bear': {'model_name': 'gru', 'path': 'models/gru_bear.onnx', 'thresholds': {'min_confidence_prob': 0.60, 'min_expected_edge': 0.05}},\n",
            "  'sideways': {'model_name': 'gbm_light', 'path': 'models/gbm_sideways.pkl', 'thresholds': {'min_confidence_prob': 0.58, 'min_expected_edge': 0.02}},\n",
            "  'baseline': {'model_name': 'baseline_robuste', 'path': 'models/baseline.pkl'}\n",
            "}\n",
            "with open(os.path.join(base_path, 'champions.json'), 'w', encoding='utf-8') as f:\n",
            "    json.dump(champions, f, indent=2, ensure_ascii=False)\n",
            "print('champions.json écrit dans', os.path.join(base_path, 'champions.json'))"
        ]
    }
    
    # Cellule 2: Test V2
    cell_test = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, asyncio\n",
            "# === Test V2: ModelSelectorV2 et HybridOrchestrator ===\n",
            "# Prérequis: repo monté (Drive), chemin racine correct, dépendances installées\n",
            "\n",
            "# Installation deps (Colab)\n",
            "try:\n",
            "    import numpy, pandas, yfinance  # noqa\n",
            "except Exception:\n",
            "    import subprocess\n",
            "    subprocess.run(['pip','install','-q','numpy','pandas','yfinance'], check=False)\n",
            "\n",
            "base_path = '/content/drive/MyDrive/Tradingbot_V2'  # Adapter si besoin\n",
            "if base_path not in sys.path:\n",
            "    sys.path.insert(0, base_path)\n",
            "\n",
            "from alphabot.core.hybrid_orchestrator import HybridOrchestrator, HybridWorkflowType\n",
            "\n",
            "print('=== Test 1: ModelSelectorV2 thresholds ===')\n",
            "orch = HybridOrchestrator(enable_ml=False)\n",
            "sel = orch.model_selector_v2\n",
            "print('baseline =', sel.get_baseline())\n",
            "for regime in ['bull','bear','sideways','unknown']:\n",
            "    th = sel.get_thresholds(regime)\n",
            "    print(f\"{regime}: min_confidence_prob={th.min_confidence_prob}, min_expected_edge={th.min_expected_edge}\")\n",
            "print('[OK] Test 1 exécuté.')\n",
            "\n",
            "print('\\n=== Test 2: Hybrid analysis ML_ENHANCED ===')\n",
            "async def main():\n",
            "    orch2 = HybridOrchestrator(enable_ml=True, ml_confidence_threshold=0.7)\n",
            "    decisions = await orch2.analyze_portfolio_hybrid(['AAPL','MSFT','GOOGL'], HybridWorkflowType.ML_ENHANCED)\n",
            "    summary = {k: (v.action, round(float(v.confidence),3), v.ml_components_used) for k,v in decisions.items()}\n",
            "    print('decisions:', summary)\n",
            "    print('metrics:', orch2.get_performance_metrics())\n",
            "    print('[OK] Test 2 exécuté.')\n",
            "\n",
            "asyncio.run(main())\n",
            "print('\\n=== Fin des tests V2 ===')"
        ]
    }
    
    # Ajouter les cellules à la fin
    notebook['cells'].append(cell_champions)
    notebook['cells'].append(cell_test)
    
    # Sauvegarder le notebook modifié
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Cellules de test ajoutées au notebook. Total cellules: {len(notebook['cells'])}")
    return len(notebook['cells'])

if __name__ == "__main__":
    insert_test_cells()
