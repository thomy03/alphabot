import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Ajouter une cellule de suivi et reprise automatique au début
progress_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🔄 Suivi de Progression et Reprise Automatique\n",
        "\n",
        "Cette cellule vérifie l'état d'avancement du notebook et permet de reprendre là où le processus s'est arrêté."
    ]
}

# Ajouter la cellule de suivi après la première cellule
nb['cells'].insert(1, progress_cell)

# Ajouter une cellule de code pour le suivi et la reprise
tracking_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# 🔄 Système de suivi et reprise automatique\n",
        "import os\n",
        "import json\n",
        "import pickle\n",
        "from datetime import datetime\n",
        "\n",
        "# Définir le chemin de base\n",
        "base_path = '/content/drive/MyDrive/AlphaBot_ML_Training'\n",
        "os.makedirs(base_path, exist_ok=True)\n",
        "\n",
        "# Fichier de suivi de progression\n",
        "progress_file = f'{base_path}/progress_tracker.json'\n",
        "\n",
        "# État initial des étapes\n",
        "default_progress = {\n",
        "    'cell_1_setup': False,\n",
        "    'cell_2_data_download': False,\n",
        "    'cell_3_data_analysis': False,\n",
        "    'cell_4_pattern_training': False,\n",
        "    'cell_5_sentiment_training': False,\n",
        "    'cell_6_rag_training': False,\n",
        "    'cell_7_integration': False,\n",
        "    'cell_8_testing': False,\n",
        "    'cell_9_deployment': False,\n",
        "    'last_cell_executed': None,\n",
        "    'start_time': None,\n",
        "    'last_update': None\n",
        "}\n",
        "\n",
        "# Charger ou initialiser le suivi\n",
        "try:\n",
        "    with open(progress_file, 'r') as f:\n",
        "        progress = json.load(f)\n",
        "    print(\"📊 Suivi de progression chargé\")\n",
        "except:\n",
        "    progress = default_progress.copy()\n",
        "    progress['start_time'] = datetime.now().isoformat()\n",
        "    print(\"🆕 Nouveau suivi de progression initialisé\")\n",
        "\n",
        "# Fonction pour mettre à jour la progression\n",
        "def update_progress(cell_name):\n",
        "    progress[cell_name] = True\n",
        "    progress['last_cell_executed'] = cell_name\n",
        "    progress['last_update'] = datetime.now().isoformat()\n",
        "    \n",
        "    with open(progress_file, 'w') as f:\n",
        "        json.dump(progress, f, indent=2)\n",
        "    \n",
        "    print(f\"✅ Progression mise à jour: {cell_name}\")\n",
        "\n",
        "# Fonction pour vérifier l'état\n",
        "def check_progress():\n",
        "    print(\"\\n📋 État actuel de la progression:\")\n",
        "    print(\"=\" * 50)\n",
        "    \n",
        "    completed = sum(progress.values()) - 4  # Exclure les métadonnées\n",
        "    total = len(default_progress) - 4\n",
        "    \n",
        "    print(f\"📊 Progression: {completed}/{total} étapes complétées ({completed/total*100:.1f}%)\")\n",
        "    print(f\"⏰ Démarré: {progress.get('start_time', 'N/A')}\")\n",
        "    print(f\"🔄 Dernière mise à jour: {progress.get('last_update', 'N/A')}\")\n",
        "    print(f\"📍 Dernière cellule: {progress.get('last_cell_executed', 'Aucune')}\")\n",
        "    \n",
        "    print(\"\\n📝 Statut des étapes:\")\n",
        "    steps = [\n",
        "        ('cell_1_setup', '1. Configuration initiale'),\n",
        "        ('cell_2_data_download', '2. Téléchargement des données'),\n",
        "        ('cell_3_data_analysis', '3. Analyse des données'),\n",
        "        ('cell_4_pattern_training', '4. Entraînement Pattern Detector'),\n",
        "        ('cell_5_sentiment_training', '5. Entraînement Sentiment Analyzer'),\n",
        "        ('cell_6_rag_training', '6. Entraînement RAG'),\n",
        "        ('cell_7_integration', '7. Intégration'),\n",
        "        ('cell_8_testing', '8. Tests'),\n",
        "        ('cell_9_deployment', '9. Déploiement')\n",
        "    ]\n",
        "    \n",
        "    for step_key, step_name in steps:\n",
        "        status = \"✅\" if progress.get(step_key, False) else \"⏳\"\n",
        "        print(f\"  {status} {step_name}\")\n",
        "    \n",
        "    print(\"=\" * 50)\n",
        "    \n",
        "    # Suggérer la prochaine étape\n",
        "    if not progress['cell_1_setup']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 1 (Configuration)\")\n",
        "    elif not progress['cell_2_data_download']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 2 (Téléchargement des données)\")\n",
        "    elif not progress['cell_3_data_analysis']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 3 (Analyse des données)\")\n",
        "    elif not progress['cell_4_pattern_training']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 4 (Pattern Detector)\")\n",
        "    elif not progress['cell_5_sentiment_training']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 5 (Sentiment Analyzer)\")\n",
        "    elif not progress['cell_6_rag_training']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 6 (RAG)\")\n",
        "    elif not progress['cell_7_integration']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 7 (Intégration)\")\n",
        "    elif not progress['cell_8_testing']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 8 (Tests)\")\n",
        "    elif not progress['cell_9_deployment']:\n",
        "        print(\"\\n🚀 Prochaine étape: Exécuter la cellule 9 (Déploiement)\")\n",
        "    else:\n",
        "        print(\"\\n🎉 Toutes les étapes sont complétées !\")\n",
        "\n",
        "# Vérifier l'état actuel\n",
        "check_progress()\n",
        "\n",
        "# Instructions pour l'utilisateur\n",
        "print(\"\\n💡 Instructions:\")\n",
        "print(\"1. Exécutez cette cellule pour voir l'état d'avancement\")\n",
        "print(\"2. Chaque cellule mettra à jour automatiquement sa progression\")\n",
        "print(\"3. Si le processus s'arrête, relancez simplement cette cellule\")\n",
        "print(\"4. Continuez avec la cellule suggérée\")\n",
        "print(\"\\n🔄 Note: Le système est conçu pour supporter les arrêts/redémarrages\")\n"
    ]
}

# Insérer la cellule de tracking après la cellule markdown
nb['cells'].insert(2, tracking_cell)

# Modifier chaque cellule pour ajouter le suivi de progression
cell_mapping = [
    ('CELLULE 1: Configuration initiale', 'cell_1_setup'),
    ('CELLULE 2: Téléchargement des données', 'cell_2_data_download'),
    ('CELLULE 3: Analyse des données', 'cell_3_data_analysis'),
    ('CELLULE 4: Entraînement Pattern Detector', 'cell_4_pattern_training'),
    ('CELLULE 5: Entraînement Sentiment Analyzer', 'cell_5_sentiment_training'),
    ('CELLULE 6: Entraînement RAG', 'cell_6_rag_training'),
    ('CELLULE 7: Intégration', 'cell_7_integration'),
    ('CELLULE 8: Tests', 'cell_8_testing'),
    ('CELLULE 9: Déploiement', 'cell_9_deployment')
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell_content = ''.join(cell['source'])
        
        for cell_title, progress_key in cell_mapping:
            if cell_title in cell_content:
                # Ajouter le suivi au début de la cellule
                tracking_code = [
                    "# Mettre à jour la progression\n",
                    "try:\n",
                    "    update_progress('" + progress_key + "')\n",
                    "    print(f\"📊 Démarrage de {cell_title}\")\n",
                    "except:\n",
                    "    print(\"⚠️ Impossible de mettre à jour la progression\")\n",
                    "\n"
                ]
                
                # Insérer après le titre de la cellule
                insert_idx = 0
                for i, line in enumerate(cell['source']):
                    if cell_title in line:
                        insert_idx = i + 1
                        break
                
                cell['source'][insert_idx:insert_idx] = tracking_code
                break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook mis à jour avec succès - système de suivi et reprise automatique ajouté!")
