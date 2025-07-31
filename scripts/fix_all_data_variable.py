import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 5 pour définir all_data
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 5: Entraînement Pattern Detector' in ''.join(cell['source']):
        source = cell['source']
        
        # Trouver où insérer la définition de all_data
        insert_idx = None
        for i, line in enumerate(source):
            if "# Préparer les données" in line:
                insert_idx = i
                break
        
        if insert_idx is not None:
            # Insérer la définition de all_data avant la préparation des données
            all_data_definition = [
                "# Charger les données depuis la cellule précédente\n",
                "try:\n",
                "    # Essayer de charger depuis le pickle sauvegardé\n",
                "    import pickle\n",
                "    with open(f'{base_path}/data/market_data.pkl', 'rb') as f:\n",
                "        all_data = pickle.load(f)\n",
                "    print(\"✅ Données chargées depuis le pickle\")\n",
                "except:\n",
                "    print(\"🔧 Re-téléchargement des données...\")\n",
                "    # Re-télécharger les données si le pickle n'est pas disponible\n",
                "    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']\n",
                "    end_date = datetime.now().strftime('%Y-%m-%d')\n",
                "    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')\n",
                "    \n",
                "    all_data = {}\n",
                "    for symbol in symbols:\n",
                "        print(f\"📥 Téléchargement des données pour {symbol}...\")\n",
                "        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)\n",
                "        if not data.empty:\n",
                "            all_data[symbol] = data\n",
                "            print(f\"✅ {symbol}: {len(data)} jours de données\")\n",
                "        else:\n",
                "            print(f\"⚠️ {symbol}: Pas de données disponibles\")\n",
                "    \n",
                "    # Sauvegarder pour éviter de re-télécharger\n",
                "    try:\n",
                "        os.makedirs(f'{base_path}/data', exist_ok=True)\n",
                "        with open(f'{base_path}/data/market_data.pkl', 'wb') as f:\n",
                "            pickle.dump(all_data, f)\n",
                "        print(f\"💾 Données sauvegardées dans: {base_path}/data/market_data.pkl\")\n",
                "    except Exception as e:\n",
                "        print(f\"⚠️ Erreur de sauvegarde: {e}\")\n",
                "\n",
                "print(f\"📊 Total symboles: {len(all_data)}\")\n",
                "\n"
            ]
            
            # Insérer la définition
            source[insert_idx:insert_idx] = all_data_definition
        
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - variable all_data définie!")
