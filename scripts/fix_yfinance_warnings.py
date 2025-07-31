import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 4 (Téléchargement des données)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 4: Téléchargement des données' in ''.join(cell['source']):
        source = cell['source']
        
        # Remplacer la section de téléchargement et d'affichage
        new_data_section = [
            "# Télécharger les données\n",
            "all_data = {}\n",
            "for symbol in symbols:\n",
            "    try:\n",
            "        # Spécifier explicitement auto_adjust pour éviter le warning\n",
            "        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)\n",
            "        if not data.empty:\n",
            "            all_data[symbol] = data\n",
            "            print(f\"✅ {symbol}: {len(data)} jours de données\")\n",
            "        else:\n",
            "            print(f\"❌ {symbol}: Pas de données disponibles\")\n",
            "    except Exception as e:\n",
            "        print(f\"❌ {symbol}: Erreur de téléchargement - {e}\")\n",
            "\n",
            "# Sauvegarder les données\n",
            "import pickle\n",
            "data_path = f\"{base_path}/data/market_data.pkl\"\n",
            "with open(data_path, 'wb') as f:\n",
            "    pickle.dump(all_data, f)\n",
            "\n",
            "print(f\"\\n💾 Données sauvegardées dans: {data_path}\")\n",
            "print(f\"📊 Total symboles: {len(all_data)}\")\n",
            "\n",
            "# Afficher un exemple\n",
            "if all_data:\n",
            "    sample_symbol = list(all_data.keys())[0]\n",
            "    sample_data = all_data[sample_symbol]\n",
            "    print(f\"\\n📈 Exemple pour {sample_symbol}:\")\n",
            "    print(f\"- Première date: {sample_data.index[0].strftime('%Y-%m-%d')}\")\n",
            "    print(f\"- Dernière date: {sample_data.index[-1].strftime('%Y-%m-%d')}\")\n",
            "    \n",
            "    # Calculer et afficher les statistiques sans warnings\n",
            "    mean_price = sample_data['Close'].mean()\n",
            "    volatility = sample_data['Close'].pct_change().std() * 100\n",
            "    \n",
            "    print(f\"- Prix moyen: ${mean_price:.2f}\")\n",
            "    print(f\"- Volatilité: {volatility:.2f}%\")\n"
        ]
        
        # Trouver l'index où commencer le remplacement
        start_idx = None
        for i, line in enumerate(source):
            if "# Télécharger les données" in line:
                start_idx = i
                break
        
        if start_idx is not None:
            # Trouver la fin de la section à remplacer
            end_idx = len(source)
            for i in range(start_idx, len(source)):
                if 'print(f"- Volatilité: {float(sample_data[\'Close\'].pct_change().std()*100):.2f}%")' in source[i]:
                    end_idx = i + 1
                    break
            
            # Remplacer la section
            source[start_idx:end_idx] = new_data_section
        
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - warnings yfinance éliminés!")
