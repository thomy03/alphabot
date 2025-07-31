import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 5 (Entraînement Pattern Detector)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 5: Entraînement Pattern Detector' in ''.join(cell['source']):
        source = cell['source']
        
        # Remplacer la fonction prepare_pattern_training_data
        new_function = [
            "# Préparer les données d'entraînement\n",
            "def prepare_pattern_training_data(all_data):\n",
            "    X_train, y_train = [], []\n",
            "    \n",
            "    for symbol, data in all_data.items():\n",
            "        if len(data) < 50:\n",
            "            continue\n",
            "            \n",
            "        # Créer des séquences temporelles\n",
            "        prices = data['Close'].values.flatten()  # S'assurer que c'est 1D\n",
            "        if len(prices) == 0:\n",
            "            continue\n",
            "            \n",
            "        # Calculer les returns de manière sécurisée\n",
            "        returns = np.diff(prices) / prices[:-1]\n",
            "        \n",
            "        # Labels: 0=DOWN, 1=SIDEWAYS, 2=UP\n",
            "        for i in range(30, len(returns)-5):\n",
            "            # Features: prix normalisés, returns, volume\n",
            "            seq_prices = prices[i-30:i] / prices[i-30]\n",
            "            seq_returns = returns[i-30:i]\n",
            "            \n",
            "            # Gérer le volume de manière sécurisée\n",
            "            volume_data = data['Volume'].values.flatten()\n",
            "            if len(volume_data) > 0:\n",
            "                mean_volume = np.mean(volume_data[i-30:i])\n",
            "                if mean_volume > 0:\n",
            "                    seq_volume = volume_data[i-30:i] / mean_volume\n",
            "                else:\n",
            "                    seq_volume = np.ones(30)\n",
            "            else:\n",
            "                seq_volume = np.ones(30)\n",
            "            \n",
            "            # Combiner features\n",
            "            features = np.column_stack([seq_prices, seq_returns, seq_volume])\n",
            "            \n",
            "            # Label basé sur le mouvement futur\n",
            "            future_return = np.mean(returns[i:i+5])\n",
            "            if future_return < -0.02:\n",
            "                label = 0  # DOWN\n",
            "            elif future_return > 0.02:\n",
            "                label = 2  # UP\n",
            "            else:\n",
            "                label = 1  # SIDEWAYS\n",
            "            \n",
            "            X_train.append(features)\n",
            "            y_train.append(label)\n",
            "    \n",
            "    if len(X_train) == 0:\n",
            "        print(\"⚠️ Aucune donnée d'entraînement générée\")\n",
            "        return np.array([]), np.array([])\n",
            "    \n",
            "    return np.array(X_train), np.array(y_train)\n"
        ]
        
        # Trouver l'index où commencer le remplacement
        start_idx = None
        for i, line in enumerate(source):
            if "def prepare_pattern_training_data(all_data):" in line:
                start_idx = i
                break
        
        if start_idx is not None:
            # Trouver la fin de la fonction
            end_idx = start_idx + 1
            indent_level = len(source[start_idx]) - len(source[start_idx].lstrip())
            
            for i in range(start_idx + 1, len(source)):
                line = source[i]
                if line.strip() == "":
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip():
                    end_idx = i
                    break
            
            # Remplacer la fonction
            source[start_idx:end_idx] = new_function
        
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - préparation des données fixée!")
