import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def patch_prepare_fn(source_lines):
    """
    Remplace la fonction prepare_pattern_training_data pour éviter
    l'erreur 'The truth value of a Series is ambiguous' et sécuriser les index.
    """
    new_fn = [
        "print(\"🔧 Préparation des données (sécurisée)...\")\n",
        "def prepare_pattern_training_data(all_data):\n",
        "    import numpy as np\n",
        "    import pandas as pd\n",
        "    X_train = []\n",
        "    y_train = []\n",
        "    \n",
        "    # Parcourir chaque symbole\n",
        "    for symbol, data in all_data.items():\n",
        "        try:\n",
        "            # Vérifications de base\n",
        "            if data is None or not hasattr(data, 'empty') or data.empty:\n",
        "                print(f\"⚠️ {symbol}: dataset vide/None, ignoré\")\n",
        "                continue\n",
        "            # S'assurer des colonnes nécessaires\n",
        "            required_cols = ['Close', 'Volume', 'High', 'Low']\n",
        "            if not all(col in data.columns for col in required_cols):\n",
        "                print(f\"⚠️ {symbol}: colonnes manquantes, ignoré\")\n",
        "                continue\n",
        "            # Trier par date et nettoyer\n",
        "            data = data.sort_index()\n",
        "            data = data.dropna(subset=required_cols)\n",
        "            \n",
        "            n = len(data)\n",
        "            if n < 36:  # 30 jours fenêtre + 5 jours horizon + 1 marge\n",
        "                print(f\"ℹ️ {symbol}: pas assez de points ({n}), ignoré\")\n",
        "                continue\n",
        "            \n",
        "            # Itérer sur des fenêtres glissantes de 30 jours\n",
        "            for i in range(0, n - 35):\n",
        "                seq = data.iloc[i:i+30]\n",
        "                next5 = data.iloc[i+30:i+35]\n",
        "                \n",
        "                # Vérifs de sécurité\n",
        "                if seq.isnull().any().any() or next5.isnull().any().any():\n",
        "                    continue\n",
        "                \n",
        "                # Caractéristiques: Close, Volume, High-Low spread (en float32)\n",
        "                close = np.asarray(seq['Close'].values, dtype=np.float32).reshape(-1, 1)\n",
        "                volume = np.asarray(seq['Volume'].values, dtype=np.float32).reshape(-1, 1)\n",
        "                spread = np.asarray((seq['High'] - seq['Low']).values, dtype=np.float32).reshape(-1, 1)\n",
        "                features = np.concatenate([close, volume, spread], axis=1)\n",
        "                if features.shape != (30, 3):\n",
        "                    continue\n",
        "                \n",
        "                # Label: moyenne des 5 prochains jours vs dernier jour de la fenêtre\n",
        "                current_price = float(seq['Close'].iloc[-1])\n",
        "                future_mean = float(np.mean(next5['Close'].values))\n",
        "                # Calcul de future_return (float scalaire)\n",
        "                if current_price == 0:\n",
        "                    continue\n",
        "                future_return = (future_mean - current_price) / current_price\n",
        "                \n",
        "                # Discrétisation en 3 classes\n",
        "                if future_return > 0.02:\n",
        "                    label = 2  # Buy\n",
        "                elif future_return < -0.02:\n",
        "                    label = 0  # Sell\n",
        "                else:\n",
        "                    label = 1  # Hold\n",
        "                \n",
        "                X_train.append(features)\n",
        "                y_train.append(label)\n",
        "        except Exception as e:\n",
        "            print(f\"⚠️ Erreur sur {symbol}, segment ignoré: {e}\")\n",
        "            continue\n",
        "    \n",
        "    X_train = np.array(X_train, dtype=np.float32)\n",
        "    y_train = np.array(y_train, dtype=np.int32)\n",
        "    print(f\"✅ Préparation terminée: X={X_train.shape}, y={y_train.shape}\")\n",
        "    return X_train, y_train\n",
        "\n",
        "# Reconstruire X/y avec la nouvelle fonction\n",
        "X_train, y_train = prepare_pattern_training_data(all_data)\n",
        "print(f\"📊 Données préparées: {X_train.shape[0]} échantillons\")\n",
    ]
    # Remplacer l’ancienne définition par la nouvelle: on cherche la ligne de définition et on remplace jusqu’à l’appel
    start_idx = None
    end_idx = None
    for i, line in enumerate(source_lines):
        if line.strip().startswith("def prepare_pattern_training_data("):
            start_idx = i
            break
    if start_idx is None:
        return source_lines  # rien à faire
    # chercher l’appel à la fin
    for j in range(start_idx, len(source_lines)):
        if "X_train, y_train = prepare_pattern_training_data(" in source_lines[j]:
            end_idx = j
            break
    if end_idx is None:
        end_idx = start_idx + 1
    # Remplacer le bloc
    return source_lines[:start_idx] + new_fn + source_lines[end_idx+1:]


# Appliquer le patch à la cellule 5 (ou 4 selon version) qui appelle prepare_pattern_training_data
patched = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and ('prepare_pattern_training_data' in ''.join(cell['source'])):
        cell['source'] = patch_prepare_fn(cell['source'])
        patched = True
        break

with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

if patched:
    print("Notebook corrigé: fonction de préparation sécurisée pour éviter l'ambiguïté des Series pandas.")
else:
    print("Aucune cellule cible trouvée pour le patch. Vérifiez la structure du notebook.")
