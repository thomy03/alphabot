import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 3 (Code AlphaBot setup)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 3: Code AlphaBot setup' in ''.join(cell['source']):
        source = cell['source']
        
        # Corriger l'import de RAGIntegrator
        for i, line in enumerate(source):
            if "from alphabot.ml.rag_integrator import RAGIntegrator" in line:
                source[i] = "        from alphabot.ml.rag_integrator import RAGIntegrator\n"
                break
        break

# Trouver et corriger la cellule 5 (Entraînement Pattern Detector)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 5: Entraînement Pattern Detector' in ''.join(cell['source']):
        source = cell['source']
        
        # Remplacer la création du modèle LSTM
        new_model_creation = [
            "# Créer le modèle LSTM\n",
            "with strategy.scope():\n",
            "    lstm_model = tf.keras.Sequential([\n",
            "        tf.keras.layers.Input(shape=(30, 3)),  # Utiliser Input au lieu d'input_shape\n",
            "        tf.keras.layers.LSTM(64, return_sequences=True),\n",
            "        tf.keras.layers.Dropout(0.2),\n",
            "        tf.keras.layers.LSTM(32, return_sequences=False),\n",
            "        tf.keras.layers.Dropout(0.2),\n",
            "        tf.keras.layers.Dense(16, activation='relu'),\n",
            "        tf.keras.layers.Dense(3, activation='softmax')\n",
            "    ])\n",
            "    \n",
            "    # Compiler avec des paramètres compatibles CPU/GPU\n",
            "    lstm_model.compile(\n",
            "        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\n",
            "        loss='sparse_categorical_crossentropy',\n",
            "        metrics=['accuracy'],\n",
            "        run_eagerly=True  # Forcer l'exécution eager pour éviter les problèmes CuDNN\n",
            "    )\n"
        ]
        
        # Trouver l'index où commencer le remplacement
        start_idx = None
        for i, line in enumerate(source):
            if "with strategy.scope():" in line and "lstm_model = tf.keras.Sequential" in source[i+1]:
                start_idx = i
                break
        
        if start_idx is not None:
            # Trouver la fin de la section à remplacer
            end_idx = start_idx + 1
            for i in range(start_idx + 1, len(source)):
                if "lstm_model.compile(" in source[i]:
                    # Trouver la fin du compile
                    for j in range(i, len(source)):
                        if source[j].strip().endswith(")") and "metrics=['accuracy']" in source[j]:
                            end_idx = j + 1
                            break
                    break
            
            # Remplacer la section
            source[start_idx:end_idx] = new_model_creation
        
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - modèle LSTM compatible CPU/GPU!")
