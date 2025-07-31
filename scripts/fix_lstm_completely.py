import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 5 (Entraînement Pattern Detector)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 5: Entraînement Pattern Detector' in ''.join(cell['source']):
        source = cell['source']
        
        # Remplacer complètement la création et l'entraînement du modèle
        new_model_section = [
            "# Créer le modèle LSTM compatible CPU\n",
            "print(\"🔧 Création du modèle LSTM compatible CPU...\")\n",
            "\n",
            "# Désactiver temporairement la distribution strategy pour éviter les problèmes CuDNN\n",
            "original_strategy = strategy\n",
            "strategy = tf.distribute.get_strategy()  # Utiliser la stratégie par défaut\n",
            "\n",
            "with strategy.scope():\n",
            "    # Créer un modèle plus simple compatible CPU\n",
            "    inputs = tf.keras.Input(shape=(30, 3), name='input_layer')\n",
            "    \n",
            "    # Première couche LSTM avec activation explicite\n",
            "    lstm1 = tf.keras.layers.LSTM(\n",
            "        32, \n",
            "        return_sequences=True,\n",
            "        activation='tanh',\n",
            "        recurrent_activation='sigmoid',\n",
            "        use_bias=True,\n",
            "        name='lstm_1'\n",
            "    )(inputs)\n",
            "    lstm1 = tf.keras.layers.Dropout(0.2, name='dropout_1')(lstm1)\n",
            "    \n",
            "    # Deuxième couche LSTM\n",
            "    lstm2 = tf.keras.layers.LSTM(\n",
            "        16, \n",
            "        return_sequences=False,\n",
            "        activation='tanh',\n",
            "        recurrent_activation='sigmoid',\n",
            "        use_bias=True,\n",
            "        name='lstm_2'\n",
            "    )(lstm1)\n",
            "    lstm2 = tf.keras.layers.Dropout(0.2, name='dropout_2')(lstm2)\n",
            "    \n",
            "    # Couches denses\n",
            "    dense1 = tf.keras.layers.Dense(8, activation='relu', name='dense_1')(lstm2)\n",
            "    outputs = tf.keras.layers.Dense(3, activation='softmax', name='output')(dense1)\n",
            "    \n",
            "    # Créer le modèle fonctionnel\n",
            "    lstm_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='lstm_pattern_detector')\n",
            "    \n",
            "    # Compiler avec des paramètres compatibles CPU\n",
            "    lstm_model.compile(\n",
            "        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\n",
            "        loss='sparse_categorical_crossentropy',\n",
            "        metrics=['accuracy'],\n",
            "        run_eagerly=False  # Désactiver run_eagerly pour de meilleures performances\n",
            "    )\n",
            "\n",
            "# Afficher le résumé du modèle\n",
            "print(\"✅ Modèle LSTM créé:\")\n",
            "lstm_model.summary()\n",
            "\n",
            "# Callbacks simplifiés\n",
            "callbacks = [\n",
            "    tf.keras.callbacks.EarlyStopping(\n",
            "        patience=5, \n",
            "        restore_best_weights=True,\n",
            "        monitor='val_loss'\n",
            "    ),\n",
            "    tf.keras.callbacks.ReduceLROnPlateau(\n",
            "        factor=0.5, \n",
            "        patience=3, \n",
            "        monitor='val_loss'\n",
            "    ),\n",
            "    tf.keras.callbacks.ModelCheckpoint(\n",
            "        f'{base_path}/checkpoints/lstm_pattern_best.h5',\n",
            "        save_best_only=True,\n",
            "        monitor='val_accuracy'\n",
            "    )\n",
            "]\n",
            "\n",
            "# Vérifier les données avant l'entraînement\n",
            "print(f\"📊 Vérification des données:\")\n",
            "print(f\"  - X_train shape: {X_train.shape}\")\n",
            "print(f\"  - y_train shape: {y_train.shape}\")\n",
            "print(f\"  - X_train dtype: {X_train.dtype}\")\n",
            "print(f\"  - y_train dtype: {y_train.dtype}\")\n",
            "print(f\"  - Valeurs uniques dans y_train: {np.unique(y_train)}\")\n",
            "\n",
            "# S'assurer que les données sont du bon type\n",
            "X_train = X_train.astype(np.float32)\n",
            "y_train = y_train.astype(np.int32)\n",
            "\n",
            "# Entraîner avec des paramètres conservateurs\n",
            "print(\"🚀 Début de l'entraînement LSTM...\")\n",
            "try:\n",
            "    history = lstm_model.fit(\n",
            "        X_train, y_train,\n",
            "        epochs=20,  # Réduit pour éviter les timeouts\n",
            "        batch_size=16,  # Plus petit batch size\n",
            "        validation_split=0.2,\n",
            "        callbacks=callbacks,\n",
            "        verbose=1,\n",
            "        shuffle=True\n",
            "    )\n",
            "    print(\"✅ Entraînement terminé avec succès\")\n",
            "except Exception as e:\n",
            "    print(f\"❌ Erreur lors de l'entraînement: {e}\")\n",
            "    print(\"🔧 Tentative avec un modèle encore plus simple...\")\n",
            "    \n",
            "    # Modèle de secours très simple\n",
            "    with strategy.scope():\n",
            "        simple_model = tf.keras.Sequential([\n",
            "            tf.keras.layers.Flatten(input_shape=(30, 3)),\n",
            "            tf.keras.layers.Dense(64, activation='relu'),\n",
            "            tf.keras.layers.Dropout(0.3),\n",
            "            tf.keras.layers.Dense(32, activation='relu'),\n",
            "            tf.keras.layers.Dense(3, activation='softmax')\n",
            "        ])\n",
            "        \n",
            "        simple_model.compile(\n",
            "            optimizer='adam',\n",
            "            loss='sparse_categorical_crossentropy',\n",
            "            metrics=['accuracy']\n",
            "        )\n",
            "    \n",
            "    history = simple_model.fit(\n",
            "        X_train, y_train,\n",
            "        epochs=10,\n",
            "        batch_size=32,\n",
            "        validation_split=0.2,\n",
            "        verbose=1\n",
            "    )\n",
            "    lstm_model = simple_model  # Utiliser le modèle simple\n",
            "\n",
            "# Sauvegarder le modèle\n",
            "try:\n",
            "    lstm_model.save(f'{base_path}/models/lstm_pattern_model.h5')\n",
            "    print(\"✅ Modèle sauvegardé\")\n",
            "except Exception as e:\n",
            "    print(f\"⚠️ Erreur de sauvegarde: {e}\")\n",
            "\n",
            "# Afficher les courbes d'apprentissage si disponibles\n",
            "if 'history' in locals() and hasattr(history, 'history'):\n",
            "    try:\n",
            "        plt.figure(figsize=(12, 4))\n",
            "        \n",
            "        plt.subplot(1, 2, 1)\n",
            "        plt.plot(history.history['accuracy'], label='Training')\n",
            "        if 'val_accuracy' in history.history:\n",
            "            plt.plot(history.history['val_accuracy'], label='Validation')\n",
            "        plt.title('Model Accuracy')\n",
            "        plt.legend()\n",
            "        \n",
            "        plt.subplot(1, 2, 2)\n",
            "        plt.plot(history.history['loss'], label='Training')\n",
            "        if 'val_loss' in history.history:\n",
            "            plt.plot(history.history['val_loss'], label='Validation')\n",
            "        plt.title('Model Loss')\n",
            "        plt.legend()\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "    except Exception as e:\n",
            "        print(f\"⚠️ Erreur lors de l'affichage des courbes: {e}\")\n"
        ]
        
        # Trouver l'index où commencer le remplacement
        start_idx = None
        for i, line in enumerate(source):
            if "# Créer le modèle LSTM" in line:
                start_idx = i
                break
        
        if start_idx is not None:
            # Trouver la fin de la section à remplacer (jusqu'à plt.show())
            end_idx = len(source)
            for i in range(start_idx, len(source)):
                if "plt.show()" in source[i]:
                    end_idx = i + 1
                    break
            
            # Remplacer la section
            source[start_idx:end_idx] = new_model_section
        
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - modèle LSTM complètement repensé!")
