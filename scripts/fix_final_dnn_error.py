import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 5 (Entraînement Pattern Detector)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 5: Entraînement Pattern Detector' in ''.join(cell['source']):
        source = cell['source']
        
        # Remplacer complètement la section d'entraînement avec une approche CPU-only
        new_training_section = [
            "# Créer un modèle compatible CPU (sans CuDNN)\n",
            "print(\"🔧 Création du modèle CPU-only...\")\n",
            "\n",
            "# Forcer l'utilisation de CPU et désactiver l'accélération matérielle\n",
            "import os\n",
            "os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Désactiver GPU\n",
            "os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'\n",
            "\n",
            "# Recréer la stratégie sans GPU\n",
            "strategy = tf.distribute.get_strategy()\n",
            "\n",
            "with strategy.scope():\n",
            "    # Utiliser un modèle simple sans LSTM pour éviter les problèmes CuDNN\n",
            "    inputs = tf.keras.Input(shape=(30, 3), name='input_layer')\n",
            "    \n",
            "    # Aplatir les données pour les utiliser dans un réseau dense\n",
            "    flatten = tf.keras.layers.Flatten(name='flatten')(inputs)\n",
            "    \n",
            "    # Couches denses avec dropout\n",
            "    dense1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(flatten)\n",
            "    dense1 = tf.keras.layers.Dropout(0.3, name='dropout_1')(dense1)\n",
            "    \n",
            "    dense2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(dense1)\n",
            "    dense2 = tf.keras.layers.Dropout(0.3, name='dropout_2')(dense2)\n",
            "    \n",
            "    dense3 = tf.keras.layers.Dense(32, activation='relu', name='dense_3')(dense2)\n",
            "    dense3 = tf.keras.layers.Dropout(0.2, name='dropout_3')(dense3)\n",
            "    \n",
            "    outputs = tf.keras.layers.Dense(3, activation='softmax', name='output')(dense3)\n",
            "    \n",
            "    # Créer le modèle\n",
            "    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cpu_pattern_detector')\n",
            "    \n",
            "    # Compiler avec des paramètres conservateurs\n",
            "    model.compile(\n",
            "        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),\n",
            "        loss='sparse_categorical_crossentropy',\n",
            "        metrics=['accuracy']\n",
            "    )\n",
            "\n",
            "# Afficher le résumé du modèle\n",
            "print(\"✅ Modèle CPU-only créé:\")\n",
            "model.summary()\n",
            "\n",
            "# Callbacks simplifiés\n",
            "callbacks = [\n",
            "    tf.keras.callbacks.EarlyStopping(\n",
            "        patience=10, \n",
            "        restore_best_weights=True,\n",
            "        monitor='val_loss',\n",
            "        verbose=1\n",
            "    ),\n",
            "    tf.keras.callbacks.ReduceLROnPlateau(\n",
            "        factor=0.5, \n",
            "        patience=5, \n",
            "        monitor='val_loss',\n",
            "        verbose=1\n",
            "    )\n",
            "]\n",
            "\n",
            "# Vérifier et préparer les données\n",
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
            "# Normaliser les données pour meilleure convergence\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)\n",
            "\n",
            "# Entraîner avec des paramètres très conservateurs\n",
            "print(\"🚀 Début de l'entraînement CPU-only...\")\n",
            "try:\n",
            "    history = model.fit(\n",
            "        X_train_scaled, y_train,\n",
            "        epochs=30,  # Plus d'epochs car pas de GPU\n",
            "        batch_size=32,  # Batch size plus grand pour CPU\n",
            "        validation_split=0.2,\n",
            "        callbacks=callbacks,\n",
            "        verbose=1,\n",
            "        shuffle=True\n",
            "    )\n",
            "    print(\"✅ Entraînement terminé avec succès\")\n",
            "    \n",
            "    # Sauvegarder le modèle et le scaler\n",
            "    try:\n",
            "        model.save(f'{base_path}/models/cpu_pattern_model.h5')\n",
            "        import pickle\n",
            "        with open(f'{base_path}/models/scaler.pkl', 'wb') as f:\n",
            "            pickle.dump(scaler, f)\n",
            "        print(\"✅ Modèle et scaler sauvegardés\")\n",
            "    except Exception as e:\n",
            "        print(f\"⚠️ Erreur de sauvegarde: {e}\")\n",
            "    \n",
            "    # Afficher les courbes d'apprentissage\n",
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
            "        print(f\"⚠️ Erreur lors de l'affichage des courbes: {e}\")\n",
            "        \n",
            "except Exception as e:\n",
            "    print(f\"❌ Erreur critique lors de l'entraînement: {e}\")\n",
            "    print(\"🔧 Création d'un modèle minimaliste de secours...\")\n",
            "    \n",
            "    # Modèle minimaliste absolument garanti de fonctionner\n",
            "    with strategy.scope():\n",
            "        minimal_model = tf.keras.Sequential([\n",
            "            tf.keras.layers.Input(shape=(30, 3), name='input'),\n",
            "            tf.keras.layers.Flatten(),\n",
            "            tf.keras.layers.Dense(16, activation='relu'),\n",
            "            tf.keras.layers.Dense(3, activation='softmax')\n",
            "        ])\n",
            "        \n",
            "        minimal_model.compile(\n",
            "            optimizer='adam',\n",
            "            loss='sparse_categorical_crossentropy',\n",
            "            metrics=['accuracy']\n",
            "        )\n",
            "    \n",
            "    # Entraînement minimaliste\n",
            "    history = minimal_model.fit(\n",
            "        X_train_scaled[:500], y_train[:500],  # Utiliser seulement 500 échantillons\n",
            "        epochs=5,\n",
            "        batch_size=16,\n",
            "        verbose=1\n",
            "    )\n",
            "    \n",
            "    model = minimal_model\n",
            "    print(\"✅ Modèle minimaliste entraîné\")\n"
        ]
        
        # Trouver l'index où commencer le remplacement
        start_idx = None
        for i, line in enumerate(source):
            if "# Créer le modèle LSTM compatible CPU" in line:
                start_idx = i
                break
        
        if start_idx is not None:
            # Trouver la fin de la section à remplacer (jusqu'à la fin du try/except)
            end_idx = len(source)
            for i in range(start_idx, len(source)):
                if 'print("✅ Modèle minimaliste entraîné")' in source[i]:
                    end_idx = i + 1
                    break
            
            # Remplacer la section
            source[start_idx:end_idx] = new_training_section
        
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - modèle CPU-only sans dépendance CuDNN!")
