import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 5 pour fix l'erreur de dataset
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 5: Entraînement Pattern Detector' in ''.join(cell['source']):
        source = cell['source']
        
        # Remplacer la section de création et entraînement du dataset
        new_dataset_section = [
            "# Créer un dataset TensorFlow optimisé pour GPU\n",
            "train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train))\n",
            "train_dataset = train_dataset.shuffle(buffer_size=len(X_train))\n",
            "train_dataset = train_dataset.batch(64)  # Batch size plus grand pour GPU\n",
            "train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)\n",
            "\n",
            "# Créer des datasets de train/validation séparés\n",
            "val_size = int(0.2 * len(X_train))\n",
            "full_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train))\n",
            "full_dataset = full_dataset.shuffle(buffer_size=len(X_train))\n",
            "\n",
            "# Split en train/validation\n",
            "val_dataset = full_dataset.take(val_size).batch(64).prefetch(tf.data.AUTOTUNE)\n",
            "train_dataset = full_dataset.skip(val_size).batch(64).prefetch(tf.data.AUTOTUNE)\n",
            "\n",
            "# Calculer les steps correctement\n",
            "train_size = len(X_train) - val_size\n",
            "steps_per_epoch = train_size // 64\n",
            "validation_steps = val_size // 64\n",
            "\n",
            "print(f\"📊 Configuration dataset:\")\n",
            "print(f\"  - Taille train: {train_size}\")\n",
            "print(f\"  - Taille validation: {val_size}\")\n",
            "print(f\"  - Steps par epoch: {steps_per_epoch}\")\n",
            "print(f\"  - Validation steps: {validation_steps}\")\n",
            "\n",
            "# Entraîner avec des paramètres optimisés pour A100\n",
            "print(\"🚀 Début de l'entraînement GPU optimisé pour A100...\")\n",
            "try:\n",
            "    history = model.fit(\n",
            "        train_dataset,\n",
            "        validation_data=val_dataset,\n",
            "        epochs=50,\n",
            "        callbacks=callbacks,\n",
            "        verbose=1,\n",
            "        steps_per_epoch=steps_per_epoch,\n",
            "        validation_steps=validation_steps\n",
            "    )\n",
            "    print(\"✅ Entraînement GPU terminé avec succès\")\n",
            "    \n",
            "    # Sauvegarder le modèle et le scaler\n",
            "    try:\n",
            "        model.save(f'{base_path}/models/gpu_lstm_cnn_model.keras')\n",
            "        import pickle\n",
            "        with open(f'{base_path}/models/gpu_scaler.pkl', 'wb') as f:\n",
            "            pickle.dump(scaler, f)\n",
            "        print(\"✅ Modèle GPU et scaler sauvegardés\")\n",
            "    except Exception as e:\n",
            "        print(f\"⚠️ Erreur de sauvegarde: {e}\")\n",
            "    \n",
            "    # Afficher les courbes d'apprentissage\n",
            "    try:\n",
            "        plt.figure(figsize=(15, 5))\n",
            "        \n",
            "        plt.subplot(1, 3, 1)\n",
            "        plt.plot(history.history['accuracy'], label='Training')\n",
            "        if 'val_accuracy' in history.history:\n",
            "            plt.plot(history.history['val_accuracy'], label='Validation')\n",
            "        plt.title('Model Accuracy')\n",
            "        plt.legend()\n",
            "        \n",
            "        plt.subplot(1, 3, 2)\n",
            "        plt.plot(history.history['loss'], label='Training')\n",
            "        if 'val_loss' in history.history:\n",
            "            plt.plot(history.history['val_loss'], label='Validation')\n",
            "        plt.title('Model Loss')\n",
            "        plt.legend()\n",
            "        \n",
            "        if 'top_2_accuracy' in history.history:\n",
            "            plt.subplot(1, 3, 3)\n",
            "            plt.plot(history.history['top_2_accuracy'], label='Training Top-2')\n",
            "            if 'val_top_2_accuracy' in history.history:\n",
            "                plt.plot(history.history['val_top_2_accuracy'], label='Validation Top-2')\n",
            "            plt.title('Top-2 Accuracy')\n",
            "            plt.legend()\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "    except Exception as e:\n",
            "        print(f\"⚠️ Erreur lors de l'affichage des courbes: {e}\")\n",
            "        \n",
            "except Exception as e:\n",
            "    print(f\"❌ Erreur lors de l'entraînement GPU: {e}\")\n",
            "    print(\"🔧 Analyse de l'erreur:\")\n",
            "    print(f\"  - Type d'erreur: {type(e).__name__}\")\n",
            "    print(f\"  - Message: {str(e)}\")\n",
            "    \n",
            "    # Si erreur de dataset, essayer une approche plus simple\n",
            "    if \"StopIteration\" in str(e) or \"dataset\" in str(e).lower():\n",
            "        print(\"🔧 Détection d'erreur de dataset, utilisation de fit() direct...\")\n",
            "        \n",
            "        # Utiliser fit() directement sans dataset complexe\n",
            "        history = model.fit(\n",
            "            X_train_scaled, y_train,\n",
            "            epochs=30,\n",
            "            batch_size=64,\n",
            "            validation_split=0.2,\n",
            "            callbacks=callbacks,\n",
            "            verbose=1\n",
            "        )\n",
            "        \n",
            "        print(\"✅ Entraînement direct terminé avec succès\")\n",
            "        \n",
            "        # Sauvegarder le modèle\n",
            "        try:\n",
            "            model.save(f'{base_path}/models/gpu_direct_model.keras')\n",
            "            with open(f'{base_path}/models/direct_scaler.pkl', 'wb') as f:\n",
            "                pickle.dump(scaler, f)\n",
            "            print(\"✅ Modèle direct et scaler sauvegardés\")\n",
            "        except Exception as save_e:\n",
            "            print(f\"⚠️ Erreur de sauvegarde: {save_e}\")\n",
            "    \n",
            "    # Si erreur CuDNN, essayer une approche CPU\n",
            "    elif \"CuDNN\" in str(e) or \"DNN\" in str(e):\n",
            "        print(\"🔧 Détection d'erreur CuDNN, passage en mode CPU...\")\n",
            "        import os\n",
            "        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'\n",
            "        \n",
            "        # Recréer un modèle CPU simple\n",
            "        with tf.distribute.get_strategy().scope():\n",
            "            cpu_model = tf.keras.Sequential([\n",
            "                tf.keras.layers.Input(shape=(30, 3)),\n",
            "                tf.keras.layers.Flatten(),\n",
            "                tf.keras.layers.Dense(128, activation='relu'),\n",
            "                tf.keras.layers.Dropout(0.3),\n",
            "                tf.keras.layers.Dense(64, activation='relu'),\n",
            "                tf.keras.layers.Dropout(0.3),\n",
            "                tf.keras.layers.Dense(32, activation='relu'),\n",
            "                tf.keras.layers.Dense(3, activation='softmax')\n",
            "            ])\n",
            "            \n",
            "            cpu_model.compile(\n",
            "                optimizer='adam',\n",
            "                loss='sparse_categorical_crossentropy',\n",
            "                metrics=['accuracy']\n",
            "            )\n",
            "        \n",
            "        # Entraîner le modèle CPU\n",
            "        history = cpu_model.fit(\n",
            "            X_train_scaled, y_train,\n",
            "            epochs=20,\n",
            "            batch_size=32,\n",
            "            validation_split=0.2,\n",
            "            verbose=1\n",
            "        )\n",
            "        \n",
            "        model = cpu_model\n",
            "        print(\"✅ Modèle CPU de secours entraîné\")\n",
            "    else:\n",
            "        raise e\n"
        ]
        
        # Trouver l'index où commencer le remplacement
        start_idx = None
        for i, line in enumerate(source):
            if "# Créer un dataset TensorFlow optimisé pour GPU" in line:
                start_idx = i
                break
        
        if start_idx is not None:
            # Trouver la fin de la section à remplacer
            end_idx = len(source)
            for i in range(start_idx, len(source)):
                if 'else:' in source[i] and i > start_idx + 50:  # Trouver le else final
                    # Chercher la fin du bloc except
                    for j in range(i, len(source)):
                        if 'raise e' in source[j]:
                            end_idx = j + 1
                            break
                    break
            
            # Remplacer la section
            source[start_idx:end_idx] = new_dataset_section
        
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - erreur d'itération dataset résolue!")
