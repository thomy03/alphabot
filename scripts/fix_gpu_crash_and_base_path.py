import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Corriger la cellule 5 - Simplifier le modèle GPU pour éviter les crashs
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 5: Entraînement Pattern Detector' in ''.join(cell['source']):
        source = cell['source']
        
        # Remplacer la configuration mixed precision qui peut causer des crashs
        for i, line in enumerate(source):
            if "tf.keras.mixed_precision.set_global_policy('mixed_float16')" in line:
                source[i] = "# tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Désactivé pour éviter les crashs\n"
                break
        
        # Modifier la création du modèle pour le rendre plus simple
        model_start_idx = None
        for i, line in enumerate(source):
            if "with strategy.scope():" in line and "Utiliser un modèle LSTM/CNN hybride" in source[i+1]:
                model_start_idx = i
                break
        
        if model_start_idx is not None:
            # Nouveau modèle simplifié
            new_model = [
                "with strategy.scope():\n",
                "    # Utiliser un modèle plus simple pour éviter les crashs GPU\n",
                "    inputs = tf.keras.Input(shape=(30, 3), name='input_layer')\n",
                "    \n",
                "    # Normalisation\n",
                "    x = tf.keras.layers.BatchNormalization()(inputs)\n",
                "    \n",
                "    # Une seule couche LSTM\n",
                "    x = tf.keras.layers.LSTM(\n",
                "        64, \n",
                "        return_sequences=False,\n",
                "        kernel_initializer='glorot_uniform',\n",
                "        recurrent_initializer='orthogonal',\n",
                "        name='lstm_main'\n",
                "    )(x)\n",
                "    x = tf.keras.layers.Dropout(0.3)(x)\n",
                "    \n",
                "    # Couches denses\n",
                "    x = tf.keras.layers.Dense(32, activation='relu')(x)\n",
                "    x = tf.keras.layers.BatchNormalization()(x)\n",
                "    x = tf.keras.layers.Dropout(0.3)(x)\n",
                "    \n",
                "    outputs = tf.keras.layers.Dense(3, activation='softmax', name='output')(x)\n",
                "    \n",
                "    # Créer le modèle\n",
                "    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='simplified_gpu_model')\n",
                "    \n",
                "    # Compiler avec des paramètres simples\n",
                "    model.compile(\n",
                "        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\n",
                "        loss='sparse_categorical_crossentropy',\n",
                "        metrics=['accuracy']\n",
                "    )\n"
            ]
            
            # Trouver la fin du bloc model
            model_end_idx = model_start_idx + 1
            brace_count = 1
            for i in range(model_start_idx + 1, len(source)):
                if "with strategy.scope():" in source[i]:
                    brace_count += 1
                if "model.compile(" in source[i]:
                    # Chercher la fin du compile
                    for j in range(i, len(source)):
                        if source[j].strip().endswith(")"):
                            model_end_idx = j + 1
                            break
                    break
            
            # Remplacer le modèle
            source[model_start_idx:model_end_idx] = new_model
        
        # Simplifier les callbacks
        callbacks_idx = None
        for i, line in enumerate(source):
            if "# Callbacks avancés pour GPU" in line:
                callbacks_idx = i
                break
        
        if callbacks_idx is not None:
            new_callbacks = [
                "# Callbacks simplifiés pour éviter les crashs\n",
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
                "]\n"
            ]
            
            # Trouver la fin des callbacks
            callbacks_end_idx = callbacks_idx + 1
            for i in range(callbacks_idx + 1, len(source)):
                if source[i].strip() == "]":
                    callbacks_end_idx = i + 1
                    break
            
            source[callbacks_idx:callbacks_end_idx] = new_callbacks
        
        # Réduire le nombre d'epochs
        for i, line in enumerate(source):
            if "epochs=50," in line:
                source[i] = "        epochs=20,  # Réduit pour éviter les crashs\n"
            elif "epochs=30," in line:
                source[i] = "            epochs=15,  # Réduit pour éviter les crashs\n"
        
        break

# Corriger la cellule 6 - Ajouter base_path
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 6: Entraînement Sentiment Analyzer' in ''.join(cell['source']):
        source = cell['source']
        
        # Ajouter la définition de base_path au début
        base_path_definition = [
            "# Définir le chemin de base\n",
            "base_path = '/content/drive/MyDrive/AlphaBot_ML_Training'\n",
            "os.makedirs(base_path, exist_ok=True)\n",
            "\n"
        ]
        
        # Insérer après les imports
        insert_idx = 0
        for i, line in enumerate(source):
            if "print(\"💭 Entraînement du Sentiment Analyzer" in line:
                insert_idx = i
                break
        
        source[insert_idx:insert_idx] = base_path_definition
        break

# Corriger toutes les autres cellules qui pourraient avoir besoin de base_path
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Vérifier si base_path est utilisé mais pas défini
        has_base_path_usage = any('base_path' in line for line in source)
        has_base_path_definition = any('base_path =' in line for line in source)
        
        if has_base_path_usage and not has_base_path_definition and 'CELLULE' in ''.join(source):
            # Ajouter la définition au début de la cellule
            base_path_definition = [
                "# Définir le chemin de base si pas déjà défini\n",
                "if 'base_path' not in globals():\n",
                "    base_path = '/content/drive/MyDrive/AlphaBot_ML_Training'\n",
                "    os.makedirs(base_path, exist_ok=True)\n",
                "\n"
            ]
            
            # Trouver où insérer (après le titre de cellule)
            insert_idx = 0
            for i, line in enumerate(source):
                if "CELLULE" in line and "#" in line:
                    insert_idx = i + 1
                    break
            
            source[insert_idx:insert_idx] = base_path_definition

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - modèle GPU simplifié et base_path défini!")
