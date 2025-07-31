import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

progress_fixed = False
lstm_fixed = False

# 1) Corriger la cellule "Suivi de progression" (TypeError: sum(NoneType))
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "progress_tracker.json" in ''.join(cell['source']) and "check_progress" in ''.join(cell['source']):
        src = ''.join(cell['source'])
        # Normaliser les valeurs avant somme
        if "completed = sum(progress.values()) - 4" in src:
            src = src.replace(
                "completed = sum(progress.values()) - 4  # Exclure les métadonnées",
                "completed = sum(1 for k,v in progress.items() if isinstance(v, bool) and v)  # Compter uniquement True\n    total = len([k for k in default_progress.keys() if k.startswith('cell_')])"
            )
            # Supprimer l'ancienne ligne total (recalcul déjà inclus)
            src = src.replace("total = len(default_progress) - 4", "")
        # Sécuriser update_progress pour initialiser les clés manquantes
        if "def update_progress(cell_name):" in src and "progress[cell_name] = True" in src:
            src = src.replace(
                "progress[cell_name] = True",
                "if cell_name not in progress:\n        progress[cell_name] = True\n    else:\n        progress[cell_name] = True"
            )
        # S'assurer que default_progress utilise bien des bools
        if "'last_cell_executed': None" in src:
            # OK de garder, c'est un champ meta, on exclut du comptage
            pass
        # Réécrire la cellule
        cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in src.splitlines()]
        progress_fixed = True
        break

# 2) Forcer un LSTM CPU-compatible (désactiver CuDNN) et fallback Dense si GPU ne supporte pas DNN
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and ("lstm_main" in ''.join(cell['source']) or "tf.keras.layers.LSTM(" in ''.join(cell['source'])) and "model = tf.keras.Model" in ''.join(cell['source']):
        src = ''.join(cell['source'])
        # Insérer désactivation GPU au début de la construction du modèle si erreur DNN
        inject_guard = """
# Désactiver l'utilisation CuDNN en forçant CPU si nécessaire
import os
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
# Important: éviter le path CuDNN en fixant un device CPU pour LSTM
use_cpu_lstm = True
"""

        # Remplacer la définition LSTM pour forcer unchemin CPU (activation=linear, recurrent_activation=sigmoid, unroll=False, use_bias=True)
        src = src.replace(
            "tf.keras.layers.LSTM(",
            "tf.keras.layers.LSTM(\n        activation='tanh', recurrent_activation='sigmoid', use_bias=True, unit_forget_bias=True, unroll=False, time_major=False,"
        )

        # Ajouter un wrapper device CPU autour du LSTM si possible
        if "with strategy.scope():" in src and "lstm_main" in src:
            src = src.replace(
                "with strategy.scope():",
                "with strategy.scope():\n    " + inject_guard
            )
            # Remplacer directement la couche LSTM pour mettre sur CPU explicitement
            src = src.replace(
                "x = tf.keras.layers.LSTM(",
                "import tensorflow as tf\n    with tf.device('/CPU:0'):\n        x = tf.keras.layers.LSTM("
            )

        # Ajouter un try/except autour de l'entraînement pour fallback Dense-only si DNN not supported
        if "history = model.fit(" in src and "if \"DNN\" in str(e)" not in src:
            src = src.replace(
                "history = model.fit(",
                "try:\n        history = model.fit("
            )
            # Trouver fin de l'appel fit et ajouter except
            src_lines = src.splitlines()
            for i in range(len(src_lines)-1, -1, -1):
                if "plt.show()" in src_lines[i] or "print(\"✅ Entraînement terminé avec succès\")" in src_lines[i]:
                    insert_idx = i + 1
                    break
            else:
                insert_idx = len(src_lines)
            except_block = [
                "    except Exception as e:\n",
                "        if \"DNN\" in str(e) or \"CuDNN\" in str(e):\n",
                "            print(\"⚠️ DNN/CuDNN non supporté sur ce GPU. Passage à un modèle Dense-only CPU...\")\n",
                "            with tf.distribute.get_strategy().scope():\n",
                "                model = tf.keras.Sequential([\n",
                "                    tf.keras.layers.Input(shape=(30,3)),\n",
                "                    tf.keras.layers.Flatten(),\n",
                "                    tf.keras.layers.Dense(64, activation='relu'),\n",
                "                    tf.keras.layers.Dropout(0.3),\n",
                "                    tf.keras.layers.Dense(32, activation='relu'),\n",
                "                    tf.keras.layers.Dense(3, activation='softmax')\n",
                "                ])\n",
                "                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
                "            history = model.fit(\n",
                "                X_train_scaled, y_train,\n",
                "                epochs=10,\n",
                "                batch_size=64,\n",
                "                validation_split=0.2,\n",
                "                verbose=1\n",
                "            )\n",
                "        else:\n",
                "            raise\n"
            ]
            src_lines[insert_idx:insert_idx] = except_block
            src = "\n".join(src_lines)

        cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in src.splitlines()]
        lstm_fixed = True
        break

with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook mis à jour - Suivi progress: {progress_fixed}, LSTM CPU fallback/DNN-safe: {lstm_fixed}")
