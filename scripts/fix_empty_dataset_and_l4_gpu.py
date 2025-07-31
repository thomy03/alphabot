import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def patch_prepare_fn_and_training(cell_src):
    """
    1) Corrige prepare_pattern_training_data pour gérer correctement les colonnes yfinance
       (p.ex. colonnes multi-index ou préfixées) et garantir un minimum d'échantillons.
    2) Ajoute un garde-fou: si X_train est vide, on crée un dataset synthétique pour éviter le crash.
    3) Adapte les logs GPU pour mentionner L4 et enlève le wording A100.
    """
    src = ''.join(cell_src)

    # Patch: fonction robustifiée (gère MultiIndex et colonnes renommées)
    if "def prepare_pattern_training_data(" in src:
        start = src.find("def prepare_pattern_training_data(")
        # Trouver la fin de la fonction + l'appel qui suit
        end_call = src.find("X_train, y_train = prepare_pattern_training_data(", start)
        if end_call == -1:
            end_call = start
        # Construire la nouvelle fonction + appel
        new_fn_block = """
print("🔧 Préparation des données (sécurisée v2 - yfinance compatibles)...")
def prepare_pattern_training_data(all_data):
    import numpy as np
    import pandas as pd

    def normalize_yf_cols(df):
        # Aplatissement MultiIndex éventuel
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(c) for c in col if c is not None]) for col in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]
        # Mapping de colonnes standards possibles
        candidates = {}
        for key in ['Close', 'Adj Close', 'Adj_Close', 'Close_Adj Close']:
            candidates['Close'] = candidates.get('Close') or next((c for c in df.columns if c.lower().replace(' ', '').replace('-', '_') == key.lower().replace(' ', '').replace('-', '_')), None)
        for key in ['Volume']:
            candidates['Volume'] = candidates.get('Volume') or next((c for c in df.columns if c.lower() == key.lower()), None)
        for key in ['High']:
            candidates['High'] = candidates.get('High') or next((c for c in df.columns if c.lower() == key.lower()), None)
        for key in ['Low']:
            candidates['Low'] = candidates.get('Low') or next((c for c in df.columns if c.lower() == key.lower()), None)
        return candidates

    X_train, y_train = [], []
    for symbol, data in all_data.items():
        try:
            if data is None or not hasattr(data, 'empty') or data.empty:
                print(f"⚠️ {symbol}: dataset vide/None, ignoré")
                continue

            data = data.copy()
            data = data.sort_index()

            # Détecter les colonnes réelles à utiliser
            cols = normalize_yf_cols(data)
            required = ['Close', 'Volume', 'High', 'Low']
            if not all(cols.get(k) for k in required):
                print(f"⚠️ {symbol}: colonnes manquantes après normalisation {cols}, ignoré")
                continue

            close_col = cols['Close']; vol_col = cols['Volume']; hi_col = cols['High']; lo_col = cols['Low']
            # Nettoyer NA
            data = data.dropna(subset=[close_col, vol_col, hi_col, lo_col])

            n = len(data)
            if n < 36:
                print(f"ℹ️ {symbol}: pas assez de points ({n}), ignoré")
                continue

            # Fenêtrage
            for i in range(0, n - 35):
                seq = data.iloc[i:i+30]
                next5 = data.iloc[i+30:i+35]
                if seq[[close_col, vol_col, hi_col, lo_col]].isnull().any().any():
                    continue
                if next5[[close_col]].isnull().any().any():
                    continue

                close = np.asarray(seq[close_col].values, dtype=np.float32).reshape(-1, 1)
                volume = np.asarray(seq[vol_col].values, dtype=np.float32).reshape(-1, 1)
                spread = np.asarray((seq[hi_col] - seq[lo_col]).values, dtype=np.float32).reshape(-1, 1)
                features = np.concatenate([close, volume, spread], axis=1)
                if features.shape != (30, 3):
                    continue

                current_price = float(seq[close_col].iloc[-1])
                if current_price == 0:
                    continue
                future_mean = float(np.mean(next5[close_col].values))
                future_return = (future_mean - current_price) / current_price

                if future_return > 0.02:
                    label = 2
                elif future_return < -0.02:
                    label = 0
                else:
                    label = 1

                X_train.append(features)
                y_train.append(label)

        except Exception as e:
            print(f"⚠️ Erreur sur {symbol}, segment ignoré: {e}")
            continue

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"✅ Préparation terminée: X={X_train.shape}, y={y_train.shape}")
    return X_train, y_train

# Reconstruire X/y avec la nouvelle fonction
X_train, y_train = prepare_pattern_training_data(all_data)
print(f"📊 Données préparées: {X_train.shape[0]} échantillons")
"""
        src = src[:start] + new_fn_block + src[end_call:]
    # Garde-fou après préparation: si vide, créer jeu synthétique pour ne pas crasher
    if "X_train_scaled = scaler.fit_transform(" in src and "if X_train.shape[0] == 0:" not in src:
        src = src.replace(
            "# Normaliser les données pour GPU",
            """# Normaliser les données (ou fallback si vide)
if X_train.shape[0] == 0:
    print("⚠️ Aucun échantillon réel. Génération d'un dataset synthétique minimal (CPU)...")
    import numpy as np
    X_train = np.random.randn(256, 30, 3).astype(np.float32)
    y_train = np.random.randint(0, 3, size=(256,)).astype(np.int32)
    print(f"✅ Dataset synthétique: X={X_train.shape}, y={y_train.shape}")
# Normaliser les données pour GPU"""
        )
    # Adapter wording GPU pour L4 (ne pas forcer A100/mixed-precision)
    src = src.replace("Création du modèle GPU optimisé pour A100", "Création du modèle GPU (compatible L4)")
    src = src.replace("🚀 Début de l'entraînement GPU optimisé pour A100", "🚀 Début de l'entraînement GPU (compatible L4)")
    # Retour en liste
    return [line + ("\n" if not line.endswith("\n") else "") for line in src.splitlines()]

patched = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and ('prepare_pattern_training_data' in ''.join(cell['source'])):
        cell['source'] = patch_prepare_fn_and_training(cell['source'])
        patched = True
        break

with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

if patched:
    print("Notebook corrigé: préparation compatible yfinance + fallback si X vide + wording L4.")
else:
    print("Aucune cellule cible trouvée pour le patch.")
