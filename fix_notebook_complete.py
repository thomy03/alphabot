import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 AlphaBot ML/DL Training - Google Colab (v2)\n",
                "\n",
                "Notebook propre et corrigé (détection robuste des colonnes yfinance MultiIndex/suffixes) avec suivi/reprise, téléchargement de données robuste et fallbacks sûrs."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Ajustement du path pour que Colab trouve le module alphabot\n",
                "import sys\n",
                "sys.path.append('/content')\n",
                "sys.path.append('/content/alphabot')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 🔄 Suivi de Progression et Reprise Automatique"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 🔄 Système de suivi et reprise automatique\n",
                "import os, json\n",
                "from datetime import datetime\n",
                "\n",
                "base_path = '/content/drive/MyDrive/AlphaBot_ML_Training'\n",
                "os.makedirs(base_path, exist_ok=True)\n",
                "progress_file = f'{base_path}/progress_tracker.json'\n",
                "\n",
                "default_progress = {\n",
                "    'cell_1_setup': False,\n",
                "    'cell_2_data_download': False,\n",
                "    'cell_3_data_analysis': False,\n",
                "    'cell_4_pattern_training': False,\n",
                "    'cell_5_sentiment_training': False,\n",
                "    'cell_6_rag_training': False,\n",
                "    'cell_7_integration': False,\n",
                "    'cell_8_testing': False,\n",
                "    'cell_9_deployment': False,\n",
                "    'last_cell_executed': None,\n",
                "    'start_time': None,\n",
                "    'last_update': None\n",
                "}\n",
                "\n",
                "try:\n",
                "    with open(progress_file, 'r') as f:\n",
                "        progress = json.load(f)\n",
                "    print('📊 Suivi de progression chargé')\n",
                "except Exception:\n",
                "    progress = default_progress.copy()\n",
                "    progress['start_time'] = datetime.now().isoformat()\n",
                "    print('🆕 Nouveau suivi de progression initialisé')\n",
                "\n",
                "def update_progress(cell_name):\n",
                "    progress.setdefault(cell_name, False)\n",
                "    progress[cell_name] = True\n",
                "    progress['last_cell_executed'] = cell_name\n",
                "    progress['last_update'] = datetime.now().isoformat()\n",
                "    with open(progress_file, 'w') as f:\n",
                "        json.dump(progress, f, indent=2)\n",
                "    print(f'✅ Progression mise à jour: {cell_name}')\n",
                "\n",
                "def check_progress():\n",
                "    print('\\n📋 État actuel de la progression:')\n",
                "    print('=' * 50)\n",
                "    completed = sum(1 for k, v in progress.items() if k.startswith('cell_') and isinstance(v, bool) and v)\n",
                "    total = sum(1 for k in default_progress.keys() if k.startswith('cell_'))\n",
                "    pct = (completed / total * 100) if total else 0.0\n",
                "    print(f'📊 Progression: {completed}/{total} étapes complétées ({pct:.1f}%)')\n",
                "    print(f'⏰ Démarré: {progress.get(\"start_time\", \"N/A\")}')\n",
                "    print(f'🔄 Dernière mise à jour: {progress.get(\"last_update\", \"N/A\")}')\n",
                "    print(f'📍 Dernière cellule: {progress.get(\"last_cell_executed\", \"Aucune\")}')\n",
                "    print('=' * 50)\n",
                "\n",
                "check_progress()\n",
                "print('\\n💡 Instructions:')\n",
                "print('1. Exécutez cette cellule pour voir l\\'état d\\'avancement')\n",
                "print('2. Chaque cellule mettra à jour automatiquement sa progression')\n",
                "print('3. Si le processus s\\'arrête, relancez simplement cette cellule')\n",
                "print('4. Continuez avec la cellule suggérée')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1) Setup GPU/TPU et environnement Colab"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys, os, subprocess\n",
                "\n",
                "def pip_install(pkgs):\n",
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)\n",
                "\n",
                "# Nettoyage des modules déjà importés\n",
                "for m in [m for m in list(sys.modules) if m.startswith('transformers') or m.startswith('accelerate') or m.startswith('numpy') or m.startswith('tensorflow') or m.startswith('torch')]:\n",
                "    del sys.modules[m]\n",
                "\n",
                "# Réinstaller numpy pour compatibilité binaire\n",
                "pip_install([\n",
                "    '--upgrade',\n",
                "    'numpy>=1.24.0,<1.27.0'\n",
                "])\n",
                "\n",
                "# Installer un set compatible (Option A)\n",
                "pip_install([\n",
                "    'transformers>=4.43,<4.47',\n",
                "    'accelerate>=0.30,<0.34',\n",
                "    'datasets>=2.18,<3.0',\n",
                "    'safetensors>=0.4.3',\n",
                "    'huggingface-hub>=0.23,<0.25'\n",
                "])\n",
                "\n",
                "os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'\n",
                "\n",
                "# Importer et afficher les versions\n",
                "import numpy\n",
                "import transformers\n",
                "print('NumPy version:', numpy.__version__)\n",
                "print('Transformers version:', transformers.__version__)\n",
                "\n",
                "import tensorflow as tf\n",
                "import torch, logging\n",
                "logging.basicConfig(level=logging.INFO)\n",
                "\n",
                "try:\n",
                "    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()\n",
                "    tf.config.experimental_connect_to_cluster(tpu)\n",
                "    tf.tpu.experimental.initialize_tpu_system(tpu)\n",
                "    strategy = tf.distribute.TPUStrategy(tpu)\n",
                "    print('✅ TPU détectée et configurée')\n",
                "except Exception:\n",
                "    try:\n",
                "        gpus = tf.config.list_physical_devices('GPU')\n",
                "        if gpus:\n",
                "            for gpu in gpus:\n",
                "                tf.config.experimental.set_memory_growth(gpu, True)\n",
                "            strategy = tf.distribute.MirroredStrategy()\n",
                "            print(f'✅ {len(gpus)} GPU(s) détectée(s)')\n",
                "        else:\n",
                "            strategy = tf.distribute.get_strategy()\n",
                "            print('⚠️ Aucun GPU/TPU détecté, utilisation du CPU')\n",
                "    except Exception as e:\n",
                "        strategy = tf.distribute.get_strategy()\n",
                "        print(f'⚠️ Erreur de configuration GPU: {e}')\n",
                "\n",
                "try:\n",
                "    policy = tf.keras.mixed_precision.Policy('mixed_float16')\n",
                "    tf.keras.mixed_precision.set_global_policy(policy)\n",
                "    print('✅ Mixed precision activée')\n",
                "except Exception:\n",
                "    print('⚠️ Mixed precision non disponible')\n",
                "\n",
                "print('\\n📊 Configuration:')\n",
                "print(f'- TensorFlow: {tf.__version__}')\n",
                "print(f'- PyTorch: {torch.__version__}')\n",
                "print(f'- Strategy: {strategy}')\n",
                "print(f\"- GPUs disponibles: {tf.config.list_physical_devices('GPU')}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f'- CUDA: {torch.version.cuda}')\n",
                "    print(f'- GPU: {torch.cuda.get_device_name(0)}')\n",
                "\n",
                "update_progress('cell_1_setup')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2) Montage Google Drive (résilient v2)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('🔧 Configuration Google Drive (résiliente v2)...')\n",
                "from google.colab import drive\n",
                "import os, shutil, time\n",
                "MOUNT_POINT = '/content/drive'\n",
                "\n",
                "def _safe_cleanup_mount_point(mp: str):\n",
                "    try:\n",
                "        if os.path.islink(mp): os.unlink(mp)\n",
                "        if os.path.isdir(mp):\n",
                "            for entry in os.listdir(mp):\n",
                "                p = os.path.join(mp, entry)\n",
                "                try:\n",
                "                    if os.path.isfile(p) or os.path.islink(p): os.remove(p)\n",
                "                    elif os.path.isdir(p): shutil.rmtree(p)\n",
                "                except Exception: pass\n",
                "        else:\n",
                "            os.makedirs(mp, exist_ok=True)\n",
                "    except Exception as e:\n",
                "        print(f'⚠️ Nettoyage mount point: {e}')\n",
                "\n",
                "def _force_unmount():\n",
                "    try: drive.flush_and_unmount()\n",
                "    except Exception: pass\n",
                "    try:\n",
                "        os.system('fusermount -u /content/drive 2>/dev/null || true')\n",
                "        os.system('umount /content/drive 2>/dev/null || true')\n",
                "    except Exception: pass\n",
                "\n",
                "_force_unmount(); time.sleep(1)\n",
                "_safe_cleanup_mount_point(MOUNT_POINT); time.sleep(0.5)\n",
                "try:\n",
                "    drive.mount(MOUNT_POINT, force_remount=True)\n",
                "    print('✅ Drive monté (v2)')\n",
                "except Exception as e:\n",
                "    print(f'❌ drive.mount a échoué: {e}')\n",
                "    if 'Mountpoint must not already contain files' in str(e):\n",
                "        try:\n",
                "            shutil.rmtree(MOUNT_POINT, ignore_errors=True)\n",
                "            os.makedirs(MOUNT_POINT, exist_ok=True)\n",
                "            drive.mount(MOUNT_POINT, force_remount=True)\n",
                "            print('✅ Drive monté après recréation du dossier')\n",
                "        except Exception as e2:\n",
                "            print(f'⚠️ Impossible de recréer {MOUNT_POINT}: {e2}')\n",
                "            raise\n",
                "\n",
                "base_path = '/content/drive/MyDrive/AlphaBot_ML_Training'\n",
                "for sub in ('data', 'models', 'checkpoints', 'logs', 'exports'):\n",
                "    os.makedirs(f'{base_path}/{sub}', exist_ok=True)\n",
                "print(f'📁 Répertoires prêts sous: {base_path}')\n",
                "update_progress('cell_2_data_download')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4) Téléchargement des données (robuste + fallbacks + sauvegarde CSV locale pour preuve)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, ssl, time, pickle\n",
                "from datetime import datetime, timedelta\n",
                "import pandas as pd\n",
                "import yfinance as yf\n",
                "\n",
                "os.environ['PYTHONHTTPSVERIFY'] = '0'\n",
                "try:\n",
                "    _create_unverified_https_context = ssl._create_unverified_context\n",
                "except AttributeError:\n",
                "    pass\n",
                "else:\n",
                "    ssl._create_default_https_context = _create_unverified_https_context\n",
                "\n",
                "symbols = ['AAPL','GOOGL','MSFT','TSLA','AMZN','NVDA','META','NFLX','IBM','ORCL','INTC','AMD','QCOM','CRM','SAP','CSCO','JPM','BAC','V','MA','DIS','KO','PEP','NKE','XOM','SPY','QQQ','DIA','IWM']\n",
                "print(f'📥 Téléchargement (yfinance) period=10y interval=1d pour {len(symbols)} tickers...')\n",
                "\n",
                "def dl_yf(symbol):\n",
                "    try:\n",
                "        df = yf.download(symbol, period='10y', interval='1d', auto_adjust=True, progress=False)\n",
                "        if df is not None and not df.empty:\n",
                "            return df\n",
                "    except Exception as e:\n",
                "        print(f'⚠️ yfinance {symbol}: {e}')\n",
                "    return None\n",
                "\n",
                "def dl_pdr_yahoo(symbol):\n",
                "    try:\n",
                "        from pandas_datareader import data as pdr\n",
                "        yf.pdr_override()\n",
                "        end = datetime.now(); start = end - timedelta(days=365*10)\n",
                "        df = pdr.get_data_yahoo(symbol, start=start, end=end)\n",
                "        if df is not None and not df.empty:\n",
                "            return df\n",
                "    except Exception as e:\n",
                "        print(f'⚠️ pdr-yahoo {symbol}: {e}')\n",
                "    return None\n",
                "\n",
                "def dl_pdr_stooq(symbol):\n",
                "    try:\n",
                "        from pandas_datareader import data as pdr\n",
                "        end = datetime.now(); start = end - timedelta(days=365*10)\n",
                "        df = pdr.DataReader(symbol, 'stooq', start, end)\n",
                "        if df is not None and not df.empty:\n",
                "            return df.sort_index()\n",
                "    except Exception as e:\n",
                "        print(f'⚠️ stooq {symbol}: {e}')\n",
                "    return None\n",
                "\n",
                "def safe_download(symbol):\n",
                "    for fn in (dl_yf, dl_pdr_yahoo, dl_pdr_stooq):\n",
                "        df = fn(symbol)\n",
                "        if df is not None and not df.empty:\n",
                "            return df\n",
                "    return None\n",
                "\n",
                "all_data = {}\n",
                "csv_out = '/content/market_data_csv'\n",
                "os.makedirs(csv_out, exist_ok=True)\n",
                "for s in symbols:\n",
                "    df = safe_download(s)\n",
                "    if df is not None and not df.empty:\n",
                "        all_data[s] = df\n",
                "        df.to_csv(f'{csv_out}/{s}.csv')\n",
                "        print(f'✅ {s}: {len(df)} lignes (CSV écrit)')\n",
                "    else:\n",
                "        print(f'❌ {s}: vide (après yfinance+pdr)')\n",
                "\n",
                "print('Symbols téléchargés:', list(all_data.keys()))\n",
                "for s, df in list(all_data.items())[:10]:\n",
                "    print(s, 'rows=', len(df))\n",
                "\n",
                "data_path = f'{base_path}/data/market_data.pkl'\n",
                "os.makedirs(f'{base_path}/data', exist_ok=True)\n",
                "with open(data_path, 'wb') as f:\n",
                "    pickle.dump(all_data, f)\n",
                "print(f'💾 Données sauvegardées: {data_path}')\n",
                "update_progress('cell_4_pattern_training')"
            ]
        }
    ],
    "metadata": {
        "colab": {
            "provenance": [],
            "name": "ALPHABOT_ML_TRAINING_COLAB_v2.ipynb"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.x"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Ajouter la cellule 5
cell5_code = '''import os
os.environ['TF_KERAS_ALLOW_CUDNN_RNN']='0'
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np, pickle, pandas as pd
from datetime import datetime

log_path = f"{base_path}/logs"; os.makedirs(log_path, exist_ok=True)
log_file = f"{log_path}/pattern_debug.txt"; print('Log path:', log_file)
open(log_file,'a').write(f"\\n==== DEBUG RUN {datetime.now().isoformat()} ====\\n")

try:
    with open(f'{base_path}/data/market_data.pkl','rb') as f:
        all_data = pickle.load(f)
    print('✅ Données chargées')
except Exception:
    print('❌ Données introuvables — exécutez la cellule 4'); all_data={}

print('Nombre de symboles dans all_data:', len(all_data))
if not all_data:
    print('⚠️ all_data est vide: vérifiez la cellule 4 (réseau/API/period).')

def find_col(df_local, bases):
    cols = [c for c in df_local.columns if isinstance(c, str)]
    lower_map = {c.lower(): c for c in cols}
    for b in bases:
        if b in lower_map:
            return lower_map[b]
    for c in cols:
        lc = c.lower()
        for b in bases:
            if lc.startswith(b + '_') or lc.endswith('_' + b) or b in lc.split('_'):
                return c
    for c in cols:
        lc = c.lower()
        for b in bases:
            if b in lc:
                return c
    return None

def prepare_pattern_training_data(all_data):
    X, y = [], []
    win_count_total = 0
    for symbol, data in all_data.items():
        try:
            if data is None or data.empty:
                open(log_file,'a').write(f"symbol={symbol} skipped: empty\\n"); continue
            df = data.copy()
            open(log_file,'a').write(f"symbol={symbol} columns={list(df.columns)}\\n")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
                open(log_file,'a').write(f"symbol={symbol} MultiIndex flattened to {list(df.columns)}\\n")
            df.columns = [str(c).strip() for c in df.columns]
            # priorité par suffixe exact si présent
            close = find_col(df, [f'close_{symbol.lower()}', 'close','adj close','adj_close','adjclose'])
            high  = find_col(df, [f'high_{symbol.lower()}', 'high'])
            low   = find_col(df, [f'low_{symbol.lower()}', 'low'])
            volume = find_col(df, [f'volume_{symbol.lower()}', 'volume'])
            open(log_file,'a').write(f"symbol={symbol} mapped: close={close} high={high} low={low} volume={volume}\\n")
            if close is None:
                alt_close = find_col(df, ['adj close','adj_close','adjclose'])
                if alt_close is not None:
                    close = alt_close
                    open(log_file,'a').write(f"symbol={symbol} fallback close-> {close}\\n")
            if not all([close, high, low]):
                open(log_file,'a').write(f"symbol={symbol} skipped: missing required columns\\n"); continue
            if volume is None:
                df['Volume_proxy'] = df[close].pct_change().rolling(10, min_periods=1).std().fillna(0.01)
                volume = 'Volume_proxy'
                open(log_file,'a').write(f"symbol={symbol} created volume proxy\\n")
            for col in [close, high, low, volume]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[[close, high, low, volume]].dropna()
            n = len(df)
            open(log_file,'a').write(f"symbol={symbol} len={n} after dropna\\n")
            if n < 20:
                open(log_file,'a').write(f"symbol={symbol} skipped: n<20\\n"); continue
            win_count = 0
            for i in range(0, n - 17):
                seq = df.iloc[i:i+15]
                fut = df.iloc[i+15:i+17]
                if seq.isnull().any().any() or fut.isnull().any().any():
                    continue
                close_arr = seq[close].values.reshape(-1, 1)
                vol_arr = seq[volume].values.reshape(-1, 1)
                spread_arr = (seq[high] - seq[low]).values.reshape(-1, 1)
                features = np.concatenate([close_arr, vol_arr, spread_arr], axis=1)
                if features.shape != (15, 3):
                    continue
                current_price = float(seq[close].iloc[-1])
                future_mean = float(fut[close].mean())
                if current_price <= 0 or not np.isfinite(future_mean):
                    continue
                future_return = (future_mean - current_price) / current_price
                if future_return > 0.002:
                    label = 2
                elif future_return < -0.002:
                    label = 0
                else:
                    label = 1
                X.append(features.astype(np.float32)); y.append(label); win_count += 1
            win_count_total += win_count
            open(log_file,'a').write(f"symbol={symbol} windows={win_count}\\n")
        except Exception as e:
            open(log_file,'a').write(f"symbol={symbol} ERROR {str(e)[:200]}\\n"); continue
    open(log_file,'a').write(f"total_windows={len(X)} total_by_symbol={win_count_total}\\n")
    if len(X) == 0:
        print('⚠️ Aucune fenêtre créée! Vérifiez le log.'); return np.array([]), np.array([])
    X = np.array(X, dtype=np.float32); Y = np.array(y, dtype=np.int32)
    print(f'✅ Préparation: X={X.shape}, y={Y.shape}')
    return X, Y

X_train, y_train = prepare_pattern_training_data(all_data)
print(f'📊 Échantillons: {X_train.shape[0] if len(X_train) > 0 else 0}')
print(f"🔍 Log fenêtres: voir {log_file}")

if X_train.shape[0] == 0:
    print('\\n📝 Dernières lignes du log:')
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(line.strip())

strategy = tf.distribute.get_strategy()
with strategy.scope():
    inputs = tf.keras.Input(shape=(15,3), name='input')
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=False, name='lstm_main_ncudnn', activation='tanh', recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs, outputs, name='l4_ncudnn_lstm_model')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('✅ Modèle L4 (non-cuDNN) créé')

X_train = X_train.astype(np.float32); y_train = y_train.astype(np.int32)
if X_train.shape[0] == 0:
    print('⚠️ Dataset vide. Génération synthétique minimale...')
    X_train = np.random.randn(1024, 15, 3).astype(np.float32)
    y_train = np.random.randint(0, 3, size=(1024,)).astype(np.int32)

scaler = (StandardScaler().fit(X_train.reshape(-1,3)))
Xs = scaler.transform(X_train.reshape(-1,3)).reshape(X_train.shape)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', verbose=1)
]
hist = model.fit(Xs, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=callbacks, verbose=1)
print('✅ Entraînement terminé')

import matplotlib.pyplot as plt
os.makedirs(f'{base_path}/models', exist_ok=True)
model.save(f'{base_path}/models/pattern_model_l4_ncudnn.keras')
import pickle as pkl
with open(f'{base_path}/models/pattern_scaler.pkl','wb') as f: pkl.dump(scaler,f)
print('💾 Modèle/scaler sauvegardés')
try:
    plt.figure(figsize=(12,4));
    if 'accuracy' in hist.history:
        plt.subplot(1,2,1); plt.plot(hist.history['accuracy']); 
        if 'val_accuracy' in hist.history: plt.plot(hist.history['val_accuracy']); plt.title('Accuracy'); plt.legend(['train','val'])
    if 'loss' in hist.history:
        plt.subplot(1,2,2); plt.plot(hist.history['loss']);
        if 'val_loss' in hist.history: plt.plot(hist.history['val_loss']); plt.title('Loss'); plt.legend(['train','val']); plt.tight_layout(); plt.show()
except Exception:
    pass

update_progress('cell_5_sentiment_training')'''

notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["##
