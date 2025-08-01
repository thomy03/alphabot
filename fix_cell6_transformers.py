import json
from copy import deepcopy

nb_path = 'ALPHABOT_ML_TRAINING_COLAB_v2.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
assert len(cells) >= 6, 'Notebook ne contient pas 6 cellules minimum.'

cell6 = cells[5]

# Correctif Option A à insérer AU DÉBUT de la cellule 6
fix_lines = [
    "import sys, os, subprocess\n",
    "\n",
    "def pip_install(pkgs):\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)\n",
    "\n",
    "# Nettoyage des modules déjà importés\n",
    "for m in [m for m in list(sys.modules) if m.startswith('transformers') or m.startswith('accelerate')]:\n",
    "    del sys.modules[m]\n",
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
    "import transformers\n",
    "print('Transformers version:', transformers.__version__)\n",
    "\n"
]

# S'assurer que la cellule est de type code
if cell6.get('cell_type') != 'code':
    c = deepcopy(cell6)
    c['cell_type'] = 'code'
    c.setdefault('metadata', {})
    c['execution_count'] = None
    c['outputs'] = []
    c['source'] = []
    cell6 = c

source = cell6.get('source', [])

# Vérifier si le correctif est déjà présent
already_has_marker = any('Transformers version:' in (s if isinstance(s, str) else '') for s in source)

if not already_has_marker:
    # Insérer le correctif au début du contenu existant
    new_source = fix_lines + (['\n\n# --- Code existant de la cellule 6 ---\n'] if source else []) + source
    cell6['source'] = new_source
    cells[5] = cell6
    print('Correctif inséré au début de la cellule 6.')
else:
    print('Le correctif est déjà présent dans la cellule 6.')

# Écrire le notebook mis à jour
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print('Notebook mis à jour avec succès.')
