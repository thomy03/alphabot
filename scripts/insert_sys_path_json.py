import json

NOTEBOOK_PATH = "ALPHABOT_ML_TRAINING_COLAB.ipynb"

def main():
    # Charger le notebook existant
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    # Créer la cellule d'ajustement du chemin
    sys_path_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('/content')\n",
            "sys.path.append('/content/alphabot')\n"
        ]
    }
    # Insérer en première position
    nb["cells"].insert(0, sys_path_cell)
    # Sauvegarder le notebook modifié
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Cellule sys.path insérée en tête de {NOTEBOOK_PATH}")

if __name__ == "__main__":
    main()
