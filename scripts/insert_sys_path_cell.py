import nbformat
import sys

NOTEBOOK_PATH = "ALPHABOT_ML_TRAINING_COLAB.ipynb"

def main():
    # Charger le notebook existant
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    # Créer la cellule d'ajustement du chemin
    sys_path_cell = nbformat.v4.new_code_cell(source=(
        "import sys\n"
        "sys.path.append('/content')\n"
        "sys.path.append('/content/alphabot')\n"
    ))
    # Insérer en première position
    nb.cells.insert(0, sys_path_cell)
    # Sauvegarder le notebook modifié
    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Cellule sys.path insérée en tête de {NOTEBOOK_PATH}")

if __name__ == "__main__":
    main()
