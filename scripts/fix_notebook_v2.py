import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 4 (téléchargement des données)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 4: Téléchargement des données' in ''.join(cell['source']):
        source = cell['source']
        for i, line in enumerate(source):
            if 'print(f"- Prix moyen: ${sample_data[\'Close\'].mean():.2f}")' in line:
                source[i] = '    print(f"- Prix moyen: ${float(sample_data[\'Close\'].mean()):.2f}")\n'
            elif 'print(f"- Volatilité: {sample_data[\'Close\'].pct_change().std()*100:.2f}%")' in line:
                source[i] = '    print(f"- Volatilité: {float(sample_data[\'Close\'].pct_change().std()*100):.2f}%")\n'
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès!")
