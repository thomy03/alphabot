import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Corriger l'erreur de formatage dans la cellule 4
cell_4_source = nb['cells'][4]['source']
for i, line in enumerate(cell_4_source):
    if 'print(f"- Prix moyen: ${sample_data[\'Close\'].mean():.2f}")' in line:
        cell_4_source[i] = '    print(f"- Prix moyen: ${float(sample_data[\'Close\'].mean()):.2f}")\n'
    if 'print(f"- Volatilité: {sample_data[\'Close\'].pct_change().std()*100:.2f}%")' in line:
        cell_4_source[i] = '    print(f"- Volatilité: {float(sample_data[\'Close\'].pct_change().std()*100):.2f}%")\n'

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès!")
