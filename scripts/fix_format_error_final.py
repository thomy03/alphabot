import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Trouver et corriger la cellule 4 (Téléchargement des données)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELLULE 4: Téléchargement des données' in ''.join(cell['source']):
        source = cell['source']
        
        # Remplacer la section d'affichage des statistiques
        new_stats_section = [
            "    # Calculer et afficher les statistiques sans warnings\n",
            "    mean_price = float(sample_data['Close'].mean())\n",
            "    volatility = float(sample_data['Close'].pct_change().std() * 100)\n",
            "    \n",
            "    print(f\"- Prix moyen: ${mean_price:.2f}\")\n",
            "    print(f\"- Volatilité: {volatility:.2f}%\")\n"
        ]
        
        # Trouver l'index où commencer le remplacement
        start_idx = None
        for i, line in enumerate(source):
            if "# Calculer et afficher les statistiques sans warnings" in line:
                start_idx = i
                break
        
        if start_idx is not None:
            # Trouver la fin de la section à remplacer
            end_idx = start_idx + 6  # Remplacer les 6 lignes de statistiques
            
            # Remplacer la section
            source[start_idx:end_idx] = new_stats_section
        
        break

# Sauvegarder le notebook corrigé
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook corrigé avec succès - erreur de formatage résolue!")
