#!/usr/bin/env python3
# Script pour réorganiser le notebook: déplacer les cellules de test à la fin

import json
import shutil
from datetime import datetime

def reorganize_notebook():
    # Créer une sauvegarde
    backup_name = f'ALPHABOT_ML_TRAINING_COLAB_v2_backup_reorganize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb'
    shutil.copy('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', backup_name)
    print(f"Sauvegarde créée: {backup_name}")
    
    # Lire le fichier
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    print(f"Total cellules: {len(cells)}")
    
    # Identifier les cellules de test (celles qui contiennent les mots-clés spécifiques)
    test_cells = []
    original_cells = []
    
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        
        # Identifier les cellules de test
        is_test_cell = (
            ('Préparation tests V2' in src) or
            ('Montage Drive et ajout du repo au PYTHONPATH' in src) or
            ('Génération champions.json de test' in src) or
            ('Test V2: ModelSelectorV2' in src) or
            ('champions.json' in src and 'test' in src.lower()) or
            ('ModelSelectorV2' in src and 'HybridOrchestrator' in src)
        )
        
        if is_test_cell:
            test_cells.append(cell)
            print(f"Cellule de test trouvée [{i}]: {cell.get('cell_type')} - {src.strip().splitlines()[0] if src.strip() else 'Vide'}")
        else:
            original_cells.append(cell)
    
    print(f"\nCellules originales: {len(original_cells)}")
    print(f"Cellules de test: {len(test_cells)}")
    
    # Réorganiser: cellules originales d'abord, puis cellules de test à la fin
    new_cells = original_cells + test_cells
    
    # Renommer les sections de test pour qu'elles soient numérotées à la fin
    section_num = 9
    for cell in test_cells:
        if cell.get('cell_type') == 'markdown':
            src = ''.join(cell.get('source', []))
            if 'Préparation tests V2' in src:
                cell['source'] = [f'## {section_num}) Préparation tests V2 (montage et chemins)\n']
                section_num += 1
            elif 'Génération champions' in src:
                cell['source'] = [f'## {section_num}) Génération champions.json de test\n']  
                section_num += 1
            elif 'Test V2' in src:
                cell['source'] = [f'## {section_num}) Test V2: ModelSelectorV2 et HybridOrchestrator\n']
                section_num += 1
    
    # Mettre à jour le notebook
    nb['cells'] = new_cells
    
    # Sauvegarder
    with open('ALPHABOT_ML_TRAINING_COLAB_v2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Notebook réorganisé: {len(new_cells)} cellules")
    
    # Vérification
    print("\n--- 3 premières cellules ---")
    for i, c in enumerate(new_cells[:3]):
        src = ''.join(c.get('source', []))
        first = src.strip().splitlines()[0] if src.strip() else 'Vide'
        print(f"[{i}] {c.get('cell_type')} | {first[:100]}")
    
    print("\n--- 3 dernières cellules ---")
    for i, c in enumerate(new_cells[-3:]):
        src = ''.join(c.get('source', []))
        first = src.strip().splitlines()[0] if src.strip() else 'Vide'
        print(f"[{len(new_cells)-3+i}] {c.get('cell_type')} | {first[:100]}")
    
    return True

if __name__ == "__main__":
    try:
        reorganize_notebook()
        print("\n✅ Réorganisation terminée!")
        print("Maintenant le notebook commence par les cellules d'entraînement ML")
        print("et les cellules de test sont à la fin (sections 9, 10, 11)")
    except Exception as e:
        print(f"❌ Erreur: {e}")
