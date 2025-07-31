import json

# Charger le notebook
with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'drive.mount' in ''.join(cell['source']) and '/content/drive' in ''.join(cell['source']):
        # Remplace toute logique de montage par une version ultra-résiliente
        cell['source'] = [
            "print(\"🔧 Configuration Google Drive (résiliente v2)...\")\n",
            "from google.colab import drive\n",
            "import os, shutil, time\n",
            "\n",
            "MOUNT_POINT = '/content/drive'\n",
            "\n",
            "def _safe_cleanup_mount_point(mp: str):\n",
            "    try:\n",
            "        # Sécuriser: si bindé ou symlink, supprimer l'entrée\n",
            "        if os.path.islink(mp):\n",
            "            print(\"ℹ️ Le point de montage est un symlink — suppression...\")\n",
            "            os.unlink(mp)\n",  # remove symlink\n",
            "        # Si dossier existe et contient des fichiers résiduels locaux (pas Drive), on nettoie\n",
            "        if os.path.isdir(mp):\n",
            "            for entry in os.listdir(mp):\n",
            "                p = os.path.join(mp, entry)\n",
            "                try:\n",
            "                    if os.path.isfile(p) or os.path.islink(p):\n",
            "                        os.remove(p)\n",
            "                    elif os.path.isdir(p):\n",
            "                        shutil.rmtree(p)\n",
            "                except Exception as e:\n",
            "                    print(f\"⚠️ Ignoré pendant nettoyage: {p} -> {e}\")\n",
            "        else:\n",
            "            os.makedirs(mp, exist_ok=True)\n",
            "    except Exception as e:\n",
            "        print(f\"⚠️ Problème nettoyage mount point: {e}\")\n",
            "\n",
            "def _force_unmount():\n",
            "    try:\n",
            "        drive.flush_and_unmount()\n",
            "        print(\"ℹ️ flush_and_unmount exécuté\")\n",
            "    except Exception as e:\n",
            "        print(f\"ℹ️ flush_and_unmount non nécessaire: {e}\")\n",
            "    # En plus, tenter umount système si nécessaire\n",
            "    try:\n",
            "        os.system('fusermount -u /content/drive 2>/dev/null || true')\n",
            "        os.system('umount /content/drive 2>/dev/null || true')\n",
            "    except Exception as e:\n",
            "        print(f\"ℹ️ umount non nécessaire: {e}\")\n",
            "\n",
            "print(\"🔎 État initial:\")\n",
            "print(f\" - ismount: {os.path.ismount(MOUNT_POINT)}\")\n",
            "print(f\" - existe: {os.path.exists(MOUNT_POINT)}\")\n",
            "try:\n",
            "    print(f\" - contenu: {os.listdir(MOUNT_POINT) if os.path.isdir(MOUNT_POINT) else 'N/A'}\")\n",
            "except Exception as _:\n",
            "    print(\" - contenu: N/A\")\n",
            "\n",
            "# Étape 1: forcer un démontage (au cas où)\n",
            "_force_unmount()\n",
            "time.sleep(1)\n",
            "\n",
            "# Étape 2: nettoyage du point de montage\n",
            "_safe_cleanup_mount_point(MOUNT_POINT)\n",
            "time.sleep(0.5)\n",
            "\n",
            "# Étape 3: montage forcé avec gestion des erreurs\n",
            "try:\n",
            "    drive.mount(MOUNT_POINT, force_remount=True)\n",
            "    print(\"✅ Drive monté (v2)\")\n",
            "except Exception as e:\n",
            "    msg = str(e)\n",
            "    print(f\"❌ drive.mount a échoué: {msg}\")\n",
            "    if 'Mountpoint must not already contain files' in msg or 'symlink' in msg.lower():\n",
            "        print(\"🔧 Correction approfondie: suppression et recréation du dossier de montage\")\n",
            "        try:\n",
            "            # Supprimer complètement et recréer /content/drive\n",
            "            if os.path.exists(MOUNT_POINT):\n",
            "                shutil.rmtree(MOUNT_POINT, ignore_errors=True)\n",
            "            os.makedirs(MOUNT_POINT, exist_ok=True)\n",
            "        except Exception as e2:\n",
            "            print(f\"⚠️ Impossible de recréer {MOUNT_POINT}: {e2}\")\n",
            "        # Retenter un dernier montage\n",
            "        drive.mount(MOUNT_POINT, force_remount=True)\n",
            "        print(\"✅ Drive monté après recréation du dossier\")\n",
            "    else:\n",
            "        raise\n",
            "\n",
            "# Étape 4: préparer l'arborescence projet\n",
            "base_path = '/content/drive/MyDrive/AlphaBot_ML_Training'\n",
            "os.makedirs(base_path, exist_ok=True)\n",
            "for sub in ('data', 'models', 'checkpoints', 'logs'):\n",
            "    os.makedirs(f\"{base_path}/{sub}\", exist_ok=True)\n",
            "print(f\"📁 Répertoires prêts sous: {base_path}\")\n",
            "\n",
            "# Vérification finale\n",
            "print(\"🔎 Vérification finale:\")\n",
            "print(f\" - ismount: {os.path.ismount(MOUNT_POINT)}\")\n",
            "try:\n",
            "    print(f\" - contenu: {os.listdir(MOUNT_POINT)}\")\n",
            "except Exception:\n",
            "    print(\" - contenu: N/A\")\n"
        ]
        updated = True
        break

if updated:
    with open('ALPHABOT_ML_TRAINING_COLAB.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook mis à jour: Cellule 2 remplacée par une logique de montage Drive résiliente (v2).")
else:
    print("Aucune modification: cellule 2 non trouvée ou déjà corrigée.")
