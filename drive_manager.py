#!/usr/bin/env python3
"""
Drive Manager - Gestion de Google Drive pour Colab
Sauvegarde, chargement, et gestion des espaces de noms
"""

import os
import json
import logging
import shutil
import zipfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class DriveManager:
    """Gestionnaire de Google Drive pour AlphaBot ML Training"""
    
    def __init__(self, base_path: str = "/content/drive/MyDrive/AlphaBot_ML_Training"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Structure des dossiers
        self.folders = {
            'models': self.base_path / 'models',
            'data': self.base_path / 'data',
            'logs': self.base_path / 'logs',
            'checkpoints': self.base_path / 'checkpoints',
            'exports': self.base_path / 'exports',
            'configs': self.base_path / 'configs'
        }
        
        # Créer tous les dossiers
        self._create_folder_structure()
        
        # Fichier de configuration
        self.config_file = self.base_path / 'drive_config.json'
        self.config = self._load_config()
        
        logger.info(f"📁 Drive Manager initialisé: {self.base_path}")
    
    def _create_folder_structure(self):
        """Créer la structure de dossiers"""
        for folder_name, folder_path in self.folders.items():
            folder_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✅ {folder_name}: {folder_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Charger la configuration"""
        default_config = {
            'created_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0.0',
            'folders': {name: str(path) for name, path in self.folders.items()},
            'models_saved': [],
            'training_sessions': [],
            'storage_stats': {}
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info("✅ Configuration Drive chargée")
                return config
            except Exception as e:
                logger.warning(f"⚠️ Échec chargement config: {e}")
        
        # Sauvegarder la configuration par défaut
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Sauvegarder la configuration"""
        try:
            config['last_updated'] = datetime.now().isoformat()
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("💾 Configuration Drive sauvegardée")
        except Exception as e:
            logger.error(f"❌ Échec sauvegarde config: {e}")
    
    def save_model(self, 
                   model: Any, 
                   model_name: str, 
                   model_type: str,
                   metadata: Dict[str, Any] = None) -> str:
        """Sauvegarder un modèle"""
        try:
            # Créer le dossier spécifique au modèle
            model_folder = self.folders['models'] / model_type / model_name
            model_folder.mkdir(parents=True, exist_ok=True)
            
            # Déterminer le format de sauvegarde
            if hasattr(model, 'save'):  # Modèle Keras
                model_path = model_folder / f"{model_name}.h5"
                model.save(str(model_path))
                format_type = "keras"
            elif hasattr(model, 'state_dict'):  # Modèle PyTorch
                model_path = model_folder / f"{model_name}.pth"
                torch.save(model.state_dict(), str(model_path))
                format_type = "pytorch"
            else:  # Autres (pickle)
                model_path = model_folder / f"{model_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                format_type = "pickle"
            
            # Sauvegarder les métadonnées
            if metadata:
                metadata_path = model_folder / f"{model_name}_metadata.json"
                metadata.update({
                    'model_name': model_name,
                    'model_type': model_type,
                    'format': format_type,
                    'saved_date': datetime.now().isoformat(),
                    'model_path': str(model_path)
                })
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Mettre à jour la configuration
            self.config['models_saved'].append({
                'name': model_name,
                'type': model_type,
                'format': format_type,
                'path': str(model_path),
                'saved_date': datetime.now().isoformat()
            })
            self._save_config(self.config)
            
            logger.info(f"💾 Modèle sauvegardé: {model_name} ({format_type})")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"❌ Échec sauvegarde modèle {model_name}: {e}")
            return ""
    
    def load_model(self, 
                   model_name: str, 
                   model_type: str,
                   custom_objects: Dict = None) -> Optional[Any]:
        """Charger un modèle"""
        try:
            model_folder = self.folders['models'] / model_type / model_name
            
            # Chercher le fichier du modèle
            model_files = list(model_folder.glob(f"{model_name}.*"))
            if not model_files:
                logger.error(f"❌ Modèle {model_name} non trouvé")
                return None
            
            model_path = model_files[0]
            
            # Charger selon le format
            if model_path.suffix == '.h5':
                from tensorflow.keras.models import load_model
                model = load_model(str(model_path), custom_objects=custom_objects)
            elif model_path.suffix == '.pth':
                # Pour PyTorch, vous devez fournir la classe du modèle
                logger.error("❌ Chargement PyTorch nécessite la classe du modèle")
                return None
            else:  # .pkl
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            logger.info(f"✅ Modèle chargé: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Échec chargement modèle {model_name}: {e}")
            return None
    
    def save_training_session(self, 
                             session_data: Dict[str, Any],
                             session_name: str = None) -> str:
        """Sauvegarder une session d'entraînement"""
        try:
            if not session_name:
                session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            session_folder = self.folders['logs'] / 'training_sessions' / session_name
            session_folder.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder les données de session
            session_file = session_folder / 'session_data.json'
            session_data.update({
                'session_name': session_name,
                'created_date': datetime.now().isoformat(),
                'session_folder': str(session_folder)
            })
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Mettre à jour la configuration
            self.config['training_sessions'].append({
                'name': session_name,
                'path': str(session_folder),
                'created_date': session_data['created_date']
            })
            self._save_config(self.config)
            
            logger.info(f"💾 Session d'entraînement sauvegardée: {session_name}")
            return str(session_folder)
            
        except Exception as e:
            logger.error(f"❌ Échec sauvegarde session: {e}")
            return ""
    
    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """Lister les modèles sauvegardés"""
        models = []
        
        if model_type:
            model_folder = self.folders['models'] / model_type
            if model_folder.exists():
                for model_dir in model_folder.iterdir():
                    if model_dir.is_dir():
                        models.append({
                            'name': model_dir.name,
                            'type': model_type,
                            'path': str(model_dir)
                        })
        else:
            # Lister tous les modèles
            for type_folder in self.folders['models'].iterdir():
                if type_folder.is_dir():
                    for model_dir in type_folder.iterdir():
                        if model_dir.is_dir():
                            models.append({
                                'name': model_dir.name,
                                'type': type_folder.name,
                                'path': str(model_dir)
                            })
        
        return models
    
    def export_models(self, 
                     model_names: List[str], 
                     export_name: str = None) -> str:
        """Exporter plusieurs modèles dans un fichier zip"""
        try:
            if not export_name:
                export_name = f"alphabot_models_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            export_folder = self.folders['exports'] / export_name
            export_folder.mkdir(parents=True, exist_ok=True)
            
            zip_path = export_folder / f"{export_name}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for model_info in self.config['models_saved']:
                    if model_info['name'] in model_names:
                        model_path = Path(model_info['path'])
                        if model_path.exists():
                            # Ajouter le fichier du modèle
                            zipf.write(model_path, model_path.relative_to(self.base_path))
                            
                            # Ajouter les métadonnées si elles existent
                            metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
                            if metadata_path.exists():
                                zipf.write(metadata_path, metadata_path.relative_to(self.base_path))
            
            logger.info(f"📦 Export créé: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"❌ Échec export modèles: {e}")
            return ""
    
    def import_models(self, zip_path: str) -> bool:
        """Importer des modèles depuis un fichier zip"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Extraire dans le dossier de base
                zipf.extractall(self.base_path)
            
            # Recharger la configuration pour détecter les nouveaux modèles
            self._scan_and_update_models()
            
            logger.info(f"📥 Import réussi depuis: {zip_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Échec import modèles: {e}")
            return False
    
    def _scan_and_update_models(self):
        """Scanner les dossiers et mettre à jour la configuration"""
        known_models = {model['name']: model for model in self.config['models_saved']}
        
        # Scanner les dossiers de modèles
        for type_folder in self.folders['models'].iterdir():
            if type_folder.is_dir():
                for model_dir in type_folder.iterdir():
                    if model_dir.is_dir() and model_dir.name not in known_models:
                        # Nouveau modèle détecté
                        model_files = list(model_dir.glob(f"{model_dir.name}.*"))
                        if model_files:
                            format_type = model_files[0].suffix[1:]  # Enlever le point
                            self.config['models_saved'].append({
                                'name': model_dir.name,
                                'type': type_folder.name,
                                'format': format_type,
                                'path': str(model_files[0]),
                                'saved_date': datetime.now().isoformat()
                            })
        
        self._save_config(self.config)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de stockage"""
        stats = {
            'total_size_mb': 0,
            'models_count': 0,
            'models_size_mb': 0,
            'data_size_mb': 0,
            'logs_size_mb': 0,
            'checkpoints_size_mb': 0,
            'exports_size_mb': 0,
            'by_type': {}
        }
        
        # Calculer les tailles
        for folder_name, folder_path in self.folders.items():
            if folder_path.exists():
                folder_size = self._get_folder_size(folder_path)
                stats[f'{folder_name}_size_mb'] = folder_size / (1024 * 1024)
                stats['total_size_mb'] += folder_size / (1024 * 1024)
                
                if folder_name == 'models':
                    stats['models_count'] = len(list(folder_path.rglob('*')))
                    # Par type de modèle
                    for type_folder in folder_path.iterdir():
                        if type_folder.is_dir():
                            type_size = self._get_folder_size(type_folder)
                            stats['by_type'][type_folder.name] = type_size / (1024 * 1024)
        
        # Mettre à jour la configuration
        self.config['storage_stats'] = stats
        self._save_config(self.config)
        
        return stats
    
    def _get_folder_size(self, folder_path: Path) -> int:
        """Calculer la taille d'un dossier"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
    
    def cleanup_old_files(self, days_old: int = 30):
        """Nettoyer les anciens fichiers"""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        cleaned_count = 0
        cleaned_size = 0
        
        # Nettoyer les checkpoints
        checkpoints_folder = self.folders['checkpoints']
        if checkpoints_folder.exists():
            for checkpoint_file in checkpoints_folder.rglob('*'):
                if checkpoint_file.is_file() and checkpoint_file.stat().st_mtime < cutoff_date:
                    cleaned_size += checkpoint_file.stat().st_size
                    checkpoint_file.unlink()
                    cleaned_count += 1
        
        # Nettoyer les logs anciens
        logs_folder = self.folders['logs']
        if logs_folder.exists():
            for log_file in logs_folder.rglob('*.log'):
                if log_file.is_file() and log_file.stat().st_mtime < cutoff_date:
                    cleaned_size += log_file.stat().st_size
                    log_file.unlink()
                    cleaned_count += 1
        
        logger.info(f"🧹 Nettoyage: {cleaned_count} fichiers supprimés ({cleaned_size/1024/1024:.1f} MB)")
        
        return cleaned_count, cleaned_size
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Vérifier l'intégrité des fichiers"""
        integrity_report = {
            'total_files': 0,
            'corrupted_files': 0,
            'missing_files': 0,
            'issues': []
        }
        
        # Vérifier les modèles
        for model_info in self.config['models_saved']:
            model_path = Path(model_info['path'])
            integrity_report['total_files'] += 1
            
            if not model_path.exists():
                integrity_report['missing_files'] += 1
                integrity_report['issues'].append(f"Modèle manquant: {model_info['name']}")
            elif model_path.stat().st_size == 0:
                integrity_report['corrupted_files'] += 1
                integrity_report['issues'].append(f"Modèle corrompu: {model_info['name']}")
        
        # Vérifier les métadonnées
        for model_info in self.config['models_saved']:
            model_path = Path(model_info['path'])
            metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
            
            if metadata_path.exists():
                integrity_report['total_files'] += 1
                try:
                    with open(metadata_path, 'r') as f:
                        json.load(f)
                except:
                    integrity_report['corrupted_files'] += 1
                    integrity_report['issues'].append(f"Métadonnées corrompues: {model_info['name']}")
        
        return integrity_report
    
    def create_backup(self, backup_name: str = None) -> str:
        """Créer une sauvegarde complète"""
        try:
            if not backup_name:
                backup_name = f"alphabot_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_folder = self.folders['exports'] / backup_name
            backup_folder.mkdir(parents=True, exist_ok=True)
            
            backup_zip = backup_folder / f"{backup_name}.zip"
            
            with zipfile.ZipFile(backup_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Sauvegarder tout le dossier de base
                for file_path in self.base_path.rglob('*'):
                    if file_path.is_file() and not str(file_path).startswith(str(self.folders['exports'])):
                        zipf.write(file_path, file_path.relative_to(self.base_path))
            
            logger.info(f"💾 Sauvegarde complète créée: {backup_zip}")
            return str(backup_zip)
            
        except Exception as e:
            logger.error(f"❌ Échec création sauvegarde: {e}")
            return ""


# Fonctions utilitaires globales
def get_drive_manager(base_path: str = None) -> DriveManager:
    """Factory pour DriveManager"""
    if base_path is None:
        base_path = "/content/drive/MyDrive/AlphaBot_ML_Training"
    return DriveManager(base_path)


def setup_drive_structure(base_path: str = None) -> DriveManager:
    """Configurer la structure Drive pour AlphaBot"""
    drive_manager = get_drive_manager(base_path)
    
    # Vérifier l'intégrité
    integrity = drive_manager.verify_integrity()
    if integrity['issues']:
        logger.warning(f"⚠️ Problèmes d'intégrité détectés: {len(integrity['issues'])}")
        for issue in integrity['issues']:
            logger.warning(f"   - {issue}")
    
    # Afficher les statistiques
    stats = drive_manager.get_storage_stats()
    logger.info(f"📊 Statistiques de stockage:")
    logger.info(f"   - Taille totale: {stats['total_size_mb']:.1f} MB")
    logger.info(f"   - Modèles: {stats['models_count']} fichiers ({stats['models_size_mb']:.1f} MB)")
    logger.info(f"   - Données: {stats['data_size_mb']:.1f} MB")
    logger.info(f"   - Logs: {stats['logs_size_mb']:.1f} MB")
    
    return drive_manager
