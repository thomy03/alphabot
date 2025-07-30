#!/usr/bin/env python3
"""
Colab Utils - Fonctions utilitaires pour Google Colab
Optimisations GPU/TPU, gestion mémoire, monitoring performance
"""

import os
import time
import logging
import gc
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ColabGPUManager:
    """Gestionnaire GPU pour Colab"""
    
    def __init__(self):
        self.gpu_available = tf.config.list_physical_devices('GPU')
        self.tpu_available = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Détecter TPU
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            self.tpu_available = True
        except ImportError:
            self.tpu_available = False
        
        self._setup_gpu_memory()
    
    def _setup_gpu_memory(self):
        """Configurer la mémoire GPU"""
        if self.gpu_available:
            try:
                # Activer la croissance de mémoire
                for gpu in self.gpu_available:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("✅ GPU memory growth activé")
            except Exception as e:
                logger.warning(f"⚠️ GPU memory setup failed: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Obtenir les informations sur les devices"""
        info = {
            'gpu_available': len(self.gpu_available) > 0,
            'tpu_available': self.tpu_available,
            'pytorch_device': str(self.device),
            'tensorflow_version': tf.__version__,
            'torch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
        
        return info
    
    def clear_gpu_memory(self):
        """Nettoyer la mémoire GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Forcer le garbage collection
        gc.collect()
        
        logger.info("🧹 GPU memory cleared")


class ColabMemoryMonitor:
    """Moniteur de mémoire pour Colab"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.start_time = time.time()
        self.memory_warnings = 0
        self.max_memory_warnings = 3
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Obtenir l'utilisation de la mémoire"""
        memory = psutil.virtual_memory()
        
        usage = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent,
            'elapsed_time_min': (time.time() - self.start_time) / 60
        }
        
        # Ajouter la mémoire GPU si disponible
        if torch.cuda.is_available():
            usage['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated(0) / (1024**3)
            usage['gpu_memory_cached_gb'] = torch.cuda.memory_reserved(0) / (1024**3)
        
        return usage
    
    def check_memory_safety(self) -> bool:
        """Vérifier si la mémoire est sûre"""
        usage = self.get_memory_usage()
        
        # Alerte si plus de 85% de RAM utilisée
        if usage['percent_used'] > 85:
            self.memory_warnings += 1
            logger.warning(f"⚠️ Mémoire critique: {usage['percent_used']:.1f}% utilisée")
            
            if self.memory_warnings >= self.max_memory_warnings:
                logger.error("❌ Limite de mémoire atteinte - nettoyage forcé")
                self._force_cleanup()
                return False
        
        return True
    
    def _force_cleanup(self):
        """Forcer le nettoyage de la mémoire"""
        logger.info("🧹 Nettoyage mémoire forcé...")
        
        # Nettoyage GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Garbage collection
        gc.collect()
        
        # Réinitialiser les compteurs
        self.memory_warnings = 0
        
        logger.info("✅ Nettoyage mémoire terminé")


class ColabCheckpointManager:
    """Gestionnaire de checkpoints pour Colab"""
    
    def __init__(self, base_path: str, model_name: str):
        self.base_path = base_path
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(base_path, 'checkpoints', model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = float('-inf')
    
    def save_checkpoint(self, 
                       model: Any, 
                       epoch: int, 
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """Sauvegarder un checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch:03d}.h5'
        )
        
        # Sauvegarder le modèle
        if hasattr(model, 'save'):
            model.save(checkpoint_path)
        else:
            # Pour les modèles PyTorch ou autres
            import pickle
            with open(checkpoint_path.replace('.h5', '.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Sauvegarder les métadonnées
        metadata = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name
        }
        
        metadata_path = checkpoint_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': metadata['timestamp']
        })
        
        # Mettre à jour le meilleur checkpoint
        if is_best:
            self.best_checkpoint = checkpoint_path
            self.best_metric = metrics.get('val_accuracy', metrics.get('accuracy', 0))
        
        logger.info(f"💾 Checkpoint sauvegardé: epoch {epoch}")
        
        # Nettoyer les anciens checkpoints (garder les 3 derniers)
        self._cleanup_old_checkpoints()
    
    def load_best_checkpoint(self) -> Optional[Any]:
        """Charger le meilleur checkpoint"""
        if not self.best_checkpoint:
            return None
        
        try:
            if self.best_checkpoint.endswith('.h5'):
                from tensorflow.keras.models import load_model
                return load_model(self.best_checkpoint)
            else:
                import pickle
                with open(self.best_checkpoint, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"❌ Échec du chargement du checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """Nettoyer les anciens checkpoints"""
        if len(self.checkpoints) > 3:
            # Trier par epoch et supprimer les plus anciens
            self.checkpoints.sort(key=lambda x: x['epoch'])
            
            for checkpoint in self.checkpoints[:-3]:
                try:
                    os.remove(checkpoint['path'])
                    os.remove(checkpoint['path'].replace('.h5', '_metadata.json'))
                    logger.info(f"🗑️ Ancien checkpoint supprimé: epoch {checkpoint['epoch']}")
                except Exception as e:
                    logger.warning(f"⚠️ Échec suppression checkpoint: {e}")
            
            self.checkpoints = self.checkpoints[-3:]


class ColabTimeoutProtection:
    """Protection contre les timeouts Colab"""
    
    def __init__(self, max_execution_time: int = 3600):  # 1 heure par défaut
        self.max_execution_time = max_execution_time
        self.start_time = time.time()
        self.checkpoints = []
        self.last_checkpoint_time = self.start_time
    
    def check_timeout(self) -> bool:
        """Vérifier si on approche du timeout"""
        elapsed_time = time.time() - self.start_time
        remaining_time = self.max_execution_time - elapsed_time
        
        # Alerte si moins de 10 minutes restantes
        if remaining_time < 600:  # 10 minutes
            logger.warning(f"⏰ Timeout approchant: {remaining_time/60:.1f} minutes restantes")
            return True
        
        return False
    
    def should_save_checkpoint(self) -> bool:
        """Déterminer si on doit sauvegarder un checkpoint"""
        elapsed_since_last = time.time() - self.last_checkpoint_time
        
        # Sauvegarder toutes les 10 minutes
        return elapsed_since_last > 600  # 10 minutes
    
    def register_checkpoint(self):
        """Enregistrer un checkpoint"""
        self.last_checkpoint_time = time.time()
        self.checkpoints.append(self.last_checkpoint_time)
        logger.info("⏰ Checkpoint timeout enregistré")


class ColabPerformanceTracker:
    """Tracker de performance pour Colab"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.metrics_history = []
        self.start_time = time.time()
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int = None):
        """Logger les métriques de performance"""
        timestamp = time.time() - self.start_time
        
        entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'metrics': metrics,
            'datetime': datetime.now().isoformat()
        }
        
        self.metrics_history.append(entry)
        
        # Logger dans la console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        epoch_str = f"Epoch {epoch}: " if epoch else ""
        logger.info(f"📊 {epoch_str}{metrics_str}")
        
        # Sauvegarder dans un fichier si spécifié
        if self.log_file:
            self._save_to_file()
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé des performances"""
        if not self.metrics_history:
            return {}
        
        # Extraire les métriques finales
        final_metrics = self.metrics_history[-1]['metrics']
        
        summary = {
            'total_time_min': (time.time() - self.start_time) / 60,
            'total_epochs': len([m for m in self.metrics_history if m['epoch'] is not None]),
            'final_metrics': final_metrics,
            'best_metrics': self._get_best_metrics()
        }
        
        return summary
    
    def _get_best_metrics(self) -> Dict[str, float]:
        """Obtenir les meilleures métriques"""
        if not self.metrics_history:
            return {}
        
        # Trouver la meilleure accuracy
        best_entry = max(
            [m for m in self.metrics_history if m['epoch'] is not None],
            key=lambda x: x['metrics'].get('val_accuracy', x['metrics'].get('accuracy', 0))
        )
        
        return best_entry['metrics']
    
    def _save_to_file(self):
        """Sauvegarder les métriques dans un fichier"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Échec sauvegarde métriques: {e}")


def create_colab_callbacks(model_name: str, 
                          save_path: str,
                          patience: int = 10) -> List[tf.keras.callbacks.Callback]:
    """Créer les callbacks Keras optimisés pour Colab"""
    
    callbacks = []
    
    # Checkpoint Manager
    checkpoint_manager = ColabCheckpointManager(save_path, model_name)
    
    # Model Checkpoint
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, 'checkpoints', model_name, 'checkpoint_epoch_{epoch:03d}.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce LR on Plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Custom Callback pour le monitoring
    class ColabMonitoringCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.memory_monitor = ColabMemoryMonitor()
            self.timeout_protection = ColabTimeoutProtection()
            self.performance_tracker = ColabPerformanceTracker()
        
        def on_epoch_end(self, epoch, logs=None):
            # Vérifier la mémoire
            if not self.memory_monitor.check_memory_safety():
                logger.warning("⚠️ Mémoire critique - arrêt possible")
            
            # Vérifier le timeout
            if self.timeout_protection.check_timeout():
                logger.warning("⚠️ Timeout approchant")
            
            # Logger les métriques
            if logs:
                self.performance_tracker.log_metrics(logs, epoch)
    
    callbacks.append(ColabMonitoringCallback())
    
    return callbacks


def optimize_batch_size(dataset_size: int, 
                       available_memory_gb: float,
                       model_complexity: str = 'medium') -> int:
    """Optimiser le batch size pour Colab"""
    
    # Facteurs de complexité
    complexity_factors = {
        'low': 1.0,
        'medium': 0.5,
        'high': 0.25
    }
    
    factor = complexity_factors.get(model_complexity, 0.5)
    
    # Calculer le batch size optimal
    memory_per_sample = 0.001  # ~1MB par sample (estimation)
    max_samples = (available_memory_gb * 0.7) / memory_per_sample  # 70% de mémoire disponible
    
    batch_size = int(min(dataset_size * 0.1, max_samples * factor))
    batch_size = max(8, min(128, batch_size))  # Entre 8 et 128
    
    # Ajuster pour être une puissance de 2
    batch_size = 2 ** int(np.log2(batch_size))
    
    logger.info(f"📊 Batch size optimisé: {batch_size}")
    return batch_size


def setup_mixed_precision():
    """Configurer la précision mixte pour Colab"""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("✅ Mixed precision activé (float16)")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Mixed precision non disponible: {e}")
        return False


# Fonctions utilitaires globales
def get_colab_environment() -> Dict[str, Any]:
    """Obtenir les informations sur l'environnement Colab"""
    gpu_manager = ColabGPUManager()
    memory_monitor = ColabMemoryMonitor()
    
    env_info = {
        'gpu_info': gpu_manager.get_device_info(),
        'memory_info': memory_monitor.get_memory_usage(),
        'colab_specific': {
            'is_colab': 'google.colab' in str(get_ipython()),
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'tpu_available': False
        }
    }
    
    # Détecter TPU
    try:
        import torch_xla
        env_info['colab_specific']['tpu_available'] = True
    except ImportError:
        pass
    
    return env_info


def print_colab_setup():
    """Afficher la configuration Colab"""
    env_info = get_colab_environment()
    
    print("🚀 Configuration Google Colab:")
    print("=" * 50)
    
    # GPU Info
    if env_info['gpu_info']['gpu_available']:
        print(f"✅ GPU: {env_info['gpu_info']['cuda_device_name']}")
        print(f"   Mémoire: {env_info['gpu_info']['cuda_memory_total']/1e9:.1f} GB")
    else:
        print("⚠️ GPU non disponible - mode CPU")
    
    # Memory Info
    mem = env_info['memory_info']
    print(f"💾 RAM: {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB ({mem['percent_used']:.1f}%)")
    
    # Framework versions
    print(f"📚 TensorFlow: {env_info['gpu_info']['tensorflow_version']}")
    print(f"📚 PyTorch: {env_info['gpu_info']['torch_version']}")
    
    print("=" * 50)
