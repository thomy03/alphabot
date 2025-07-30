"""
Template Agent pour AlphaBot - Architecture standalone
Squelette de base pour tous les agents du système multi-agent
Compatible avec CrewAI mais sans héritage direct pour éviter les conflits Pydantic
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import yaml
from pathlib import Path


class AlphaBotAgentTemplate:
    """
    Template de base pour tous les agents AlphaBot
    Implémente les hooks standard et la configuration commune
    """
    
    def __init__(
        self,
        agent_name: str,
        description: str,
        config_path: str = "risk_policy.yaml",
        **kwargs
    ):
        self.agent_name = agent_name
        self.description = description
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.start_time = None
        
        # Initialisation des attributs
        self._initialize_agent(**kwargs)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier YAML"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                # Logger pas encore créé à ce stade
                print(f"Warning: Config file {config_path} not found, using defaults")
                return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _initialize_agent(self, **kwargs):
        """Initialise l'agent avec les paramètres spécifiques"""
        # Hook d'initialisation pour les sous-classes
        pass
    
    def _setup_logger(self) -> logging.Logger:
        """Configure le logger pour l'agent"""
        logger = logging.getLogger(f"alphabot.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def on_start(self) -> None:
        """Hook appelé au démarrage de l'agent"""
        self.logger.info(f"Starting {self.agent_name} agent")
        self.start_time = datetime.now()
        
        # Validation de la configuration
        self._validate_config()
    
    def on_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook appelé à chaque message reçu
        
        Args:
            message: Message reçu par l'agent
            
        Returns:
            Dict contenant la réponse de l'agent
        """
        self.logger.debug(f"Received message: {message}")
        
        try:
            # Traitement du message - à implémenter dans les sous-classes
            response = self._process_message(message)
            
            # Log de la réponse
            self.logger.debug(f"Generated response: {response}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def on_stop(self) -> None:
        """Hook appelé à l'arrêt de l'agent"""
        if hasattr(self, 'start_time'):
            runtime = datetime.now() - self.start_time
            self.logger.info(
                f"Stopping {self.agent_name} agent. Runtime: {runtime}"
            )
        else:
            self.logger.info(f"Stopping {self.agent_name} agent")
    
    def _validate_config(self) -> None:
        """Valide la configuration de l'agent"""
        # Validation basique - à étendre dans les sous-classes
        if not self.config:
            self.logger.warning("No configuration loaded")
    
    def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un message reçu - DOIT être implémenté dans les sous-classes
        
        Args:
            message: Message à traiter
            
        Returns:
            Dict contenant la réponse
        """
        raise NotImplementedError("Subclasses must implement _process_message")
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel de l'agent"""
        return {
            "agent_name": self.agent_name,
            "status": "running",
            "start_time": getattr(self, 'start_time', None),
            "config_loaded": bool(self.config),
            "timestamp": datetime.now().isoformat()
        }
    
    def health_check(self) -> bool:
        """Vérifie la santé de l'agent"""
        try:
            # Vérifications basiques
            return (
                hasattr(self, 'logger') and
                hasattr(self, 'agent_name') and
                self.agent_name is not None
            )
        except Exception:
            return False


    def create_crewai_agent(self):
        """
        Crée un agent CrewAI compatible à partir de cet agent
        Utilise la composition plutôt que l'héritage
        """
        try:
            from crewai import Agent
            return Agent(
                role=self.agent_name,
                goal=self.description,
                backstory=f"Expert {self.agent_name} agent for AlphaBot trading system",
                verbose=True
            )
        except ImportError:
            self.logger.warning("CrewAI not available, agent running in standalone mode")
            return None