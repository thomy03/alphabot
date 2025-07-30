#!/usr/bin/env python3
"""
Signal HUB Central - AlphaBot Multi-Agent Trading System
Orchestrateur central pour la communication entre agents via Redis
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import redis.asyncio as redis
from alphabot.core.config import get_settings

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types de signaux dans le système"""
    PRICE_UPDATE = "price_update"
    TECHNICAL_SIGNAL = "technical_signal"
    SENTIMENT_SIGNAL = "sentiment_signal"
    FUNDAMENTAL_SIGNAL = "fundamental_signal"
    RISK_ALERT = "risk_alert"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    EXECUTION_ORDER = "execution_order"
    SYSTEM_STATUS = "system_status"
    
    # Signaux de trading
    BUY_SIGNAL = "buy_signal"
    SELL_SIGNAL = "sell_signal"
    TRADE_EXECUTION = "trade_execution"


class SignalPriority(Enum):
    """Priorités des signaux"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Signal:
    """Signal standard pour communication inter-agents"""
    id: str
    type: SignalType
    source_agent: str
    target_agent: Optional[str] = None  # None = broadcast
    priority: SignalPriority = SignalPriority.MEDIUM
    timestamp: datetime = None
    symbol: Optional[str] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    ttl_seconds: int = 300  # 5 minutes par défaut
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le signal pour Redis"""
        result = asdict(self)
        result['type'] = self.type.value
        result['priority'] = self.priority.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Désérialise un signal depuis Redis"""
        data['type'] = SignalType(data['type'])
        data['priority'] = SignalPriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class SignalHub:
    """Hub central de communication via Redis"""
    
    def __init__(self, redis_url: str = None):
        self.settings = get_settings()
        self.redis_url = redis_url or f"redis://{self.settings.redis_host}:{self.settings.redis_port}"
        self.redis: Optional[redis.Redis] = None
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.running = False
        
        # Channels Redis
        self.SIGNALS_CHANNEL = "alphabot:signals"
        self.AGENTS_STATUS_CHANNEL = "alphabot:agents:status"
        self.SYSTEM_CHANNEL = "alphabot:system"
        
        # Métriques
        self.metrics = {
            'signals_sent': 0,
            'signals_received': 0,
            'signals_processed': 0,
            'errors': 0,
            'latency_ms': []
        }
    
    async def connect(self) -> bool:
        """Connexion à Redis"""
        try:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            logger.info(f"🔌 Signal HUB connecté à Redis: {self.redis_url}")
            return True
        except Exception as e:
            logger.error(f"❌ Échec connexion Redis: {e}")
            return False
    
    async def disconnect(self):
        """Déconnexion Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("🔌 Signal HUB déconnecté")
    
    async def publish_signal(self, signal: Signal) -> bool:
        """Publier un signal sur le hub"""
        if not self.redis:
            logger.error("Redis non connecté")
            return False
        
        try:
            start_time = time.time()
            
            # Générer ID si absent
            if not signal.id:
                signal.id = str(uuid.uuid4())
            
            # Sérialiser et publier
            signal_json = json.dumps(signal.to_dict())
            
            # Publication sur le channel principal
            await self.redis.publish(self.SIGNALS_CHANNEL, signal_json)
            
            # Stocker dans Redis avec TTL pour historique
            key = f"signal:{signal.id}"
            await self.redis.setex(key, signal.ttl_seconds, signal_json)
            
            # Métriques
            latency_ms = (time.time() - start_time) * 1000
            self.metrics['signals_sent'] += 1
            self.metrics['latency_ms'].append(latency_ms)
            
            # Garder seulement les 100 dernières latences
            if len(self.metrics['latency_ms']) > 100:
                self.metrics['latency_ms'] = self.metrics['latency_ms'][-100:]
            
            logger.debug(f"📡 Signal publié: {signal.type.value} [{signal.id[:8]}] en {latency_ms:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur publication signal: {e}")
            self.metrics['errors'] += 1
            return False
    
    async def subscribe_to_signals(self, 
                                 agent_name: str,
                                 callback: Callable[[Signal], None],
                                 signal_types: Optional[List[SignalType]] = None,
                                 symbol_filter: Optional[str] = None) -> bool:
        """S'abonner aux signaux avec filtres optionnels"""
        if not self.redis:
            logger.error("Redis non connecté")
            return False
        
        try:
            # Enregistrer le callback
            subscription_key = f"{agent_name}:{id(callback)}"
            if subscription_key not in self.subscriptions:
                self.subscriptions[subscription_key] = []
            
            self.subscriptions[subscription_key].append({
                'callback': callback,
                'signal_types': signal_types,
                'symbol_filter': symbol_filter,
                'agent_name': agent_name
            })
            
            logger.info(f"📥 Agent {agent_name} abonné aux signaux")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur abonnement signaux: {e}")
            return False
    
    async def start_listening(self):
        """Démarrer l'écoute des signaux Redis"""
        if not self.redis:
            logger.error("Redis non connecté")
            return
        
        self.running = True
        pubsub = self.redis.pubsub()
        
        try:
            # S'abonner aux channels
            await pubsub.subscribe(self.SIGNALS_CHANNEL)
            await pubsub.subscribe(self.AGENTS_STATUS_CHANNEL)
            await pubsub.subscribe(self.SYSTEM_CHANNEL)
            
            logger.info("🎧 Signal HUB en écoute...")
            
            async for message in pubsub.listen():
                if not self.running:
                    break
                
                if message['type'] == 'message':
                    await self._process_message(message)
                    
        except Exception as e:
            logger.error(f"❌ Erreur écoute Redis: {e}")
        finally:
            await pubsub.close()
            self.running = False
    
    async def _process_message(self, message: Dict[str, Any]):
        """Traiter un message reçu"""
        try:
            channel = message['channel']
            data = json.loads(message['data'])
            
            if channel == self.SIGNALS_CHANNEL:
                await self._handle_signal(data)
            elif channel == self.AGENTS_STATUS_CHANNEL:
                await self._handle_agent_status(data)
            elif channel == self.SYSTEM_CHANNEL:
                await self._handle_system_message(data)
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
            self.metrics['errors'] += 1
    
    async def _handle_signal(self, signal_data: Dict[str, Any]):
        """Traiter un signal reçu"""
        try:
            signal = Signal.from_dict(signal_data)
            self.metrics['signals_received'] += 1
            
            # Distribuer aux agents abonnés
            for subscription_key, subscription_list in self.subscriptions.items():
                for subscription in subscription_list:
                    if self._signal_matches_subscription(signal, subscription):
                        try:
                            # Appel asynchrone du callback
                            if asyncio.iscoroutinefunction(subscription['callback']):
                                await subscription['callback'](signal)
                            else:
                                subscription['callback'](signal)
                            
                            self.metrics['signals_processed'] += 1
                            
                        except Exception as e:
                            logger.error(f"❌ Erreur callback {subscription['agent_name']}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal: {e}")
    
    def _signal_matches_subscription(self, signal: Signal, subscription: Dict[str, Any]) -> bool:
        """Vérifier si un signal correspond à un abonnement"""
        
        # Filtre par type de signal
        if subscription['signal_types'] and signal.type not in subscription['signal_types']:
            return False
        
        # Filtre par symbole
        if subscription['symbol_filter'] and signal.symbol != subscription['symbol_filter']:
            return False
        
        # Filtre par agent cible (si spécifié)
        if signal.target_agent and signal.target_agent != subscription['agent_name']:
            return False
        
        return True
    
    async def _handle_agent_status(self, status_data: Dict[str, Any]):
        """Traiter un changement de statut d'agent"""
        logger.debug(f"📊 Statut agent: {status_data}")
    
    async def _handle_system_message(self, system_data: Dict[str, Any]):
        """Traiter un message système"""
        logger.info(f"🔧 Message système: {system_data}")
    
    async def get_signal_history(self, 
                               signal_type: Optional[SignalType] = None,
                               symbol: Optional[str] = None,
                               limit: int = 100) -> List[Signal]:
        """Récupérer l'historique des signaux"""
        if not self.redis:
            return []
        
        try:
            # Pattern de recherche des clés
            pattern = "signal:*"
            keys = await self.redis.keys(pattern)
            
            signals = []
            for key in keys[-limit:]:  # Limiter le nombre
                signal_json = await self.redis.get(key)
                if signal_json:
                    signal_data = json.loads(signal_json)
                    signal = Signal.from_dict(signal_data)
                    
                    # Appliquer les filtres
                    if signal_type and signal.type != signal_type:
                        continue
                    if symbol and signal.symbol != symbol:
                        continue
                    
                    signals.append(signal)
            
            # Trier par timestamp (plus récent en premier)
            signals.sort(key=lambda s: s.timestamp, reverse=True)
            return signals[:limit]
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération historique: {e}")
            return []
    
    async def cleanup_expired_signals(self):
        """Nettoyer les signaux expirés"""
        if not self.redis:
            return
        
        try:
            pattern = "signal:*"
            keys = await self.redis.keys(pattern)
            
            cleaned = 0
            for key in keys:
                # Redis TTL se charge automatiquement des expirations
                # Ici on peut ajouter une logique supplémentaire si nécessaire
                ttl = await self.redis.ttl(key)
                if ttl == -1:  # Pas de TTL défini
                    await self.redis.expire(key, 3600)  # 1 heure par défaut
                    cleaned += 1
            
            if cleaned > 0:
                logger.info(f"🧹 {cleaned} signaux nettoyés")
                
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage signaux: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtenir les métriques du hub"""
        avg_latency = sum(self.metrics['latency_ms']) / len(self.metrics['latency_ms']) if self.metrics['latency_ms'] else 0
        max_latency = max(self.metrics['latency_ms']) if self.metrics['latency_ms'] else 0
        
        return {
            'signals_sent': self.metrics['signals_sent'],
            'signals_received': self.metrics['signals_received'],
            'signals_processed': self.metrics['signals_processed'],
            'errors': self.metrics['errors'],
            'avg_latency_ms': round(avg_latency, 2),
            'max_latency_ms': round(max_latency, 2),
            'active_subscriptions': len(self.subscriptions),
            'connected': self.redis is not None and self.running
        }
    
    async def publish_agent_status(self, agent_name: str, status: str, metadata: Dict[str, Any] = None):
        """Publier le statut d'un agent"""
        status_data = {
            'agent_name': agent_name,
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        await self.redis.publish(self.AGENTS_STATUS_CHANNEL, json.dumps(status_data))
    
    async def stop(self):
        """Arrêter le hub"""
        self.running = False
        await self.disconnect()


# Instance globale du hub
_signal_hub_instance: Optional[SignalHub] = None


def get_signal_hub() -> SignalHub:
    """Obtenir l'instance globale du Signal HUB"""
    global _signal_hub_instance
    if _signal_hub_instance is None:
        _signal_hub_instance = SignalHub()
    return _signal_hub_instance


async def init_signal_hub() -> bool:
    """Initialiser le Signal HUB"""
    hub = get_signal_hub()
    connected = await hub.connect()
    if connected:
        # Démarrer l'écoute en arrière-plan
        asyncio.create_task(hub.start_listening())
    return connected


async def shutdown_signal_hub():
    """Arrêter le Signal HUB"""
    global _signal_hub_instance
    if _signal_hub_instance:
        await _signal_hub_instance.stop()
        _signal_hub_instance = None