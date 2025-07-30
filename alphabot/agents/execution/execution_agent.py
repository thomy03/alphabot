#!/usr/bin/env python3
"""
Execution Agent - AlphaBot Multi-Agent Trading System
Exécution d'ordres via Interactive Brokers API
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from alphabot.core.signal_hub import (
    Signal, SignalType, SignalPriority, get_signal_hub
)
from alphabot.core.config import get_settings

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types d'ordres"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Statuts d'ordres"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Côtés d'ordres"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Ordre de trading"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: float = 0.0
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class Position:
    """Position en portefeuille"""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime


@dataclass
class ExecutionReport:
    """Rapport d'exécution"""
    orders_submitted: int
    orders_filled: int
    orders_rejected: int
    total_volume: float
    total_commission: float
    slippage_bps: float
    execution_time_ms: float
    fill_rate: float


class ExecutionAgent:
    """Agent d'exécution via Interactive Brokers"""
    
    def __init__(self):
        self.settings = get_settings()
        self.signal_hub = get_signal_hub()
        self.is_running = False
        
        # Configuration IBKR
        self.ibkr_config = {
            'host': self.settings.ibkr_host,
            'port': self.settings.ibkr_port,
            'client_id': self.settings.ibkr_client_id,
            'connect_timeout': 30,
            'request_timeout': 10
        }
        
        # Mock connection pour Phase 4 (sera remplacé par ib_insync en Phase 5)
        self.ib_connection = None
        self.is_connected = False
        
        # États internes
        self.pending_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.execution_history: List[Order] = []
        
        # Paramètres d'exécution
        self.execution_params = {
            'max_order_size': 1000000,  # $1M max par ordre
            'min_order_size': 100,      # $100 min par ordre
            'slippage_tolerance_bps': 10,  # 10 bps max slippage
            'position_size_limit': 0.20,   # 20% max du capital
            'daily_volume_limit': 0.10,    # 10% du volume quotidien
            'order_timeout_minutes': 60,   # 1h timeout
        }
        
        # Métriques d'exécution
        self.metrics = {
            'orders_today': 0,
            'volume_today': 0.0,
            'commission_today': 0.0,
            'avg_slippage_bps': 0.0,
            'fill_rate': 0.0,
            'connection_uptime': 0.0
        }
    
    async def start(self):
        """Démarrer l'agent d'exécution"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("⚡ Execution Agent démarré")
        
        # Connexion IBKR (simulée pour Phase 4)
        await self._connect_to_ibkr()
        
        # S'abonner aux signaux d'exécution
        await self.signal_hub.subscribe_to_signals(
            agent_name="execution_agent",
            callback=self._handle_signal,
            signal_types=[
                SignalType.EXECUTION_ORDER,
                SignalType.PORTFOLIO_REBALANCE,
                SignalType.RISK_ALERT
            ]
        )
        
        # Démarrer les tâches de monitoring
        asyncio.create_task(self._monitor_orders())
        asyncio.create_task(self._update_positions())
        
        # Publier le statut
        await self.signal_hub.publish_agent_status(
            "execution_agent",
            "started",
            {
                "version": "1.0",
                "broker": "Interactive Brokers",
                "connection_status": "connected" if self.is_connected else "disconnected",
                "order_types": [ot.value for ot in OrderType]
            }
        )
    
    async def stop(self):
        """Arrêter l'agent d'exécution"""
        self.is_running = False
        
        # Annuler les ordres en attente
        await self._cancel_pending_orders()
        
        # Déconnecter IBKR
        await self._disconnect_from_ibkr()
        
        await self.signal_hub.publish_agent_status("execution_agent", "stopped")
        logger.info("⚡ Execution Agent arrêté")
    
    async def _connect_to_ibkr(self):
        """Connexion à Interactive Brokers (simulée pour Phase 4)"""
        try:
            logger.info(f"🔌 Connexion à IBKR {self.ibkr_config['host']}:{self.ibkr_config['port']}")
            
            # SIMULATION - En Phase 5, utiliser ib_insync
            await asyncio.sleep(0.1)  # Simuler délai connexion
            
            # Mock de la connexion
            self.ib_connection = {
                'connected': True,
                'account': 'DU123456',  # Compte démo
                'buying_power': 100000.0,  # $100k capital
                'currency': 'USD'
            }
            
            self.is_connected = True
            logger.info(f"✅ Connecté à IBKR - Compte: {self.ib_connection['account']}")
            
            # Initialiser les positions (vides pour démo)
            await self._load_positions()
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion IBKR: {e}")
            self.is_connected = False
    
    async def _disconnect_from_ibkr(self):
        """Déconnexion d'Interactive Brokers"""
        if self.ib_connection:
            logger.info("🔌 Déconnexion IBKR")
            self.ib_connection = None
            self.is_connected = False
    
    async def _handle_signal(self, signal: Signal):
        """Traiter un signal reçu"""
        try:
            if signal.type == SignalType.EXECUTION_ORDER:
                await self._process_execution_order(signal)
            
            elif signal.type == SignalType.PORTFOLIO_REBALANCE:
                await self._process_rebalancing_order(signal)
            
            elif signal.type == SignalType.RISK_ALERT:
                await self._process_risk_alert(signal)
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal execution: {e}")
    
    async def _process_execution_order(self, signal: Signal):
        """Traiter un ordre d'exécution direct"""
        
        order_data = signal.data
        symbol = signal.symbol
        
        if not symbol or not order_data:
            logger.warning("⚠️ Signal d'exécution incomplet")
            return
        
        # Extraire les paramètres d'ordre
        action = order_data.get('action', 'HOLD')
        quantity = order_data.get('quantity', 0)
        order_type = order_data.get('order_type', 'MARKET')
        price = order_data.get('price')
        
        if action == 'HOLD' or quantity == 0:
            logger.info(f"📊 {symbol}: HOLD - Pas d'action requise")
            return
        
        # Créer l'ordre
        side = OrderSide.BUY if action == 'BUY' else OrderSide.SELL
        
        order = Order(
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            order_type=OrderType(order_type.lower()),
            price=price,
            created_at=datetime.utcnow()
        )
        
        # Soumettre l'ordre
        await self._submit_order(order)
    
    async def _process_rebalancing_order(self, signal: Signal):
        """Traiter un ordre de rebalancement de portefeuille"""
        
        trades = signal.data.get('trades', {})
        weights = signal.data.get('weights', {})
        
        if not trades and not weights:
            logger.warning("⚠️ Signal de rebalancement sans trades")
            return
        
        logger.info(f"🔄 Rebalancement: {len(trades)} trades à exécuter")
        
        # Calculer les ordres nécessaires
        orders = []
        portfolio_value = self._get_portfolio_value()
        
        for symbol, weight_change in trades.items():
            if abs(weight_change) < 0.01:  # Ignorer les petits changements < 1%
                continue
            
            # Calculer la valeur en dollars
            dollar_amount = portfolio_value * weight_change
            
            # Estimer la quantité (prix simulé)
            estimated_price = self._get_estimated_price(symbol)
            quantity = abs(dollar_amount) / estimated_price
            
            side = OrderSide.BUY if weight_change > 0 else OrderSide.SELL
            
            order = Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                created_at=datetime.utcnow()
            )
            
            orders.append(order)
        
        # Soumettre tous les ordres
        for order in orders:
            await self._submit_order(order)
        
        logger.info(f"📋 {len(orders)} ordres de rebalancement soumis")
    
    async def _process_risk_alert(self, signal: Signal):
        """Traiter une alerte de risque (ordres défensifs)"""
        
        risk_level = signal.data.get('risk_level', 'MEDIUM')
        
        if risk_level == 'CRITICAL':
            logger.warning("🚨 Alerte critique - Réduction d'exposition")
            await self._emergency_risk_reduction()
        
        elif risk_level == 'HIGH':
            logger.warning("⚠️ Risque élevé - Annulation ordres non essentiels")
            await self._cancel_non_essential_orders()
    
    async def _submit_order(self, order: Order) -> bool:
        """Soumettre un ordre à IBKR"""
        
        if not self.is_connected:
            logger.error("❌ Pas de connexion IBKR")
            return False
        
        try:
            # Validation pré-trade
            if not await self._validate_order(order):
                return False
            
            start_time = time.time()
            
            # Générer ID d'ordre
            order.order_id = f"ORD_{int(time.time() * 1000)}"
            
            # SIMULATION - En Phase 5, utiliser ib_insync
            logger.info(f"📤 Soumission ordre: {order.side.value.upper()} {order.quantity:.0f} {order.symbol}")
            
            # Simuler délai de soumission
            await asyncio.sleep(0.05)  # 50ms
            
            # Simuler réponse IBKR
            order.status = OrderStatus.SUBMITTED
            self.pending_orders[order.order_id] = order
            
            # Simuler exécution rapide (90% de chance)
            if np.random.random() < 0.9:
                await asyncio.sleep(0.1)  # 100ms délai d'exécution
                await self._simulate_order_fill(order)
            
            submission_time = (time.time() - start_time) * 1000
            
            # Publier signal de confirmation
            await self._publish_execution_update(order, submission_time)
            
            # Métriques
            self.metrics['orders_today'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur soumission ordre {order.symbol}: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return False
    
    async def _validate_order(self, order: Order) -> bool:
        """Valider un ordre avant soumission"""
        
        # Vérifier taille d'ordre
        estimated_value = order.quantity * self._get_estimated_price(order.symbol)
        
        if estimated_value < self.execution_params['min_order_size']:
            logger.warning(f"⚠️ Ordre trop petit: ${estimated_value:.0f} < ${self.execution_params['min_order_size']}")
            return False
        
        if estimated_value > self.execution_params['max_order_size']:
            logger.warning(f"⚠️ Ordre trop grand: ${estimated_value:.0f} > ${self.execution_params['max_order_size']}")
            return False
        
        # Vérifier limites de position
        current_position = self.positions.get(order.symbol, Position(order.symbol, 0, 0, 0, 0, 0, datetime.utcnow()))
        portfolio_value = self._get_portfolio_value()
        
        new_quantity = current_position.quantity
        if order.side == OrderSide.BUY:
            new_quantity += order.quantity
        else:
            new_quantity -= order.quantity
        
        new_position_value = abs(new_quantity * self._get_estimated_price(order.symbol))
        position_weight = new_position_value / portfolio_value
        
        if position_weight > self.execution_params['position_size_limit']:
            logger.warning(f"⚠️ Position trop importante: {position_weight:.1%} > {self.execution_params['position_size_limit']:.1%}")
            return False
        
        # Vérifier power d'achat (pour les achats)
        if order.side == OrderSide.BUY:
            buying_power = self.ib_connection.get('buying_power', 0)
            if estimated_value > buying_power:
                logger.warning(f"⚠️ Pouvoir d'achat insuffisant: ${estimated_value:.0f} > ${buying_power:.0f}")
                return False
        
        return True
    
    async def _simulate_order_fill(self, order: Order):
        """Simuler l'exécution d'un ordre"""
        
        # Simuler prix d'exécution avec slippage
        estimated_price = self._get_estimated_price(order.symbol)
        
        # Slippage réaliste (0-5 bps pour ordres market)
        slippage_bps = np.random.uniform(0, 5)
        slippage_factor = 1 + (slippage_bps / 10000)
        
        if order.side == OrderSide.BUY:
            fill_price = estimated_price * slippage_factor
        else:
            fill_price = estimated_price / slippage_factor
        
        # Simuler commission (IBKR: $0.005/action, min $1)
        commission = max(1.0, order.quantity * 0.005)
        
        # Mettre à jour l'ordre
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        order.filled_at = datetime.utcnow()
        
        # Mettre à jour la position
        await self._update_position(order)
        
        # Retirer des ordres en attente
        if order.order_id in self.pending_orders:
            del self.pending_orders[order.order_id]
        
        # Ajouter à l'historique
        self.execution_history.append(order)
        
        # Métriques
        self.metrics['volume_today'] += order.quantity * fill_price
        self.metrics['commission_today'] += commission
        
        logger.info(f"✅ Ordre exécuté: {order.side.value.upper()} {order.quantity:.0f} {order.symbol} @ ${fill_price:.2f}")
    
    async def _update_position(self, order: Order):
        """Mettre à jour une position après exécution"""
        
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_cost=0,
                market_value=0,
                unrealized_pnl=0,
                realized_pnl=0,
                last_updated=datetime.utcnow()
            )
        
        position = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            # Achat - augmenter position
            total_cost = position.quantity * position.avg_cost + order.filled_quantity * order.avg_fill_price
            position.quantity += order.filled_quantity
            position.avg_cost = total_cost / position.quantity if position.quantity > 0 else 0
        
        else:
            # Vente - réduire position
            realized_pnl = order.filled_quantity * (order.avg_fill_price - position.avg_cost)
            position.realized_pnl += realized_pnl
            position.quantity -= order.filled_quantity
            
            # Si position fermée
            if position.quantity <= 0:
                position.quantity = 0
                position.avg_cost = 0
        
        # Mettre à jour valeur de marché
        current_price = self._get_estimated_price(symbol)
        position.market_value = position.quantity * current_price
        position.unrealized_pnl = position.quantity * (current_price - position.avg_cost)
        position.last_updated = datetime.utcnow()
    
    def _get_estimated_price(self, symbol: str) -> float:
        """Obtenir prix estimé d'un symbole (simulation)"""
        # Simulation - En Phase 5, utiliser données temps réel
        
        # Prix de base par symbole
        base_prices = {
            'AAPL': 175.0,
            'MSFT': 300.0,
            'GOOGL': 2800.0,
            'AMZN': 3200.0,
            'TSLA': 800.0,
            'SPY': 450.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Ajouter un peu de volatilité
        volatility = np.random.normal(0, 0.01)  # 1% vol
        return base_price * (1 + volatility)
    
    def _get_portfolio_value(self) -> float:
        """Obtenir la valeur totale du portefeuille"""
        if not self.ib_connection:
            return 100000.0  # Valeur par défaut
        
        total_value = 0.0
        
        # Valeur des positions
        for position in self.positions.values():
            total_value += position.market_value
        
        # Cash disponible
        total_value += self.ib_connection.get('buying_power', 0)
        
        return total_value
    
    async def _publish_execution_update(self, order: Order, processing_time_ms: float):
        """Publier une mise à jour d'exécution"""
        
        signal = Signal(
            id=None,
            type=SignalType.SYSTEM_STATUS,
            source_agent="execution_agent",
            symbol=order.symbol,
            priority=SignalPriority.MEDIUM,
            data={
                'order_id': order.order_id,
                'status': order.status.value,
                'side': order.side.value,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'avg_fill_price': order.avg_fill_price,
                'commission': order.commission,
                'processing_time_ms': processing_time_ms
            },
            metadata={
                'execution_venue': 'IBKR',
                'order_type': order.order_type.value,
                'created_at': order.created_at.isoformat() if order.created_at else None,
                'filled_at': order.filled_at.isoformat() if order.filled_at else None
            }
        )
        
        await self.signal_hub.publish_signal(signal)
    
    async def _monitor_orders(self):
        """Monitor les ordres en attente"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                timeout_minutes = self.execution_params['order_timeout_minutes']
                
                expired_orders = []
                for order_id, order in self.pending_orders.items():
                    if order.created_at and (current_time - order.created_at).total_seconds() > timeout_minutes * 60:
                        expired_orders.append(order_id)
                
                # Annuler les ordres expirés
                for order_id in expired_orders:
                    await self._cancel_order(order_id, "Timeout")
                
                await asyncio.sleep(30)  # Check toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"❌ Erreur monitoring ordres: {e}")
                await asyncio.sleep(60)
    
    async def _update_positions(self):
        """Mettre à jour les positions périodiquement"""
        while self.is_running:
            try:
                # Mettre à jour les valeurs de marché
                for position in self.positions.values():
                    current_price = self._get_estimated_price(position.symbol)
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = position.quantity * (current_price - position.avg_cost)
                    position.last_updated = datetime.utcnow()
                
                await asyncio.sleep(60)  # Update toutes les minutes
                
            except Exception as e:
                logger.error(f"❌ Erreur update positions: {e}")
                await asyncio.sleep(120)
    
    async def _cancel_order(self, order_id: str, reason: str = "Manual"):
        """Annuler un ordre"""
        if order_id not in self.pending_orders:
            return False
        
        order = self.pending_orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.error_message = f"Cancelled: {reason}"
        
        del self.pending_orders[order_id]
        
        logger.info(f"❌ Ordre annulé: {order_id} - {reason}")
        return True
    
    async def _cancel_pending_orders(self):
        """Annuler tous les ordres en attente"""
        order_ids = list(self.pending_orders.keys())
        for order_id in order_ids:
            await self._cancel_order(order_id, "Agent shutdown")
    
    async def _cancel_non_essential_orders(self):
        """Annuler les ordres non essentiels (risque élevé)"""
        cancelled_count = 0
        
        for order_id, order in list(self.pending_orders.items()):
            # Garder seulement les ordres de vente (réduction risque)
            if order.side == OrderSide.BUY:
                await self._cancel_order(order_id, "Risk management")
                cancelled_count += 1
        
        logger.info(f"🛡️ {cancelled_count} ordres d'achat annulés (gestion risque)")
    
    async def _emergency_risk_reduction(self):
        """Réduction d'urgence de l'exposition"""
        logger.warning("🚨 Mode réduction d'urgence activé")
        
        # Annuler tous les ordres d'achat
        await self._cancel_non_essential_orders()
        
        # Créer des ordres de vente pour réduire les positions importantes
        portfolio_value = self._get_portfolio_value()
        emergency_orders = []
        
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                position_weight = position.market_value / portfolio_value
                
                # Réduire les positions > 10%
                if position_weight > 0.10:
                    reduction_quantity = position.quantity * 0.5  # Réduire de 50%
                    
                    emergency_order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=reduction_quantity,
                        order_type=OrderType.MARKET,
                        created_at=datetime.utcnow()
                    )
                    
                    emergency_orders.append(emergency_order)
        
        # Soumettre les ordres d'urgence
        for order in emergency_orders:
            await self._submit_order(order)
        
        logger.warning(f"🚨 {len(emergency_orders)} ordres d'urgence soumis")
    
    async def _load_positions(self):
        """Charger les positions existantes (simulation)"""
        # En Phase 5, charger depuis IBKR
        # Pour Phase 4, créer des positions de test
        
        test_positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100,
                avg_cost=150.0,
                market_value=17500.0,
                unrealized_pnl=2500.0,
                realized_pnl=0.0,
                last_updated=datetime.utcnow()
            )
        }
        
        self.positions.update(test_positions)
        logger.info(f"📊 {len(self.positions)} positions chargées")
    
    def get_execution_report(self) -> ExecutionReport:
        """Générer un rapport d'exécution"""
        
        today_orders = [o for o in self.execution_history 
                       if o.created_at and o.created_at.date() == datetime.utcnow().date()]
        
        filled_orders = [o for o in today_orders if o.status == OrderStatus.FILLED]
        rejected_orders = [o for o in today_orders if o.status == OrderStatus.REJECTED]
        
        total_volume = sum(o.filled_quantity * (o.avg_fill_price or 0) for o in filled_orders)
        total_commission = sum(o.commission for o in filled_orders)
        
        fill_rate = len(filled_orders) / len(today_orders) if today_orders else 0
        
        return ExecutionReport(
            orders_submitted=len(today_orders),
            orders_filled=len(filled_orders),
            orders_rejected=len(rejected_orders),
            total_volume=total_volume,
            total_commission=total_commission,
            slippage_bps=self.metrics['avg_slippage_bps'],
            execution_time_ms=50.0,  # Moyenne simulée
            fill_rate=fill_rate
        )
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """Résumé des positions"""
        
        total_value = sum(p.market_value for p in self.positions.values())
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in self.positions.values())
        
        return {
            'total_positions': len(self.positions),
            'total_market_value': total_value,
            'total_pnl': total_pnl,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_cost': pos.avg_cost,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'weight': pos.market_value / total_value if total_value > 0 else 0
                }
                for symbol, pos in self.positions.items()
                if pos.quantity > 0
            }
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Obtenir le statut de l'agent"""
        return {
            'name': 'execution_agent',
            'version': '1.0',
            'is_running': self.is_running,
            'connection_status': 'connected' if self.is_connected else 'disconnected',
            'broker': 'Interactive Brokers',
            'account': self.ib_connection.get('account') if self.ib_connection else None,
            'pending_orders': len(self.pending_orders),
            'positions_count': len([p for p in self.positions.values() if p.quantity > 0]),
            'metrics': self.metrics,
            'execution_params': self.execution_params
        }