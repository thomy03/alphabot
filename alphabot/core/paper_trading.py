"""
AlphaBot Paper Trading Engine
Simulation de trading en temps réel avec données live
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import yfinance as yf
import json
import logging
from enum import Enum

from alphabot.core.config import get_settings
from alphabot.core.signal_hub import Signal, SignalType, SignalPriority, get_signal_hub
from alphabot.core.crew_orchestrator import CrewOrchestrator

class PaperOrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PaperOrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

@dataclass
class PaperOrder:
    """Ordre de paper trading"""
    id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    order_type: PaperOrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    status: PaperOrderStatus = PaperOrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    
    commission: float = 0.0
    fees: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'filled_price': self.filled_price,
            'filled_quantity': self.filled_quantity,
            'commission': self.commission,
            'fees': self.fees
        }

@dataclass  
class PaperPosition:
    """Position de paper trading"""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_market_value(self, current_price: float):
        """Met à jour la valeur de marché"""
        self.last_price = current_price
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = self.market_value - (self.quantity * self.avg_cost)
        self.last_updated = datetime.utcnow()

@dataclass
class PaperTradingConfig:
    """Configuration du paper trading"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_bps: int = 5  # 5 basis points
    
    # Contraintes de risque
    max_position_size: float = 0.05  # 5% max par position
    max_total_exposure: float = 0.95  # 95% max investi
    min_order_value: float = 1000.0   # $1000 minimum
    
    # Fréquences
    price_update_interval: int = 30   # 30 secondes
    portfolio_update_interval: int = 60  # 1 minute
    signal_check_interval: int = 10   # 10 secondes
    
    # Données
    data_source: str = "yfinance"
    market_hours_only: bool = True
    
    # Persistence
    save_trades: bool = True
    save_positions: bool = True
    data_directory: str = "paper_trading_data"

class PaperTradingEngine:
    """Engine de paper trading en temps réel"""
    
    def __init__(self, config: PaperTradingConfig = None):
        self.config = config or PaperTradingConfig()
        self.settings = get_settings()
        
        # État du portfolio
        self.cash = self.config.initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.execution_history: List[PaperOrder] = []
        
        # Données de marché
        self.current_prices: Dict[str, float] = {}
        self.last_price_update: Optional[datetime] = None
        
        # Agents
        self.crew_orchestrator: Optional[CrewOrchestrator] = None
        self.signal_hub = get_signal_hub()
        
        # État
        self.is_running = False
        self.is_market_open = False
        
        # Métriques en temps réel
        self.metrics = {
            'total_value': self.config.initial_capital,
            'total_pnl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'daily_returns': []
        }
        
        # Logger
        self.logger = logging.getLogger('PaperTrading')
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le logging"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def start(self):
        """Démarre le paper trading"""
        self.logger.info("🚀 Démarrage Paper Trading Engine...")
        
        # Initialiser agents
        self.crew_orchestrator = CrewOrchestrator()
        await self.crew_orchestrator.initialize()
        
        # Créer répertoires
        data_dir = Path(self.config.data_directory)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # S'abonner aux signaux
        await self.signal_hub.subscribe_to_signals(self._handle_trading_signal)
        
        self.is_running = True
        
        # Lancer tâches parallèles
        tasks = [
            asyncio.create_task(self._price_update_loop()),
            asyncio.create_task(self._portfolio_update_loop()),
            asyncio.create_task(self._signal_generation_loop()),
            asyncio.create_task(self._order_management_loop())
        ]
        
        self.logger.info("✅ Paper Trading démarré")
        
        # Attendre arrêt
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("🛑 Arrêt du Paper Trading")
        
    async def stop(self):
        """Arrête le paper trading"""
        self.is_running = False
        
        # Sauvegarder état final
        await self._save_session_data()
        
        if self.crew_orchestrator:
            await self.crew_orchestrator.cleanup()
        
        self.logger.info("✅ Paper Trading arrêté")
    
    async def _price_update_loop(self):
        """Boucle de mise à jour des prix"""
        while self.is_running:
            try:
                if self._is_market_open():
                    await self._update_market_prices()
                    await self._process_pending_orders()
                
                await asyncio.sleep(self.config.price_update_interval)
                
            except Exception as e:
                self.logger.error(f"Erreur mise à jour prix: {e}")
                await asyncio.sleep(5)
    
    async def _portfolio_update_loop(self):
        """Boucle de mise à jour du portfolio"""
        while self.is_running:
            try:
                await self._update_portfolio_metrics()
                await self._save_portfolio_snapshot()
                
                await asyncio.sleep(self.config.portfolio_update_interval)
                
            except Exception as e:
                self.logger.error(f"Erreur mise à jour portfolio: {e}")
                await asyncio.sleep(10)
    
    async def _signal_generation_loop(self):
        """Boucle de génération des signaux"""
        while self.is_running:
            try:
                if self._is_market_open() and self.crew_orchestrator:
                    # Déclencher analyse des agents
                    await self._trigger_agent_analysis()
                
                await asyncio.sleep(self.config.signal_check_interval)
                
            except Exception as e:
                self.logger.error(f"Erreur génération signaux: {e}")
                await asyncio.sleep(30)
    
    async def _order_management_loop(self):
        """Boucle de gestion des ordres"""
        while self.is_running:
            try:
                # Nettoyer ordres expirés
                await self._cleanup_expired_orders()
                
                # Vérifier ordres stop
                await self._check_stop_orders()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Erreur gestion ordres: {e}")
                await asyncio.sleep(5)
    
    def _is_market_open(self) -> bool:
        """Vérifie si le marché est ouvert"""
        if not self.config.market_hours_only:
            return True
        
        now = datetime.now()
        
        # Simplification: marché ouvert 9h30-16h EST, lundi-vendredi
        if now.weekday() >= 5:  # Weekend
            return False
        
        # Approximation heures de marché (à affiner avec timezone)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    async def _update_market_prices(self):
        """Met à jour les prix de marché"""
        try:
            # Obtenir tous les symboles actifs
            symbols = set()
            symbols.update(self.positions.keys())
            symbols.update([order.symbol for order in self.orders.values()])
            
            if not symbols:
                return
            
            # Télécharger prix actuels (simplification avec yfinance)
            symbols_list = list(symbols)
            
            # Pour demo, simuler prix avec variation aléatoire
            for symbol in symbols_list:
                if symbol in self.current_prices:
                    # Variation ±0.5% max
                    change = np.random.normal(0, 0.005)
                    self.current_prices[symbol] *= (1 + change)
                else:
                    # Prix initial simulé
                    self.current_prices[symbol] = np.random.uniform(50, 500)
            
            self.last_price_update = datetime.utcnow()
            
            # Mettre à jour positions
            for position in self.positions.values():
                if position.symbol in self.current_prices:
                    position.update_market_value(self.current_prices[position.symbol])
            
        except Exception as e:
            self.logger.error(f"Erreur mise à jour prix: {e}")
    
    async def _process_pending_orders(self):
        """Traite les ordres en attente"""
        filled_orders = []
        
        for order in self.orders.values():
            if order.status != PaperOrderStatus.PENDING:
                continue
            
            current_price = self.current_prices.get(order.symbol)
            if current_price is None:
                continue
            
            # Simuler exécution selon type d'ordre
            should_fill, fill_price = self._should_fill_order(order, current_price)
            
            if should_fill:
                await self._fill_order(order, fill_price)
                filled_orders.append(order.id)
        
        # Nettoyer ordres exécutés
        for order_id in filled_orders:
            if order_id in self.orders:
                del self.orders[order_id]
    
    def _should_fill_order(self, order: PaperOrder, current_price: float) -> tuple[bool, float]:
        """Détermine si un ordre doit être exécuté"""
        
        if order.order_type == PaperOrderType.MARKET:
            # Ordre au marché: exécution immédiate avec slippage
            slippage = self.config.slippage_bps / 10000
            if order.side == "BUY":
                fill_price = current_price * (1 + slippage)
            else:
                fill_price = current_price * (1 - slippage)
            return True, fill_price
        
        elif order.order_type == PaperOrderType.LIMIT:
            # Ordre à cours limité
            if order.price is None:
                return False, 0.0
            
            if order.side == "BUY" and current_price <= order.price:
                return True, order.price
            elif order.side == "SELL" and current_price >= order.price:
                return True, order.price
        
        # Autres types d'ordres à implémenter
        return False, 0.0
    
    async def _fill_order(self, order: PaperOrder, fill_price: float):
        """Exécute un ordre"""
        try:
            # Calculer commission
            order_value = order.quantity * fill_price
            commission = order_value * self.config.commission_rate
            
            # Vérifier liquidité
            if order.side == "BUY":
                total_cost = order_value + commission
                if total_cost > self.cash:
                    order.status = PaperOrderStatus.REJECTED
                    self.logger.warning(f"Ordre {order.id} rejeté: liquidité insuffisante")
                    return
                
                self.cash -= total_cost
            else:
                self.cash += (order_value - commission)
            
            # Mettre à jour position
            await self._update_position(order.symbol, order.side, order.quantity, fill_price)
            
            # Finaliser ordre
            order.status = PaperOrderStatus.FILLED
            order.filled_at = datetime.utcnow()
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.commission = commission
            
            self.execution_history.append(order)
            self.metrics['total_trades'] += 1
            
            self.logger.info(f"✅ Ordre exécuté: {order.side} {order.quantity} {order.symbol} @ ${fill_price:.2f}")
            
            # Publier signal d'exécution
            execution_signal = Signal(
                id=f"execution_{order.id}",
                type=SignalType.TRADE_EXECUTION,
                source_agent="paper_trading",
                priority=SignalPriority.MEDIUM,
                data=order.to_dict()
            )
            await self.signal_hub.publish_signal(execution_signal)
            
        except Exception as e:
            self.logger.error(f"Erreur exécution ordre {order.id}: {e}")
            order.status = PaperOrderStatus.REJECTED
    
    async def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """Met à jour une position"""
        
        if symbol not in self.positions:
            if side == "BUY":
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=price
                )
            # Vente à découvert non implémentée pour l'instant
            return
        
        position = self.positions[symbol]
        
        if side == "BUY":
            # Augmenter position
            total_cost = (position.quantity * position.avg_cost) + (quantity * price)
            total_quantity = position.quantity + quantity
            position.avg_cost = total_cost / total_quantity
            position.quantity = total_quantity
            
        else:  # SELL
            # Réduire position
            if quantity >= position.quantity:
                # Fermeture complète
                realized_pnl = (price - position.avg_cost) * position.quantity
                position.realized_pnl += realized_pnl
                del self.positions[symbol]
            else:
                # Fermeture partielle
                realized_pnl = (price - position.avg_cost) * quantity
                position.realized_pnl += realized_pnl
                position.quantity -= quantity
    
    async def _handle_trading_signal(self, signal: Signal):
        """Traite un signal de trading"""
        try:
            if signal.type == SignalType.BUY_SIGNAL:
                await self._process_buy_signal(signal)
            elif signal.type == SignalType.SELL_SIGNAL:
                await self._process_sell_signal(signal)
            elif signal.type == SignalType.PORTFOLIO_REBALANCE:
                await self._process_rebalance_signal(signal)
                
        except Exception as e:
            self.logger.error(f"Erreur traitement signal {signal.id}: {e}")
    
    async def _process_buy_signal(self, signal: Signal):
        """Traite un signal d'achat"""
        symbol = signal.data.get('symbol')
        target_weight = signal.data.get('weight', 0.02)  # 2% par défaut
        
        if not symbol:
            return
        
        # Calculer quantité cible
        portfolio_value = await self._get_portfolio_value()
        target_value = portfolio_value * target_weight
        
        # Vérifier contraintes
        if target_value < self.config.min_order_value:
            return
        
        current_price = self.current_prices.get(symbol)
        if not current_price:
            return
        
        quantity = target_value / current_price
        
        # Créer ordre
        order = PaperOrder(
            id=f"buy_{symbol}_{datetime.now().strftime('%H%M%S')}",
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            order_type=PaperOrderType.MARKET
        )
        
        self.orders[order.id] = order
        self.logger.info(f"📈 Ordre d'achat créé: {quantity:.2f} {symbol}")
    
    async def _process_sell_signal(self, signal: Signal):
        """Traite un signal de vente"""
        symbol = signal.data.get('symbol')
        sell_ratio = signal.data.get('ratio', 1.0)  # 100% par défaut
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        sell_quantity = position.quantity * sell_ratio
        
        # Créer ordre
        order = PaperOrder(
            id=f"sell_{symbol}_{datetime.now().strftime('%H%M%S')}",
            symbol=symbol,
            side="SELL",
            quantity=sell_quantity,
            order_type=PaperOrderType.MARKET
        )
        
        self.orders[order.id] = order
        self.logger.info(f"📉 Ordre de vente créé: {sell_quantity:.2f} {symbol}")
    
    async def _process_rebalance_signal(self, signal: Signal):
        """Traite un signal de rebalancement"""
        target_weights = signal.data.get('weights', {})
        
        portfolio_value = await self._get_portfolio_value()
        
        for symbol, target_weight in target_weights.items():
            if symbol == 'CASH':
                continue
            
            target_value = portfolio_value * target_weight
            current_value = 0.0
            
            if symbol in self.positions:
                current_value = self.positions[symbol].market_value
            
            delta_value = target_value - current_value
            
            if abs(delta_value) < self.config.min_order_value:
                continue
            
            current_price = self.current_prices.get(symbol)
            if not current_price:
                continue
            
            if delta_value > 0:
                # Acheter
                quantity = delta_value / current_price
                order = PaperOrder(
                    id=f"rebal_buy_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    order_type=PaperOrderType.MARKET
                )
                self.orders[order.id] = order
            else:
                # Vendre
                quantity = abs(delta_value) / current_price
                if symbol in self.positions:
                    quantity = min(quantity, self.positions[symbol].quantity)
                    order = PaperOrder(
                        id=f"rebal_sell_{symbol}_{datetime.now().strftime('%H%M%S')}",
                        symbol=symbol,
                        side="SELL",
                        quantity=quantity,
                        order_type=PaperOrderType.MARKET
                    )
                    self.orders[order.id] = order
    
    async def _trigger_agent_analysis(self):
        """Déclenche l'analyse des agents"""
        if not self.crew_orchestrator:
            return
        
        try:
            # Obtenir données de marché récentes
            market_data = {
                'current_prices': self.current_prices,
                'portfolio': await self.get_portfolio_summary(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Déclencher workflow CrewAI
            await self.crew_orchestrator.run_trading_workflow(market_data)
            
        except Exception as e:
            self.logger.error(f"Erreur analyse agents: {e}")
    
    async def _update_portfolio_metrics(self):
        """Met à jour les métriques du portfolio"""
        portfolio_value = await self._get_portfolio_value()
        
        # P&L total
        total_pnl = portfolio_value - self.config.initial_capital
        self.metrics['total_pnl'] = total_pnl
        self.metrics['total_value'] = portfolio_value
        
        # Rendement quotidien
        if len(self.metrics['daily_returns']) > 0:
            previous_value = self.metrics.get('previous_value', self.config.initial_capital)
            daily_return = (portfolio_value - previous_value) / previous_value
            self.metrics['daily_returns'].append(daily_return)
            
            # Garder seulement 252 jours (1 an)
            if len(self.metrics['daily_returns']) > 252:
                self.metrics['daily_returns'] = self.metrics['daily_returns'][-252:]
            
            # Calculer Sharpe
            if len(self.metrics['daily_returns']) > 30:
                returns = np.array(self.metrics['daily_returns'])
                self.metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Calculer max drawdown
            values = [self.config.initial_capital * (1 + sum(self.metrics['daily_returns'][:i+1])) 
                     for i in range(len(self.metrics['daily_returns']))]
            if values:
                running_max = np.maximum.accumulate(values)
                drawdowns = (np.array(values) - running_max) / running_max
                self.metrics['max_drawdown'] = np.min(drawdowns)
        
        self.metrics['previous_value'] = portfolio_value
        
        # Win rate
        if self.execution_history:
            profitable_trades = sum(1 for order in self.execution_history 
                                  if order.side == "SELL" and order.filled_price and order.symbol in self.positions)
            self.metrics['win_rate'] = profitable_trades / len(self.execution_history)
    
    async def _get_portfolio_value(self) -> float:
        """Calcule la valeur totale du portfolio"""
        total_value = self.cash
        
        for position in self.positions.values():
            total_value += position.market_value
        
        return total_value
    
    async def get_portfolio_summary(self) -> dict:
        """Retourne un résumé du portfolio"""
        portfolio_value = await self._get_portfolio_value()
        
        positions_summary = []
        for position in self.positions.values():
            weight = position.market_value / portfolio_value if portfolio_value > 0 else 0
            positions_summary.append({
                'symbol': position.symbol,
                'quantity': position.quantity,
                'avg_cost': position.avg_cost,
                'market_value': position.market_value,
                'weight': weight,
                'unrealized_pnl': position.unrealized_pnl,
                'last_price': position.last_price
            })
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_value': portfolio_value,
            'cash': self.cash,
            'invested_value': portfolio_value - self.cash,
            'total_pnl': self.metrics['total_pnl'],
            'total_return': self.metrics['total_pnl'] / self.config.initial_capital,
            'positions': positions_summary,
            'metrics': self.metrics,
            'active_orders': len(self.orders),
            'total_trades': len(self.execution_history)
        }
    
    async def _save_portfolio_snapshot(self):
        """Sauvegarde un snapshot du portfolio"""
        if not self.config.save_positions:
            return
        
        try:
            snapshot = await self.get_portfolio_summary()
            
            # Sauvegarder en JSON
            data_dir = Path(self.config.data_directory) / "snapshots"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(data_dir / f"portfolio_{timestamp}.json", 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde snapshot: {e}")
    
    async def _save_session_data(self):
        """Sauvegarde les données de session"""
        try:
            data_dir = Path(self.config.data_directory)
            
            # Sauvegarder historique des trades
            if self.config.save_trades and self.execution_history:
                trades_data = [order.to_dict() for order in self.execution_history]
                with open(data_dir / "trades_history.json", 'w') as f:
                    json.dump(trades_data, f, indent=2, default=str)
            
            # Sauvegarder état final
            final_summary = await self.get_portfolio_summary()
            with open(data_dir / "final_portfolio.json", 'w') as f:
                json.dump(final_summary, f, indent=2, default=str)
            
            self.logger.info("💾 Données de session sauvegardées")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde session: {e}")
    
    async def _cleanup_expired_orders(self):
        """Nettoie les ordres expirés"""
        # Implémenter logique d'expiration (ex: ordres > 1 jour)
        pass
    
    async def _check_stop_orders(self):
        """Vérifie les ordres stop"""
        # Implémenter logique stop-loss/take-profit
        pass

def create_paper_trading_engine(
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001
) -> PaperTradingEngine:
    """Factory pour créer un engine de paper trading"""
    config = PaperTradingConfig(
        initial_capital=initial_capital,
        commission_rate=commission_rate
    )
    return PaperTradingEngine(config)