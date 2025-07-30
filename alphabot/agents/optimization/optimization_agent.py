#!/usr/bin/env python3
"""
Optimization Agent - AlphaBot Multi-Agent Trading System
Optimisation de portefeuille avec Hierarchical Risk Parity (HRP)
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import cluster
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import warnings

from alphabot.core.signal_hub import (
    Signal, SignalType, SignalPriority, get_signal_hub
)
from alphabot.core.config import get_settings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class PortfolioWeights:
    """Poids d'allocation de portefeuille"""
    symbols: List[str]
    weights: np.ndarray
    method: str
    risk_contribution: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    timestamp: datetime


@dataclass
class OptimizationSignal:
    """Signal d'optimisation de portefeuille"""
    portfolio_weights: PortfolioWeights
    rebalancing_trades: Dict[str, float]  # symbole -> poids delta
    confidence: float
    risk_budget_utilization: float
    diversification_ratio: float
    key_insights: List[str]


class OptimizationAgent:
    """Agent d'optimisation de portefeuille avec HRP"""
    
    def __init__(self):
        self.settings = get_settings()
        self.signal_hub = get_signal_hub()
        self.is_running = False
        
        # Paramètres d'optimisation
        self.optimization_params = {
            'lookback_days': 252,  # 1 an de données
            'min_weight': 0.01,    # 1% minimum par position
            'max_weight': 0.20,    # 20% maximum par position
            'risk_target': 0.15,   # 15% volatilité cible
            'rebalancing_threshold': 0.05,  # 5% drift pour rebalancement
            'correlation_threshold': 0.7,   # Seuil corrélation élevée
        }
        
        # Cache des données
        self.price_data_cache = {}
        self.correlation_matrix_cache = {}
        self.current_portfolio = None
        
        # Métriques
        self.metrics = {
            'optimizations_performed': 0,
            'avg_sharpe_improvement': 0.0,
            'avg_diversification_ratio': 0.0,
            'rebalancing_frequency': 0.0
        }
    
    async def start(self):
        """Démarrer l'agent"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("⚖️ Optimization Agent démarré")
        
        # S'abonner aux signaux pertinents
        await self.signal_hub.subscribe_to_signals(
            agent_name="optimization_agent",
            callback=self._handle_signal,
            signal_types=[
                SignalType.PORTFOLIO_REBALANCE,
                SignalType.RISK_ALERT,
                SignalType.FUNDAMENTAL_SIGNAL
            ]
        )
        
        # Publier le statut
        await self.signal_hub.publish_agent_status(
            "optimization_agent",
            "started",
            {
                "version": "1.0",
                "methods": ["hrp", "equal_weight", "risk_parity"],
                "risk_target": self.optimization_params['risk_target']
            }
        )
    
    async def stop(self):
        """Arrêter l'agent"""
        self.is_running = False
        await self.signal_hub.publish_agent_status("optimization_agent", "stopped")
        logger.info("⚖️ Optimization Agent arrêté")
    
    async def _handle_signal(self, signal: Signal):
        """Traiter un signal reçu"""
        try:
            if signal.type == SignalType.PORTFOLIO_REBALANCE:
                symbols = signal.data.get('symbols', [])
                if symbols:
                    await self._optimize_portfolio(symbols)
            
            elif signal.type == SignalType.RISK_ALERT:
                # Réoptimiser en cas d'alerte risque
                if self.current_portfolio:
                    await self._emergency_rebalance()
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal optimization: {e}")
    
    async def _optimize_portfolio(self, symbols: List[str], method: str = "hrp"):
        """Optimiser un portefeuille"""
        try:
            start_time = time.time()
            logger.info(f"⚖️ Optimisation portefeuille: {len(symbols)} actifs, méthode: {method}")
            
            # Récupérer les données de prix
            price_data = await self._fetch_price_data(symbols)
            if price_data is None or price_data.empty:
                logger.warning("⚠️ Pas de données de prix pour l'optimisation")
                return
            
            # Calculer les rendements
            returns = price_data.pct_change().dropna()
            
            # Optimiser selon la méthode
            if method == "hrp":
                weights = self._hierarchical_risk_parity(returns)
            elif method == "equal_weight":
                weights = self._equal_weight_optimization(symbols)
            elif method == "risk_parity":
                weights = self._risk_parity_optimization(returns)
            else:
                logger.error(f"Méthode d'optimisation inconnue: {method}")
                return
            
            # Calculer les métriques de performance
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            # Créer l'objet PortfolioWeights
            portfolio_weights = PortfolioWeights(
                symbols=symbols,
                weights=weights,
                method=method,
                risk_contribution=self._calculate_risk_contribution(returns, weights),
                expected_return=portfolio_metrics['expected_return'],
                expected_volatility=portfolio_metrics['expected_volatility'],
                sharpe_ratio=portfolio_metrics['sharpe_ratio'],
                max_drawdown=portfolio_metrics['max_drawdown'],
                timestamp=datetime.utcnow()
            )
            
            # Calculer les trades de rebalancement
            rebalancing_trades = self._calculate_rebalancing_trades(portfolio_weights)
            
            # Créer le signal d'optimisation
            optimization_signal = OptimizationSignal(
                portfolio_weights=portfolio_weights,
                rebalancing_trades=rebalancing_trades,
                confidence=self._calculate_optimization_confidence(portfolio_weights),
                risk_budget_utilization=self._calculate_risk_budget_utilization(portfolio_weights),
                diversification_ratio=self._calculate_diversification_ratio(returns, weights),
                key_insights=self._generate_optimization_insights(portfolio_weights, returns)
            )
            
            # Publier le signal
            await self._publish_optimization_signal(optimization_signal)
            
            # Mettre à jour le portfolio courant
            self.current_portfolio = portfolio_weights
            
            # Métriques
            self.metrics['optimizations_performed'] += 1
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Optimisation terminée en {processing_time:.1f}ms - Sharpe: {portfolio_weights.sharpe_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Erreur optimisation portefeuille: {e}")
    
    def _hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Implémentation Hierarchical Risk Parity (HRP)"""
        
        # 1. Calculer la matrice de corrélation
        corr_matrix = returns.corr()
        
        # 2. Calculer la matrice de distance
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # 3. Clustering hiérarchique
        distance_condensed = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(distance_condensed, method='ward')
        
        # 4. Réorganiser selon le clustering
        sorted_indices = self._get_quasi_diagonal_order(linkage_matrix)
        
        # 5. Allocation récursive
        weights = self._recursive_bisection(returns.iloc[:, sorted_indices])
        
        # 6. Remettre dans l'ordre original
        final_weights = np.zeros(len(returns.columns))
        for i, original_idx in enumerate(sorted_indices):
            final_weights[original_idx] = weights[i]
        
        # 7. Appliquer les contraintes
        final_weights = self._apply_weight_constraints(final_weights)
        
        return final_weights
    
    def _get_quasi_diagonal_order(self, linkage_matrix: np.ndarray) -> List[int]:
        """Obtenir l'ordre quasi-diagonal du dendrogramme"""
        n = linkage_matrix.shape[0] + 1
        
        # Créer le dendrogramme et extraire l'ordre
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro['leaves']
        
        return order
    
    def _recursive_bisection(self, returns: pd.DataFrame) -> np.ndarray:
        """Allocation récursive par bisection"""
        
        def _get_cluster_variance(cluster_returns):
            """Calculer la variance du cluster"""
            if cluster_returns.shape[1] == 1:
                return cluster_returns.var().iloc[0]
            
            # Weights égaux pour le cluster
            weights = np.ones(cluster_returns.shape[1]) / cluster_returns.shape[1]
            portfolio_returns = (cluster_returns * weights).sum(axis=1)
            return portfolio_returns.var()
        
        def _bisect_allocation(cluster_returns, weights):
            """Allocation par bisection récursive"""
            if cluster_returns.shape[1] == 1:
                return weights
            
            # Diviser en deux clusters
            mid = cluster_returns.shape[1] // 2
            left_cluster = cluster_returns.iloc[:, :mid]
            right_cluster = cluster_returns.iloc[:, mid:]
            
            # Calculer les variances
            left_var = _get_cluster_variance(left_cluster)
            right_var = _get_cluster_variance(right_cluster)
            
            # Allocation inverse de la variance
            total_var = left_var + right_var
            if total_var > 0:
                left_weight = right_var / total_var
                right_weight = left_var / total_var
            else:
                left_weight = right_weight = 0.5
            
            # Normaliser
            left_weight = max(0.1, min(0.9, left_weight))  # Contraintes
            right_weight = 1 - left_weight
            
            # Allocation récursive
            left_weights = _bisect_allocation(left_cluster, weights[:mid] * left_weight)
            right_weights = _bisect_allocation(right_cluster, weights[mid:] * right_weight)
            
            return np.concatenate([left_weights, right_weights])
        
        # Initialiser avec des poids égaux
        initial_weights = np.ones(returns.shape[1]) / returns.shape[1]
        
        return _bisect_allocation(returns, initial_weights)
    
    def _equal_weight_optimization(self, symbols: List[str]) -> np.ndarray:
        """Optimisation poids égaux (1/N)"""
        n = len(symbols)
        weights = np.ones(n) / n
        return self._apply_weight_constraints(weights)
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> np.ndarray:
        """Optimisation Risk Parity (contribution égale au risque)"""
        
        # Calculer la matrice de covariance
        cov_matrix = returns.cov().values
        
        # Algorithme itératif pour Risk Parity
        n = len(returns.columns)
        weights = np.ones(n) / n  # Initialisation
        
        for _ in range(50):  # Max 50 itérations
            # Calculer les contributions de risque
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Mise à jour des poids (algorithme simple)
            target_contrib = portfolio_vol / n
            weights = weights * (target_contrib / contrib) ** 0.5
            
            # Renormaliser
            weights = weights / weights.sum()
            
            # Vérifier convergence
            contrib_std = np.std(contrib)
            if contrib_std < 1e-6:
                break
        
        return self._apply_weight_constraints(weights)
    
    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Appliquer les contraintes de poids"""
        
        # Contraintes min/max
        weights = np.clip(weights, 
                         self.optimization_params['min_weight'],
                         self.optimization_params['max_weight'])
        
        # Renormaliser
        weights = weights / weights.sum()
        
        return weights
    
    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Calculer les métriques de performance du portefeuille"""
        
        # Rendements du portefeuille
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Métriques annualisées
        expected_return = portfolio_returns.mean() * 252
        expected_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_risk_contribution(self, returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        """Calculer la contribution de chaque actif au risque total"""
        
        cov_matrix = returns.cov().values
        portfolio_variance = weights.T @ cov_matrix @ weights
        
        if portfolio_variance > 0:
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_variance
        else:
            risk_contrib = weights / len(weights)
        
        return risk_contrib
    
    def _calculate_rebalancing_trades(self, new_portfolio: PortfolioWeights) -> Dict[str, float]:
        """Calculer les trades nécessaires pour le rebalancement"""
        
        if self.current_portfolio is None:
            # Premier portefeuille - tous les poids sont des achats
            return {symbol: weight for symbol, weight in zip(new_portfolio.symbols, new_portfolio.weights)}
        
        trades = {}
        
        # Créer un dictionnaire des anciens poids
        old_weights = {symbol: weight for symbol, weight in 
                      zip(self.current_portfolio.symbols, self.current_portfolio.weights)}
        
        # Calculer les différences
        for symbol, new_weight in zip(new_portfolio.symbols, new_portfolio.weights):
            old_weight = old_weights.get(symbol, 0.0)
            weight_diff = new_weight - old_weight
            
            # Ne trader que si la différence dépasse le seuil
            if abs(weight_diff) >= self.optimization_params['rebalancing_threshold']:
                trades[symbol] = weight_diff
        
        return trades
    
    def _calculate_optimization_confidence(self, portfolio: PortfolioWeights) -> float:
        """Calculer la confiance dans l'optimisation"""
        
        confidence = 0.5  # Base
        
        # Bonus pour Sharpe ratio élevé
        if portfolio.sharpe_ratio > 1.0:
            confidence += 0.3
        elif portfolio.sharpe_ratio > 0.5:
            confidence += 0.1
        
        # Bonus pour diversification (poids équilibrés)
        weight_concentration = np.sum(portfolio.weights ** 2)  # Herfindahl index
        max_concentration = 1.0 / len(portfolio.weights)
        
        if weight_concentration < 2 * max_concentration:
            confidence += 0.2
        
        # Malus pour volatilité excessive
        if portfolio.expected_volatility > 0.25:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_risk_budget_utilization(self, portfolio: PortfolioWeights) -> float:
        """Calculer l'utilisation du budget de risque"""
        
        risk_target = self.optimization_params['risk_target']
        utilization = portfolio.expected_volatility / risk_target
        
        return min(1.0, utilization)
    
    def _calculate_diversification_ratio(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculer le ratio de diversification"""
        
        # Volatilité pondérée des actifs individuels
        individual_vols = returns.std() * np.sqrt(252)
        weighted_vol = np.sum(weights * individual_vols)
        
        # Volatilité du portefeuille
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Ratio de diversification
        if portfolio_vol > 0:
            diversification_ratio = weighted_vol / portfolio_vol
        else:
            diversification_ratio = 1.0
        
        return diversification_ratio
    
    def _generate_optimization_insights(self, portfolio: PortfolioWeights, returns: pd.DataFrame) -> List[str]:
        """Générer des insights sur l'optimisation"""
        
        insights = []
        
        # Insight sur la concentration
        max_weight = np.max(portfolio.weights)
        max_symbol = portfolio.symbols[np.argmax(portfolio.weights)]
        
        if max_weight > 0.15:
            insights.append(f"Concentration élevée: {max_symbol} ({max_weight:.1%})")
        
        # Insight sur la diversification
        div_ratio = self._calculate_diversification_ratio(returns, portfolio.weights)
        if div_ratio > 1.3:
            insights.append(f"Excellente diversification (ratio: {div_ratio:.2f})")
        elif div_ratio < 1.1:
            insights.append(f"Diversification limitée (ratio: {div_ratio:.2f})")
        
        # Insight sur le Sharpe ratio
        if portfolio.sharpe_ratio > 1.5:
            insights.append("Excellent ratio risque/rendement")
        elif portfolio.sharpe_ratio < 0.5:
            insights.append("Ratio risque/rendement faible")
        
        # Insight sur la volatilité
        if portfolio.expected_volatility < 0.10:
            insights.append("Portefeuille conservateur (faible volatilité)")
        elif portfolio.expected_volatility > 0.20:
            insights.append("Portefeuille agressif (volatilité élevée)")
        
        return insights[:3]  # Max 3 insights
    
    async def _publish_optimization_signal(self, optimization_signal: OptimizationSignal):
        """Publier un signal d'optimisation"""
        
        portfolio = optimization_signal.portfolio_weights
        
        signal = Signal(
            id=None,
            type=SignalType.PORTFOLIO_REBALANCE,
            source_agent="optimization_agent",
            priority=SignalPriority.HIGH,
            data={
                'method': portfolio.method,
                'weights': {symbol: float(weight) for symbol, weight in 
                           zip(portfolio.symbols, portfolio.weights)},
                'trades': optimization_signal.rebalancing_trades,
                'expected_return': portfolio.expected_return,
                'expected_volatility': portfolio.expected_volatility,
                'sharpe_ratio': portfolio.sharpe_ratio,
                'confidence': optimization_signal.confidence,
                'diversification_ratio': optimization_signal.diversification_ratio,
                'insights': optimization_signal.key_insights
            },
            metadata={
                'optimization_method': portfolio.method,
                'risk_budget_utilization': optimization_signal.risk_budget_utilization,
                'max_drawdown': portfolio.max_drawdown,
                'num_assets': len(portfolio.symbols)
            }
        )
        
        await self.signal_hub.publish_signal(signal)
        logger.info(f"📊 Signal optimisation publié: {portfolio.method} - Sharpe {portfolio.sharpe_ratio:.2f}")
    
    async def _fetch_price_data(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Récupérer les données de prix (simulation pour Phase 4)"""
        try:
            # SIMULATION - En production, utiliser yfinance ou APIs
            
            lookback_days = self.optimization_params['lookback_days']
            dates = pd.date_range(
                end=datetime.now(),
                periods=lookback_days,
                freq='D'
            )
            
            # Générer des prix réalistes avec corrélations
            np.random.seed(42)  # Reproductible
            
            price_data = {}
            base_price = 100.0
            
            for i, symbol in enumerate(symbols):
                # Prix avec drift et volatilité
                returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% drift daily, 2% vol
                
                # Ajouter corrélation entre actifs
                if i > 0:
                    correlation = 0.3  # 30% corrélation
                    prev_returns = list(price_data.values())[0].pct_change().fillna(0)
                    returns = correlation * prev_returns.values + np.sqrt(1 - correlation**2) * returns
                
                # Calculer les prix
                prices = [base_price * (1 + i * 0.1)]  # Prix initial différent par actif
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                price_data[symbol] = pd.Series(prices, index=dates)
            
            df = pd.DataFrame(price_data)
            
            # Mettre en cache
            cache_key = "_".join(sorted(symbols))
            self.price_data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données prix: {e}")
            return None
    
    async def _emergency_rebalance(self):
        """Rebalancement d'urgence en cas d'alerte risque"""
        if not self.current_portfolio:
            return
        
        logger.warning("⚠️ Rebalancement d'urgence déclenché")
        
        # Réduire l'exposition au risque (méthode conservative)
        conservative_weights = self.current_portfolio.weights * 0.8  # Réduire de 20%
        conservative_weights = conservative_weights / conservative_weights.sum()  # Renormaliser
        
        # Le reste en cash (simulation)
        cash_allocation = 0.2
        
        logger.info(f"💰 Allocation cash d'urgence: {cash_allocation:.1%}")
    
    def get_current_portfolio(self) -> Optional[PortfolioWeights]:
        """Obtenir le portefeuille courant"""
        return self.current_portfolio
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Obtenir le statut de l'agent"""
        return {
            'name': 'optimization_agent',
            'version': '1.0',
            'is_running': self.is_running,
            'optimization_methods': ['hrp', 'equal_weight', 'risk_parity'],
            'current_portfolio': {
                'symbols': self.current_portfolio.symbols if self.current_portfolio else [],
                'method': self.current_portfolio.method if self.current_portfolio else None,
                'sharpe_ratio': self.current_portfolio.sharpe_ratio if self.current_portfolio else None
            },
            'metrics': self.metrics,
            'cache_size': len(self.price_data_cache)
        }