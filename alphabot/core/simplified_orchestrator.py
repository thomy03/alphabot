#!/usr/bin/env python3
"""
Simplified Orchestrator - AlphaBot Optimized 3-Agent System
Orchestrateur simplifié avec 3 agents core pour performance maximale
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from alphabot.core.signal_hub import (
    Signal, SignalType, SignalPriority, get_signal_hub
)
from alphabot.core.config import get_settings
from alphabot.agents.risk.enhanced_risk_agent import EnhancedRiskAgent
from alphabot.agents.technical.simplified_technical_agent import SimplifiedTechnicalAgent
from alphabot.agents.execution.execution_agent import ExecutionAgent

logger = logging.getLogger(__name__)


class SimplifiedWorkflowType(Enum):
    """Types de workflows simplifiés"""
    WEEKLY_REBALANCE = "weekly_rebalance"
    RISK_CHECK = "risk_check"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SimplifiedTradingDecision:
    """Décision de trading simplifiée - focus core metrics"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    target_weight: float
    reasoning: List[str]
    risk_score: float
    technical_score: float
    cvar_95: float  # 🆕 CVaR instead of VaR
    ulcer_index: float  # 🆕 Downside volatility
    execution_timestamp: datetime


class SimplifiedOrchestrator:
    """
    Orchestrateur simplifié - 3 agents core avec pipeline asynchrone
    
    Architecture optimisée selon recommandations expert :
    - Technical Agent (EMA + RSI seulement)
    - Risk Agent (CVaR + TVaR)
    - Execution Agent (weekly rebalancing)
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.signal_hub = get_signal_hub()
        
        # 3 agents core seulement
        self.technical_agent = SimplifiedTechnicalAgent()
        self.risk_agent = EnhancedRiskAgent()
        self.execution_agent = ExecutionAgent()
        
        # Métriques performance
        self.last_rebalance = None
        self.performance_metrics = {}
        
        # Configuration simplifiée
        self.rebalance_frequency = timedelta(weeks=1)  # Weekly vs daily
        self.trade_threshold = 0.05  # 5% vs 0.5%
        
        logger.info("✅ Simplified Orchestrator initialized - 3 agents core")
    
    async def analyze_portfolio_async(self, symbols: List[str]) -> Dict[str, SimplifiedTradingDecision]:
        """
        Analyse portfolio avec pipeline asynchrone pour <50ms latency
        """
        start_time = time.time()
        
        try:
            # Pipeline asynchrone - tous agents en parallèle
            tasks = []
            
            for symbol in symbols:
                # Créer tâches parallèles pour chaque symbole
                task = self._analyze_symbol_parallel(symbol)
                tasks.append(task)
            
            # Exécution parallèle
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Consolidation résultats
            decisions = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing {symbols[i]}: {result}")
                    continue
                
                if result:
                    decisions[symbols[i]] = result
            
            latency = (time.time() - start_time) * 1000
            logger.info(f"⚡ Portfolio analysis completed in {latency:.1f}ms (target <50ms)")
            
            # Métriques
            self.performance_metrics['last_latency_ms'] = latency
            self.performance_metrics['symbols_analyzed'] = len(symbols)
            self.performance_metrics['success_rate'] = len(decisions) / len(symbols)
            
            return decisions
            
        except Exception as e:
            logger.error(f"❌ Portfolio analysis failed: {e}")
            return {}
    
    async def _analyze_symbol_parallel(self, symbol: str) -> Optional[SimplifiedTradingDecision]:
        """
        Analyse symbole avec 3 agents en parallèle
        """
        try:
            # Lancer analyses en parallèle
            technical_task = self._get_technical_signals_simplified(symbol)
            risk_task = self._get_risk_assessment_cvar(symbol)
            execution_task = self._get_execution_context(symbol)
            
            # Attendre résultats
            technical_result, risk_result, execution_result = await asyncio.gather(
                technical_task, risk_task, execution_task
            )
            
            # Combiner signaux (logique simplifiée)
            decision = self._combine_signals_simplified(
                symbol, technical_result, risk_result, execution_result
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return None
    
    async def _get_technical_signals_simplified(self, symbol: str) -> Dict[str, Any]:
        """
        Signaux techniques simplifiés - utilise le SimplifiedTechnicalAgent
        """
        try:
            # Utiliser le simplified technical agent directement
            signals = await self.technical_agent.get_simplified_signals(symbol)
            
            return {
                'score': signals.get('score', 0.5),
                'ema_signal': signals.get('ema_signal', 0),
                'rsi_value': signals.get('rsi_value', 50),
                'reasoning': signals.get('reasoning', [])
            }
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            return {'score': 0.5, 'reasoning': ['Technical analysis unavailable']}
    
    async def _get_risk_assessment_cvar(self, symbol: str) -> Dict[str, Any]:
        """
        Évaluation risque avec CVaR - utilise EnhancedRiskAgent
        """
        try:
            # Utiliser l'enhanced risk agent directement
            risk_assessment = await self.risk_agent.assess_single_asset_risk(symbol)
            
            return {
                'risk_score': risk_assessment.get('risk_score', 0.5),
                'cvar_95': risk_assessment.get('cvar_95', -0.05),
                'ulcer_index': risk_assessment.get('ulcer_index', 5.0),
                'reasoning': [
                    f"CVaR 95%: {risk_assessment.get('cvar_95', -0.05):.2%}",
                    f"Ulcer Index: {risk_assessment.get('ulcer_index', 5.0):.2f}"
                ]
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed for {symbol}: {e}")
            return {
                'risk_score': 0.5,
                'cvar_95': -0.05,
                'ulcer_index': 5.0,
                'reasoning': ['Risk assessment unavailable']
            }
    
    async def _get_execution_context(self, symbol: str) -> Dict[str, Any]:
        """
        Contexte d'exécution - position actuelle et contraintes
        """
        try:
            current_position = await self.execution_agent.get_position(symbol)
            sector_exposure = await self.execution_agent.get_sector_exposure(symbol)
            
            # Vérifier contraintes
            can_trade = True
            constraints = []
            
            if sector_exposure > 0.30:  # Max 30% par secteur
                can_trade = False
                constraints.append("Sector exposure limit exceeded")
            
            return {
                'current_position': current_position,
                'sector_exposure': sector_exposure,
                'can_trade': can_trade,
                'constraints': constraints
            }
            
        except Exception as e:
            logger.error(f"Execution context failed for {symbol}: {e}")
            return {
                'current_position': 0.0,
                'sector_exposure': 0.0,
                'can_trade': True,
                'constraints': []
            }
    
    def _combine_signals_simplified(self, symbol: str, technical: Dict, risk: Dict, execution: Dict) -> SimplifiedTradingDecision:
        """
        Combinaison signaux avec logique simplifiée
        """
        # Score technique (60% weight)
        technical_score = technical['score']
        
        # Score risque (40% weight)
        risk_score = risk['risk_score']
        
        # Score final pondéré
        final_score = 0.6 * technical_score + 0.4 * risk_score
        
        # Décision d'action
        if final_score > 0.7 and execution['can_trade']:
            action = "BUY"
            target_weight = min(0.05, final_score * 0.07)  # Max 5% par position
        elif final_score < 0.3:
            action = "SELL"
            target_weight = 0.0
        else:
            action = "HOLD"
            target_weight = execution['current_position']
        
        # Ajuster selon threshold de trade
        if abs(target_weight - execution['current_position']) < self.trade_threshold:
            action = "HOLD"
            target_weight = execution['current_position']
        
        # Reasoning consolidé
        reasoning = []
        reasoning.extend(technical.get('reasoning', []))
        reasoning.extend(risk.get('reasoning', []))
        reasoning.extend(execution.get('constraints', []))
        
        return SimplifiedTradingDecision(
            symbol=symbol,
            action=action,
            confidence=final_score,
            target_weight=target_weight,
            reasoning=reasoning,
            risk_score=risk_score,
            technical_score=technical_score,
            cvar_95=risk['cvar_95'],
            ulcer_index=risk['ulcer_index'],
            execution_timestamp=datetime.now()
        )
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """
        Calcul CVaR (Conditional Value at Risk) - recommandation expert
        Plus robuste que VaR pour tail risks
        """
        if len(returns) == 0:
            return -0.05
        
        var = np.percentile(returns, confidence_level * 100)
        cvar = returns[returns <= var].mean()
        return float(cvar)
    
    def _calculate_ulcer_index(self, prices: np.ndarray) -> float:
        """
        Calcul Ulcer Index - recommandation expert
        Focus sur downside volatility uniquement
        """
        if len(prices) < 2:
            return 5.0
        
        # Calcul des drawdowns
        cummax = np.maximum.accumulate(prices)
        drawdowns = (prices / cummax - 1) * 100
        
        # Ulcer Index = RMS des drawdowns
        ulcer = np.sqrt(np.mean(drawdowns ** 2))
        return float(ulcer)
    
    def _calculate_risk_score(self, cvar: float, ulcer: float) -> float:
        """
        Score de risque basé sur CVaR et Ulcer Index
        """
        # Normalisation CVaR (plus négatif = plus risqué)
        cvar_score = max(0, min(1, 1 + cvar / 0.1))  # -10% CVaR = score 0
        
        # Normalisation Ulcer Index (plus élevé = plus risqué)
        ulcer_score = max(0, min(1, 1 - ulcer / 10))  # 10 Ulcer = score 0
        
        # Combinaison
        risk_score = 0.6 * cvar_score + 0.4 * ulcer_score
        return risk_score
    
    async def should_rebalance(self) -> bool:
        """
        Vérifier si rebalancing nécessaire (weekly frequency)
        """
        if self.last_rebalance is None:
            return True
        
        time_since_rebalance = datetime.now() - self.last_rebalance
        return time_since_rebalance >= self.rebalance_frequency
    
    async def execute_weekly_rebalance(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Exécution rebalancing hebdomadaire
        """
        if not await self.should_rebalance():
            logger.info("🟡 Rebalancing not needed - weekly frequency")
            return {'status': 'skipped', 'reason': 'frequency_check'}
        
        try:
            # Analyse portfolio
            decisions = await self.analyze_portfolio_async(symbols)
            
            # Exécution trades
            execution_results = {}
            for symbol, decision in decisions.items():
                if decision.action != "HOLD":
                    result = await self.execution_agent.execute_trade(
                        symbol=symbol,
                        action=decision.action,
                        target_weight=decision.target_weight,
                        reasoning=decision.reasoning
                    )
                    execution_results[symbol] = result
            
            # Mettre à jour timestamp
            self.last_rebalance = datetime.now()
            
            # Métriques
            metrics = {
                'status': 'completed',
                'decisions_count': len(decisions),
                'trades_executed': len(execution_results),
                'latency_ms': self.performance_metrics.get('last_latency_ms', 0),
                'timestamp': self.last_rebalance.isoformat()
            }
            
            logger.info(f"✅ Weekly rebalance completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Weekly rebalance failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Métriques de performance du système simplifié
        """
        return {
            **self.performance_metrics,
            'agents_active': 3,  # vs 6 original
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'rebalance_frequency': 'weekly',
            'trade_threshold': self.trade_threshold,
            'architecture': 'simplified_3_agents'
        }


# Factory function
def get_simplified_orchestrator() -> SimplifiedOrchestrator:
    """Factory pour orchestrateur simplifié"""
    return SimplifiedOrchestrator()