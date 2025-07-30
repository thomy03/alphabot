#!/usr/bin/env python3
"""
Hybrid Orchestrator - AlphaBot Optimized Hybrid System
Architecture hybride : Core System (3 agents) + ML/DL stratégique
Combine la simplicité d'exécution avec la puissance du ML/DL
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from alphabot.core.signal_hub import (
    Signal, SignalType, SignalPriority, get_signal_hub
)
from alphabot.core.config import get_settings
from alphabot.agents.risk.enhanced_risk_agent import EnhancedRiskAgent
from alphabot.agents.technical.simplified_technical_agent import SimplifiedTechnicalAgent
from alphabot.agents.execution.execution_agent import ExecutionAgent

# Import ML/DL components (optionnels - chargés seulement si disponibles)
try:
    from alphabot.ml.pattern_detector import MLPatternDetector
    ML_PATTERNS_AVAILABLE = True
except ImportError:
    ML_PATTERNS_AVAILABLE = False

try:
    from alphabot.ml.sentiment_analyzer import SentimentDLAnalyzer
    SENTIMENT_DL_AVAILABLE = True
except ImportError:
    SENTIMENT_DL_AVAILABLE = False

try:
    from alphabot.ml.rag_integrator import RAGIntegrator
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridWorkflowType(Enum):
    """Types de workflows hybrides"""
    CORE_ANALYSIS = "core_analysis"      # Analyse sans ML/DL
    ML_ENHANCED = "ml_enhanced"         # Analyse avec ML/DL
    EMERGENCY_STOP = "emergency_stop"   # Arrêt d'urgence


@dataclass
class HybridTradingDecision:
    """Décision de trading hybride - combine core et ML/DL"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    target_weight: float
    reasoning: List[str]
    
    # Core metrics
    risk_score: float
    technical_score: float
    cvar_95: float
    ulcer_index: float
    
    # ML/DL metrics (optionnelles)
    ml_pattern_score: Optional[float] = None
    ml_sentiment_score: Optional[float] = None
    rag_confidence: Optional[float] = None
    
    # Méta-données
    execution_timestamp: datetime
    workflow_type: str
    ml_components_used: List[str]


class HybridOrchestrator:
    """
    Orchestrateur Hybride - Architecture 3 phases
    
    Phase 1: Core System (Technical + Risk + Execution)
    Phase 2: ML/DL Enhancement (Pattern + Sentiment + RAG)
    Phase 3: Decision Fusion intelligente
    """
    
    def __init__(self, enable_ml: bool = True, ml_confidence_threshold: float = 0.7):
        self.settings = get_settings()
        self.signal_hub = get_signal_hub()
        
        # Core System - toujours actif
        self.technical_agent = SimplifiedTechnicalAgent()
        self.risk_agent = EnhancedRiskAgent()
        self.execution_agent = ExecutionAgent()
        
        # ML/DL Components - optionnels
        self.enable_ml = enable_ml
        self.ml_confidence_threshold = ml_confidence_threshold
        
        self.ml_pattern_detector = None
        self.ml_sentiment_analyzer = None
        self.rag_integrator = None
        
        # Initialiser les composants ML/DL si disponibles et activés
        if self.enable_ml:
            self._init_ml_components()
        
        # Métriques de performance
        self.performance_metrics = {
            'core_decisions': 0,
            'ml_enhanced_decisions': 0,
            'ml_confidence_avg': 0.0,
            'latency_core_ms': 0.0,
            'latency_ml_ms': 0.0,
            'ml_success_rate': 0.0
        }
        
        # Configuration
        self.rebalance_frequency = timedelta(weeks=1)  # Weekly
        self.trade_threshold = 0.05  # 5%
        
        logger.info(f"🚀 Hybrid Orchestrator initialized - ML: {self.enable_ml}")
    
    def _init_ml_components(self):
        """Initialiser les composants ML/DL si disponibles"""
        if ML_PATTERNS_AVAILABLE:
            try:
                self.ml_pattern_detector = MLPatternDetector()
                logger.info("✅ ML Pattern Detector initialized")
            except Exception as e:
                logger.warning(f"⚠️ ML Pattern Detector failed: {e}")
        
        if SENTIMENT_DL_AVAILABLE:
            try:
                self.ml_sentiment_analyzer = SentimentDLAnalyzer()
                logger.info("✅ ML Sentiment Analyzer initialized")
            except Exception as e:
                logger.warning(f"⚠️ ML Sentiment Analyzer failed: {e}")
        
        if RAG_AVAILABLE:
            try:
                self.rag_integrator = RAGIntegrator()
                logger.info("✅ RAG Integrator initialized")
            except Exception as e:
                logger.warning(f"⚠️ RAG Integrator failed: {e}")
    
    async def analyze_portfolio_hybrid(self, 
                                     symbols: List[str], 
                                     workflow_type: HybridWorkflowType = HybridWorkflowType.CORE_ANALYSIS
                                     ) -> Dict[str, HybridTradingDecision]:
        """
        Analyse portfolio avec approche hybride
        
        Phase 1: Core System (toujours exécuté)
        Phase 2: ML/DL Enhancement (si activé et workflow_type approprié)
        Phase 3: Fusion intelligente
        """
        start_time = time.time()
        
        try:
            # Phase 1: Core System Analysis (obligatoire)
            core_decisions = await self._core_analysis_parallel(symbols)
            
            # Phase 2: ML/DL Enhancement (conditionnel)
            ml_enhancements = {}
            if (self.enable_ml and 
                workflow_type == HybridWorkflowType.ML_ENHANCED and
                self._has_ml_components()):
                
                ml_enhancements = await self._ml_analysis_parallel(symbols)
            
            # Phase 3: Decision Fusion
            final_decisions = await self._fuse_decisions(
                core_decisions, ml_enhancements, workflow_type
            )
            
            # Métriques
            latency = (time.time() - start_time) * 1000
            self._update_performance_metrics(final_decisions, latency, workflow_type)
            
            logger.info(f"⚡ Hybrid analysis completed in {latency:.1f}ms")
            return final_decisions
            
        except Exception as e:
            logger.error(f"❌ Hybrid analysis failed: {e}")
            return {}
    
    async def _core_analysis_parallel(self, symbols: List[str]) -> Dict[str, HybridTradingDecision]:
        """
        Phase 1: Core System Analysis - 3 agents en parallèle
        Cible: <50ms latence
        """
        start_time = time.time()
        
        try:
            tasks = []
            for symbol in symbols:
                task = self._analyze_symbol_core(symbol)
                tasks.append(task)
            
            # Exécution parallèle
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Consolidation
            decisions = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Core analysis failed for {symbols[i]}: {result}")
                    continue
                
                if result:
                    decisions[symbols[i]] = result
            
            latency = (time.time() - start_time) * 1000
            self.performance_metrics['latency_core_ms'] = latency
            
            logger.info(f"🎯 Core analysis: {len(decisions)}/{len(symbols)} symbols in {latency:.1f}ms")
            return decisions
            
        except Exception as e:
            logger.error(f"❌ Core analysis failed: {e}")
            return {}
    
    async def _analyze_symbol_core(self, symbol: str) -> Optional[HybridTradingDecision]:
        """
        Analyse symbole avec Core System (3 agents)
        """
        try:
            # Pipeline parallèle Core
            technical_task = self._get_technical_signals_core(symbol)
            risk_task = self._get_risk_assessment_core(symbol)
            execution_task = self._get_execution_context(symbol)
            
            # Exécution parallèle
            technical_result, risk_result, execution_result = await asyncio.gather(
                technical_task, risk_task, execution_task
            )
            
            # Fusion Core
            decision = self._fuse_core_signals(
                symbol, technical_result, risk_result, execution_result
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Core analysis failed for {symbol}: {e}")
            return None
    
    async def _ml_analysis_parallel(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Phase 2: ML/DL Enhancement - analyse en parallèle
        """
        start_time = time.time()
        
        try:
            ml_tasks = {}
            
            # Pattern Detection
            if self.ml_pattern_detector:
                ml_tasks['patterns'] = self._detect_patterns_batch(symbols)
            
            # Sentiment Analysis
            if self.ml_sentiment_analyzer:
                ml_tasks['sentiment'] = self._analyze_sentiment_batch(symbols)
            
            # RAG Analysis
            if self.rag_integrator:
                ml_tasks['rag'] = self._rag_analysis_batch(symbols)
            
            # Exécuter toutes les tâches ML en parallèle
            if ml_tasks:
                ml_results = await asyncio.gather(*ml_tasks.values(), return_exceptions=True)
                
                # Organiser les résultats
                enhancements = {}
                task_names = list(ml_tasks.keys())
                
                for i, result in enumerate(ml_results):
                    task_name = task_names[i]
                    if isinstance(result, Exception):
                        logger.warning(f"ML task {task_name} failed: {result}")
                        continue
                    
                    enhancements[task_name] = result
                
                latency = (time.time() - start_time) * 1000
                self.performance_metrics['latency_ml_ms'] = latency
                
                logger.info(f"🧠 ML analysis completed in {latency:.1f}ms")
                return enhancements
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ ML analysis failed: {e}")
            return {}
    
    async def _fuse_decisions(self, 
                             core_decisions: Dict[str, HybridTradingDecision],
                             ml_enhancements: Dict[str, Any],
                             workflow_type: HybridWorkflowType) -> Dict[str, HybridTradingDecision]:
        """
        Phase 3: Fusion intelligente des décisions Core et ML/DL
        """
        final_decisions = {}
        
        for symbol, core_decision in core_decisions.items():
            try:
                # Commencer avec la décision Core
                final_decision = core_decision
                
                # Appliquer les améliorations ML/DL si disponibles
                if ml_enhancements:
                    final_decision = self._apply_ml_enhancements(
                        final_decision, ml_enhancements, symbol
                    )
                
                # Mettre à jour le workflow type
                final_decision.workflow_type = workflow_type.value
                
                final_decisions[symbol] = final_decision
                
            except Exception as e:
                logger.error(f"Decision fusion failed for {symbol}: {e}")
                # Garder la décision Core en fallback
                final_decisions[symbol] = core_decision
        
        return final_decisions
    
    def _apply_ml_enhancements(self, 
                              core_decision: HybridTradingDecision,
                              ml_enhancements: Dict[str, Any],
                              symbol: str) -> HybridTradingDecision:
        """
        Appliquer les améliorations ML/DL à la décision Core
        """
        ml_components_used = []
        confidence_adjustment = 0.0
        
        # Pattern Detection Enhancement
        if 'patterns' in ml_enhancements and symbol in ml_enhancements['patterns']:
            pattern_result = ml_enhancements['patterns'][symbol]
            
            if pattern_result.get('confidence', 0) > self.ml_confidence_threshold:
                pattern_score = pattern_result.get('score', 0.5)
                core_decision.ml_pattern_score = pattern_score
                
                # Ajuster la confiance
                if pattern_score > 0.7:
                    confidence_adjustment += 0.1
                elif pattern_score < 0.3:
                    confidence_adjustment -= 0.1
                
                ml_components_used.append('pattern_detection')
                
                # Ajouter au reasoning
                core_decision.reasoning.extend(pattern_result.get('reasoning', []))
        
        # Sentiment Analysis Enhancement
        if 'sentiment' in ml_enhancements and symbol in ml_enhancements['sentiment']:
            sentiment_result = ml_enhancements['sentiment'][symbol]
            
            if sentiment_result.get('confidence', 0) > self.ml_confidence_threshold:
                sentiment_score = sentiment_result.get('score', 0.5)
                core_decision.ml_sentiment_score = sentiment_score
                
                # Ajuster la confiance
                if sentiment_score > 0.65:
                    confidence_adjustment += 0.08
                elif sentiment_score < 0.35:
                    confidence_adjustment -= 0.08
                
                ml_components_used.append('sentiment_analysis')
                
                # Ajouter au reasoning
                core_decision.reasoning.extend(sentiment_result.get('reasoning', []))
        
        # RAG Enhancement
        if 'rag' in ml_enhancements and symbol in ml_enhancements['rag']:
            rag_result = ml_enhancements['rag'][symbol]
            
            if rag_result.get('confidence', 0) > self.ml_confidence_threshold:
                rag_confidence = rag_result.get('confidence', 0.5)
                core_decision.rag_confidence = rag_confidence
                
                # Ajuster la confiance
                if rag_confidence > 0.8:
                    confidence_adjustment += 0.05
                
                ml_components_used.append('rag_analysis')
                
                # Ajouter au reasoning
                core_decision.reasoning.extend(rag_result.get('reasoning', []))
        
        # Appliquer l'ajustement de confiance
        core_decision.confidence = np.clip(
            core_decision.confidence + confidence_adjustment, 
            0.0, 
            1.0
        )
        
        # Mettre à jour les composants ML utilisés
        core_decision.ml_components_used = ml_components_used
        
        # Ajuster l'action si la confiance change significativement
        if confidence_adjustment > 0.15 and core_decision.action == "HOLD":
            if core_decision.technical_score > 0.6:
                core_decision.action = "BUY"
        elif confidence_adjustment < -0.15 and core_decision.action == "HOLD":
            if core_decision.risk_score > 0.7:
                core_decision.action = "SELL"
        
        return core_decision
    
    # Méthodes Core System
    async def _get_technical_signals_core(self, symbol: str) -> Dict[str, Any]:
        """Signaux techniques Core"""
        try:
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
    
    async def _get_risk_assessment_core(self, symbol: str) -> Dict[str, Any]:
        """Évaluation risque Core"""
        try:
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
        """Contexte d'exécution"""
        try:
            current_position = await self.execution_agent.get_position(symbol)
            sector_exposure = await self.execution_agent.get_sector_exposure(symbol)
            
            can_trade = True
            constraints = []
            
            if sector_exposure > 0.30:
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
    
    def _fuse_core_signals(self, symbol: str, technical: Dict, risk: Dict, execution: Dict) -> HybridTradingDecision:
        """Fusion des signaux Core"""
        # Score technique (60% weight)
        technical_score = technical['score']
        
        # Score risque (40% weight)
        risk_score = risk['risk_score']
        
        # Score final pondéré
        final_score = 0.6 * technical_score + 0.4 * risk_score
        
        # Décision d'action
        if final_score > 0.7 and execution['can_trade']:
            action = "BUY"
            target_weight = min(0.05, final_score * 0.07)
        elif final_score < 0.3:
            action = "SELL"
            target_weight = 0.0
        else:
            action = "HOLD"
            target_weight = execution['current_position']
        
        # Ajuster selon threshold
        if abs(target_weight - execution['current_position']) < self.trade_threshold:
            action = "HOLD"
            target_weight = execution['current_position']
        
        # Reasoning consolidé
        reasoning = []
        reasoning.extend(technical.get('reasoning', []))
        reasoning.extend(risk.get('reasoning', []))
        reasoning.extend(execution.get('constraints', []))
        
        return HybridTradingDecision(
            symbol=symbol,
            action=action,
            confidence=final_score,
            target_weight=target_weight,
            reasoning=reasoning,
            risk_score=risk_score,
            technical_score=technical_score,
            cvar_95=risk['cvar_95'],
            ulcer_index=risk['ulcer_index'],
            execution_timestamp=datetime.now(),
            workflow_type="core_analysis",
            ml_components_used=[]
        )
    
    # Méthodes ML/DL (implémentées avec intégration réelle)
    async def _detect_patterns_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Détection de patterns ML par batch - intégration réelle"""
        if not self.ml_pattern_detector:
            return {}
        
        try:
            # Récupérer les données de prix pour tous les symboles
            price_data = {}
            for symbol in symbols:
                prices = await self._get_cached_prices_for_ml(symbol)
                if prices is not None and len(prices) >= 50:  # Minimum pour analyse
                    price_data[symbol] = prices
            
            if not price_data:
                logger.warning("No price data available for pattern detection")
                return {}
            
            # Appeler le détecteur de patterns
            pattern_results = await self.ml_pattern_detector.detect_patterns_batch(price_data)
            
            # Formater les résultats pour l'orchestrateur
            results = {}
            for symbol, pattern_result in pattern_results.items():
                results[symbol] = {
                    'score': pattern_result.score,
                    'confidence': pattern_result.confidence,
                    'expected_move': pattern_result.expected_move,
                    'reasoning': pattern_result.reasoning,
                    'pattern_type': pattern_result.pattern_type
                }
            
            logger.info(f"🧠 Pattern detection completed for {len(results)} symbols")
            return {'patterns': results}
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {}
    
    async def _analyze_sentiment_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyse de sentiment ML par batch - intégration réelle"""
        if not self.ml_sentiment_analyzer:
            return {}
        
        try:
            # Récupérer les données de news pour les symboles
            news_data = await self._get_news_data_for_symbols(symbols)
            
            if not news_data:
                logger.warning("No news data available for sentiment analysis")
                return {}
            
            # Appeler l'analyseur de sentiment
            sentiment_results = await self.ml_sentiment_analyzer.analyze_sentiment_batch(symbols, news_data)
            
            # Formater les résultats pour l'orchestrateur
            results = {}
            for symbol, sentiment_result in sentiment_results.items():
                results[symbol] = {
                    'score': sentiment_result.sentiment_score,
                    'confidence': sentiment_result.confidence,
                    'sentiment_label': sentiment_result.sentiment_label,
                    'momentum': sentiment_result.momentum,
                    'reasoning': sentiment_result.reasoning,
                    'key_phrases': sentiment_result.key_phrases
                }
            
            logger.info(f"📰 Sentiment analysis completed for {len(results)} symbols")
            return {'sentiment': results}
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {}
    
    async def _rag_analysis_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyse RAG par batch - placeholder pour implémentation future"""
        if not self.rag_integrator:
            return {}
        
        try:
            # Pour l'instant, retourner des résultats simulés
            # À implémenter avec un vrai RAG integrator plus tard
            results = {}
            for symbol in symbols:
                # Simulation de résultat RAG basique
                results[symbol] = {
                    'confidence': 0.7,  # Confiance modérée
                    'reasoning': ['RAG analysis placeholder - to be implemented'],
                    'context_score': 0.6
                }
            
            logger.info(f"🔍 RAG analysis completed for {len(results)} symbols (placeholder)")
            return {'rag': results}
            
        except Exception as e:
            logger.error(f"RAG analysis failed: {e}")
            return {}
    
    # Méthodes utilitaires pour l'intégration ML
    async def _get_cached_prices_for_ml(self, symbol: str, days: int = 100) -> Optional[pd.Series]:
        """Récupérer les prix de manière optimisée pour l'analyse ML"""
        try:
            # Utiliser yfinance pour récupérer les données
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if data.empty:
                return None
            
            return data['Close']
            
        except Exception as e:
            logger.error(f"Failed to get prices for ML analysis {symbol}: {e}")
            return None
    
    async def _get_news_data_for_symbols(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Récupérer les données de news pour l'analyse de sentiment"""
        try:
            # Pour l'instant, simuler des données de news
            # À remplacer par un vrai appel à News API ou autre source
            news_data = {}
            
            # Simuler quelques headlines pour chaque symbole
            sample_news = {
                'AAPL': [
                    "Apple reports strong quarterly earnings beating expectations",
                    "iPhone sales exceed analyst estimates",
                    "Apple announces new AI features for iOS",
                    "Analysts maintain buy rating on Apple stock"
                ],
                'MSFT': [
                    "Microsoft cloud growth accelerates in Q2",
                    "Azure AI services see increased adoption",
                    "Microsoft raises dividend by 10%",
                    "Tech analysts positive on Microsoft outlook"
                ],
                'GOOGL': [
                    "Google parent Alphabet beats revenue expectations",
                    "Search advertising shows strong growth",
                    "Google invests heavily in AI infrastructure",
                    "Analysts see upside potential for Google shares"
                ]
            }
            
            for symbol in symbols:
                # Utiliser des données simulées ou données réelles si disponibles
                if symbol in sample_news:
                    news_data[symbol] = sample_news[symbol]
                else:
                    # Données génériques pour les autres symboles
                    news_data[symbol] = [
                        f"{symbol} shows positive market momentum",
                        f"Analysts maintain neutral stance on {symbol}",
                        f"{symbol} reports steady financial performance",
                        f"Market sentiment for {symbol} remains mixed"
                    ]
            
            return news_data
            
        except Exception as e:
            logger.error(f"Failed to get news data: {e}")
            return {}
    
    # Méthodes utilitaires
    def _has_ml_components(self) -> bool:
        """Vérifier si des composants ML sont disponibles"""
        return (self.ml_pattern_detector is not None or 
                self.ml_sentiment_analyzer is not None or 
                self.rag_integrator is not None)
    
    def _update_performance_metrics(self, decisions: Dict[str, HybridTradingDecision], 
                                   latency: float, workflow_type: HybridWorkflowType):
        """Mettre à jour les métriques de performance"""
        if workflow_type == HybridWorkflowType.CORE_ANALYSIS:
            self.performance_metrics['core_decisions'] += len(decisions)
        else:
            self.performance_metrics['ml_enhanced_decisions'] += len(decisions)
            
            # Calculer la confiance moyenne ML
            ml_confidences = []
            for decision in decisions.values():
                if decision.ml_components_used:
                    ml_confidences.append(decision.confidence)
            
            if ml_confidences:
                avg_confidence = np.mean(ml_confidences)
                self.performance_metrics['ml_confidence_avg'] = (
                    (self.performance_metrics['ml_confidence_avg'] * 
                     (self.performance_metrics['ml_enhanced_decisions'] - len(decisions)) +
                     avg_confidence * len(decisions)
                    ) / self.performance_metrics['ml_enhanced_decisions']
                )
        
        # Taux de succès ML
        if self.enable_ml:
            ml_used_count = sum(1 for d in decisions.values() if d.ml_components_used)
            if ml_used_count > 0:
                self.performance_metrics['ml_success_rate'] = ml_used_count / len(decisions)
    
    async def should_rebalance(self) -> bool:
        """Vérifier si rebalancing nécessaire"""
        # À implémenter avec logique de temps
        return True
    
    async def execute_weekly_rebalance(self, symbols: List[str]) -> Dict[str, Any]:
        """Exécution rebalancement hebdomadaire"""
        if not await self.should_rebalance():
            return {'status': 'skipped', 'reason': 'frequency_check'}
        
        try:
            # Analyse hybride
            decisions = await self.analyze_portfolio_hybrid(
                symbols, HybridWorkflowType.ML_ENHANCED
            )
            
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
            
            return {
                'status': 'completed',
                'decisions_count': len(decisions),
                'trades_executed': len(execution_results),
                'ml_enhanced': sum(1 for d in decisions.values() if d.ml_components_used),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Weekly rebalance failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtenir les métriques de performance"""
        return {
            **self.performance_metrics,
            'ml_enabled': self.enable_ml,
            'ml_components_available': {
                'pattern_detector': self.ml_pattern_detector is not None,
                'sentiment_analyzer': self.ml_sentiment_analyzer is not None,
                'rag_integrator': self.rag_integrator is not None
            },
            'architecture': 'hybrid_core_plus_ml'
        }


# Factory function
def get_hybrid_orchestrator(enable_ml: bool = True) -> HybridOrchestrator:
    """Factory pour orchestrateur hybride"""
    return HybridOrchestrator(enable_ml=enable_ml)
