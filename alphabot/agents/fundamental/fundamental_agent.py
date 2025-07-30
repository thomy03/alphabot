#!/usr/bin/env python3
"""
Fundamental Agent - AlphaBot Multi-Agent Trading System
Analyse fondamentale avec ratios P/E, Piotroski F-Score, ROE, etc.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from alphabot.core.signal_hub import (
    Signal, SignalType, SignalPriority, get_signal_hub
)
from alphabot.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FundamentalMetrics:
    """Métriques fondamentales d'une action"""
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    piotroski_score: Optional[int] = None
    altman_z_score: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    last_updated: Optional[datetime] = None


@dataclass
class FundamentalSignal:
    """Signal d'analyse fondamentale"""
    symbol: str
    score: float  # Score composite 0-100
    recommendation: str  # BUY, HOLD, SELL
    confidence: float  # Niveau de confiance 0-1
    key_factors: List[str]  # Facteurs clés de la décision
    metrics: FundamentalMetrics
    risk_level: str  # LOW, MEDIUM, HIGH


class FundamentalAgent:
    """Agent d'analyse fondamentale"""
    
    def __init__(self):
        self.settings = get_settings()
        self.signal_hub = get_signal_hub()
        self.is_running = False
        
        # Cache des métriques
        self.metrics_cache: Dict[str, FundamentalMetrics] = {}
        self.cache_ttl = timedelta(hours=24)  # Données fondamentales changent lentement
        
        # Seuils de scoring
        self.scoring_weights = {
            'valuation': 0.25,     # P/E, P/B ratios
            'profitability': 0.25, # ROE, ROA, marges
            'financial_health': 0.20, # Ratios de liquidité, dette
            'growth': 0.20,        # Croissance revenus/bénéfices
            'quality': 0.10        # Piotroski score, dividendes
        }
        
        # Benchmarks sectoriels (simplifié)
        self.sector_benchmarks = {
            'technology': {'pe_median': 25, 'roe_median': 15},
            'finance': {'pe_median': 12, 'roe_median': 12},
            'healthcare': {'pe_median': 20, 'roe_median': 10},
            'consumer': {'pe_median': 18, 'roe_median': 14},
            'industrial': {'pe_median': 16, 'roe_median': 11},
            'energy': {'pe_median': 14, 'roe_median': 8},
            'utilities': {'pe_median': 15, 'roe_median': 9},
            'materials': {'pe_median': 13, 'roe_median': 10},
            'real_estate': {'pe_median': 20, 'roe_median': 8},
            'default': {'pe_median': 18, 'roe_median': 12}
        }
    
    async def start(self):
        """Démarrer l'agent"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🔍 Fundamental Agent démarré")
        
        # S'abonner aux demandes d'analyse
        await self.signal_hub.subscribe_to_signals(
            agent_name="fundamental_agent",
            callback=self._handle_signal,
            signal_types=[SignalType.PRICE_UPDATE, SignalType.SYSTEM_STATUS]
        )
        
        # Publier le statut
        await self.signal_hub.publish_agent_status(
            "fundamental_agent", 
            "started",
            {"version": "1.0", "capabilities": ["fundamental_analysis", "piotroski_score"]}
        )
    
    async def stop(self):
        """Arrêter l'agent"""
        self.is_running = False
        await self.signal_hub.publish_agent_status("fundamental_agent", "stopped")
        logger.info("🔍 Fundamental Agent arrêté")
    
    async def _handle_signal(self, signal: Signal):
        """Traiter un signal reçu"""
        try:
            if signal.type == SignalType.PRICE_UPDATE and signal.symbol:
                # Analyser si on n'a pas de données récentes
                if await self._should_analyze(signal.symbol):
                    await self._analyze_symbol(signal.symbol)
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal: {e}")
    
    async def _should_analyze(self, symbol: str) -> bool:
        """Vérifier si une analyse est nécessaire"""
        if symbol not in self.metrics_cache:
            return True
        
        metrics = self.metrics_cache[symbol]
        if not metrics.last_updated:
            return True
        
        # Analyser si données trop anciennes
        return datetime.utcnow() - metrics.last_updated > self.cache_ttl
    
    async def _analyze_symbol(self, symbol: str):
        """Analyser un symbole"""
        try:
            start_time = time.time()
            
            # Récupérer les données fondamentales
            metrics = await self._fetch_fundamental_data(symbol)
            if not metrics:
                logger.warning(f"⚠️ Pas de données fondamentales pour {symbol}")
                return
            
            # Calculer le score composite
            signal_data = await self._calculate_fundamental_score(metrics)
            
            # Publier le signal
            signal = Signal(
                id=None,
                type=SignalType.FUNDAMENTAL_SIGNAL,
                source_agent="fundamental_agent",
                symbol=symbol,
                priority=SignalPriority.MEDIUM,
                data={
                    'recommendation': signal_data.recommendation,
                    'score': signal_data.score,
                    'confidence': signal_data.confidence,
                    'key_factors': signal_data.key_factors,
                    'risk_level': signal_data.risk_level,
                    'pe_ratio': metrics.pe_ratio,
                    'roe': metrics.roe,
                    'piotroski_score': metrics.piotroski_score
                },
                metadata={
                    'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                    'data_freshness_hours': 24,
                    'agent_version': '1.0'
                }
            )
            
            await self.signal_hub.publish_signal(signal)
            logger.info(f"📊 Analyse fondamentale {symbol}: {signal_data.recommendation} (score: {signal_data.score:.1f})")
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse {symbol}: {e}")
    
    async def _fetch_fundamental_data(self, symbol: str) -> Optional[FundamentalMetrics]:
        """Récupérer les données fondamentales (simulation pour Phase 4)"""
        try:
            # SIMULATION - En production, utiliser FinancialModelingPrep ou SEC filings
            # Générer des données réalistes pour les tests
            
            np.random.seed(hash(symbol) % 2**32)  # Reproductible par symbole
            
            # Métriques simulées mais réalistes
            base_pe = 15 + np.random.normal(0, 8)
            base_roe = 0.12 + np.random.normal(0, 0.08)
            
            metrics = FundamentalMetrics(
                symbol=symbol,
                pe_ratio=max(5, base_pe),
                pb_ratio=max(0.5, 2.5 + np.random.normal(0, 1.5)),
                roe=max(0, base_roe),
                roa=max(0, base_roe * 0.7),
                debt_to_equity=max(0, 0.5 + np.random.normal(0, 0.4)),
                current_ratio=max(0.5, 1.5 + np.random.normal(0, 0.5)),
                quick_ratio=max(0.3, 1.2 + np.random.normal(0, 0.4)),
                gross_margin=max(0, 0.35 + np.random.normal(0, 0.15)),
                operating_margin=max(0, 0.15 + np.random.normal(0, 0.10)),
                net_margin=max(0, 0.08 + np.random.normal(0, 0.08)),
                revenue_growth=np.random.normal(0.08, 0.20),
                earnings_growth=np.random.normal(0.10, 0.30),
                dividend_yield=max(0, np.random.exponential(0.02)),
                market_cap=np.random.lognormal(22, 2),  # ~10B médian
                last_updated=datetime.utcnow()
            )
            
            # Calculer le Piotroski F-Score
            metrics.piotroski_score = self._calculate_piotroski_score(metrics)
            
            # Calculer l'Altman Z-Score
            metrics.altman_z_score = self._calculate_altman_z_score(metrics)
            
            # Mettre en cache
            self.metrics_cache[symbol] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données {symbol}: {e}")
            return None
    
    def _calculate_piotroski_score(self, metrics: FundamentalMetrics) -> int:
        """Calculer le Piotroski F-Score (9 critères)"""
        score = 0
        
        # Profitabilité (4 points max)
        if metrics.roe and metrics.roe > 0:
            score += 1
        if metrics.roa and metrics.roa > 0:
            score += 1
        if metrics.operating_margin and metrics.operating_margin > 0:
            score += 1
        if metrics.net_margin and metrics.net_margin > 0:
            score += 1
        
        # Levier/Liquidité (3 points max)
        if metrics.debt_to_equity and metrics.debt_to_equity < 0.4:
            score += 1
        if metrics.current_ratio and metrics.current_ratio > 1.25:
            score += 1
        if metrics.quick_ratio and metrics.quick_ratio > 1.0:
            score += 1
        
        # Efficacité opérationnelle (2 points max)
        if metrics.gross_margin and metrics.gross_margin > 0.25:
            score += 1
        if metrics.revenue_growth and metrics.revenue_growth > 0:
            score += 1
        
        return score
    
    def _calculate_altman_z_score(self, metrics: FundamentalMetrics) -> float:
        """Calculer l'Altman Z-Score (prédiction faillite)"""
        try:
            # Version simplifiée de l'Altman Z-Score
            # Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*ME/TL + 1.0*S/TA
            
            # Approximations avec les données disponibles
            wc_ta = (metrics.current_ratio or 1.0) * 0.1  # Working capital / Total assets
            re_ta = (metrics.roe or 0.0) * 0.5  # Retained earnings / Total assets
            ebit_ta = (metrics.operating_margin or 0.0)  # EBIT / Total assets
            me_tl = 1.0 / max(0.1, metrics.debt_to_equity or 0.5)  # Market equity / Total liabilities
            s_ta = 1.0  # Sales / Total assets (assumé à 1 pour simplifier)
            
            z_score = (1.2 * wc_ta + 
                      1.4 * re_ta + 
                      3.3 * ebit_ta + 
                      0.6 * me_tl + 
                      1.0 * s_ta)
            
            return round(z_score, 2)
            
        except:
            return 1.8  # Score neutre par défaut
    
    async def _calculate_fundamental_score(self, metrics: FundamentalMetrics) -> FundamentalSignal:
        """Calculer le score composite et la recommandation"""
        
        scores = {}
        key_factors = []
        
        # 1. Score de valorisation (25%)
        valuation_score = 50  # Score neutre par défaut
        if metrics.pe_ratio:
            if metrics.pe_ratio < 15:
                valuation_score = 80
                key_factors.append("P/E attractif")
            elif metrics.pe_ratio < 25:
                valuation_score = 60
            elif metrics.pe_ratio > 40:
                valuation_score = 20
                key_factors.append("P/E élevé")
        
        if metrics.pb_ratio:
            if metrics.pb_ratio < 1.5:
                valuation_score = min(100, valuation_score + 20)
                key_factors.append("P/B attractif")
            elif metrics.pb_ratio > 3:
                valuation_score = max(0, valuation_score - 20)
        
        scores['valuation'] = valuation_score
        
        # 2. Score de profitabilité (25%)
        profitability_score = 50
        if metrics.roe:
            if metrics.roe > 0.15:
                profitability_score = 90
                key_factors.append("ROE élevé")
            elif metrics.roe > 0.10:
                profitability_score = 70
            elif metrics.roe < 0.05:
                profitability_score = 20
        
        if metrics.net_margin:
            if metrics.net_margin > 0.10:
                profitability_score = min(100, profitability_score + 15)
            elif metrics.net_margin < 0.02:
                profitability_score = max(0, profitability_score - 30)
        
        scores['profitability'] = profitability_score
        
        # 3. Score de santé financière (20%)
        health_score = 50
        if metrics.debt_to_equity:
            if metrics.debt_to_equity < 0.3:
                health_score = 85
                key_factors.append("Dette faible")
            elif metrics.debt_to_equity > 1.0:
                health_score = 25
                key_factors.append("Dette élevée")
        
        if metrics.current_ratio:
            if metrics.current_ratio > 1.5:
                health_score = min(100, health_score + 15)
            elif metrics.current_ratio < 1.0:
                health_score = max(0, health_score - 25)
        
        scores['financial_health'] = health_score
        
        # 4. Score de croissance (20%)
        growth_score = 50
        if metrics.revenue_growth:
            if metrics.revenue_growth > 0.15:
                growth_score = 85
                key_factors.append("Forte croissance")
            elif metrics.revenue_growth > 0.05:
                growth_score = 65
            elif metrics.revenue_growth < -0.05:
                growth_score = 25
                key_factors.append("Revenus en baisse")
        
        scores['growth'] = growth_score
        
        # 5. Score de qualité (10%)
        quality_score = 50
        if metrics.piotroski_score:
            if metrics.piotroski_score >= 7:
                quality_score = 90
                key_factors.append("Piotroski élevé")
            elif metrics.piotroski_score >= 5:
                quality_score = 60
            else:
                quality_score = 30
        
        scores['quality'] = quality_score
        
        # Score composite pondéré
        composite_score = sum(
            scores[category] * weight 
            for category, weight in self.scoring_weights.items()
        )
        
        # Déterminer la recommandation
        if composite_score >= 75:
            recommendation = "BUY"
            confidence = min(0.95, composite_score / 100 * 1.1)
        elif composite_score >= 55:
            recommendation = "HOLD"
            confidence = 0.6 + (composite_score - 55) / 20 * 0.2
        else:
            recommendation = "SELL"
            confidence = min(0.9, (55 - composite_score) / 55 * 0.8 + 0.5)
        
        # Niveau de risque
        risk_level = "MEDIUM"
        if metrics.debt_to_equity and metrics.debt_to_equity > 1.5:
            risk_level = "HIGH"
        elif (metrics.current_ratio and metrics.current_ratio > 2.0 and 
              metrics.debt_to_equity and metrics.debt_to_equity < 0.3):
            risk_level = "LOW"
        
        # Ajustements finaux
        if len(key_factors) == 0:
            key_factors.append("Analyse standard")
        
        return FundamentalSignal(
            symbol=metrics.symbol,
            score=round(composite_score, 1),
            recommendation=recommendation,
            confidence=round(confidence, 2),
            key_factors=key_factors[:3],  # Max 3 facteurs
            metrics=metrics,
            risk_level=risk_level
        )
    
    async def analyze_portfolio(self, symbols: List[str]) -> Dict[str, FundamentalSignal]:
        """Analyser un portefeuille de symboles"""
        results = {}
        
        for symbol in symbols:
            try:
                if await self._should_analyze(symbol):
                    await self._analyze_symbol(symbol)
                
                # Récupérer depuis le cache
                if symbol in self.metrics_cache:
                    metrics = self.metrics_cache[symbol]
                    signal = await self._calculate_fundamental_score(metrics)
                    results[symbol] = signal
                    
            except Exception as e:
                logger.error(f"❌ Erreur analyse portefeuille {symbol}: {e}")
        
        return results
    
    def get_cached_metrics(self, symbol: str) -> Optional[FundamentalMetrics]:
        """Obtenir les métriques en cache"""
        return self.metrics_cache.get(symbol)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Obtenir le statut de l'agent"""
        return {
            'name': 'fundamental_agent',
            'version': '1.0',
            'is_running': self.is_running,
            'cache_size': len(self.metrics_cache),
            'capabilities': [
                'pe_ratio_analysis',
                'piotroski_f_score',
                'altman_z_score',
                'composite_scoring',
                'sector_comparison'
            ],
            'last_analysis': max(
                (m.last_updated for m in self.metrics_cache.values() if m.last_updated),
                default=None
            )
        }