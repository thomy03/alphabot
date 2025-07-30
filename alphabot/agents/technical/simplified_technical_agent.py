"""
Simplified Technical Agent - EMA + RSI only
Agent technique simplifié selon recommandations expert
Focus sur les 2 indicateurs les plus robustes
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


class SimplifiedTechnicalAgent:
    """
    Agent technique simplifié - EMA + RSI seulement
    
    Recommandations expert appliquées :
    - Focus sur 2 indicateurs robustes
    - Latence optimisée <15ms
    - Signaux clairs sans overfitting
    """
    
    def __init__(self):
        self.ema_short = 20
        self.ema_long = 50
        self.rsi_period = 14
        
        # Seuils simplifiés
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Cache optimisé
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes TTL
        
        logger.info("✅ Simplified Technical Agent initialized - EMA + RSI only")
    
    async def get_ema_crossover(self, symbol: str, short_period: int = 20, long_period: int = 50) -> float:
        """
        Signal EMA crossover - indicateur momentum principal
        
        Returns:
            float: Signal strength [-1, 1]
                   >0 = bullish crossover
                   <0 = bearish crossover
        """
        try:
            # Récupérer données avec cache
            prices = await self._get_cached_prices(symbol)
            if prices is None or len(prices) < long_period:
                return 0.0
            
            # Calcul EMAs
            ema_short = self._calculate_ema(prices, short_period)
            ema_long = self._calculate_ema(prices, long_period)
            
            # Signal crossover
            current_diff = ema_short.iloc[-1] - ema_long.iloc[-1]
            previous_diff = ema_short.iloc[-2] - ema_long.iloc[-2]
            
            # Normaliser signal [-1, 1]
            signal_strength = np.tanh(current_diff / ema_long.iloc[-1] * 100)
            
            # Détecter crossover récent (boost signal)
            crossover_boost = 0.0
            if (current_diff > 0 and previous_diff <= 0):  # Bullish crossover
                crossover_boost = 0.3
            elif (current_diff < 0 and previous_diff >= 0):  # Bearish crossover
                crossover_boost = -0.3
            
            final_signal = np.clip(signal_strength + crossover_boost, -1.0, 1.0)
            
            logger.debug(f"{symbol} EMA signal: {final_signal:.3f} (current_diff: {current_diff:.4f})")
            return float(final_signal)
            
        except Exception as e:
            logger.error(f"EMA crossover calculation failed for {symbol}: {e}")
            return 0.0
    
    async def get_rsi(self, symbol: str, period: int = 14) -> float:
        """
        Calcul RSI - indicateur mean reversion
        
        Returns:
            float: RSI value [0, 100]
        """
        try:
            # Récupérer données
            prices = await self._get_cached_prices(symbol)
            if prices is None or len(prices) < period + 1:
                return 50.0  # Neutral
            
            # Calcul RSI
            rsi_value = self._calculate_rsi(prices, period)
            
            logger.debug(f"{symbol} RSI: {rsi_value:.1f}")
            return float(rsi_value)
            
        except Exception as e:
            logger.error(f"RSI calculation failed for {symbol}: {e}")
            return 50.0
    
    async def get_simplified_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Génère signaux techniques simplifiés (EMA + RSI)
        
        Logic simplifiée selon expert :
        - EMA crossover = momentum
        - RSI = mean reversion opportunity
        """
        try:
            # Analyses parallèles
            ema_task = self.get_ema_crossover(symbol)
            rsi_task = self.get_rsi(symbol)
            
            ema_signal, rsi_value = await asyncio.gather(ema_task, rsi_task)
            
            # Logique de scoring simplifiée
            score = 0.0
            reasoning = []
            
            # EMA contribution (60% weight)
            if ema_signal > 0.2:
                score += 0.6 * min(ema_signal, 1.0)
                reasoning.append(f"EMA bullish crossover (strength: {ema_signal:.2f})")
            elif ema_signal < -0.2:
                score += 0.6 * max(ema_signal, -1.0)
                reasoning.append(f"EMA bearish crossover (strength: {ema_signal:.2f})")
            
            # RSI contribution (40% weight)
            if rsi_value < self.rsi_oversold:
                rsi_boost = (self.rsi_oversold - rsi_value) / self.rsi_oversold * 0.4
                score += rsi_boost
                reasoning.append(f"RSI oversold ({rsi_value:.1f}) - reversal opportunity")
            elif rsi_value > self.rsi_overbought:
                rsi_penalty = (rsi_value - self.rsi_overbought) / (100 - self.rsi_overbought) * 0.4
                score -= rsi_penalty
                reasoning.append(f"RSI overbought ({rsi_value:.1f}) - caution")
            
            # Score final normalisé
            final_score = np.clip(score, -1.0, 1.0)
            
            # Déterminer action
            if final_score > 0.4:
                action = "BUY"
                confidence = final_score
            elif final_score < -0.4:
                action = "SELL"
                confidence = abs(final_score)
            else:
                action = "HOLD"
                confidence = 1.0 - abs(final_score)
            
            return {
                'symbol': symbol,
                'action': action,
                'score': final_score,
                'confidence': confidence,
                'ema_signal': ema_signal,
                'rsi_value': rsi_value,
                'reasoning': reasoning,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Simplified signals failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': "HOLD",
                'score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def _get_cached_prices(self, symbol: str, days: int = 100) -> Optional[pd.Series]:
        """
        Récupère prix avec cache TTL pour optimiser latence
        """
        cache_key = f"{symbol}_prices"
        current_time = datetime.now()
        
        # Vérifier cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if (current_time - timestamp).total_seconds() < self._cache_ttl:
                return cached_data
        
        try:
            # Télécharger nouvelles données
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if data.empty:
                logger.warning(f"No price data for {symbol}")
                return None
            
            prices = data['Close']
            
            # Mettre en cache
            self._cache[cache_key] = (prices, current_time)
            
            return prices
            
        except Exception as e:
            logger.error(f"Failed to fetch prices for {symbol}: {e}")
            return None
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcul EMA optimisé"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """
        Calcul RSI optimisé - dernière valeur seulement
        """
        # Calcul des variations
        delta = prices.diff()
        
        # Gains et pertes
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Moyennes mobiles exponentielles
        avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
        
        # RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Métriques de performance de l'agent simplifié"""
        return {
            'agent_name': 'SimplifiedTechnical',
            'indicators': ['EMA_20_50', 'RSI_14'],
            'cache_size': len(self._cache),
            'cache_ttl_seconds': self._cache_ttl,
            'architecture': 'simplified_core_indicators'
        }
    
    def clear_cache(self):
        """Nettoyer cache (utile pour tests)"""
        self._cache.clear()
        logger.info("Technical agent cache cleared")


# Test rapide
async def test_simplified_technical():
    """Test rapide de l'agent simplifié"""
    agent = SimplifiedTechnicalAgent()
    
    # Test sur quelques symboles
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        # Test signaux
        signals = await agent.get_simplified_signals(symbol)
        print(f"  Action: {signals['action']}")
        print(f"  Score: {signals['score']:.3f}")
        print(f"  Confidence: {signals['confidence']:.3f}")
        print(f"  Reasoning: {signals.get('reasoning', [])}")


if __name__ == "__main__":
    # Test standalone
    asyncio.run(test_simplified_technical())