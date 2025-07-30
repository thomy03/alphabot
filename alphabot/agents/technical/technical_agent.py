"""
Technical Agent - Analyse technique pour AlphaBot
Calcule EMA 20/50, RSI, ATR et génère des signaux techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from alphabot.agents.TEMPLATE_agent import AlphaBotAgentTemplate


class TechnicalAgent(AlphaBotAgentTemplate):
    """Agent d'analyse technique utilisant EMA, RSI, ATR"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(
            agent_name="TechnicalAgent",
            description="Analyse technique avec EMA 20/50, RSI, ATR",
            config_path=config_path
        )
        self.logger = logging.getLogger(__name__)
        
        # Paramètres techniques par défaut
        self.ema_short = 20
        self.ema_long = 50
        self.rsi_period = 14
        self.atr_period = 14
        
        # Seuils de signaux
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.atr_multiplier = 2.0
        
        # Cache pour les calculs
        self._indicator_cache = {}
    
    def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les messages pour l'analyse technique"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'analyze_technical':
                return self._analyze_technical(message)
            elif msg_type == 'calculate_indicators':
                return self._calculate_indicators(message)
            elif msg_type == 'generate_signals':
                return self._generate_signals(message)
            elif msg_type == 'calculate_stops':
                return self._calculate_stops(message)
            else:
                return {
                    'status': 'error',
                    'message': f'Type de message non supporté: {msg_type}'
                }
                
        except Exception as e:
            self.logger.error(f"Erreur processing message: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _analyze_technical(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse technique complète d'un actif"""
        ticker = message.get('ticker')
        price_data = message.get('price_data')  # DataFrame avec OHLCV
        
        if not ticker or price_data is None:
            return {
                'status': 'error',
                'message': 'Ticker et price_data requis'
            }
        
        # Convertir en DataFrame si nécessaire
        if isinstance(price_data, list):
            # Liste de dictionnaires (format records)
            df = pd.DataFrame(price_data)
        elif isinstance(price_data, dict):
            # Dictionnaire de listes
            df = pd.DataFrame(price_data)
        else:
            # DataFrame existant
            df = price_data.copy()
        
        # Calculer tous les indicateurs
        indicators = self._calculate_all_indicators(df)
        
        # Générer les signaux
        signals = self._generate_trading_signals(df, indicators)
        
        # Calculer stops et targets
        stops_targets = self._calculate_stops_and_targets(df, indicators)
        
        return {
            'status': 'success',
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'indicators': indicators,
            'signals': signals,
            'stops_targets': stops_targets,
            'current_price': float(df['close'].iloc[-1]) if 'close' in df else None
        }
    
    def _calculate_indicators(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les indicateurs techniques"""
        price_data = message.get('price_data')
        indicators_requested = message.get('indicators', ['ema', 'rsi', 'atr'])
        
        if isinstance(price_data, list):
            # Liste de dictionnaires (format records)
            df = pd.DataFrame(price_data)
        elif isinstance(price_data, dict):
            # Dictionnaire de listes
            df = pd.DataFrame(price_data)
        else:
            # DataFrame existant
            df = price_data.copy()
        
        results = {}
        
        if 'ema' in indicators_requested:
            results['ema_short'] = self._calculate_ema(df['close'], self.ema_short)
            results['ema_long'] = self._calculate_ema(df['close'], self.ema_long)
        
        if 'rsi' in indicators_requested:
            results['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        if 'atr' in indicators_requested:
            results['atr'] = self._calculate_atr(df, self.atr_period)
        
        if 'bollinger' in indicators_requested:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
            results['bb_upper'] = bb_upper
            results['bb_middle'] = bb_middle
            results['bb_lower'] = bb_lower
        
        return {
            'status': 'success',
            'indicators': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcule tous les indicateurs pour un DataFrame"""
        indicators = {}
        
        # EMAs
        indicators['ema_short'] = self._calculate_ema(df['close'], self.ema_short)
        indicators['ema_long'] = self._calculate_ema(df['close'], self.ema_long)
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # ATR
        indicators['atr'] = self._calculate_atr(df, self.atr_period)
        
        # MACD
        indicators['macd'], indicators['macd_signal'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = \
            self._calculate_bollinger_bands(df['close'])
        
        return indicators
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcule EMA (Exponential Moving Average)"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcule RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calcule ATR (Average True Range)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calcule MACD et signal line"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd, signal)
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _generate_trading_signals(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des signaux de trading basés sur les indicateurs"""
        signals = {
            'ema_signal': 'neutral',
            'rsi_signal': 'neutral',
            'macd_signal': 'neutral',
            'bollinger_signal': 'neutral',
            'overall_signal': 'neutral',
            'signal_strength': 0.0
        }
        
        current_price = df['close'].iloc[-1]
        ema_short_current = indicators['ema_short'].iloc[-1]
        ema_long_current = indicators['ema_long'].iloc[-1]
        rsi_current = indicators['rsi'].iloc[-1]
        macd_current = indicators['macd'].iloc[-1]
        macd_signal_current = indicators['macd_signal'].iloc[-1]
        
        signal_count = 0
        bullish_signals = 0
        
        # Signal EMA
        if ema_short_current > ema_long_current:
            signals['ema_signal'] = 'bullish'
            bullish_signals += 1
        elif ema_short_current < ema_long_current:
            signals['ema_signal'] = 'bearish'
        signal_count += 1
        
        # Signal RSI
        if rsi_current < self.rsi_oversold:
            signals['rsi_signal'] = 'bullish'  # Oversold = buy signal
            bullish_signals += 1
        elif rsi_current > self.rsi_overbought:
            signals['rsi_signal'] = 'bearish'  # Overbought = sell signal
        signal_count += 1
        
        # Signal MACD
        if macd_current > macd_signal_current:
            signals['macd_signal'] = 'bullish'
            bullish_signals += 1
        elif macd_current < macd_signal_current:
            signals['macd_signal'] = 'bearish'
        signal_count += 1
        
        # Signal Bollinger
        bb_upper_current = indicators['bb_upper'].iloc[-1]
        bb_lower_current = indicators['bb_lower'].iloc[-1]
        if current_price < bb_lower_current:
            signals['bollinger_signal'] = 'bullish'  # Price below lower band
            bullish_signals += 1
        elif current_price > bb_upper_current:
            signals['bollinger_signal'] = 'bearish'  # Price above upper band
        signal_count += 1
        
        # Signal global et force
        bullish_ratio = bullish_signals / signal_count
        if bullish_ratio >= 0.75:
            signals['overall_signal'] = 'strong_bullish'
            signals['signal_strength'] = bullish_ratio
        elif bullish_ratio >= 0.5:
            signals['overall_signal'] = 'bullish'
            signals['signal_strength'] = bullish_ratio
        elif bullish_ratio <= 0.25:
            signals['overall_signal'] = 'strong_bearish'
            signals['signal_strength'] = 1 - bullish_ratio
        elif bullish_ratio < 0.5:
            signals['overall_signal'] = 'bearish'
            signals['signal_strength'] = 1 - bullish_ratio
        
        return signals
    
    def _generate_signals(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des signaux pour une liste d'actifs"""
        tickers = message.get('tickers', [])
        price_data_dict = message.get('price_data_dict', {})
        
        results = {}
        
        for ticker in tickers:
            if ticker in price_data_dict:
                df = pd.DataFrame(price_data_dict[ticker])
                indicators = self._calculate_all_indicators(df)
                signals = self._generate_trading_signals(df, indicators)
                
                results[ticker] = {
                    'signals': signals,
                    'current_price': float(df['close'].iloc[-1]),
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'status': 'success',
            'results': results,
            'total_analyzed': len(results)
        }
    
    def _calculate_stops_and_targets(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, float]:
        """Calcule stops et targets basés sur ATR"""
        current_price = df['close'].iloc[-1]
        atr_current = indicators['atr'].iloc[-1]
        
        # Stop loss basé sur ATR
        stop_distance = atr_current * self.atr_multiplier
        
        return {
            'stop_loss_long': current_price - stop_distance,
            'stop_loss_short': current_price + stop_distance,
            'target_long': current_price + (stop_distance * 2),  # Risk/Reward 1:2
            'target_short': current_price - (stop_distance * 2),
            'atr_value': atr_current,
            'stop_distance': stop_distance
        }
    
    def _calculate_stops(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule stops pour une position"""
        ticker = message.get('ticker')
        current_price = message.get('current_price')
        position_type = message.get('position_type', 'long')  # 'long' ou 'short'
        atr_value = message.get('atr_value')
        
        if not all([ticker, current_price, atr_value]):
            return {
                'status': 'error',
                'message': 'ticker, current_price et atr_value requis'
            }
        
        stop_distance = atr_value * self.atr_multiplier
        
        if position_type == 'long':
            stop_loss = current_price - stop_distance
            target = current_price + (stop_distance * 2)
        else:  # short
            stop_loss = current_price + stop_distance
            target = current_price - (stop_distance * 2)
        
        return {
            'status': 'success',
            'ticker': ticker,
            'position_type': position_type,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'target': target,
            'stop_distance': stop_distance,
            'risk_reward_ratio': 2.0
        }
    
    def get_market_regime(self, price_data: pd.DataFrame) -> str:
        """Détermine le régime de marché actuel"""
        indicators = self._calculate_all_indicators(price_data)
        
        ema_short = indicators['ema_short'].iloc[-1]
        ema_long = indicators['ema_long'].iloc[-1]
        atr_current = indicators['atr'].iloc[-1]
        atr_avg = indicators['atr'].rolling(50).mean().iloc[-1]
        
        # Tendance
        if ema_short > ema_long * 1.02:
            trend = "bullish"
        elif ema_short < ema_long * 0.98:
            trend = "bearish"
        else:
            trend = "sideways"
        
        # Volatilité
        if atr_current > atr_avg * 1.3:
            volatility = "high"
        elif atr_current < atr_avg * 0.7:
            volatility = "low"
        else:
            volatility = "normal"
        
        return f"{trend}_{volatility}"
    
    def health_check(self) -> bool:
        """Vérifie la santé de l'agent"""
        try:
            # Test calcul simple
            test_data = pd.Series([1, 2, 3, 4, 5])
            ema = self._calculate_ema(test_data, 3)
            return len(ema) == 5 and not ema.isna().all()
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut de l'agent"""
        return {
            'agent_name': self.agent_name,
            'health': self.health_check(),
            'cache_size': len(self._indicator_cache),
            'parameters': {
                'ema_short': self.ema_short,
                'ema_long': self.ema_long,
                'rsi_period': self.rsi_period,
                'atr_period': self.atr_period,
                'atr_multiplier': self.atr_multiplier
            }
        }


if __name__ == "__main__":
    # Test rapide
    agent = TechnicalAgent()
    print(f"Technical Agent initialized: {agent.get_status()}")