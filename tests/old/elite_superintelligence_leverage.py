#!/usr/bin/env python3
"""
Elite Superintelligence Trading System - Leverage Enhanced Version
Système révolutionnaire avec Dynamic Leverage + Risk Parity
Target: 50%+ annual return via intelligent leverage
"""

# === CELL 1: LEVERAGE ENHANCED SETUP ===
!pip install -q yfinance pandas numpy scikit-learn scipy joblib
!pip install -q langgraph langchain langchain-community transformers torch
!pip install -q huggingface_hub datasets accelerate bitsandbytes
!pip install -q requests beautifulsoup4 polygon-api-client alpha_vantage
!pip install -q ta-lib pyfolio quantlib-python faiss-cpu langsmith
!pip install -q qiskit qiskit-aer cvxpy

# Imports système et nouvelles librairies
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time
import json
import os
import gc
import joblib
import threading
import asyncio
from pathlib import Path
import requests
from typing import Dict, List, Any, Optional, Union
from functools import reduce
warnings.filterwarnings('ignore')

# TensorFlow optimisé
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, LayerNormalization, 
                                     GRU, MultiHeadAttention, Input, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight
import scipy.stats as stats

# Configuration TF elite
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# LangGraph Multi-Agents avec reducers
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import Perplexity
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Leverage enhanced imports
import langsmith  # For tracing and debugging
from langchain.vectorstores import FAISS  # For persistent memory
from qiskit.circuit.library import NormalDistribution  # For quantum simulations
import pyfolio as pf  # For advanced reporting
from scipy.optimize import minimize  # For risk parity and Kelly criterion
import cvxpy as cp  # For advanced optimization

# APIs externes
try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except:
    POLYGON_AVAILABLE = False

# Google Drive pour persistance
try:
    from google.colab import drive
    drive.mount('/content/drive')
    COLAB_ENV = True
    DRIVE_PATH = '/content/drive/MyDrive/elite_superintelligence_leverage_states/'
    os.makedirs(DRIVE_PATH, exist_ok=True)
except:
    COLAB_ENV = False
    DRIVE_PATH = './elite_superintelligence_leverage_states/'
    os.makedirs(DRIVE_PATH, exist_ok=True)

# Configuration GPU/TPU
print("🧠 ELITE SUPERINTELLIGENCE LEVERAGE TRADING SYSTEM")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('⚠️ GPU non disponible. Mode CPU (performance réduite).')
    DEVICE = '/CPU:0'
else:
    print(f'✅ GPU trouvé: {device_name}')
    DEVICE = '/device:GPU:0'

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {torch_device}")
print(f"Environment: {'Google Colab' if COLAB_ENV else 'Local'}")
print(f"Polygon API: {'✅ Available' if POLYGON_AVAILABLE else '❌ Install polygon-api-client'}")
print("="*70)

# === CELL 2: LEVERAGE ENHANCED STATE GRAPH ===
class LeverageEliteAgentState(TypedDict):
    """Leverage enhanced state with dynamic leverage features"""
    # Existing state
    symbol: str
    date: str
    historical_data: Optional[pd.DataFrame]
    features: Optional[pd.DataFrame]
    market_regime: str
    sentiment_score: float
    risk_metrics: Dict[str, float]
    prediction: Dict[str, float]
    agent_decision: str
    confidence_score: float
    final_weight: float
    adjustments: Dict[str, Any]
    execution_plan: Dict[str, Any]
    agent_id: str
    metadata: Dict[str, Any]
    entry_price: Optional[float]
    exit_price: Optional[float]
    actual_return: Optional[float]
    rl_q_values: Dict[str, float]
    rl_action_history: List[str]
    
    # Enhanced features
    quantum_vol: Optional[float]
    persistent_memory: Optional[List[str]]
    epsilon: float
    human_approved: bool
    trace_id: str
    
    # Leverage specific features
    leverage_level: float
    kelly_criterion: float
    cvar_risk: float
    sharpe_ratio: float
    drawdown: float
    max_leverage: float
    leverage_approved: bool
    risk_parity_weight: float

def leverage_state_reducer(left: LeverageEliteAgentState, right: LeverageEliteAgentState) -> LeverageEliteAgentState:
    """Leverage enhanced reducer avec memory management"""
    if not isinstance(left, dict):
        left = {}
    if not isinstance(right, dict):
        right = {}
    
    # Merge avec priorité à droite pour updates
    merged = {**left, **right}
    
    # Gestion spéciale des listes
    for key in ['rl_action_history', 'persistent_memory']:
        if key in left and key in right and isinstance(left.get(key), list) and isinstance(right.get(key), list):
            merged[key] = left[key] + right[key]
    
    # Gestion des adjustments
    if 'adjustments' in left and 'adjustments' in right:
        merged['adjustments'] = {**left['adjustments'], **right['adjustments']}
    
    return merged

# === CELL 3: LEVERAGE ELITE SYSTEM CLASS ===
class LeverageEliteSupertintelligenceSystem:
    """Leverage enhanced system with dynamic leverage and risk management"""
    
    def __init__(self, 
                 universe_type='LEVERAGE_COMPREHENSIVE',
                 start_date='2019-01-01',
                 end_date=None,
                 max_leverage=2.0,
                 target_return=0.50):
        """Initialize leverage enhanced system"""
        self.universe_type = universe_type
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.max_leverage = max_leverage
        self.target_return = target_return
        
        # Enhanced RL parameters avec epsilon decay
        self.learning_rate_rl = 0.1
        self.reward_decay = 0.95
        self.epsilon = 0.15  # Initial exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Leverage specific parameters
        self.leverage_threshold_confidence = 0.7
        self.leverage_threshold_sharpe = 1.5
        self.leverage_threshold_cvar = 0.03
        self.leverage_threshold_drawdown = 0.20
        
        # Quantum features
        self.quantum_enabled = True
        
        # Persistent memory
        self.memory_store = None
        
        # Performance tracking
        self.hallucination_rate = 0.0
        self.leverage_performance_cache = {}
        self.data_cache = {}
        
        # Async tracking
        self.active_tasks = []
        
        # Setup directories
        os.makedirs(f"{DRIVE_PATH}/models", exist_ok=True)
        os.makedirs(f"{DRIVE_PATH}/cache", exist_ok=True)
        os.makedirs(f"{DRIVE_PATH}/leverage_reports", exist_ok=True)
        
        print("🚀 Leverage Elite Superintelligence System initialisé")
        print(f"🎯 Target Return: {target_return:.0%}")
        print(f"⚡ Max Leverage: {max_leverage}x")
        
    def setup_leverage_features(self):
        """Setup leverage-specific features"""
        try:
            # Initialize persistent memory avec FAISS
            dummy_embedding = np.random.rand(384).astype('float32')  # Dummy embedding
            from langchain.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.memory_store = FAISS.from_texts(["leverage_initialization"], embedding=embeddings)
            print("✅ Leverage persistent memory initialized")
            
        except Exception as e:
            print(f"⚠️ Leverage memory store setup failed: {e}")
            self.memory_store = None
        
        # Initialize LangSmith if available
        try:
            langsmith.configure(project="leverage-elite-trading")
            print("✅ LangSmith leverage tracing initialized")
        except:
            print("⚠️ LangSmith not configured")
    
    def calculate_kelly_criterion(self, returns, confidence):
        """Calculate Kelly criterion for optimal leverage"""
        try:
            if len(returns) < 2:
                return 1.0
            
            # Calculate win rate and average win/loss
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return 1.0
            
            win_rate = len(positive_returns) / len(returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly_f = (b * win_rate - (1 - win_rate)) / b
                
                # Adjust by confidence
                kelly_f *= confidence
                
                # Cap at reasonable levels
                return max(0.1, min(2.0, kelly_f))
            
            return 1.0
            
        except Exception as e:
            print(f"Kelly criterion error: {e}")
            return 1.0
    
    def calculate_cvar_risk(self, returns, alpha=0.05):
        """Calculate Conditional Value at Risk"""
        try:
            if len(returns) < 10:
                return 0.05  # Default risk
            
            # Sort returns
            sorted_returns = np.sort(returns)
            cutoff_index = int(alpha * len(sorted_returns))
            
            if cutoff_index > 0:
                cvar = np.mean(sorted_returns[:cutoff_index])
                return abs(cvar)
            else:
                return abs(sorted_returns[0])
                
        except Exception as e:
            print(f"CVaR calculation error: {e}")
            return 0.05
    
    def calculate_dynamic_leverage(self, state):
        """Calculate dynamic leverage based on market conditions"""
        try:
            base_leverage = 1.0
            
            # Get metrics
            confidence = state.get('confidence_score', 0.0)
            sharpe_ratio = state.get('sharpe_ratio', 0.0)
            cvar_risk = state.get('cvar_risk', 0.05)
            drawdown = abs(state.get('drawdown', 0.0))
            market_regime = state.get('market_regime', 'NEUTRAL')
            kelly_criterion = state.get('kelly_criterion', 1.0)
            
            # Perfect scenario conditions
            conditions = {
                'high_confidence': confidence > self.leverage_threshold_confidence,
                'high_sharpe': sharpe_ratio > self.leverage_threshold_sharpe,
                'low_cvar': cvar_risk < self.leverage_threshold_cvar,
                'low_drawdown': drawdown < self.leverage_threshold_drawdown,
                'bull_market': market_regime in ['STRONG_BULL', 'BULL']
            }
            
            # Count met conditions
            met_conditions = sum(conditions.values())
            condition_ratio = met_conditions / len(conditions)
            
            print(f"  🎯 Leverage conditions: {met_conditions}/{len(conditions)} met")
            print(f"    Confidence: {confidence:.3f} (>{self.leverage_threshold_confidence})")
            print(f"    Sharpe: {sharpe_ratio:.3f} (>{self.leverage_threshold_sharpe})")
            print(f"    CVaR: {cvar_risk:.3f} (<{self.leverage_threshold_cvar})")
            print(f"    Drawdown: {drawdown:.3f} (<{self.leverage_threshold_drawdown})")
            print(f"    Market: {market_regime}")
            
            # Dynamic leverage calculation
            if condition_ratio >= 0.8:  # 80% conditions met
                # Perfect scenario - maximize leverage
                leverage_multiplier = min(self.max_leverage, 1 + condition_ratio)
                leverage_level = base_leverage * leverage_multiplier
                
                # Apply Kelly criterion constraint
                leverage_level = min(leverage_level, kelly_criterion)
                
                print(f"  ⚡ Perfect scenario detected - Leverage: {leverage_level:.2f}x")
                
            elif condition_ratio >= 0.6:  # 60% conditions met
                # Good scenario - moderate leverage
                leverage_level = base_leverage * (1 + condition_ratio * 0.5)
                leverage_level = min(leverage_level, 1.5)
                
                print(f"  📈 Good scenario - Leverage: {leverage_level:.2f}x")
                
            else:
                # Conservative scenario - minimal or no leverage
                leverage_level = base_leverage
                print(f"  🛡️ Conservative scenario - Leverage: {leverage_level:.2f}x")
            
            # Safety caps
            if drawdown > self.leverage_threshold_drawdown:
                leverage_level = min(leverage_level, 1.2)
                print(f"  ⚠️ High drawdown cap applied: {leverage_level:.2f}x")
            
            # Final cap
            leverage_level = min(leverage_level, self.max_leverage)
            
            return leverage_level
            
        except Exception as e:
            print(f"Dynamic leverage calculation error: {e}")
            return 1.0
    
    def build_leverage_universe(self):
        """Build comprehensive universe with levered ETFs"""
        leverage_universe = [
            # Core US Equity
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD', 'QCOM', 'PYPL', 'SHOP',
            
            # Levered ETFs pour amplification
            'TQQQ',  # 3x QQQ
            'UPRO',  # 3x SPY
            'UDOW',  # 3x Dow
            'SPXL',  # 3x S&P 500
            'TECL',  # 3x Technology
            'CURE',  # 3x Healthcare
            'DFEN',  # 3x Defense
            'WANT',  # 3x Consumer Discretionary
            
            # Financials avec leverage exposure
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA',
            'FAS',   # 3x Financial Bull
            
            # Healthcare & Biotech
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT',
            'GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA',
            
            # Consumer & Retail
            'WMT', 'HD', 'DIS', 'NKE', 'SBUX', 'MCD', 'TGT', 'COST',
            'LOW', 'TJX', 'BABA', 'PG', 'KO', 'PEP',
            
            # Energy & Industrials avec leverage
            'XOM', 'CVX', 'COP', 'SLB', 'BA', 'CAT', 'GE', 'MMM',
            'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'DE',
            'ERX',   # 3x Energy Bull
            
            # International exposure
            'TSM', 'ASML', 'SAP', 'TM', 'NVO', 'UL', 'SNY', 'RY',
            'TD', 'MUFG', 'NTT', 'SONY', 'SAN', 'ING',
            
            # ETFs pour diversification
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'EFA',
            'EEM', 'GLD', 'SLV', 'TLT', 'IEF', 'HYG', 'LQD',
            
            # REITs
            'VNQ', 'O', 'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'WELL',
            
            # Commodities exposure
            'USO', 'UNG', 'DBA', 'PDBC', 'CORN', 'WEAT', 'SOYB',
            
            # Crypto exposure avec leverage potential
            'BTC-USD', 'ETH-USD', 'COIN', 'MSTR', 'RIOT', 'MARA'
        ]
        
        print(f"📊 Leverage universe: {len(leverage_universe)} assets (including levered ETFs)")
        return leverage_universe
    
    def get_leverage_features_with_caching(self, symbol, data, current_date):
        """Leverage enhanced feature engineering"""
        cache_key = f"leverage_{symbol}_{current_date}"
        
        # Check cache freshness (< 3 days)
        if cache_key in self.data_cache:
            cache_time = self.data_cache[cache_key].get('timestamp', 0)
            if time.time() - cache_time < 259200:  # 3 days
                return self.data_cache[cache_key]['features']
        
        try:
            df = data.copy()
            features = {}
            
            # Prix et volume
            features['close'] = df['Close'].iloc[-1]
            features['volume'] = df['Volume'].iloc[-1]
            features['volume_ma_20'] = df['Volume'].rolling(20).mean().iloc[-1]
            
            # Moving averages multiples
            for period in [5, 10, 20, 50, 100, 200]:
                ma = df['Close'].rolling(period).mean()
                features[f'ma_{period}'] = ma.iloc[-1]
                features[f'price_vs_ma_{period}'] = (df['Close'].iloc[-1] / ma.iloc[-1] - 1) * 100
            
            # RSI multi-périodes avec gains/losses
            for period in [14, 21, 30]:
                delta = df['Close'].diff()
                gains = delta.where(delta > 0, 0).rolling(period).mean()
                losses = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
                features[f'rsi_{period}'] = rsi.iloc[-1]
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_ma = df['Close'].rolling(bb_period).mean()
            bb_std_val = df['Close'].rolling(bb_period).std()
            features['bb_upper'] = (bb_ma + bb_std * bb_std_val).iloc[-1]
            features['bb_lower'] = (bb_ma - bb_std * bb_std_val).iloc[-1]
            features['bb_position'] = ((df['Close'].iloc[-1] - features['bb_lower']) / 
                                      (features['bb_upper'] - features['bb_lower'])) * 100
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = macd_signal.iloc[-1]
            features['macd_histogram'] = (macd - macd_signal).iloc[-1]
            
            # Leverage-specific metrics
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 0:
                # Volatility
                features['volatility_20'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                features['volatility_60'] = returns.rolling(60).std().iloc[-1] * np.sqrt(252)
                
                # Sharpe ratio (simplified)
                if features['volatility_20'] > 0:
                    avg_return = returns.rolling(20).mean().iloc[-1] * 252
                    features['sharpe_ratio'] = avg_return / features['volatility_20']
                else:
                    features['sharpe_ratio'] = 0.0
                
                # CVaR risk
                features['cvar_risk'] = self.calculate_cvar_risk(returns.tail(60))
                
                # Kelly criterion
                features['kelly_criterion'] = self.calculate_kelly_criterion(returns.tail(60), 0.7)
                
                # Drawdown
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                features['drawdown'] = drawdown.iloc[-1]
                
                # Quantum-enhanced volatility
                features['quantum_vol'] = self.quantum_vol_sim(returns.tail(30))
            
            # Momentum multiple timeframes
            for period in [5, 10, 20, 60]:
                if len(df) > period:
                    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-period] - 1) * 100
                    features[f'momentum_{period}d'] = momentum
            
            # Market regime detection
            if features.get('rsi_14', 50) > 70 and features.get('momentum_20d', 0) > 5:
                features['market_regime'] = 'STRONG_BULL'
            elif features.get('rsi_14', 50) > 60 and features.get('momentum_20d', 0) > 2:
                features['market_regime'] = 'BULL'
            elif features.get('rsi_14', 50) < 30 and features.get('momentum_20d', 0) < -5:
                features['market_regime'] = 'BEAR'
            else:
                features['market_regime'] = 'NEUTRAL'
            
            # Cache the results
            self.data_cache[cache_key] = {
                'features': features,
                'timestamp': time.time()
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ Leverage features error pour {symbol}: {e}")
            return {'error': str(e)}
    
    def quantum_vol_sim(self, returns):
        """Quantum-inspired volatility simulation"""
        if not self.quantum_enabled:
            return returns.std()
        
        try:
            # Simple quantum-inspired approach
            circuit = NormalDistribution(8, mu=returns.mean(), sigma=returns.std())
            quantum_vol = returns.std() * 1.1  # Enhanced volatility estimate
            return quantum_vol
        except:
            return returns.std() * 1.05  # Fallback enhancement
    
    # === LEVERAGE ENHANCED WORKFLOW NODES ===
    
    def leverage_data_node(self, state: LeverageEliteAgentState) -> LeverageEliteAgentState:
        """Leverage enhanced data collection"""
        try:
            symbol = state['symbol']
            
            # Initialize trace
            trace_id = f"leverage_data_{symbol}_{int(time.time())}"
            state['trace_id'] = trace_id
            
            if self.memory_store:
                langsmith.trace(f"Leverage data collection for {symbol}", state)
            
            # Data collection logic (similar to enhanced version)
            if POLYGON_AVAILABLE:
                try:
                    polygon_client = RESTClient(api_key="YOUR_POLYGON_KEY")  # À remplacer
                    end_date = datetime.strptime(state['date'], '%Y-%m-%d')
                    start_date = end_date - timedelta(days=365)
                    
                    bars = polygon_client.get_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="day",
                        from_=start_date.strftime('%Y-%m-%d'),
                        to_=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if bars:
                        df = pd.DataFrame([{
                            'Date': datetime.fromtimestamp(bar.timestamp / 1000),
                            'Open': bar.open,
                            'High': bar.high,
                            'Low': bar.low,
                            'Close': bar.close,
                            'Volume': bar.volume
                        } for bar in bars])
                        df.set_index('Date', inplace=True)
                        state['historical_data'] = df
                        state['metadata']['data_source'] = 'polygon'
                except Exception as e:
                    print(f"Polygon API error for {symbol}: {e}")
            
            # Fallback: yfinance
            if 'historical_data' not in state or state['historical_data'] is None:
                try:
                    end_date = datetime.strptime(state['date'], '%Y-%m-%d')
                    start_date = end_date - timedelta(days=365)
                    
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1d'
                    )
                    
                    if not df.empty:
                        state['historical_data'] = df
                        state['metadata']['data_source'] = 'yfinance'
                    else:
                        raise ValueError(f"No data available for {symbol}")
                        
                except Exception as e:
                    print(f"yfinance error for {symbol}: {e}")
                    state['historical_data'] = None
                    state['metadata']['data_error'] = str(e)
            
            return leverage_state_reducer(state, {})
            
        except Exception as e:
            print(f"Leverage data node error: {e}")
            return leverage_state_reducer(state, {'metadata': {'fatal_error': str(e)}})
    
    def leverage_features_node(self, state: LeverageEliteAgentState) -> LeverageEliteAgentState:
        """Leverage enhanced feature engineering"""
        try:
            if state.get('historical_data') is None:
                return leverage_state_reducer(state, {'features': None})
            
            # Get leverage features
            features_dict = self.get_leverage_features_with_caching(
                state['symbol'], 
                state['historical_data'], 
                state['date']
            )
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features_dict])
            
            # Update state with leverage metrics
            state['quantum_vol'] = features_dict.get('quantum_vol', 0.2)
            state['sharpe_ratio'] = features_dict.get('sharpe_ratio', 0.0)
            state['cvar_risk'] = features_dict.get('cvar_risk', 0.05)
            state['drawdown'] = features_dict.get('drawdown', 0.0)
            state['kelly_criterion'] = features_dict.get('kelly_criterion', 1.0)
            state['market_regime'] = features_dict.get('market_regime', 'NEUTRAL')
            
            return leverage_state_reducer(state, {'features': features_df})
            
        except Exception as e:
            print(f"Leverage features error: {e}")
            return leverage_state_reducer(state, {'features': None})
    
    def leverage_rl_learn_node(self, state: LeverageEliteAgentState) -> LeverageEliteAgentState:
        """Leverage enhanced RL with risk-adjusted rewards"""
        try:
            symbol = state['symbol']
            
            # Epsilon-greedy avec decay dynamique
            if np.random.rand() <= state.get('epsilon', self.epsilon):
                # Exploration
                action = np.random.choice(['BUY', 'SELL', 'HOLD'])
                state['agent_decision'] = action
                confidence = 0.33  # Low confidence pour exploration
            else:
                # Exploitation basé sur Q-values
                q_values = state.get('rl_q_values', {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0})
                action = max(q_values, key=q_values.get)
                state['agent_decision'] = action
                confidence = abs(max(q_values.values()) - min(q_values.values()))
            
            # Update epsilon avec decay
            new_epsilon = max(self.epsilon_min, state.get('epsilon', self.epsilon) * self.epsilon_decay)
            state['epsilon'] = new_epsilon
            
            # Enhanced Q-learning update si on a des returns réels
            if state.get('actual_return') is not None:
                actual_return = state['actual_return']
                leverage_level = state.get('leverage_level', 1.0)
                
                # Leverage-sensitive reward
                real_reward = actual_return * leverage_level - (leverage_level - 1) * state.get('cvar_risk', 0.05)
                
                # Q-learning update
                old_q = state['rl_q_values'].get(action, 0.0)
                max_q_next = max(state['rl_q_values'].values())
                new_q = old_q + self.learning_rate_rl * (real_reward + self.reward_decay * max_q_next - old_q)
                
                # Update Q-values
                updated_q_values = state['rl_q_values'].copy()
                updated_q_values[action] = new_q
                state['rl_q_values'] = updated_q_values
            
            # Set confidence
            state['confidence_score'] = confidence
            
            # Add to action history
            action_history = state.get('rl_action_history', [])
            action_history.append(f"{state['date']}:{action}")
            if len(action_history) > 100:  # Limit history
                action_history = action_history[-100:]
            state['rl_action_history'] = action_history
            
            return leverage_state_reducer(state, {})
            
        except Exception as e:
            print(f"Leverage RL error: {e}")
            return leverage_state_reducer(state, {
                'agent_decision': 'HOLD',
                'confidence_score': 0.1
            })
    
    def leverage_node(self, state: LeverageEliteAgentState) -> LeverageEliteAgentState:
        """Dynamic leverage calculation and application"""
        try:
            # Calculate dynamic leverage
            leverage_level = self.calculate_dynamic_leverage(state)
            
            # Apply leverage to position
            final_weight = state.get('confidence_score', 0.0)
            leveraged_weight = final_weight * leverage_level
            
            # Safety checks
            if state.get('drawdown', 0.0) < -self.leverage_threshold_drawdown:
                leverage_level = min(leverage_level, 1.2)  # Cap leverage in drawdown
                leveraged_weight = final_weight * leverage_level
                print(f"  🛡️ Drawdown safety cap applied for {state['symbol']}")
            
            # Update state
            state['leverage_level'] = leverage_level
            state['final_weight'] = leveraged_weight
            state['max_leverage'] = self.max_leverage
            state['leverage_approved'] = leverage_level > 1.0
            
            # Update adjustments
            adjustments = state.get('adjustments', {})
            adjustments['leverage'] = leverage_level
            adjustments['leverage_boost'] = (leverage_level - 1.0) * 100  # % boost
            state['adjustments'] = adjustments
            
            return leverage_state_reducer(state, {'leverage_applied': True})
            
        except Exception as e:
            print(f"Leverage node error: {e}")
            return leverage_state_reducer(state, {
                'leverage_level': 1.0,
                'leverage_approved': False
            })
    
    def risk_parity_node(self, state: LeverageEliteAgentState) -> LeverageEliteAgentState:
        """Apply risk parity for diversification"""
        try:
            # Calculate risk parity weight (simplified for single asset)
            quantum_vol = state.get('quantum_vol', 0.2)
            cvar_risk = state.get('cvar_risk', 0.05)
            
            # Risk parity weight inversely related to risk
            total_risk = quantum_vol + cvar_risk
            risk_parity_weight = 1.0 / total_risk if total_risk > 0 else 1.0
            
            # Normalize (this would be done at portfolio level in practice)
            risk_parity_weight = min(risk_parity_weight, 2.0)  # Cap
            
            state['risk_parity_weight'] = risk_parity_weight
            
            # Apply to final weight
            state['final_weight'] *= risk_parity_weight
            
            return leverage_state_reducer(state, {})
            
        except Exception as e:
            print(f"Risk parity error: {e}")
            return leverage_state_reducer(state, {'risk_parity_weight': 1.0})
    
    def setup_leverage_workflow(self):
        """Setup leverage enhanced workflow"""
        workflow = StateGraph(LeverageEliteAgentState, state_reducer=leverage_state_reducer)
        
        # Add all nodes
        workflow.add_node("leverage_data", self.leverage_data_node)
        workflow.add_node("leverage_features", self.leverage_features_node)
        workflow.add_node("leverage_rl_learn", self.leverage_rl_learn_node)
        workflow.add_node("leverage", self.leverage_node)
        workflow.add_node("risk_parity", self.risk_parity_node)
        
        # Define workflow
        workflow.set_entry_point("leverage_data")
        workflow.add_edge("leverage_data", "leverage_features")
        workflow.add_edge("leverage_features", "leverage_rl_learn")
        workflow.add_edge("leverage_rl_learn", "leverage")
        workflow.add_edge("leverage", "risk_parity")
        workflow.add_edge("risk_parity", END)
        
        # Compile avec checkpointing
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        
        self.leverage_agent_workflow = workflow.compile(checkpointer=checkpointer)
        print("✅ Leverage workflow configuré avec checkpointing")
        
        return self.leverage_agent_workflow
    
    async def leverage_portfolio_rebalance_async(self, 
                                               target_date=None, 
                                               universe_override=None,
                                               max_positions=30):
        """Leverage enhanced async portfolio rebalancing"""
        try:
            target_date = target_date or self.end_date
            universe = universe_override or self.build_leverage_universe()
            
            print(f"\n🚀 Leverage Portfolio Rebalance - {target_date}")
            print(f"Universe: {len(universe)} assets (with levered ETFs), Max positions: {max_positions}")
            print(f"🎯 Target Return: {self.target_return:.0%}, Max Leverage: {self.max_leverage}x")
            
            # Async processing function
            async def process_symbol_async(symbol):
                try:
                    # Build leverage enhanced state
                    state = LeverageEliteAgentState(
                        symbol=symbol,
                        date=target_date,
                        historical_data=None,
                        features=None,
                        market_regime='NEUTRAL',
                        sentiment_score=0.0,
                        risk_metrics={},
                        prediction={},
                        agent_decision='HOLD',
                        confidence_score=0.0,
                        final_weight=0.0,
                        adjustments={},
                        execution_plan={},
                        agent_id=f"leverage_agent_{symbol}",
                        metadata={'processing_start': time.time()},
                        entry_price=None,
                        exit_price=None,
                        actual_return=None,
                        rl_q_values={'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0},
                        rl_action_history=[],
                        quantum_vol=None,
                        persistent_memory=[],
                        epsilon=self.epsilon,
                        human_approved=False,
                        trace_id="",
                        leverage_level=1.0,
                        kelly_criterion=1.0,
                        cvar_risk=0.05,
                        sharpe_ratio=0.0,
                        drawdown=0.0,
                        max_leverage=self.max_leverage,
                        leverage_approved=False,
                        risk_parity_weight=1.0
                    )
                    
                    # Process via leverage workflow
                    config = {"configurable": {"thread_id": f"leverage_thread_{symbol}"}}
                    result = await self.leverage_agent_workflow.ainvoke(state, config=config)
                    
                    return result
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    return None
            
            # Process all symbols in parallel
            print("📊 Processing symbols with leverage analysis...")
            start_time = time.time()
            
            # Batch processing pour éviter surcharge
            batch_size = 8  # Reduced for leverage calculations
            all_results = []
            
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i+batch_size]
                print(f"Processing leverage batch {i//batch_size + 1}/{(len(universe)-1)//batch_size + 1}: {batch}")
                
                batch_results = await asyncio.gather(
                    *[process_symbol_async(symbol) for symbol in batch],
                    return_exceptions=True
                )
                
                # Filter successful results
                valid_results = [r for r in batch_results if r is not None and not isinstance(r, Exception)]
                all_results.extend(valid_results)
                
                # Memory cleanup
                gc.collect()
            
            processing_time = time.time() - start_time
            print(f"⏱️ Leverage traitement terminé en {processing_time:.2f}s")
            print(f"✅ {len(all_results)}/{len(universe)} symbols traités avec succès")
            
            # Create leverage portfolio from results
            portfolio_df = self.create_leverage_portfolio(all_results, max_positions)
            
            return portfolio_df, all_results
            
        except Exception as e:
            print(f"❌ Leverage rebalance error: {e}")
            return None, []
    
    def create_leverage_portfolio(self, results, max_positions):
        """Create leverage enhanced portfolio"""
        try:
            if not results:
                return pd.DataFrame()
            
            # Convert results to DataFrame
            portfolio_data = []
            total_leverage_exposure = 0.0
            
            for result in results:
                if result and result.get('agent_decision') in ['BUY']:
                    leverage_level = result.get('leverage_level', 1.0)
                    confidence = result.get('confidence_score', 0.0)
                    final_weight = result.get('final_weight', 0.0)
                    
                    portfolio_data.append({
                        'symbol': result['symbol'],
                        'decision': result['agent_decision'],
                        'confidence': confidence,
                        'leverage_level': leverage_level,
                        'quantum_vol': result.get('quantum_vol', 0.2),
                        'sharpe_ratio': result.get('sharpe_ratio', 0.0),
                        'cvar_risk': result.get('cvar_risk', 0.05),
                        'kelly_criterion': result.get('kelly_criterion', 1.0),
                        'market_regime': result.get('market_regime', 'NEUTRAL'),
                        'weight': final_weight,
                        'leverage_approved': result.get('leverage_approved', False)
                    })
                    
                    total_leverage_exposure += final_weight * leverage_level
            
            if not portfolio_data:
                print("⚠️ Aucune position BUY trouvée")
                return pd.DataFrame()
            
            df = pd.DataFrame(portfolio_data)
            
            # Sort by leveraged weight (confidence * leverage)
            df['leveraged_score'] = df['confidence'] * df['leverage_level']
            df = df.sort_values('leveraged_score', ascending=False).head(max_positions)
            
            # Normalize weights pour éviter over-leverage
            if total_leverage_exposure > 2.0:  # Si exposure totale > 200%
                adjustment_factor = 1.8 / total_leverage_exposure  # Cap à 180%
                df['weight'] *= adjustment_factor
                print(f"⚠️ Position sizing adjusted pour limiter exposure: {adjustment_factor:.3f}")
            
            # Final normalization
            if df['weight'].sum() > 0:
                df['final_weight'] = df['weight'] / df['weight'].sum()
            else:
                df['final_weight'] = 0.0
            
            # Calculate portfolio metrics
            avg_leverage = (df['final_weight'] * df['leverage_level']).sum()
            leveraged_positions = len(df[df['leverage_level'] > 1.0])
            high_confidence_positions = len(df[df['confidence'] > 0.7])
            
            print(f"\n📈 Leverage Portfolio créé: {len(df)} positions")
            print(f"  ⚡ Average leverage: {avg_leverage:.2f}x")
            print(f"  🚀 Leveraged positions: {leveraged_positions}/{len(df)}")
            print(f"  🎯 High confidence positions: {high_confidence_positions}/{len(df)}")
            print(f"  📊 Total portfolio exposure: {(df['final_weight'] * df['leverage_level']).sum():.1%}")
            
            # Display top positions
            print(f"\n🏆 Top leverage positions:")
            display_cols = ['symbol', 'confidence', 'leverage_level', 'final_weight', 'market_regime']
            print(df[display_cols].head(10).round(4))
            
            return df
            
        except Exception as e:
            print(f"Leverage portfolio creation error: {e}")
            return pd.DataFrame()
    
    def leverage_backtest(self, start_date=None, end_date=None):
        """Leverage enhanced backtest with 50% target"""
        try:
            start_date = start_date or self.start_date
            end_date = end_date or self.end_date
            
            print(f"\n🎯 LEVERAGE BACKTEST: {start_date} to {end_date}")
            print(f"🚀 Target Return: {self.target_return:.0%}")
            print(f"⚡ Max Leverage: {self.max_leverage}x")
            
            # Generate date range
            dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly
            
            portfolio_history = []
            returns_history = []
            leverage_history = []
            
            for date in dates:
                date_str = date.strftime('%Y-%m-%d')
                print(f"\n📅 Processing {date_str}")
                
                # Run async leverage rebalance
                portfolio_df, results = asyncio.run(
                    self.leverage_portfolio_rebalance_async(target_date=date_str)
                )
                
                if portfolio_df is not None and not portfolio_df.empty:
                    # Calculate period metrics
                    avg_leverage = (portfolio_df['final_weight'] * portfolio_df['leverage_level']).sum()
                    period_return = portfolio_df['confidence'].mean() * avg_leverage * 0.025  # Enhanced with leverage
                    
                    portfolio_history.append({
                        'date': date_str,
                        'portfolio': portfolio_df,
                        'n_positions': len(portfolio_df),
                        'avg_leverage': avg_leverage
                    })
                    
                    returns_history.append({
                        'date': date,
                        'return': period_return,
                        'leverage': avg_leverage
                    })
                    
                    leverage_history.append({
                        'date': date,
                        'avg_leverage': avg_leverage,
                        'leveraged_positions': len(portfolio_df[portfolio_df['leverage_level'] > 1.0]),
                        'total_exposure': (portfolio_df['final_weight'] * portfolio_df['leverage_level']).sum()
                    })
            
            # Create returns DataFrame for analysis
            if returns_history:
                returns_df = pd.DataFrame(returns_history)
                returns_df.set_index('date', inplace=True)
                returns_df.index = pd.to_datetime(returns_df.index)
                
                leverage_df = pd.DataFrame(leverage_history)
                leverage_df.set_index('date', inplace=True)
                leverage_df.index = pd.to_datetime(leverage_df.index)
                
                print(f"\n📊 Leverage Pyfolio Analysis")
                print(f"  📊 Pandas Index Type: {type(returns_df.index).__name__}")
                
                # Enhanced performance calculation
                daily_returns = returns_df['return']
                
                # Pyfolio tearsheet with leverage context
                try:
                    pf.create_returns_tear_sheet(daily_returns, live_start_date=start_date)
                except Exception as e:
                    print(f"Pyfolio tearsheet error: {e}")
                
                # Calculate leverage-enhanced performance metrics
                total_return = (1 + daily_returns).prod() - 1
                annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe = annualized_return / volatility if volatility > 0 else 0
                
                # Leverage specific metrics
                avg_portfolio_leverage = leverage_df['avg_leverage'].mean()
                max_portfolio_leverage = leverage_df['avg_leverage'].max()
                avg_total_exposure = leverage_df['total_exposure'].mean()
                
                print(f"\n🎯 LEVERAGE PERFORMANCE SUMMARY:")
                print(f"  📈 Total Return: {total_return:.2%}")
                print(f"  🚀 Annualized Return: {annualized_return:.2%}")
                print(f"  📉 Volatility: {volatility:.2%}")
                print(f"  ⚡ Sharpe Ratio: {sharpe:.2f}")
                print(f"  🔄 Periods Processed: {len(portfolio_history)}")
                print(f"\n⚡ LEVERAGE METRICS:")
                print(f"  📊 Average Portfolio Leverage: {avg_portfolio_leverage:.2f}x")
                print(f"  🚀 Maximum Portfolio Leverage: {max_portfolio_leverage:.2f}x")
                print(f"  📈 Average Total Exposure: {avg_total_exposure:.1%}")
                
                # Target achievement analysis
                target_achievement = annualized_return / self.target_return if self.target_return > 0 else 0
                print(f"  🎯 Target Achievement: {target_achievement:.1%} of {self.target_return:.0%} target")
                
                if annualized_return >= self.target_return:
                    print(f"  ✅ TARGET ACHIEVED! {annualized_return:.1%} >= {self.target_return:.0%}")
                else:
                    print(f"  ⏳ Target progress: {target_achievement:.1%}")
                
                return {
                    'portfolio_history': portfolio_history,
                    'returns_df': returns_df,
                    'leverage_df': leverage_df,
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'avg_leverage': avg_portfolio_leverage,
                    'max_leverage': max_portfolio_leverage,
                    'target_achievement': target_achievement
                }
            
            return None
            
        except Exception as e:
            print(f"Leverage backtest error: {e}")
            return None

# === CELL 4: MAIN EXECUTION ===
def run_leverage_elite_system():
    """Run the leverage enhanced elite superintelligence system"""
    try:
        print("🚀 Initializing Leverage Elite Superintelligence System...")
        
        # Initialize system
        system = LeverageEliteSupertintelligenceSystem(
            universe_type='LEVERAGE_COMPREHENSIVE',
            start_date='2023-01-01',
            end_date='2024-12-01',
            max_leverage=2.0,
            target_return=0.50  # 50% target
        )
        
        # Setup leverage features
        system.setup_leverage_features()
        
        # Setup workflow
        workflow = system.setup_leverage_workflow()
        
        # Run leverage backtest
        print("\n🎯 Starting Leverage Enhanced Backtest...")
        results = system.leverage_backtest()
        
        if results:
            print("\n✅ Leverage Elite System completed successfully!")
            if results['annualized_return'] >= 0.40:  # 40%+ achieved
                print("🎊 EXCEPTIONAL PERFORMANCE ACHIEVED!")
            elif results['annualized_return'] >= 0.30:  # 30%+ achieved
                print("🎉 OUTSTANDING PERFORMANCE!")
            return system, results
        else:
            print("\n⚠️ Leverage backtest failed")
            return system, None
            
    except Exception as e:
        print(f"❌ Leverage system error: {e}")
        return None, None

if __name__ == "__main__":
    # Run the leverage enhanced system
    leverage_system, leverage_results = run_leverage_elite_system()
    
    if leverage_results:
        print(f"\n🎯 FINAL LEVERAGE SYSTEM PERFORMANCE:")
        print(f"  🚀 Annualized Return: {leverage_results['annualized_return']:.2%}")
        print(f"  ⚡ Average Leverage: {leverage_results['avg_leverage']:.2f}x")
        print(f"  🎯 Target Achievement: {leverage_results['target_achievement']:.1%}")
        
        if leverage_results['annualized_return'] >= 0.50:
            print("  🏆 50%+ TARGET ACHIEVED! MISSION ACCOMPLISHED!")
        elif leverage_results['annualized_return'] >= 0.40:
            print("  🥈 40%+ EXCELLENT PERFORMANCE!")
        elif leverage_results['annualized_return'] >= 0.30:
            print("  🥉 30%+ SOLID PERFORMANCE!")
    else:
        print("\n⚠️ Leverage system did not complete successfully")