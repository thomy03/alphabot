#!/usr/bin/env python3
"""
Elite Superintelligence Trading System - Version Finale
Système révolutionnaire avec RL Online + Polygon API + Expanded Universe
Target: 35%+ annual return via superintelligence
"""

# === CELL 1: SETUP ELITE SUPERINTELLIGENCE ===
# Installations complètes avec APIs externes
!pip install -q yfinance pandas numpy scikit-learn scipy joblib
!pip install -q langgraph langchain langchain-community transformers torch
!pip install -q huggingface_hub datasets accelerate bitsandbytes
!pip install -q requests beautifulsoup4 polygon-api-client alpha_vantage
!pip install -q ta-lib pyfolio quantlib-python

# Imports système
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
    DRIVE_PATH = '/content/drive/MyDrive/elite_superintelligence_states/'
    os.makedirs(DRIVE_PATH, exist_ok=True)
except:
    COLAB_ENV = False
    DRIVE_PATH = './elite_superintelligence_states/'
    os.makedirs(DRIVE_PATH, exist_ok=True)

# Configuration GPU/TPU
print("🧠 ELITE SUPERINTELLIGENCE TRADING SYSTEM")
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

# === CELL 2: ELITE AGENT STATES AVEC REDUCERS ===
class EliteAgentState(TypedDict):
    """État elite avec reducers pour performance"""
    # Core data
    symbol: str
    date: str
    current_price: float
    features: List[float]
    
    # ML predictions
    ml_prediction: float
    confidence_score: float
    
    # Risk metrics
    cvar_risk: float
    drawdown: float
    market_regime: str
    
    # External intelligence
    news_sentiment: float
    financial_metrics: Dict[str, float]
    external_signals: Dict[str, float]
    
    # Agent decisions
    agent_decision: str
    reasoning: str
    supervisor_guidance: str
    
    # RL & learning
    rl_feedback: Dict[str, float]
    q_value: float
    reward_history: List[float]
    
    # Meta
    adjustments: Dict[str, Any]
    rethink_count: int
    execution_time: float
    portfolio_context: Dict[str, Any]
    
    # Agent stats
    hallucination_detected: bool
    confidence_adjusted: bool
    rl_learning_active: bool

def state_reducer(old_state: EliteAgentState, updates: Dict[str, Any]) -> EliteAgentState:
    """Reducer pour updates partielles optimisées"""
    return {**old_state, **updates}

class EliteSupertintelligenceSystem:
    """Système Elite Superintelligence avec univers étendu"""
    
    def __init__(self):
        # UNIVERS ÉTENDU ELITE (120+ assets)
        self.elite_universe = self.build_elite_universe()
        
        # Configuration superintelligence
        self.start_date = "2019-01-01"
        self.end_date = "2025-07-13"
        self.initial_capital = 100000
        
        # Paramètres DL optimisés
        self.lookback_days = 50  # Plus de contexte
        self.features_count = 28  # Enrichi avec Polygon
        self.lstm_units = 128     # Plus de puissance
        self.attention_heads = 4
        self.attention_dim = 64
        self.epochs = 30
        self.batch_size = 64
        
        # Paramètres RL online learning
        self.learning_rate_rl = 0.05  # Plus agressif
        self.q_learning_enabled = True
        self.reward_decay = 0.95
        self.exploration_rate = 0.15
        
        # Multi-agents avancés
        self.max_rethink_loops = 3  # Plus de sophistication
        self.agent_timeout = 400
        self.confidence_base = 0.62
        self.dynamic_thresholds = {}
        
        # Trading parameters elite
        self.max_positions = 12    # Plus de diversification
        self.max_position_size = 0.10
        self.rebalance_frequency = 2  # Plus fréquent
        
        # Storage avec optimisations
        self.data = {}
        self.models = {}
        self.scalers = {}
        self.per_symbol_scalers = {}
        self.news_cache = {}
        self.financial_cache = {}
        self.rl_memory = {}
        self.agent_stats = {
            'decisions_made': 0,
            'hallucinations_detected': 0,
            'rl_updates': 0,
            'successful_trades': 0,
            'avg_execution_time': 0.0
        }
        
        # APIs setup
        self.setup_external_apis()
        
        # Setup multi-agents avec reducers
        self.setup_elite_llms()
        self.setup_elite_workflow()
        
        print(f"🧠 ELITE SUPERINTELLIGENCE INITIALISÉ")
        print(f"📊 Univers Elite: {len(self.elite_universe)} assets premium")
        print(f"🎯 Target: 35%+ rendement via superintelligence")
        print(f"🤖 Multi-Agents: Analyst + Supervisor + RL + Validator")
        print(f"📰 External APIs: News + Financial + Economic")
        print(f"🧠 RL Online Learning: Q-Learning adaptatif")
        print(f"🛡️ Anti-Hallucination: Guards + Validation")
    
    def build_elite_universe(self) -> List[str]:
        """Construire univers élite étendu (120+ assets)"""
        
        # Mega caps tech (croissance)
        mega_tech = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'NVDA', 'META', 'AMZN', 'TSLA']
        
        # Growth tech leaders
        growth_tech = ['CRM', 'ADBE', 'AMD', 'QCOM', 'ORCL', 'NFLX', 'UBER', 'SQ', 
                      'PYPL', 'SHOP', 'SNOW', 'PLTR', 'NET', 'DDOG', 'ZM']
        
        # Cloud & SaaS
        cloud_saas = ['NOW', 'WDAY', 'OKTA', 'ZS', 'CRWD', 'MDB', 'ESTC', 'SPLK']
        
        # Semiconductors
        semiconductors = ['TSM', 'AVGO', 'TXN', 'LRCX', 'KLAC', 'AMAT', 'MRVL', 'MCHP']
        
        # Quality large caps
        quality_large = ['V', 'MA', 'UNH', 'HD', 'JPM', 'BAC', 'WFC', 'COST', 'PG', 
                        'JNJ', 'KO', 'PEP', 'WMT', 'DIS', 'NKE', 'ADSK']
        
        # Growth consumer
        growth_consumer = ['LULU', 'SBUX', 'CMG', 'DXCM', 'ISRG', 'ILMN', 'REGN', 'GILD']
        
        # Financial innovation
        fintech = ['BLK', 'GS', 'MS', 'SPGI', 'CME', 'ICE', 'COIN', 'HOOD']
        
        # ETFs sectoriels
        sector_etfs = ['QQQ', 'XLK', 'VGT', 'SOXX', 'XLF', 'XLV', 'XLI', 'XLE', 
                      'XLB', 'XLP', 'XLU', 'XLRE', 'XLY']
        
        # Broad market & factors
        broad_market = ['SPY', 'IWM', 'VTI', 'VEA', 'VWO', 'EFA', 'EEM']
        
        # Alternative assets
        alternatives = ['GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'VNQ', 'DBA', 'USO']
        
        # Innovation & disruption
        innovation = ['ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ICLN', 'PBW', 'JETS', 'DRIV']
        
        # Combine all
        elite_universe = (mega_tech + growth_tech + cloud_saas + semiconductors + 
                         quality_large + growth_consumer + fintech + sector_etfs + 
                         broad_market + alternatives + innovation)
        
        # Remove duplicates et trier
        elite_universe = sorted(list(set(elite_universe)))
        
        print(f"🌟 Elite Universe construit: {len(elite_universe)} assets")
        return elite_universe
    
    def setup_external_apis(self):
        """Setup APIs externes avec fallbacks"""
        self.external_apis = {
            'polygon_enabled': POLYGON_AVAILABLE,
            'polygon_client': None,
            'alpha_vantage_key': None,
            'news_api_key': None
        }
        
        if POLYGON_AVAILABLE:
            try:
                # API key depuis environment ou prompt
                polygon_key = os.getenv('POLYGON_API_KEY', 'your_polygon_key_here')
                self.external_apis['polygon_client'] = RESTClient(polygon_key)
                print("✅ Polygon API configuré")
            except Exception as e:
                print(f"⚠️ Polygon API error: {e}")
        
        print("✅ External APIs initialisés")
    
    def setup_elite_llms(self):
        """Configuration LLMs élite avec guards"""
        try:
            # Primary: Mistral pour finance
            self.primary_llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.25,  # Plus conservateur
                max_new_tokens=400,
                timeout=150
            )
            
            # News: Perplexity Sonar
            self.news_llm = Perplexity(
                temperature=0.1,
                model="sonar-medium-online",
                timeout=90
            )
            
            # Supervisor: Plus stricte
            self.supervisor_llm = HuggingFaceEndpoint(
                repo_id="microsoft/DialoGPT-medium",
                temperature=0.05,
                max_new_tokens=250,
                timeout=120
            )
            
            # Validator: Anti-hallucination
            self.validator_llm = self.primary_llm  # Même modèle, différent prompt
            
            print("✅ Elite LLMs configurés avec guards")
            
        except Exception as e:
            print(f"⚠️ LLM setup error: {e}")
            # Fallback robuste
            self.primary_llm = pipeline("text-generation", model="microsoft/DialoGPT-medium", 
                                      device=0 if torch.cuda.is_available() else -1, max_length=300)
            self.news_llm = self.primary_llm
            self.supervisor_llm = self.primary_llm
            self.validator_llm = self.primary_llm
            print("✅ Fallback LLMs activés")
    
    def setup_elite_workflow(self):
        """Setup workflow elite avec reducers et guards"""
        
        def predict_node(state: EliteAgentState) -> EliteAgentState:
            """Prédiction ML + External APIs"""
            start_time = time.time()
            
            try:
                # ML prediction standard
                features = np.array(state['features']).reshape(1, self.lookback_days, self.features_count)
                
                with tf.device(DEVICE):
                    trend_pred = self.models['trend'].predict(features, verbose=0)[0][0]
                    momentum_pred = self.models['momentum'].predict(features, verbose=0)[0][0]
                
                # Prédiction combinée
                ml_score = 0.65 * trend_pred + 0.35 * momentum_pred
                
                # External APIs boost
                external_boost = 0.0
                
                # News sentiment
                if state['news_sentiment'] > 0.7:
                    external_boost += 0.12
                elif state['news_sentiment'] < 0.3:
                    external_boost -= 0.12
                
                # Financial metrics (Polygon)
                financial_metrics = state.get('financial_metrics', {})
                roe = financial_metrics.get('roe', 0)
                if roe > 0.20:  # Excellent ROE
                    external_boost += 0.08
                elif roe < 0.05:  # Poor ROE
                    external_boost -= 0.08
                
                # Final prediction
                final_prediction = np.clip(ml_score + external_boost, 0.0, 1.0)
                
                updates = {
                    'ml_prediction': float(final_prediction),
                    'confidence_score': float(max(trend_pred, momentum_pred)),
                    'execution_time': time.time() - start_time
                }
                
                return state_reducer(state, updates)
                
            except Exception as e:
                print(f"⚠️ Predict error: {e}")
                updates = {
                    'ml_prediction': 0.5,
                    'confidence_score': 0.5,
                    'execution_time': time.time() - start_time
                }
                return state_reducer(state, updates)
        
        def analyze_node(state: EliteAgentState) -> EliteAgentState:
            """Analyse agentique avec anti-hallucination"""
            start_time = time.time()
            
            try:
                # Get enhanced data
                news_insight = self.get_news_sentiment_cached(state['symbol'], state['date'])
                financial_data = self.get_financial_metrics_cached(state['symbol'])
                
                # Elite analyst prompt
                prompt = f"""
                ELITE HEDGE FUND ANALYSIS - {state['symbol']}:
                
                Technical Signals:
                - Price: ${state['current_price']:.2f}
                - ML Score: {state['ml_prediction']:.3f}
                - Confidence: {state['confidence_score']:.3f}
                - CVaR Risk: {state['cvar_risk']:.3f}
                
                Market Context:
                - Regime: {state['market_regime']}
                - Drawdown: {state['drawdown']:.1%}
                
                Intelligence Sources:
                - News Sentiment: {news_insight['sentiment_score']:.3f}
                - Recent Headlines: {news_insight['summary'][:80]}
                - ROE: {financial_data.get('roe', 0):.1%}
                - Revenue Growth: {financial_data.get('revenue_growth', 0):.1%}
                
                Portfolio Context:
                - Positions: {len(state['portfolio_context'].get('positions', {}))}
                - Cash: ${state['portfolio_context'].get('cash', 0):,.0f}
                
                Decision Required: BUY/SELL/HOLD
                Confidence: 0-100%
                Reasoning: Max 80 words, focus on key factors.
                """
                
                # Generate with timeout
                if hasattr(self.primary_llm, 'invoke'):
                    response = self.primary_llm.invoke(prompt)
                else:
                    response = self.primary_llm(prompt, max_length=200)[0]['generated_text']
                
                # Anti-hallucination guards
                response_text = str(response)
                hallucination_detected = self.detect_hallucination(response_text, state)
                
                if hallucination_detected:
                    # Fallback to conservative decision
                    decision = 'HOLD'
                    reasoning = "Conservative decision due to response validation failure"
                    confidence_adj = 0.3
                    self.agent_stats['hallucinations_detected'] += 1
                else:
                    # Parse normal response
                    response_upper = response_text.upper()
                    if 'BUY' in response_upper and 'SELL' not in response_upper:
                        decision = 'BUY'
                    elif 'SELL' in response_upper:
                        decision = 'SELL'
                    else:
                        decision = 'HOLD'
                    
                    reasoning = response_text[:250]
                    confidence_adj = 1.0
                
                # External data adjustments
                confidence_multiplier = confidence_adj
                if news_insight['sentiment_score'] > 0.75:
                    confidence_multiplier *= 1.15
                elif news_insight['sentiment_score'] < 0.25:
                    confidence_multiplier *= 0.75
                
                updates = {
                    'agent_decision': decision,
                    'reasoning': reasoning,
                    'news_sentiment': news_insight['sentiment_score'],
                    'financial_metrics': financial_data,
                    'confidence_score': state['confidence_score'] * confidence_multiplier,
                    'hallucination_detected': hallucination_detected,
                    'confidence_adjusted': confidence_adj != 1.0,
                    'execution_time': state['execution_time'] + (time.time() - start_time)
                }
                
                return state_reducer(state, updates)
                
            except Exception as e:
                print(f"⚠️ Analyze error: {e}")
                updates = {
                    'agent_decision': 'HOLD',
                    'reasoning': f"Error in analysis: {e}",
                    'hallucination_detected': True,
                    'execution_time': state['execution_time'] + (time.time() - start_time)
                }
                return state_reducer(state, updates)
        
        def supervisor_node(state: EliteAgentState) -> EliteAgentState:
            """Superviseur portfolio avec risk management"""
            start_time = time.time()
            
            try:
                # Portfolio risk assessment
                portfolio_metrics = self.calculate_portfolio_risk(state['portfolio_context'])
                
                supervisor_prompt = f"""
                PORTFOLIO SUPERVISOR - RISK REVIEW:
                
                Proposed Trade: {state['agent_decision']} {state['symbol']}
                Agent Confidence: {state['confidence_score']:.3f}
                
                Portfolio Risk Metrics:
                - Positions: {portfolio_metrics['position_count']}/12
                - Sector Concentration: {portfolio_metrics['max_sector_weight']:.1%}
                - Cash Ratio: {portfolio_metrics['cash_ratio']:.1%}
                - Current Drawdown: {state['drawdown']:.1%}
                
                Risk Flags:
                - CVaR Risk: {state['cvar_risk']:.3f}
                - Hallucination Detected: {state['hallucination_detected']}
                - Market Regime: {state['market_regime']}
                
                Supervisor Decision: APPROVE/MODIFY/REJECT
                Risk Adjustment: Specify confidence multiplier (0.5-1.5)
                """
                
                if hasattr(self.supervisor_llm, 'invoke'):
                    supervisor_response = self.supervisor_llm.invoke(supervisor_prompt)
                else:
                    supervisor_response = self.supervisor_llm(supervisor_prompt, max_length=120)[0]['generated_text']
                
                response_text = str(supervisor_response).upper()
                
                # Parse supervisor decision
                confidence_multiplier = 1.0
                if 'REJECT' in response_text:
                    final_decision = 'HOLD'
                    confidence_multiplier = 0.3
                elif 'MODIFY' in response_text:
                    final_decision = state['agent_decision']
                    confidence_multiplier = 0.7
                else:  # APPROVE
                    final_decision = state['agent_decision']
                    confidence_multiplier = 1.0
                
                # Risk-based adjustments
                if state['drawdown'] < -0.25:
                    confidence_multiplier *= 0.6
                elif portfolio_metrics['position_count'] >= 10:
                    confidence_multiplier *= 0.8
                
                updates = {
                    'agent_decision': final_decision,
                    'confidence_score': state['confidence_score'] * confidence_multiplier,
                    'supervisor_guidance': str(supervisor_response)[:200],
                    'execution_time': state['execution_time'] + (time.time() - start_time)
                }
                
                return state_reducer(state, updates)
                
            except Exception as e:
                print(f"⚠️ Supervisor error: {e}")
                updates = {
                    'supervisor_guidance': f"Supervisor error: {e}",
                    'execution_time': state['execution_time'] + (time.time() - start_time)
                }
                return state_reducer(state, updates)
        
        def rl_learn_node(state: EliteAgentState) -> EliteAgentState:
            """RL Online Learning avec Q-Learning"""
            start_time = time.time()
            
            try:
                symbol = state['symbol']
                
                # Initialize RL memory if needed
                if symbol not in self.rl_memory:
                    self.rl_memory[symbol] = {
                        'q_values': {},
                        'decisions': [],
                        'returns': [],
                        'avg_return': 0.0,
                        'accuracy': 0.5,
                        'entry_prices': {},
                        'last_update': datetime.now(),
                        'learning_active': True
                    }
                
                # Get RL state
                rl_data = self.rl_memory[symbol]
                
                # Q-Learning update si available
                current_q = rl_data.get('q_values', {}).get(state['agent_decision'], 0.0)
                
                # Estimate reward based on recent performance
                estimated_reward = self.estimate_immediate_reward(state)
                
                # Q-Learning update (simplified Bellman)
                if self.q_learning_enabled:
                    new_q = current_q + self.learning_rate_rl * (estimated_reward - current_q)
                    rl_data['q_values'][state['agent_decision']] = new_q
                    
                    # Store for later real reward update
                    rl_data['entry_prices'][state['date']] = state['current_price']
                
                # RL-based confidence adjustment
                rl_confidence_adj = 1.0
                
                if symbol in self.rl_memory:
                    historical_accuracy = rl_data['accuracy']
                    avg_return = rl_data['avg_return']
                    
                    # Boost if historically good
                    if historical_accuracy > 0.65 and avg_return > 0.03:
                        rl_confidence_adj = 1.2
                    # Penalize if historically bad
                    elif historical_accuracy < 0.45 or avg_return < -0.05:
                        rl_confidence_adj = 0.7
                
                # Final RL adjustment
                final_confidence = state['confidence_score'] * rl_confidence_adj
                final_threshold = self.confidence_base + state['adjustments'].get('threshold_adj', 0)
                
                # Decision based on RL-adjusted confidence
                if final_confidence >= final_threshold:
                    final_decision = state['agent_decision']
                else:
                    final_decision = 'HOLD'
                
                updates = {
                    'agent_decision': final_decision,
                    'confidence_score': final_confidence,
                    'q_value': new_q if self.q_learning_enabled else current_q,
                    'rl_learning_active': True,
                    'adjustments': {**state['adjustments'], 'rl_threshold': final_threshold},
                    'execution_time': state['execution_time'] + (time.time() - start_time)
                }
                
                # Update stats
                self.agent_stats['rl_updates'] += 1
                
                return state_reducer(state, updates)
                
            except Exception as e:
                print(f"⚠️ RL learn error: {e}")
                updates = {
                    'rl_learning_active': False,
                    'execution_time': state['execution_time'] + (time.time() - start_time)
                }
                return state_reducer(state, updates)
        
        def should_rethink(state: EliteAgentState) -> str:
            """Décision rethink avec conditions elite"""
            try:
                # Limite loops
                if state['rethink_count'] >= self.max_rethink_loops:
                    return END
                
                # Conditions rethink sophistiquées
                needs_rethink = False
                
                # Drawdown critique
                if state['drawdown'] < -0.30:
                    needs_rethink = True
                
                # Confiance très faible
                if state['confidence_score'] < 0.20:
                    needs_rethink = True
                
                # Hallucination détectée
                if state['hallucination_detected']:
                    needs_rethink = True
                
                # Contradiction supervisor sévère
                if 'REJECT' in state.get('supervisor_guidance', '').upper() and state['confidence_score'] > 0.8:
                    needs_rethink = True
                
                # RL performance très mauvaise
                symbol = state['symbol']
                if symbol in self.rl_memory:
                    if self.rl_memory[symbol]['avg_return'] < -0.20:
                        needs_rethink = True
                
                # Execution time excessive
                if state['execution_time'] > self.agent_timeout:
                    print(f"⚠️ Timeout reached for {state['symbol']}")
                    return END
                
                if needs_rethink:
                    updates = {'rethink_count': state['rethink_count'] + 1}
                    updated_state = state_reducer(state, updates)
                    self.save_agent_state(updated_state)
                    return "predict"
                else:
                    return END
                    
            except Exception as e:
                print(f"⚠️ Rethink error: {e}")
                return END
        
        # Construire workflow avec reducers
        workflow = StateGraph(EliteAgentState)
        
        # Ajouter nœuds avec reducers
        workflow.add_node("predict", predict_node)
        workflow.add_node("analyze", analyze_node) 
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("rl_learn", rl_learn_node)
        
        # Edges séquentiels
        workflow.add_edge("predict", "analyze")
        workflow.add_edge("analyze", "supervisor")
        workflow.add_edge("supervisor", "rl_learn")
        workflow.add_conditional_edges(
            "rl_learn",
            should_rethink,
            {
                "predict": "predict",
                END: END
            }
        )
        
        # Entry point
        workflow.set_entry_point("predict")
        
        # Compiler
        self.agent_workflow = workflow.compile()
        
        print("✅ Elite Workflow avec Reducers configuré")
    
    def detect_hallucination(self, response: str, state: EliteAgentState) -> bool:
        """Détecter hallucinations LLM"""
        try:
            response_lower = response.lower()
            
            # Checks basiques
            if len(response) < 30:
                return True
            
            if any(word in response_lower for word in ['error', 'unable', 'cannot', 'sorry']):
                return True
            
            # Checks financiers
            if state['current_price'] > 0:
                # Chercher mentions prix aberrants
                import re
                price_mentions = re.findall(r'\$(\d+(?:\.\d+)?)', response)
                for price_str in price_mentions:
                    price = float(price_str)
                    if abs(price - state['current_price']) / state['current_price'] > 0.5:  # 50% différence
                        return True
            
            # Check cohérence decision
            response_upper = response.upper()
            buy_count = response_upper.count('BUY')
            sell_count = response_upper.count('SELL')
            if buy_count > 0 and sell_count > 0:  # Contradiction
                return True
            
            return False
            
        except:
            return True  # Safe fallback
    
    def calculate_portfolio_risk(self, portfolio_context: Dict) -> Dict[str, float]:
        """Calculer métriques risque portfolio"""
        try:
            positions = portfolio_context.get('positions', {})
            total_value = portfolio_context.get('total_value', 100000)
            cash = portfolio_context.get('cash', 0)
            
            # Basic metrics
            position_count = len(positions)
            cash_ratio = cash / max(total_value, 1)
            
            # Sector concentration (simplifié)
            max_sector_weight = 0.3  # Placeholder
            
            return {
                'position_count': position_count,
                'cash_ratio': cash_ratio,
                'max_sector_weight': max_sector_weight,
                'total_value': total_value
            }
            
        except:
            return {
                'position_count': 0,
                'cash_ratio': 1.0,
                'max_sector_weight': 0.0,
                'total_value': 100000
            }
    
    def estimate_immediate_reward(self, state: EliteAgentState) -> float:
        """Estimer reward immédiat pour RL"""
        try:
            # Base reward sur confidence et market conditions
            base_reward = 0.0
            
            # Reward basé sur confidence
            if state['confidence_score'] > 0.8:
                base_reward += 0.1
            elif state['confidence_score'] < 0.4:
                base_reward -= 0.1
            
            # Penalty pour risk
            if state['cvar_risk'] > 0.04:
                base_reward -= 0.05
            
            # Bonus pour good market regime
            if state['market_regime'] in ['BULL', 'STRONG_BULL']:
                base_reward += 0.02
            elif state['market_regime'] in ['BEAR', 'STRONG_BEAR']:
                base_reward -= 0.02
            
            # News sentiment impact
            if state['news_sentiment'] > 0.7:
                base_reward += 0.03
            elif state['news_sentiment'] < 0.3:
                base_reward -= 0.03
            
            return base_reward
            
        except:
            return 0.0
    
    def get_news_sentiment_cached(self, symbol: str, date: str) -> Dict[str, Any]:
        """News sentiment avec cache amélioré"""
        try:
            cache_key = f"{symbol}_{date}"
            if cache_key in self.news_cache:
                cached_data = self.news_cache[cache_key]
                # Check if cache is fresh (< 1 day)
                cache_time = cached_data.get('timestamp', datetime.min)
                if (datetime.now() - cache_time).days < 1:
                    return cached_data['data']
            
            # Fetch new data
            news_query = f"Latest financial news and analyst sentiment for {symbol} stock on {date}. Include earnings, guidance, and price targets."
            
            try:
                if hasattr(self.news_llm, 'invoke'):
                    news_response = self.news_llm.invoke(news_query)
                else:
                    news_response = f"Neutral market sentiment for {symbol}, no major catalysts."
                
                # Enhanced sentiment scoring
                response_text = str(news_response).lower()
                
                # Positive indicators
                positive_words = ['bullish', 'buy', 'strong buy', 'outperform', 'upgrade', 
                                'beat earnings', 'raised guidance', 'strong growth', 'positive']
                
                # Negative indicators  
                negative_words = ['bearish', 'sell', 'underperform', 'downgrade', 
                                'missed earnings', 'lowered guidance', 'weak', 'negative']
                
                positive_score = sum(response_text.count(word) for word in positive_words)
                negative_score = sum(response_text.count(word) for word in negative_words)
                
                # Calculate sentiment (0-1 scale)
                if positive_score + negative_score == 0:
                    sentiment_score = 0.5  # Neutral
                else:
                    sentiment_score = positive_score / (positive_score + negative_score)
                
                # Adjust for intensity
                if positive_score > 3:
                    sentiment_score = min(sentiment_score + 0.1, 1.0)
                elif negative_score > 3:
                    sentiment_score = max(sentiment_score - 0.1, 0.0)
                
                result = {
                    'sentiment_score': sentiment_score,
                    'summary': str(news_response)[:250],
                    'positive_signals': positive_score,
                    'negative_signals': negative_score,
                    'source': 'perplexity'
                }
                
            except Exception as e:
                result = {
                    'sentiment_score': 0.5,
                    'summary': f"News fetch error: {e}",
                    'positive_signals': 0,
                    'negative_signals': 0,
                    'source': 'fallback'
                }
            
            # Cache result
            self.news_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            return {
                'sentiment_score': 0.5,
                'summary': f"News system error: {e}",
                'positive_signals': 0,
                'negative_signals': 0,
                'source': 'error'
            }
    
    def get_financial_metrics_cached(self, symbol: str) -> Dict[str, float]:
        """Metrics financières avec Polygon API"""
        try:
            # Check cache first
            if symbol in self.financial_cache:
                cached_data = self.financial_cache[symbol]
                cache_time = cached_data.get('timestamp', datetime.min)
                if (datetime.now() - cache_time).days < 7:  # Weekly refresh
                    return cached_data['data']
            
            # Try Polygon API
            financial_data = {}
            
            if self.external_apis['polygon_enabled'] and self.external_apis['polygon_client']:
                try:
                    client = self.external_apis['polygon_client']
                    
                    # Get ticker details
                    ticker_details = client.get_ticker_details(symbol)
                    if hasattr(ticker_details, 'results'):
                        details = ticker_details.results
                        
                        # Extract key metrics
                        financial_data = {
                            'market_cap': getattr(details, 'market_cap', 0) / 1e9,  # Billions
                            'pe_ratio': getattr(details, 'price_earnings_ratio', 0),
                            'revenue_growth': getattr(details, 'revenue_growth_rate', 0),
                            'roe': getattr(details, 'return_on_equity', 0) / 100 if hasattr(details, 'return_on_equity') else 0,
                            'debt_to_equity': getattr(details, 'debt_to_equity_ratio', 0),
                            'current_ratio': getattr(details, 'current_ratio', 1.0),
                            'source': 'polygon'
                        }
                    
                except Exception as e:
                    print(f"⚠️ Polygon API error for {symbol}: {e}")
                    financial_data = self.fallback_financial_metrics(symbol)
            else:
                financial_data = self.fallback_financial_metrics(symbol)
            
            # Cache result
            self.financial_cache[symbol] = {
                'data': financial_data,
                'timestamp': datetime.now()
            }
            
            return financial_data
            
        except Exception as e:
            return self.fallback_financial_metrics(symbol)
    
    def fallback_financial_metrics(self, symbol: str) -> Dict[str, float]:
        """Fallback financial metrics"""
        # Simplified metrics based on sector/symbol
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN']
        
        if symbol in tech_symbols:
            return {
                'market_cap': 500.0,  # Billions
                'pe_ratio': 25.0,
                'revenue_growth': 0.15,
                'roe': 0.20,
                'debt_to_equity': 0.3,
                'current_ratio': 1.5,
                'source': 'fallback_tech'
            }
        else:
            return {
                'market_cap': 100.0,
                'pe_ratio': 18.0,
                'revenue_growth': 0.08,
                'roe': 0.12,
                'debt_to_equity': 0.6,
                'current_ratio': 1.2,
                'source': 'fallback_general'
            }
    
    def update_rl_memory_online(self, symbol: str, decision: str, actual_return: float, entry_price: float, exit_price: float):
        """Update RL memory avec vrais returns"""
        try:
            if symbol not in self.rl_memory:
                return
            
            rl_data = self.rl_memory[symbol]
            
            # Add new observation
            rl_data['decisions'].append(decision)
            rl_data['returns'].append(actual_return)
            
            # Limit history
            max_history = 100
            if len(rl_data['returns']) > max_history:
                rl_data['decisions'] = rl_data['decisions'][-max_history:]
                rl_data['returns'] = rl_data['returns'][-max_history:]
            
            # Update metrics
            returns = rl_data['returns']
            decisions = rl_data['decisions']
            
            rl_data['avg_return'] = np.mean(returns)
            
            # Calculate accuracy
            correct_predictions = 0
            total_predictions = len(decisions)
            
            for i, pred_decision in enumerate(decisions):
                actual_return_i = returns[i]
                
                if pred_decision == 'BUY' and actual_return_i > 0:
                    correct_predictions += 1
                elif pred_decision == 'SELL' and actual_return_i < 0:
                    correct_predictions += 1
                elif pred_decision == 'HOLD':
                    correct_predictions += 0.5  # Neutral
            
            rl_data['accuracy'] = correct_predictions / max(total_predictions, 1)
            
            # Q-Learning update with real reward
            if self.q_learning_enabled and decision in rl_data['q_values']:
                old_q = rl_data['q_values'][decision]
                
                # Real reward based on actual return
                real_reward = actual_return
                if decision == 'BUY' and actual_return > 0.02:  # Good buy
                    real_reward *= 1.5
                elif decision == 'SELL' and actual_return < -0.02:  # Good sell
                    real_reward = abs(real_reward) * 1.5
                
                # Q-update with decay
                new_q = old_q + self.learning_rate_rl * (real_reward + self.reward_decay * old_q - old_q)
                rl_data['q_values'][decision] = new_q
            
            rl_data['last_update'] = datetime.now()
            
            # Update global stats
            if actual_return > 0.01:  # Successful trade
                self.agent_stats['successful_trades'] += 1
            
            print(f"🧠 RL Updated {symbol}: Return={actual_return:.2%}, Accuracy={rl_data['accuracy']:.1%}")
            
        except Exception as e:
            print(f"⚠️ RL update error: {e}")
    
    def save_agent_state(self, state: EliteAgentState):
        """Sauvegarder état avec optimisations"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{DRIVE_PATH}/elite_state_{state['symbol']}_{timestamp}.json"
            
            # Serialize only essential data
            essential_state = {
                'symbol': state['symbol'],
                'date': state['date'],
                'decision': state['agent_decision'],
                'confidence': state['confidence_score'],
                'rethink_count': state['rethink_count'],
                'execution_time': state['execution_time'],
                'hallucination_detected': state['hallucination_detected'],
                'rl_active': state['rl_learning_active']
            }
            
            with open(filename, 'w') as f:
                json.dump(essential_state, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Save state error: {e}")

    def download_data_intelligent_caching(self):
        """Téléchargement avec cache ultra-intelligent"""
        print("\\n📊 TÉLÉCHARGEMENT UNIVERS ELITE (CACHE INTELLIGENT)...")
        
        # Cache management
        cache_file = f"{DRIVE_PATH}/elite_data_cache_v2.pkl"
        
        # Try loading cache
        if os.path.exists(cache_file):
            try:
                print("  🔄 Vérification cache existant...")
                cached_data = joblib.load(cache_file)
                
                # Validation cache
                if isinstance(cached_data, dict) and len(cached_data) > 50:
                    # Check freshness
                    sample_symbol = list(cached_data.keys())[0]
                    if sample_symbol in cached_data:
                        last_date = cached_data[sample_symbol].index[-1]
                        days_old = (datetime.now() - last_date).days
                        
                        if days_old <= 3:  # Fresh cache
                            self.data = cached_data
                            print(f"  ✅ Cache valide chargé: {len(self.data)} assets ({days_old} jours)")
                            return self
                        else:
                            print(f"  🔄 Cache ancien ({days_old} jours), mise à jour partielle...")
                
            except Exception as e:
                print(f"  ⚠️ Erreur cache: {e}")
        
        # Download with intelligent batching
        print(f"  📈 Téléchargement {len(self.elite_universe)} assets elite...")
        
        successful_downloads = 0
        batch_size = 10
        
        for i in range(0, len(self.elite_universe), batch_size):
            batch = self.elite_universe[i:i+batch_size]
            
            for j, symbol in enumerate(batch):
                global_idx = i + j + 1
                try:
                    print(f"    🧠 ({global_idx:3d}/{len(self.elite_universe)}) {symbol:8s}...", end=" ")
                    
                    # Skip if already cached and recent
                    if symbol in self.data:
                        print("cached")
                        continue
                    
                    ticker_data = yf.download(
                        symbol,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        auto_adjust=True,
                        threads=True
                    )
                    
                    if len(ticker_data) > 800:  # Minimum data requirement
                        if isinstance(ticker_data.columns, pd.MultiIndex):
                            ticker_data.columns = ticker_data.columns.droplevel(1)
                        
                        # Store with optimized dtypes
                        self.data[symbol] = ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float32')
                        successful_downloads += 1
                        print(f"✅ {len(ticker_data)} jours")
                    else:
                        print(f"❌ Données insuffisantes ({len(ticker_data)} jours)")
                        
                except Exception as e:
                    print(f"❌ Erreur: {str(e)[:30]}")
            
            # Rate limiting et memory management
            if (i // batch_size + 1) % 3 == 0:
                time.sleep(2)  # Pause between batches
                gc.collect()
                print(f"    💾 Batch {i//batch_size + 1} terminé, nettoyage mémoire...")
        
        # Save enhanced cache
        try:
            print(f"  💾 Sauvegarde cache elite...")
            # Compress cache
            joblib.dump(self.data, cache_file, compress=3)
            print(f"  ✅ Cache sauvegardé: {len(self.data)} assets")
        except Exception as e:
            print(f"  ⚠️ Erreur sauvegarde: {e}")
        
        print(f"\\n✅ Univers Elite chargé: {successful_downloads}/{len(self.elite_universe)} assets")
        
        # Validate minimum requirements
        if len(self.data) < 50:
            print("⚠️ ATTENTION: Univers trop petit pour entraînement optimal")
            print("   Recommandation: Vérifiez connexion internet et reessayez")
        
        return self

    def build_enhanced_elite_models(self):
        """Construire modèles avec architecture elite"""
        print("\\n🧠 CONSTRUCTION MODÈLES ELITE SUPERINTELLIGENCE...")
        
        # Modèle 1: Elite Trend avec architecture sophistiquée
        print("  🧠 Elite Trend Model (LSTM + Multi-Head Attention)...")
        
        inputs = Input(shape=(self.lookback_days, self.features_count))
        
        # Premier bloc LSTM
        x = LSTM(self.lstm_units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)(inputs)
        x = LayerNormalization()(x)
        
        # Multi-Head Attention
        attention_output = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.attention_dim,
            dropout=0.25
        )(x, x)
        
        # Residual connection
        x = tf.keras.layers.Add()([x, attention_output])
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Deuxième bloc LSTM
        x = LSTM(self.lstm_units//2, return_sequences=False, dropout=0.25)(x)
        
        # Dense layers sophistiquées
        x = Dense(64, activation='swish')(x)  # Swish activation
        x = Dropout(0.4)(x)
        x = Dense(32, activation='swish')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='relu')(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        elite_trend_model = Model(inputs=inputs, outputs=outputs)
        elite_trend_model.compile(
            optimizer=Adam(learning_rate=0.0008, beta_1=0.9, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['trend'] = elite_trend_model
        self.scalers['trend'] = RobustScaler()  # Plus robuste que MinMaxScaler
        
        # Modèle 2: Elite Momentum avec GRU optimisé
        print("  🧠 Elite Momentum Model (GRU + Dense)...")
        
        momentum_model = Sequential([
            GRU(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                input_shape=(self.lookback_days, self.features_count)),
            LayerNormalization(),
            GRU(48, return_sequences=True, dropout=0.2),
            GlobalAveragePooling1D(),  # Better than last timestep
            Dense(48, activation='swish'),
            Dropout(0.35),
            Dense(24, activation='swish'),
            Dropout(0.25),
            Dense(12, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        momentum_model.compile(
            optimizer=Adam(learning_rate=0.0008),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['momentum'] = momentum_model
        self.scalers['momentum'] = RobustScaler()
        
        print("  ✅ Modèles Elite construits avec architectures sophistiquées")
        return self

    def prepare_elite_features(self, symbol, date):
        """Préparer 28 features élites avec external data"""
        try:
            if symbol not in self.data:
                return None
            
            data = self.data[symbol]
            prices = data['Close']
            volumes = data['Volume']
            highs = data['High']
            lows = data['Low']
            opens = data['Open']
            
            historical_prices = prices[prices.index <= date]
            historical_volumes = volumes[volumes.index <= date]
            historical_highs = highs[highs.index <= date]
            historical_lows = lows[lows.index <= date]
            historical_opens = opens[opens.index <= date]
            
            if len(historical_prices) < self.lookback_days + 50:
                return None
            
            features = []
            
            # Features pour chaque jour (28 features)
            for i in range(self.lookback_days):
                idx = -(self.lookback_days - i)
                
                # Basic features
                returns = historical_prices.pct_change().dropna()
                daily_return = returns.iloc[idx] if len(returns) >= abs(idx) else 0
                
                # Enhanced moving averages
                ma_3 = historical_prices.iloc[max(idx-3, -len(historical_prices)):idx].mean()
                ma_7 = historical_prices.iloc[max(idx-7, -len(historical_prices)):idx].mean()
                ma_15 = historical_prices.iloc[max(idx-15, -len(historical_prices)):idx].mean()
                ma_30 = historical_prices.iloc[max(idx-30, -len(historical_prices)):idx].mean()
                ma_60 = historical_prices.iloc[max(idx-60, -len(historical_prices)):idx].mean()
                
                current_price = historical_prices.iloc[idx]
                current_high = historical_highs.iloc[idx]
                current_low = historical_lows.iloc[idx]
                
                # Price position features
                price_vs_ma3 = (current_price / ma_3) - 1 if ma_3 > 0 else 0
                price_vs_ma7 = (current_price / ma_7) - 1 if ma_7 > 0 else 0
                price_vs_ma15 = (current_price / ma_15) - 1 if ma_15 > 0 else 0
                price_vs_ma30 = (current_price / ma_30) - 1 if ma_30 > 0 else 0
                price_vs_ma60 = (current_price / ma_60) - 1 if ma_60 > 0 else 0
                
                # Advanced volatility
                volatility_5 = returns.iloc[max(idx-5, -len(returns)):idx].std() if len(returns) >= abs(idx) else 0
                volatility_20 = returns.iloc[max(idx-20, -len(returns)):idx].std() if len(returns) >= abs(idx) else 0
                
                # Volume features
                vol_ma_3 = historical_volumes.iloc[max(idx-3, -len(historical_volumes)):idx].mean()
                vol_ma_10 = historical_volumes.iloc[max(idx-10, -len(historical_volumes)):idx].mean()
                vol_ratio_3 = historical_volumes.iloc[idx] / vol_ma_3 if vol_ma_3 > 0 else 1
                vol_ratio_10 = historical_volumes.iloc[idx] / vol_ma_10 if vol_ma_10 > 0 else 1
                
                # Multi-timeframe momentum
                momentum_2 = (current_price / historical_prices.iloc[max(idx-2, -len(historical_prices))]) - 1 if len(historical_prices) >= abs(idx-2) else 0
                momentum_5 = (current_price / historical_prices.iloc[max(idx-5, -len(historical_prices))]) - 1 if len(historical_prices) >= abs(idx-5) else 0
                momentum_15 = (current_price / historical_prices.iloc[max(idx-15, -len(historical_prices))]) - 1 if len(historical_prices) >= abs(idx-15) else 0
                momentum_30 = (current_price / historical_prices.iloc[max(idx-30, -len(historical_prices))]) - 1 if len(historical_prices) >= abs(idx-30) else 0
                
                # Advanced technical indicators
                # RSI multi-period
                recent_returns_10 = returns.iloc[max(idx-10, -len(returns)):idx]
                rsi_10 = self.calculate_rsi(recent_returns_10)
                
                recent_returns_20 = returns.iloc[max(idx-20, -len(returns)):idx]
                rsi_20 = self.calculate_rsi(recent_returns_20)
                
                # MACD
                ema_8 = historical_prices.iloc[max(idx-8, -len(historical_prices)):idx].ewm(span=8).mean().iloc[-1] if len(historical_prices) >= abs(idx-8) else current_price
                ema_21 = historical_prices.iloc[max(idx-21, -len(historical_prices)):idx].ewm(span=21).mean().iloc[-1] if len(historical_prices) >= abs(idx-21) else current_price
                macd_line = (ema_8 - ema_21) / current_price if current_price > 0 else 0
                
                # Bollinger Bands position
                bb_position = self.calculate_bb_position(historical_prices.iloc[max(idx-20, -len(historical_prices)):idx], current_price)
                
                # Price patterns
                high_low_range = (current_high - current_low) / current_price if current_price > 0 else 0
                open_close_range = (current_price - historical_opens.iloc[idx]) / current_price if current_price > 0 else 0
                
                # Momentum acceleration
                momentum_accel = momentum_2 - momentum_5 if abs(momentum_5) > 0.001 else 0
                
                # 28 features élites
                day_features = [
                    daily_return,                           # 1
                    price_vs_ma3,                         # 2
                    price_vs_ma7,                         # 3
                    price_vs_ma15,                        # 4
                    price_vs_ma30,                        # 5
                    price_vs_ma60,                        # 6
                    volatility_5,                         # 7
                    volatility_20,                        # 8
                    min(vol_ratio_3, 4),                  # 9
                    min(vol_ratio_10, 4),                 # 10
                    momentum_2,                           # 11
                    momentum_5,                           # 12
                    momentum_15,                          # 13
                    momentum_30,                          # 14
                    rsi_10,                               # 15
                    rsi_20,                               # 16
                    macd_line,                            # 17
                    bb_position,                          # 18
                    high_low_range,                       # 19
                    open_close_range,                     # 20
                    momentum_accel,                       # 21
                    abs(daily_return),                    # 22
                    max(momentum_5, 0),                   # 23
                    max(momentum_15, 0),                  # 24
                    1 if daily_return > 0 else 0,        # 25
                    1 if momentum_5 > 0 else 0,           # 26
                    i / self.lookback_days,               # 27
                    np.tanh(daily_return * 12)            # 28
                ]
                
                features.append(day_features)
            
            return np.array(features, dtype='float32')
            
        except Exception as e:
            return None
    
    def calculate_rsi(self, returns_series):
        """Calculer RSI robuste"""
        try:
            if len(returns_series) == 0:
                return 0.5
            
            gains = returns_series[returns_series > 0].sum()
            losses = abs(returns_series[returns_series < 0].sum())
            
            if gains + losses == 0:
                return 0.5
            
            rs = gains / losses if losses > 0 else 10
            rsi = 1 - (1 / (1 + rs))
            
            return np.clip(rsi, 0, 1)
        except:
            return 0.5
    
    def calculate_bb_position(self, price_series, current_price):
        """Calculer position Bollinger Bands"""
        try:
            if len(price_series) < 10:
                return 0.5
            
            bb_mean = price_series.mean()
            bb_std = price_series.std()
            
            if bb_std == 0:
                return 0.5
            
            # Position between bands (-1 to +1)
            bb_position = (current_price - bb_mean) / (2 * bb_std)
            return np.clip(bb_position + 0.5, 0, 1)  # Normalize to 0-1
        except:
            return 0.5

    def train_elite_models(self):
        """Entraîner modèles avec techniques elite"""
        print("\\n🎓 ENTRAÎNEMENT ELITE SUPERINTELLIGENCE...")
        
        for model_name, model in self.models.items():
            print(f"  🧠 Entraînement Elite {model_name}...")
            
            X_train, y_train = self.prepare_elite_training_data(model_name)
            
            if X_train is not None and len(X_train) > 2000:  # Plus exigeant
                # Advanced class balancing
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
                
                # Elite callbacks
                callbacks = [
                    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss', verbose=1),
                    ReduceLROnPlateau(patience=8, factor=0.6, min_lr=0.0001, monitor='val_loss', verbose=1)
                ]
                
                print(f"    📊 Training samples: {len(X_train)}, Positive rate: {y_train.mean():.2%}")
                
                # Elite training
                with tf.device(DEVICE):
                    history = model.fit(
                        X_train, y_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_split=0.25,  # Plus de validation
                        verbose=1,
                        callbacks=callbacks,
                        class_weight=class_weight_dict
                    )
                
                # Advanced threshold optimization
                X_val = X_train[int(len(X_train) * 0.75):]
                y_val = y_train[int(len(y_train) * 0.75):]
                
                y_pred = model.predict(X_val, verbose=0)
                precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
                
                # Optimize for F1 with precision bias
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                precision_weighted_f1 = f1_scores * (precision + 0.1)  # Bias towards precision
                
                optimal_idx = np.argmax(precision_weighted_f1)
                optimal_threshold = thresholds[optimal_idx]
                
                self.dynamic_thresholds[model_name] = {
                    'threshold': optimal_threshold,
                    'f1_score': f1_scores[optimal_idx],
                    'precision': precision[optimal_idx],
                    'recall': recall[optimal_idx],
                    'accuracy': max(history.history['val_accuracy']),
                    'performance': 0.0
                }
                
                print(f"    ✅ {model_name} Elite - Acc: {max(history.history['val_accuracy']):.3f}")
                print(f"    🎯 Optimal: F1={f1_scores[optimal_idx]:.3f}, Precision={precision[optimal_idx]:.3f}, Threshold={optimal_threshold:.3f}")
                
                # Memory cleanup
                del X_train, y_train, X_val, y_val, y_pred
                gc.collect()
                
            else:
                print(f"    ❌ {model_name} données insuffisantes ({len(X_train) if X_train is not None else 0} samples)")
        
        # Update global confidence
        if self.dynamic_thresholds:
            # Weighted average by F1 score
            weights = np.array([v['f1_score'] for v in self.dynamic_thresholds.values()])
            thresholds = np.array([v['threshold'] for v in self.dynamic_thresholds.values()])
            
            self.confidence_base = np.average(thresholds, weights=weights)
            print(f"  🎯 Elite confidence threshold: {self.confidence_base:.3f}")
        
        return self

    def prepare_elite_training_data(self, model_type):
        """Préparer données d'entraînement elite avec full universe"""
        try:
            all_features = []
            all_targets = []
            
            # Use larger subset for better diversity
            training_symbols = list(self.data.keys())[:min(40, len(self.data))]  # Plus de symboles
            print(f"    📊 Training sur {len(training_symbols)} symboles")
            
            for symbol in training_symbols:
                if symbol not in self.data:
                    continue
                    
                prices = self.data[symbol]['Close']
                
                # Per-symbol scaling
                if symbol not in self.per_symbol_scalers:
                    self.per_symbol_scalers[symbol] = RobustScaler()
                
                # Dense sampling for more data
                step = 3
                sample_count = 0
                target_samples_per_symbol = 300  # Limite par symbole
                
                for i in range(self.lookback_days + 50, len(prices) - 15, step):
                    if sample_count >= target_samples_per_symbol:
                        break
                        
                    date = prices.index[i]
                    
                    features = self.prepare_elite_features(symbol, date)
                    
                    if features is not None:
                        # Scaling per-symbol
                        features_scaled = self.per_symbol_scalers[symbol].fit_transform(
                            features.reshape(-1, self.features_count)
                        ).reshape(features.shape)
                        
                        # Elite targets with multiple horizons
                        if model_type == 'trend':
                            # Long-term trend
                            future_return_10 = (prices.iloc[i+10] / prices.iloc[i]) - 1
                            future_return_15 = (prices.iloc[i+15] / prices.iloc[i]) - 1
                            # Combine horizons
                            combined_return = 0.6 * future_return_10 + 0.4 * future_return_15
                            target = 1 if combined_return > 0.015 else 0  # 1.5% threshold
                        else:  # momentum
                            # Short-term momentum
                            future_return_3 = (prices.iloc[i+3] / prices.iloc[i]) - 1
                            future_return_7 = (prices.iloc[i+7] / prices.iloc[i]) - 1
                            combined_return = 0.7 * future_return_3 + 0.3 * future_return_7
                            target = 1 if combined_return > 0.008 else 0  # 0.8% threshold
                        
                        all_features.append(features_scaled)
                        all_targets.append(target)
                        sample_count += 1
                        
                        # Global limit for memory
                        if len(all_features) >= 15000:
                            break
                
                if len(all_features) >= 15000:
                    break
            
            if len(all_features) > 2000:
                features_array = np.array(all_features)
                targets_array = np.array(all_targets)
                
                print(f"    ✅ Elite dataset: {len(features_array)} samples, {targets_array.mean():.1%} positive")
                return features_array, targets_array
            else:
                print(f"    ❌ Insufficient data: {len(all_features)} samples")
                return None, None
                
        except Exception as e:
            print(f"    ❌ Training data error: {e}")
            return None, None

    def detect_market_regime_advanced(self, date):
        """Détecter régime marché avec 5 niveaux"""
        try:
            if 'SPY' not in self.data:
                return 'NEUTRAL'
            
            spy_prices = self.data['SPY']['Close']
            spy_volumes = self.data['SPY']['Volume']
            
            historical_prices = spy_prices[spy_prices.index <= date].tail(120)
            historical_volumes = spy_volumes[spy_volumes.index <= date].tail(120)
            
            if len(historical_prices) < 120:
                return 'NEUTRAL'
            
            current = historical_prices.iloc[-1]
            ma_10 = historical_prices.tail(10).mean()
            ma_20 = historical_prices.tail(20).mean()
            ma_50 = historical_prices.tail(50).mean()
            ma_100 = historical_prices.tail(100).mean()
            
            # Volatility analysis
            returns = historical_prices.pct_change().dropna()
            vol_20 = returns.tail(20).std()
            vol_100 = returns.tail(100).std()
            
            # Volume analysis
            vol_ratio = historical_volumes.tail(10).mean() / historical_volumes.tail(50).mean()
            
            # Trend strength
            trend_strength = (current - ma_100) / ma_100 if ma_100 > 0 else 0
            
            # 5-level regime classification
            if current > ma_10 > ma_20 > ma_50 > ma_100 and vol_20 < 0.015 and trend_strength > 0.1:
                return 'STRONG_BULL'
            elif current > ma_10 > ma_20 > ma_50 and trend_strength > 0.02:
                return 'BULL'
            elif current < ma_10 < ma_20 < ma_50 < ma_100 and vol_20 > 0.03 and trend_strength < -0.1:
                return 'STRONG_BEAR'
            elif current < ma_10 < ma_20 < ma_50 and trend_strength < -0.02:
                return 'BEAR'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            return 'NEUTRAL'

    def calculate_cvar_risk_advanced(self, symbol, lookback_days=40):
        """CVaR avancé avec multiple quantiles"""
        try:
            if symbol not in self.data:
                return 0
            
            returns = self.data[symbol]['Close'].pct_change().dropna().tail(lookback_days)
            if len(returns) < 20:
                return 0
            
            # Multiple CVaR levels
            cvar_5 = self.calculate_cvar(returns, 0.05)
            cvar_1 = self.calculate_cvar(returns, 0.01)
            
            # Weighted CVaR
            weighted_cvar = 0.7 * cvar_5 + 0.3 * cvar_1
            
            return abs(weighted_cvar) if not np.isnan(weighted_cvar) else 0
            
        except:
            return 0
    
    def calculate_cvar(self, returns, alpha):
        """Calculer CVaR pour un quantile donné"""
        try:
            var_alpha = np.percentile(returns, alpha * 100)
            cvar = returns[returns <= var_alpha].mean()
            return cvar
        except:
            return 0

    def elite_portfolio_rebalance(self, portfolio, date, current_drawdown):
        """Rebalancement elite avec multi-agents sophistiqués"""
        print(f"    🧠 Elite Rebalancement - {date.strftime('%Y-%m-%d')}")
        
        start_time = time.time()
        signals = {}
        agent_performance = {
            'processed': 0,
            'successful': 0,
            'hallucinations': 0,
            'rethinks': 0,
            'avg_confidence': 0,
            'avg_execution_time': 0
        }
        
        # Portfolio context enrichi
        portfolio_context = {
            'positions': portfolio['positions'],
            'cash': portfolio['cash'],
            'total_value': portfolio['value'],
            'peak_value': portfolio['peak_value'],
            'current_drawdown': current_drawdown,
            'date': date
        }
        
        # Traitement parallèle des symboles (simulation)
        processing_symbols = list(self.data.keys())[:30]  # Plus de symboles
        
        for symbol in processing_symbols:
            try:
                agent_performance['processed'] += 1
                
                features = self.prepare_elite_features(symbol, date)
                
                if features is not None:
                    current_price = self.data[symbol]['Close'][self.data[symbol]['Close'].index <= date].iloc[-1]
                    
                    # État elite agent
                    elite_state = EliteAgentState(
                        symbol=symbol,
                        date=date.strftime('%Y-%m-%d'),
                        current_price=float(current_price),
                        features=features.flatten().tolist(),
                        ml_prediction=0.0,
                        confidence_score=0.5,
                        cvar_risk=self.calculate_cvar_risk_advanced(symbol),
                        drawdown=current_drawdown,
                        market_regime=self.detect_market_regime_advanced(date),
                        news_sentiment=0.5,
                        financial_metrics={},
                        external_signals={},
                        agent_decision='HOLD',
                        reasoning='',
                        supervisor_guidance='',
                        rl_feedback=self.rl_memory.get(symbol, {}),
                        q_value=0.0,
                        reward_history=[],
                        adjustments={},
                        rethink_count=0,
                        execution_time=0.0,
                        portfolio_context=portfolio_context,
                        hallucination_detected=False,
                        confidence_adjusted=False,
                        rl_learning_active=True
                    )
                    
                    # Exécuter workflow elite
                    try:
                        agent_start = time.time()
                        result = self.agent_workflow.invoke(elite_state)
                        agent_execution_time = time.time() - agent_start
                        
                        if agent_execution_time > self.agent_timeout:
                            print(f"    ⚠️ Agent timeout {symbol} ({agent_execution_time:.1f}s)")
                            continue
                        
                        # Analyser résultats elite
                        if result['agent_decision'] == 'BUY' and result['confidence_score'] >= self.confidence_base:
                            signals[symbol] = result['confidence_score']
                            agent_performance['successful'] += 1
                            
                            if result['rethink_count'] > 0:
                                agent_performance['rethinks'] += result['rethink_count']
                            
                            if result['hallucination_detected']:
                                agent_performance['hallucinations'] += 1
                            
                            agent_performance['avg_confidence'] += result['confidence_score']
                            agent_performance['avg_execution_time'] += agent_execution_time
                            
                            print(f"    ✅ {symbol}: {result['confidence_score']:.3f} " +
                                  f"({result['rethink_count']}R, {agent_execution_time:.1f}s)")
                        
                        # Update RL with simulated immediate feedback
                        if symbol in self.rl_memory:
                            immediate_reward = self.estimate_immediate_reward(result)
                            self.rl_memory[symbol]['reward_history'] = \
                                self.rl_memory[symbol].get('reward_history', [])[-49:] + [immediate_reward]
                        
                    except Exception as e:
                        print(f"    ⚠️ Agent error {symbol}: {str(e)[:50]}")
                        continue
                        
            except Exception as e:
                print(f"    ❌ Processing error {symbol}: {str(e)[:30]}")
                continue
        
        # Statistiques agents
        if agent_performance['successful'] > 0:
            agent_performance['avg_confidence'] /= agent_performance['successful']
            agent_performance['avg_execution_time'] /= agent_performance['successful']
        
        success_rate = agent_performance['successful'] / max(agent_performance['processed'], 1)
        hallucination_rate = agent_performance['hallucinations'] / max(agent_performance['successful'], 1)
        
        print(f"    📊 Elite Agents: {agent_performance['successful']}/{agent_performance['processed']} " +
              f"({success_rate:.1%} success, {hallucination_rate:.1%} halluc, " +
              f"{agent_performance['avg_execution_time']:.1f}s avg)")
        
        # Update global stats
        self.agent_stats['decisions_made'] += agent_performance['successful']
        self.agent_stats['hallucinations_detected'] += agent_performance['hallucinations']
        self.agent_stats['avg_execution_time'] = (
            (self.agent_stats['avg_execution_time'] * 0.9) + 
            (agent_performance['avg_execution_time'] * 0.1)
        )
        
        # Elite signal filtering
        qualified_signals = [(s, score) for s, score in signals.items() 
                           if score >= self.confidence_base * 0.95]  # Slightly more permissive
        qualified_signals.sort(key=lambda x: x[1], reverse=True)
        
        print(f"    🎯 Signals qualifiés: {len(qualified_signals)}/{len(signals)} " +
              f"(seuil: {self.confidence_base:.3f})")
        
        # Adaptive position management
        max_positions = self.max_positions
        if current_drawdown < -0.30:
            max_positions = 6
        elif current_drawdown < -0.25:
            max_positions = 8
        elif current_drawdown < -0.20:
            max_positions = 10
        
        # Portfolio heat control
        if len(portfolio['positions']) >= 10:
            max_positions = min(max_positions, len(portfolio['positions']) + 1)
        
        top_signals = qualified_signals[:max_positions]
        
        # Elite allocation strategy
        base_allocation = 0.88
        if current_drawdown < -0.30:
            base_allocation = 0.50
        elif current_drawdown < -0.25:
            base_allocation = 0.65
        elif current_drawdown < -0.20:
            base_allocation = 0.75
        elif current_drawdown > -0.05:  # Low drawdown, more aggressive
            base_allocation = 0.92
        
        # Market regime adjustment
        market_regime = self.detect_market_regime_advanced(date)
        if market_regime == 'STRONG_BULL':
            base_allocation = min(base_allocation * 1.1, 0.95)
        elif market_regime == 'STRONG_BEAR':
            base_allocation *= 0.7
        
        # Execute elite trades
        self.execute_elite_trades(portfolio, date, top_signals, base_allocation)
        
        total_time = time.time() - start_time
        print(f"    ⏱️ Elite rebalance completed in {total_time:.1f}s")

    def execute_elite_trades(self, portfolio, date, top_signals, allocation):
        """Exécuter trades avec sophistication elite"""
        if not top_signals:
            return
        
        current_positions = list(portfolio['positions'].keys())
        
        # Close positions not in top signals
        for symbol in current_positions:
            if symbol not in [s[0] for s in top_signals]:
                self.close_position_with_rl_update(portfolio, symbol, date)
        
        # Elite position sizing
        for symbol, score in top_signals:
            base_weight = allocation / len(top_signals)
            
            # Score-based adjustment with confidence scaling
            confidence_multiplier = (score / self.confidence_base) ** 0.8  # Sublinear scaling
            score_weight = base_weight * min(confidence_multiplier, 2.0)
            
            # RL-based adjustment
            rl_multiplier = 1.0
            if symbol in self.rl_memory:
                rl_data = self.rl_memory[symbol]
                if rl_data['accuracy'] > 0.65:
                    rl_multiplier = 1.15
                elif rl_data['accuracy'] < 0.45:
                    rl_multiplier = 0.85
                
                if rl_data['avg_return'] > 0.03:
                    rl_multiplier *= 1.1
                elif rl_data['avg_return'] < -0.02:
                    rl_multiplier *= 0.9
            
            # Final weight calculation
            final_weight = min(score_weight * rl_multiplier, self.max_position_size)
            
            # Minimum position filter
            if final_weight < 0.02:  # 2% minimum
                continue
            
            self.adjust_position_with_rl_tracking(portfolio, symbol, date, final_weight)

    def close_position_with_rl_update(self, portfolio, symbol, date):
        """Fermer position avec RL update complet"""
        if symbol in portfolio['positions'] and symbol in self.data:
            try:
                shares = portfolio['positions'][symbol]
                current_price = self.data[symbol]['Close'][self.data[symbol]['Close'].index <= date].iloc[-1]
                proceeds = shares * current_price * 0.9995
                
                # Calculate real return for RL
                if symbol in self.rl_memory:
                    rl_data = self.rl_memory[symbol]
                    entry_prices = rl_data.get('entry_prices', {})
                    
                    # Find most recent entry
                    relevant_entries = [(d, p) for d, p in entry_prices.items() 
                                      if pd.to_datetime(d) <= pd.to_datetime(date)]
                    
                    if relevant_entries:
                        entry_date, entry_price = max(relevant_entries, key=lambda x: pd.to_datetime(x[0]))
                        actual_return = (current_price - entry_price) / entry_price
                        
                        # Get the decision that led to this entry
                        last_decision = rl_data.get('decisions', ['HOLD'])[-1]
                        
                        # Update RL with real performance
                        self.update_rl_memory_online(symbol, last_decision, actual_return, 
                                                   entry_price, current_price)
                        
                        # Clean up old entry
                        if entry_date in entry_prices:
                            del entry_prices[entry_date]
                
                portfolio['cash'] += proceeds
                del portfolio['positions'][symbol]
                
            except Exception as e:
                print(f"⚠️ Close position error {symbol}: {e}")

    def adjust_position_with_rl_tracking(self, portfolio, symbol, date, target_weight):
        """Ajuster position avec tracking RL"""
        if symbol not in self.data:
            return
        
        try:
            current_price = self.data[symbol]['Close'][self.data[symbol]['Close'].index <= date].iloc[-1]
            portfolio_value = portfolio['value']
            
            target_value = portfolio_value * target_weight
            target_shares = target_value / current_price
            
            current_shares = portfolio['positions'].get(symbol, 0)
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff * current_price) > portfolio_value * 0.01:
                cost = shares_diff * current_price
                
                if shares_diff > 0 and portfolio['cash'] >= cost * 1.0005:
                    # Buy operation
                    portfolio['cash'] -= cost * 1.0005
                    portfolio['positions'][symbol] = target_shares
                    
                    # Track entry for RL
                    if symbol in self.rl_memory:
                        entry_prices = self.rl_memory[symbol].get('entry_prices', {})
                        entry_prices[date.strftime('%Y-%m-%d')] = current_price
                        self.rl_memory[symbol]['entry_prices'] = entry_prices
                    
                elif shares_diff < 0:
                    # Sell operation
                    proceeds = -cost * 0.9995
                    portfolio['cash'] += proceeds
                    
                    if target_shares > 0:
                        portfolio['positions'][symbol] = target_shares
                    else:
                        portfolio['positions'].pop(symbol, None)
                        
        except Exception as e:
            print(f"⚠️ Adjust position error {symbol}: {e}")

    def update_portfolio_value_with_tracking(self, portfolio, date):
        """Update portfolio value avec tracking performance"""
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in self.data and shares > 0:
                try:
                    price = self.data[symbol]['Close'][self.data[symbol]['Close'].index <= date].iloc[-1]
                    position_value = shares * price
                    total_value += position_value
                except:
                    pass
        
        portfolio['value'] = total_value
        return total_value

    def execute_elite_strategy(self):
        """Exécuter stratégie elite complète"""
        print("\\n🧠 EXÉCUTION STRATÉGIE ELITE SUPERINTELLIGENCE...")
        
        portfolio = {
            'cash': float(self.initial_capital),
            'positions': {},
            'value': float(self.initial_capital),
            'peak_value': float(self.initial_capital)
        }
        
        trading_dates = pd.bdate_range(start=self.start_date, end=self.end_date).tolist()
        history = []
        
        # Start with more historical data for better RL learning
        start_idx = 300
        
        print(f"  🧠 Elite backtest: {len(trading_dates)-start_idx} jours")
        print(f"  📊 Rebalancing: Tous les {self.rebalance_frequency} jours")
        print(f"  🎯 Target: 35%+ rendement elite")
        
        for i, date in enumerate(trading_dates[start_idx:], start_idx):
            if (i - start_idx) % 250 == 0:
                year = (i - start_idx) // 250 + 1
                progress = (i - start_idx) / (len(trading_dates) - start_idx)
                print(f"    📅 Année {year}/6 - {date.strftime('%Y-%m-%d')} ({progress:.1%})")
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value_with_tracking(portfolio, date)
            
            # Track peak and drawdown
            if portfolio_value > portfolio['peak_value']:
                portfolio['peak_value'] = portfolio_value
            
            current_drawdown = (portfolio_value / portfolio['peak_value']) - 1
            
            # Elite rebalancing
            if i % self.rebalance_frequency == 0:
                self.elite_portfolio_rebalance(portfolio, date, current_drawdown)
            
            # Track history
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'drawdown': current_drawdown,
                'positions': len(portfolio['positions']),
                'cash_ratio': portfolio['cash'] / portfolio_value
            })
            
            # Performance monitoring
            if (i - start_idx) % 500 == 0 and i > start_idx:
                current_return = (portfolio_value / self.initial_capital) ** (252 / (i - start_idx)) - 1
                print(f"    📊 Current annual return: {current_return:.1%}, Drawdown: {current_drawdown:.1%}")
            
            # Memory management
            if i % 100 == 0:
                gc.collect()
        
        print(f"\\n✅ Elite strategy execution completed")
        print(f"📊 Final portfolio value: ${portfolio['value']:,.0f}")
        print(f"📈 Total return: {(portfolio['value'] / self.initial_capital - 1):.1%}")
        
        return history

    def calculate_elite_performance(self, history):
        """Calculer performance avec métriques elite"""
        try:
            df = pd.DataFrame(history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            values = df['portfolio_value']
            daily_returns = values.pct_change().dropna()
            
            # Core metrics
            total_return = (values.iloc[-1] / values.iloc[0]) - 1
            years = len(values) / 252
            annual_return = (1 + total_return) ** (1/years) - 1
            
            # Risk metrics
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0
            
            # Drawdown analysis
            max_drawdown = df['drawdown'].min()
            
            # Advanced metrics
            win_rate = (daily_returns > 0).mean()
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Elite metrics
            information_ratio = annual_return / volatility if volatility > 0 else 0
            max_consecutive_losses = self.calculate_max_consecutive_losses(daily_returns)
            recovery_time = self.calculate_recovery_time(df['drawdown'])
            
            # Risk-adjusted metrics
            var_5 = np.percentile(daily_returns, 5)
            cvar_5 = daily_returns[daily_returns <= var_5].mean()
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'information_ratio': information_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'max_consecutive_losses': max_consecutive_losses,
                'recovery_time_days': recovery_time,
                'var_5': var_5,
                'cvar_5': cvar_5,
                'final_value': values.iloc[-1],
                'trading_days': len(values)
            }
            
        except Exception as e:
            print(f"❌ Elite performance calculation error: {e}")
            return None
    
    def calculate_max_consecutive_losses(self, returns):
        """Calculer perte consécutive max"""
        try:
            consecutive = 0
            max_consecutive = 0
            
            for ret in returns:
                if ret < 0:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            
            return max_consecutive
        except:
            return 0
    
    def calculate_recovery_time(self, drawdowns):
        """Calculer temps de récupération moyen"""
        try:
            in_drawdown = False
            recovery_times = []
            start_dd = None
            
            for i, dd in enumerate(drawdowns):
                if dd < -0.05 and not in_drawdown:  # Start of drawdown
                    in_drawdown = True
                    start_dd = i
                elif dd >= -0.01 and in_drawdown:  # Recovery
                    if start_dd is not None:
                        recovery_times.append(i - start_dd)
                    in_drawdown = False
                    start_dd = None
            
            return np.mean(recovery_times) if recovery_times else 0
        except:
            return 0

    def generate_elite_report(self, performance):
        """Générer rapport elite superintelligence"""
        if not performance:
            print("❌ Pas de données de performance elite")
            return
        
        print("\\n" + "="*80)
        print("🧠 ELITE SUPERINTELLIGENCE TRADING SYSTEM - RAPPORT FINAL")
        print("="*80)
        
        # Performance headline
        annual_return = performance['annual_return']
        if annual_return >= 0.35:
            status = "🌟 SUPERINTELLIGENCE EXCEPTIONNELLE"
        elif annual_return >= 0.30:
            status = "🏆 ELITE PERFORMANCE ATTEINTE"
        elif annual_return >= 0.25:
            status = "🔥 TRÈS HAUTE PERFORMANCE"
        elif annual_return >= 0.20:
            status = "✅ HAUTE PERFORMANCE"
        else:
            status = "📊 PERFORMANCE SOLIDE"
        
        print(f"\\n{status}")
        
        print(f"\\n🧠 PERFORMANCE ELITE SUPERINTELLIGENCE (~6.5 ANS):")
        print(f"  📈 Rendement Annuel:         {performance['annual_return']:>8.1%}")
        print(f"  📊 Rendement Total:          {performance['total_return']:>8.1%}")
        print(f"  💰 Valeur Finale:            ${performance['final_value']:>12,.0f}")
        print(f"  📉 Drawdown Maximum:         {performance['max_drawdown']:>8.1%}")
        print(f"  ⚡ Volatilité:               {performance['volatility']:>8.1%}")
        print(f"  🎯 Ratio Sharpe:             {performance['sharpe_ratio']:>8.2f}")
        print(f"  📊 Ratio Sortino:            {performance['sortino_ratio']:>8.2f}")
        print(f"  📊 Ratio Information:        {performance['information_ratio']:>8.2f}")
        print(f"  📊 Ratio Calmar:             {performance['calmar_ratio']:>8.2f}")
        print(f"  ✅ Taux de Gain:             {performance['win_rate']:>8.1%}")
        print(f"  📉 VaR 5%:                   {performance['var_5']:>8.2%}")
        print(f"  📉 CVaR 5%:                  {performance['cvar_5']:>8.2%}")
        print(f"  🔄 Pertes Consécutives Max:  {performance['max_consecutive_losses']:>8.0f}")
        print(f"  ⏱️ Temps Récupération Moy:   {performance['recovery_time_days']:>8.0f} jours")
        
        # Elite benchmarks
        nasdaq_6y = 0.184
        spy_6y = 0.132
        hedge_fund_elite = 0.15
        quant_funds = 0.22
        
        print(f"\\n🎯 COMPARAISON BENCHMARKS ELITE:")
        print(f"  📊 vs NASDAQ (18.4%):       {annual_return - nasdaq_6y:>+7.1%}")
        print(f"  📊 vs S&P 500 (13.2%):      {annual_return - spy_6y:>+7.1%}")
        print(f"  🏦 vs Elite Hedge Funds:    {annual_return - hedge_fund_elite:>+7.1%}")
        print(f"  🤖 vs Top Quant Funds:      {annual_return - quant_funds:>+7.1%}")
        
        # Risk-adjusted performance
        print(f"\\n📊 ANALYSE RISQUE-RENDEMENT:")
        risk_score = "EXCELLENT" if performance['max_drawdown'] > -0.25 else "BON"
        consistency = "ÉLEVÉE" if performance['sharpe_ratio'] > 1.5 else "BONNE"
        
        print(f"  🛡️ Contrôle Risque:         {risk_score}")
        print(f"  📊 Consistance:              {consistency}")
        print(f"  🎯 Risk/Return Score:        {performance['sharpe_ratio'] * 10:.0f}/10")
        
        # Elite technologies
        print(f"\\n🧠 TECHNOLOGIES ELITE SUPERINTELLIGENCE:")
        print(f"  ✅ Univers Elite: {len(self.elite_universe)} assets premium")
        print(f"  ✅ Deep Learning: LSTM + GRU + Multi-Head Attention")
        print(f"  ✅ Multi-Agents: 4 agents spécialisés avec reducers")
        print(f"  ✅ RL Online Learning: Q-Learning adaptatif")
        print(f"  ✅ External APIs: News + Financial metrics + Economic")
        print(f"  ✅ Anti-Hallucination: Guards + Validation LLM")
        print(f"  ✅ Régimes Marché: 5 niveaux sophistiqués")
        print(f"  ✅ CVaR Multi-Quantile: Risk management institutionnel")
        print(f"  ✅ Cache Intelligent: Optimisations mémoire et vitesse")
        print(f"  ✅ Rethinking Loops: Auto-amélioration continue")
        
        # Agent statistics
        if hasattr(self, 'agent_stats') and self.agent_stats['decisions_made'] > 0:
            print(f"\\n🤖 STATISTIQUES AGENTS ELITE:")
            print(f"  🎯 Décisions Prises:         {self.agent_stats['decisions_made']:>8,d}")
            print(f"  🛡️ Hallucinations Détectées: {self.agent_stats['hallucinations_detected']:>8,d}")
            print(f"  📊 Taux Fiabilité:          {1 - self.agent_stats['hallucinations_detected']/max(self.agent_stats['decisions_made'], 1):>8.1%}")
            print(f"  ⏱️ Temps Exécution Moyen:    {self.agent_stats['avg_execution_time']:>8.1f}s")
            print(f"  🎯 Trades Réussis:          {self.agent_stats['successful_trades']:>8,d}")
            print(f"  📈 Taux de Réussite:        {self.agent_stats['successful_trades']/max(self.agent_stats['decisions_made'], 1):>8.1%}")
        
        # RL Learning statistics
        if hasattr(self, 'rl_memory') and self.rl_memory:
            rl_accuracies = [data['accuracy'] for data in self.rl_memory.values() if 'accuracy' in data]
            rl_returns = [data['avg_return'] for data in self.rl_memory.values() if 'avg_return' in data]
            
            if rl_accuracies and rl_returns:
                print(f"\\n🧠 APPRENTISSAGE RL SUPERINTELLIGENCE:")
                print(f"  🎯 Précision Moyenne RL:     {np.mean(rl_accuracies):>8.1%}")
                print(f"  📈 Return Moyen RL:          {np.mean(rl_returns):>8.2%}")
                print(f"  🧠 Symboles Appris:          {len(self.rl_memory):>8d}")
                print(f"  📊 Mises à Jour RL:          {self.agent_stats.get('rl_updates', 0):>8,d}")
                
                # Best performing symbols
                best_rl = sorted([(symbol, data['avg_return']) for symbol, data in self.rl_memory.items() 
                                if 'avg_return' in data], key=lambda x: x[1], reverse=True)[:5]
                print(f"  🏆 Top 5 RL Symbols:         {', '.join([f'{s}({r:.1%})' for s, r in best_rl])}")
        
        # Final assessment
        print(f"\\n🏆 ÉVALUATION FINALE:")
        
        total_score = 0
        max_score = 6
        
        # Performance score
        if annual_return >= 0.35:
            perf_score = 6
        elif annual_return >= 0.30:
            perf_score = 5
        elif annual_return >= 0.25:
            perf_score = 4
        elif annual_return >= 0.20:
            perf_score = 3
        else:
            perf_score = 2
        
        total_score += perf_score
        
        # Risk score
        if performance['max_drawdown'] > -0.20:
            risk_score = 6
        elif performance['max_drawdown'] > -0.25:
            risk_score = 5
        elif performance['max_drawdown'] > -0.30:
            risk_score = 4
        else:
            risk_score = 3
        
        # Combine scores
        final_score = (total_score + risk_score) / 2
        
        if final_score >= 5.5:
            rating = "🌟 ELITE WORLD-CLASS"
        elif final_score >= 5.0:
            rating = "🏆 INSTITUTIONAL GRADE"
        elif final_score >= 4.0:
            rating = "🔥 PROFESSIONAL GRADE"
        else:
            rating = "✅ SOLID PERFORMANCE"
        
        print(f"  📊 Score Performance:        {perf_score}/6")
        print(f"  🛡️ Score Risque:             {risk_score}/6")
        print(f"  🎯 Score Final:              {final_score:.1f}/6")
        print(f"  🏆 Classification:           {rating}")
        
        # Recommendations
        print(f"\\n💡 RECOMMANDATIONS ELITE:")
        if annual_return >= 0.30:
            print(f"  🌟 Système prêt pour capital institutionnel (>$10M)")
            print(f"  🏦 Considérer licensing à hedge funds")
            print(f"  🚀 Scaling vers univers global (500+ assets)")
        elif annual_return >= 0.25:
            print(f"  🏆 Système excellent pour capital privé ($1-10M)")
            print(f"  📈 Optimiser pour réduire drawdown <20%")
            print(f"  🔧 Fine-tuner RL learning parameters")
        else:
            print(f"  🔧 Continuer optimisation RL et agents")
            print(f"  📊 Augmenter univers et features externes")
            print(f"  ⚡ Tester sur hardware plus puissant (TPU v5)")
        
        print(f"\\n" + "="*80)
        print(f"🧠 ELITE SUPERINTELLIGENCE SYSTEM - MISSION ACCOMPLIE")
        print(f"="*80)

# === CELL 3: EXÉCUTION ELITE SUPERINTELLIGENCE ===
def run_elite_superintelligence():
    """Exécuter système elite superintelligence complet"""
    print("🧠 DÉMARRAGE ELITE SUPERINTELLIGENCE SYSTEM...")
    print("🎯 Target: 35%+ rendement via AI superintelligence")
    print("🚀 Technologies: RL + Multi-Agents + External APIs + Anti-Hallucination")
    
    start_time = time.time()
    
    try:
        # Initialize system
        system = EliteSupertintelligenceSystem()
        
        # Execute pipeline
        print("\\n🔄 PIPELINE ELITE SUPERINTELLIGENCE:")
        print("1️⃣ Téléchargement univers elite...")
        system.download_data_intelligent_caching()
        
        print("2️⃣ Construction modèles elite...")
        system.build_enhanced_elite_models()
        
        print("3️⃣ Entraînement superintelligence...")
        system.train_elite_models()
        
        print("4️⃣ Exécution stratégie elite...")
        history = system.execute_elite_strategy()
        
        print("5️⃣ Calcul performance elite...")
        performance = system.calculate_elite_performance(history)
        
        print("6️⃣ Génération rapport final...")
        system.generate_elite_report(performance)
        
        execution_time = time.time() - start_time
        print(f"\\n⏱️ Temps d'exécution total: {execution_time:.1f} secondes")
        print(f"⚡ Performance système: {len(system.elite_universe)} assets en {execution_time/60:.1f} minutes")
        
        # Save final state
        final_state = {
            'performance': performance,
            'agent_stats': system.agent_stats,
            'rl_memory_summary': {symbol: {
                'accuracy': data.get('accuracy', 0),
                'avg_return': data.get('avg_return', 0),
                'decisions_count': len(data.get('decisions', []))
            } for symbol, data in system.rl_memory.items()},
            'execution_time': execution_time,
            'universe_size': len(system.elite_universe),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{DRIVE_PATH}/elite_final_results.json", 'w') as f:
            json.dump(final_state, f, indent=2, default=str)
        
        print("💾 Résultats finaux sauvegardés")
        print("🎉 ELITE SUPERINTELLIGENCE SYSTEM - TERMINÉ AVEC SUCCÈS!")
        
        return performance
        
    except Exception as e:
        print(f"❌ Erreur système elite: {e}")
        print("🛠️ Recommandation: Vérifiez dépendances et relancez")
        return None

# === CELL 4: LANCEMENT ELITE SUPERINTELLIGENCE ===
if __name__ == "__main__":
    # Vérifications avancées
    print("🧠 VÉRIFICATION ENVIRONNEMENT ELITE SUPERINTELLIGENCE...")
    
    dependencies_status = {
        'yfinance': False,
        'tensorflow': False,
        'langgraph': False,
        'langchain_community': False,
        'joblib': False,
        'polygon': False
    }
    
    try:
        import yfinance
        dependencies_status['yfinance'] = True
    except ImportError:
        print("❌ yfinance manquant")
    
    try:
        import tensorflow as tf
        dependencies_status['tensorflow'] = True
    except ImportError:
        print("❌ tensorflow manquant")
    
    try:
        from langgraph.graph import StateGraph
        dependencies_status['langgraph'] = True
    except ImportError:
        print("❌ langgraph manquant")
    
    try:
        from langchain_community.llms import Perplexity
        dependencies_status['langchain_community'] = True
    except ImportError:
        print("❌ langchain-community manquant")
    
    try:
        import joblib
        dependencies_status['joblib'] = True
    except ImportError:
        print("❌ joblib manquant")
    
    try:
        from polygon import RESTClient
        dependencies_status['polygon'] = True
    except ImportError:
        print("⚠️ polygon-api-client optionnel (recommandé)")
    
    # Check minimum requirements
    required_deps = ['yfinance', 'tensorflow', 'langgraph', 'langchain_community', 'joblib']
    missing_required = [dep for dep in required_deps if not dependencies_status[dep]]
    
    if missing_required:
        print(f"❌ Dépendances manquantes: {', '.join(missing_required)}")
        print("🛠️ Installez avec:")
        print("!pip install yfinance tensorflow langgraph langchain-community joblib")
        if not dependencies_status['polygon']:
            print("!pip install polygon-api-client  # Optionnel pour financial metrics")
        exit(1)
    
    print("✅ Toutes les dépendances elite sont installées")
    
    # Lancement système
    print("\\n🚀 LANCEMENT ELITE SUPERINTELLIGENCE...")
    performance = run_elite_superintelligence()
    
    # Résumé final executif
    if performance:
        annual_return = performance['annual_return']
        max_drawdown = performance['max_drawdown']
        sharpe_ratio = performance['sharpe_ratio']
        
        print(f"\\n📊 RÉSUMÉ EXÉCUTIF ELITE:")
        print(f"💎 Rendement Annuel: {annual_return:.1%}")
        print(f"🛡️ Drawdown Max: {max_drawdown:.1%}")
        print(f"🎯 Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"💰 Capital Final: ${performance['final_value']:,.0f}")
        
        # Executive decision
        if annual_return >= 0.35:
            decision = "🌟 DEPLOY WITH INSTITUTIONAL CAPITAL"
        elif annual_return >= 0.30:
            decision = "🏆 DEPLOY WITH PRIVATE CAPITAL"
        elif annual_return >= 0.25:
            decision = "🔥 DEPLOY WITH CAUTION"
        elif annual_return >= 0.20:
            decision = "✅ CONTINUE OPTIMIZATION"
        else:
            decision = "🔧 REQUIRES FURTHER DEVELOPMENT"
        
        print(f"\\n🎯 DÉCISION EXÉCUTIVE: {decision}")
        
        # Market position
        nasdaq_beat = annual_return > 0.184
        hedge_fund_beat = annual_return > 0.15
        
        print(f"📈 Beats NASDAQ: {'✅ YES' if nasdaq_beat else '❌ NO'}")
        print(f"🏦 Beats Hedge Funds: {'✅ YES' if hedge_fund_beat else '❌ NO'}")
        
        if annual_return >= 0.30:
            print("\\n🌟 FÉLICITATIONS! Système de niveau mondial créé!")
            print("🏆 Performance digne des meilleurs hedge funds quantitatifs")
            print("🚀 Prêt pour le déploiement à grande échelle")
    else:
        print("\\n❌ Échec d'exécution - Voir logs pour diagnostics")
        print("🛠️ Recommandations: Vérifier environnement et relancer")