#!/usr/bin/env python3
"""
Elite Superintelligence Trading System - Paper Trading Version
Système prêt pour déploiement Interactive Brokers paper trading
Target: 30-50% annual return with real broker integration
"""

# === PAPER TRADING SETUP FOR INTERACTIVE BROKERS ===
# Required: pip install ib_insync yfinance pandas numpy scikit-learn
# Required: pip install langgraph langchain transformers torch
# Required: pip install faiss-cpu rank-bm25 feedparser requests
# Required: pip install ta-lib cvxpy matplotlib plotly

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time
import json
import os
import asyncio
from pathlib import Path
import requests
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Interactive Brokers integration
try:
    from ib_insync import IB, Stock, MarketOrder, util
    util.startLoop()  # Required for Jupyter/async
    IBKR_AVAILABLE = True
    print("✅ Interactive Brokers API available")
except ImportError:
    IBKR_AVAILABLE = False
    print("⚠️ ib_insync not available - install: pip install ib_insync")

# Enhanced sentiment analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available for advanced sentiment")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - using basic sentiment")

# RAG components
try:
    from rank_bm25 import BM25Okapi
    import feedparser
    RAG_AVAILABLE = True
    print("✅ RAG components available")
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ RAG components not available")

# Core ML libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats

# LangGraph for multi-agent orchestration
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph available")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not available - using simplified workflow")

# Configuration
IBKR_HOST = '127.0.0.1'  # TWS/Gateway host
IBKR_PORT = 7497  # Paper trading port (7496 for live)
CLIENT_ID = 1
DRIVE_PATH = './paper_trading_states/'
os.makedirs(DRIVE_PATH, exist_ok=True)
os.makedirs(f"{DRIVE_PATH}/trades/", exist_ok=True)
os.makedirs(f"{DRIVE_PATH}/logs/", exist_ok=True)

# State definition for LangGraph
class TradingState(TypedDict):
    symbols: List[str]
    market_data: Dict[str, pd.DataFrame]
    features: Dict[str, np.ndarray]
    rag_context: Dict[str, Any]
    portfolio: pd.DataFrame
    risk_metrics: Dict[str, float]
    trades: List[Dict[str, Any]]
    timestamp: str

class ElitePaperTradingSystem:
    def __init__(self, initial_capital=100000, target_return=0.40):
        self.initial_capital = initial_capital
        self.target_return = target_return
        self.current_capital = initial_capital
        
        # Trading parameters
        self.max_positions = 20
        self.max_leverage = 1.4
        self.risk_free_rate = 0.02
        
        # RAG components
        if RAG_AVAILABLE:
            self.bm25_corpus = []
            self.bm25_index = None
            self.news_cache = {}
            
        # Enhanced sentiment pipeline
        if TRANSFORMERS_AVAILABLE:
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis', 
                model='distilbert-base-uncased-finetuned-sst-2-english'
            )
        
        # IBKR connection
        self.ib = None
        
        # Performance tracking
        self.trades_history = []
        self.portfolio_history = []
        
        print(f"🚀 Elite Paper Trading System initialized")
        print(f"   💰 Capital: ${initial_capital:,}")
        print(f"   🎯 Target: {target_return:.0%}")
        print(f"   📊 Max positions: {self.max_positions}")
        print(f"   ⚡ Max leverage: {self.max_leverage}x")

    def connect_ibkr(self):
        """Connect to Interactive Brokers for paper trading"""
        if not IBKR_AVAILABLE:
            print("❌ IBKR not available - trading disabled")
            return False
            
        try:
            self.ib = IB()
            self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID)
            print(f"✅ Connected to IBKR paper trading on {IBKR_HOST}:{IBKR_PORT}")
            
            # Get account info
            account = self.ib.managedAccounts()[0]
            account_values = self.ib.accountValues(account)
            
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    self.current_capital = float(av.value)
                    print(f"   💰 Account value: ${self.current_capital:,.2f}")
                    break
                    
            return True
            
        except Exception as e:
            print(f"❌ IBKR connection failed: {e}")
            print("   📝 Ensure TWS/Gateway is running on paper trading mode")
            return False

    def disconnect_ibkr(self):
        """Disconnect from IBKR"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            print("✅ Disconnected from IBKR")

    def fetch_real_market_data(self, symbols: List[str], period='1mo') -> Dict[str, pd.DataFrame]:
        """Fetch real market data instead of simulation"""
        print(f"📊 Fetching real data for {len(symbols)} symbols...")
        
        market_data = {}
        
        for symbol in symbols:
            try:
                # Get real historical data
                data = yf.download(symbol, period=period, progress=False)
                
                if not data.empty:
                    # Calculate technical indicators
                    data['Returns'] = data['Close'].pct_change()
                    data['SMA_20'] = data['Close'].rolling(20).mean()
                    data['SMA_50'] = data['Close'].rolling(50).mean()
                    data['RSI'] = self.calculate_rsi(data['Close'])
                    data['Volatility'] = data['Returns'].rolling(20).std()
                    
                    market_data[symbol] = data
                    print(f"  ✅ {symbol}: {len(data)} days")
                else:
                    print(f"  ❌ {symbol}: No data")
                    
            except Exception as e:
                print(f"  ❌ {symbol}: Error - {e}")
                
        return market_data

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI with proper clipping"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Fix division by zero with clipping
        rs = gain / loss.replace(0, np.nan)
        rs = np.clip(rs, 0, 100)  # Prevent infinity
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI for NaN values

    async def fetch_external_news_async(self, symbol: str, max_articles=3):
        """Async news fetching with corpus limitation"""
        if not RAG_AVAILABLE:
            return {'sentiment': 0.5, 'articles': [], 'rag_score': 0.5}
            
        # Check cache first
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
        if cache_key in self.news_cache:
            return self.news_cache[cache_key]
            
        try:
            # Multiple RSS sources for better coverage
            rss_urls = [
                f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}",
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.nasdaq.com/feed/rssoutbound?category=stocks"
            ]
            
            articles = []
            
            for rss_url in rss_urls[:2]:  # Limit sources to avoid overhead
                try:
                    feed = feedparser.parse(rss_url)
                    for entry in feed.entries[:max_articles]:
                        article_text = f"{entry.title} {entry.get('summary', '')}"
                        articles.append({
                            'title': entry.title,
                            'text': article_text,
                            'link': entry.link,
                            'published': entry.get('published', '')
                        })
                except:
                    continue
                    
                if len(articles) >= max_articles:
                    break
            
            # Calculate sentiment
            combined_text = " ".join([art['text'] for art in articles])
            sentiment = self.calculate_enhanced_sentiment(combined_text)
            
            # Update BM25 corpus with limitation
            if articles:
                new_docs = [art['text'] for art in articles]
                self.bm25_corpus.extend(new_docs)
                
                # Limit corpus size to prevent memory growth
                if len(self.bm25_corpus) > 5000:
                    self.bm25_corpus = self.bm25_corpus[-3000:]  # Keep recent 3000
                    
                self.bm25_index = BM25Okapi(self.bm25_corpus)
            
            # Calculate RAG score
            rag_score = self.calculate_rag_score(symbol, articles)
            
            result = {
                'sentiment': sentiment,
                'articles': articles,
                'rag_score': rag_score,
                'news_count': len(articles)
            }
            
            # Cache result
            self.news_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"  ⚠️ News fetch error for {symbol}: {e}")
            return {'sentiment': 0.5, 'articles': [], 'rag_score': 0.5}

    def calculate_enhanced_sentiment(self, text: str) -> float:
        """Enhanced sentiment using transformers if available"""
        if not text:
            return 0.5
            
        if TRANSFORMERS_AVAILABLE and hasattr(self, 'sentiment_pipeline'):
            try:
                # Use LLM for better sentiment analysis
                result = self.sentiment_pipeline(text[:512])[0]  # Limit text length
                score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
                return np.clip(score, 0.1, 0.9)  # Bounded sentiment
            except:
                pass
                
        # Fallback to keyword-based sentiment
        positive_words = ['growth', 'profit', 'gain', 'bullish', 'optimistic', 'positive', 'strong']
        negative_words = ['loss', 'decline', 'bearish', 'pessimistic', 'negative', 'weak', 'risk']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.5
            
        sentiment = pos_count / (pos_count + neg_count)
        return np.clip(sentiment, 0.1, 0.9)

    def calculate_rag_score(self, symbol: str, articles: List[Dict]) -> float:
        """Calculate RAG relevance score"""
        if not articles or not RAG_AVAILABLE:
            return 0.5
            
        try:
            # Query expansion
            queries = [
                symbol,
                f"{symbol} earnings revenue",
                f"{symbol} technical analysis"
            ]
            
            scores = []
            for query in queries:
                if self.bm25_index:
                    query_tokens = query.lower().split()
                    doc_scores = self.bm25_index.get_scores(query_tokens)
                    if len(doc_scores) > 0:
                        scores.append(np.mean(doc_scores[-len(articles):]))
            
            return np.clip(np.mean(scores) if scores else 0.5, 0.1, 0.9)
            
        except:
            return 0.5

    def calculate_real_expected_returns(self, market_data: Dict[str, pd.DataFrame], 
                                       rag_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected returns using real data + RAG enhancement"""
        expected_returns = {}
        
        for symbol, data in market_data.items():
            if data.empty:
                expected_returns[symbol] = 0.0
                continue
                
            # Base return from historical data
            recent_returns = data['Returns'].dropna()
            if len(recent_returns) < 5:
                base_return = 0.01  # Default 1%
            else:
                base_return = recent_returns.tail(20).mean() * 21  # Monthly estimate
            
            # RAG enhancement
            rag_data = rag_context.get(symbol, {})
            sentiment = rag_data.get('sentiment', 0.5)
            rag_score = rag_data.get('rag_score', 0.5)
            
            # Adjust return based on sentiment and RAG
            sentiment_adj = base_return * (0.8 + sentiment * 0.4)
            rag_adj = sentiment_adj * (0.9 + rag_score * 0.2)
            
            # Risk adjustment for volatility
            volatility = recent_returns.std() if len(recent_returns) > 0 else 0.02
            risk_adj = rag_adj * (1 - volatility * 0.5)  # Penalize high volatility
            
            expected_returns[symbol] = np.clip(risk_adj, -0.1, 0.15)  # Realistic bounds
            
        return expected_returns

    def optimize_portfolio_real(self, expected_returns: Dict[str, float], 
                               market_data: Dict[str, pd.DataFrame],
                               rag_context: Dict[str, Any]) -> pd.DataFrame:
        """Portfolio optimization with real constraints"""
        
        symbols = list(expected_returns.keys())
        returns = np.array([expected_returns[sym] for sym in symbols])
        
        # Calculate covariance matrix from real data
        returns_matrix = []
        for symbol in symbols:
            data = market_data.get(symbol, pd.DataFrame())
            if not data.empty and 'Returns' in data.columns:
                returns_matrix.append(data['Returns'].dropna().tail(60))  # 60 days
        
        if len(returns_matrix) > 0:
            # Align all return series
            min_length = min(len(r) for r in returns_matrix)
            aligned_returns = np.array([r.tail(min_length).values for r in returns_matrix]).T
            
            if aligned_returns.shape[0] > 5:  # Need sufficient data
                cov_matrix = np.cov(aligned_returns.T)
            else:
                cov_matrix = np.eye(len(symbols)) * 0.01  # Fallback
        else:
            cov_matrix = np.eye(len(symbols)) * 0.01
        
        # Risk parity with return adjustment
        try:
            n_assets = len(symbols)
            weights = np.ones(n_assets) / n_assets  # Equal weight start
            
            # Simple optimization: weight by return/risk ratio
            risk_scores = np.diag(cov_matrix)
            return_risk_ratios = returns / (risk_scores + 1e-6)
            
            # Normalize weights
            weights = np.maximum(return_risk_ratios, 0)
            weights = weights / np.sum(weights)
            
            # Apply concentration limits
            weights = np.minimum(weights, 0.15)  # Max 15% per position
            weights = weights / np.sum(weights)  # Renormalize
            
        except:
            # Fallback to equal weights
            weights = np.ones(len(symbols)) / len(symbols)
        
        # Create portfolio DataFrame
        portfolio_data = []
        for i, symbol in enumerate(symbols):
            rag_data = rag_context.get(symbol, {})
            
            # Calculate leverage based on confidence
            confidence = min(weights[i] * 5, 1.0)  # Scale to confidence
            sentiment = rag_data.get('sentiment', 0.5)
            rag_score = rag_data.get('rag_score', 0.5)
            
            # Conservative leverage calculation
            if sentiment > 0.7 and rag_score > 0.6:
                leverage = min(1.3, 1.0 + confidence * 0.3)
            else:
                leverage = 1.0
            
            portfolio_data.append({
                'symbol': symbol,
                'weight': weights[i],
                'expected_return': expected_returns[symbol],
                'confidence': confidence,
                'sentiment': sentiment,
                'rag_score': rag_score,
                'leverage': leverage,
                'final_weight': weights[i] * leverage
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Ensure total exposure doesn't exceed limits
        total_exposure = portfolio_df['final_weight'].sum()
        if total_exposure > 1.5:  # Max 150% exposure
            portfolio_df['final_weight'] *= 1.5 / total_exposure
            
        return portfolio_df

    def execute_paper_trades(self, portfolio_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Execute trades on IBKR paper account"""
        if not self.ib or not self.ib.isConnected():
            print("❌ IBKR not connected - saving trades to CSV only")
            return self.save_trades_csv(portfolio_df)
        
        executed_trades = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"🔄 Executing {len(portfolio_df)} paper trades...")
        
        for _, row in portfolio_df.iterrows():
            symbol = row['symbol']
            target_weight = row['final_weight']
            
            if target_weight < 0.01:  # Skip tiny positions
                continue
                
            try:
                # Create contract
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                
                # Get current price
                ticker = self.ib.reqMktData(contract)
                time.sleep(0.5)  # Brief wait for price
                
                current_price = ticker.marketPrice()
                if not current_price or current_price <= 0:
                    current_price = ticker.close
                
                if not current_price or current_price <= 0:
                    print(f"  ❌ {symbol}: No price available")
                    continue
                
                # Calculate position size
                position_value = self.current_capital * target_weight
                shares = int(position_value / current_price)
                
                if shares < 1:
                    print(f"  ⏭️ {symbol}: Position too small ({shares} shares)")
                    continue
                
                # Place market order
                order = MarketOrder('BUY', shares)
                trade = self.ib.placeOrder(contract, order)
                
                # Track trade
                trade_data = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': current_price,
                    'value': shares * current_price,
                    'weight': target_weight,
                    'confidence': row['confidence'],
                    'sentiment': row['sentiment'],
                    'rag_score': row['rag_score'],
                    'leverage': row['leverage'],
                    'order_id': trade.order.orderId
                }
                
                executed_trades.append(trade_data)
                self.trades_history.append(trade_data)
                
                print(f"  ✅ {symbol}: {shares} shares @ ${current_price:.2f} = ${shares * current_price:,.0f}")
                
                # Brief pause between orders
                time.sleep(0.2)
                
            except Exception as e:
                print(f"  ❌ {symbol}: Trade error - {e}")
        
        # Save trades
        self.save_trades_csv(portfolio_df, executed_trades, timestamp)
        
        print(f"✅ Executed {len(executed_trades)} paper trades")
        return executed_trades

    def save_trades_csv(self, portfolio_df: pd.DataFrame, 
                       executed_trades: List[Dict] = None, 
                       timestamp: str = None) -> List[Dict[str, Any]]:
        """Save trading decisions to CSV"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save portfolio allocation
        portfolio_file = f"{DRIVE_PATH}/trades/portfolio_{timestamp}.csv"
        portfolio_df.to_csv(portfolio_file, index=False)
        
        # Save executed trades if any
        if executed_trades:
            trades_file = f"{DRIVE_PATH}/trades/executed_{timestamp}.csv"
            pd.DataFrame(executed_trades).to_csv(trades_file, index=False)
            print(f"💾 Saved {len(executed_trades)} trades to {trades_file}")
        
        print(f"💾 Saved portfolio to {portfolio_file}")
        
        # Convert portfolio to trade format for consistency
        trade_decisions = []
        for _, row in portfolio_df.iterrows():
            if row['final_weight'] > 0.01:
                trade_decisions.append({
                    'timestamp': timestamp,
                    'symbol': row['symbol'],
                    'action': 'BUY',
                    'weight': row['final_weight'],
                    'confidence': row['confidence'],
                    'sentiment': row['sentiment'],
                    'rag_score': row['rag_score'],
                    'leverage': row['leverage'],
                    'expected_return': row['expected_return']
                })
        
        return trade_decisions

    async def run_paper_trading_cycle(self, symbols: List[str] = None):
        """Run one complete paper trading cycle"""
        
        if symbols is None:
            # Default universe - top liquid stocks
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 
                'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'ARKK', 'TQQQ', 'AMD',
                'CRM', 'ADBE', 'PYPL', 'DIS'
            ]
        
        print(f"\n🚀 Starting paper trading cycle: {datetime.now()}")
        print(f"📊 Universe: {len(symbols)} symbols")
        
        # 1. Fetch real market data
        market_data = self.fetch_real_market_data(symbols)
        
        if not market_data:
            print("❌ No market data available")
            return None
        
        # 2. Fetch news and RAG context
        print("📰 Fetching news and RAG context...")
        rag_context = {}
        
        for symbol in list(market_data.keys())[:15]:  # Limit for speed
            news_data = await self.fetch_external_news_async(symbol)
            rag_context[symbol] = news_data
            print(f"  📰 {symbol}: sentiment={news_data['sentiment']:.3f}, rag={news_data['rag_score']:.3f}")
        
        # 3. Calculate expected returns
        expected_returns = self.calculate_real_expected_returns(market_data, rag_context)
        
        # 4. Optimize portfolio
        portfolio_df = self.optimize_portfolio_real(expected_returns, market_data, rag_context)
        
        print(f"\n📊 Portfolio Summary:")
        print(f"  🎯 Total positions: {len(portfolio_df)}")
        print(f"  ⚡ Total exposure: {portfolio_df['final_weight'].sum():.1%}")
        print(f"  📈 Avg expected return: {portfolio_df['expected_return'].mean():.2%}")
        print(f"  😊 Avg sentiment: {portfolio_df['sentiment'].mean():.3f}")
        
        # Display top positions
        top_positions = portfolio_df.nlargest(5, 'final_weight')
        print(f"\n🏆 Top 5 Positions:")
        for _, row in top_positions.iterrows():
            print(f"  {row['symbol']}: {row['final_weight']:.1%} "
                  f"(ret: {row['expected_return']:.1%}, sent: {row['sentiment']:.2f})")
        
        # 5. Execute trades
        if IBKR_AVAILABLE and self.ib and self.ib.isConnected():
            executed_trades = self.execute_paper_trades(portfolio_df)
        else:
            print("💾 IBKR not connected - saving trades to CSV only")
            executed_trades = self.save_trades_csv(portfolio_df)
        
        # 6. Update tracking
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'portfolio': portfolio_df,
            'executed_trades': executed_trades,
            'market_data_count': len(market_data),
            'rag_context_count': len(rag_context)
        })
        
        return {
            'portfolio': portfolio_df,
            'executed_trades': executed_trades,
            'market_data': market_data,
            'rag_context': rag_context
        }

    def run_backtest_with_real_data(self, start_date: str = None, end_date: str = None):
        """Run backtest with real historical data"""
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n📊 Running backtest: {start_date} to {end_date}")
        
        # Default universe
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
        
        # Fetch historical data
        historical_data = {}
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    data['Returns'] = data['Close'].pct_change()
                    historical_data[symbol] = data
            except:
                continue
        
        if not historical_data:
            print("❌ No historical data available")
            return None
        
        # Simulate monthly rebalancing
        returns_history = []
        portfolio_history = []
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        for date in date_range:
            print(f"📅 Processing {date.strftime('%Y-%m')}...")
            
            # Get data up to current date
            current_data = {}
            for symbol, data in historical_data.items():
                mask = data.index <= date
                current_data[symbol] = data[mask].tail(60)  # Last 60 days
            
            if not current_data:
                continue
            
            # Simulate RAG context (basic)
            rag_context = {}
            for symbol in current_data.keys():
                recent_returns = current_data[symbol]['Returns'].dropna().tail(20)
                if len(recent_returns) > 0:
                    momentum = recent_returns.mean()
                    sentiment = 0.6 if momentum > 0 else 0.4
                else:
                    sentiment = 0.5
                
                rag_context[symbol] = {
                    'sentiment': sentiment,
                    'rag_score': 0.5 + np.random.normal(0, 0.1),  # Simulated
                    'articles': []
                }
            
            # Calculate returns and optimize
            expected_returns = self.calculate_real_expected_returns(current_data, rag_context)
            portfolio_df = self.optimize_portfolio_real(expected_returns, current_data, rag_context)
            
            # Calculate period return
            period_return = 0.0
            for _, row in portfolio_df.iterrows():
                symbol = row['symbol']
                weight = row['final_weight']
                
                # Get actual return for the period
                symbol_data = historical_data.get(symbol)
                if symbol_data is not None:
                    # Get return for the month
                    month_data = symbol_data[symbol_data.index.month == date.month]
                    if not month_data.empty and len(month_data) > 1:
                        actual_return = (month_data['Close'].iloc[-1] / month_data['Close'].iloc[0] - 1)
                        period_return += weight * actual_return
            
            returns_history.append({
                'date': date,
                'return': period_return,
                'positions': len(portfolio_df)
            })
            
            portfolio_history.append({
                'date': date,
                'portfolio': portfolio_df
            })
        
        # Analyze performance
        if returns_history:
            returns_df = pd.DataFrame(returns_history)
            returns_df.set_index('date', inplace=True)
            
            total_return = (1 + returns_df['return']).prod() - 1
            annualized_return = (1 + total_return) ** (12 / len(returns_df)) - 1
            volatility = returns_df['return'].std() * np.sqrt(12)
            sharpe = annualized_return / volatility if volatility > 0 else 0
            
            print(f"\n🎯 BACKTEST RESULTS:")
            print(f"  📈 Total Return: {total_return:.2%}")
            print(f"  🚀 Annualized Return: {annualized_return:.2%}")
            print(f"  📉 Volatility: {volatility:.2%}")
            print(f"  ⚡ Sharpe Ratio: {sharpe:.2f}")
            print(f"  📊 Periods: {len(returns_df)}")
            
            # Save backtest results
            backtest_file = f"{DRIVE_PATH}/backtest_results_{datetime.now().strftime('%Y%m%d')}.csv"
            returns_df.to_csv(backtest_file)
            print(f"💾 Saved backtest to {backtest_file}")
            
            return {
                'returns_df': returns_df,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe
            }
        
        return None

def main():
    """Main execution function"""
    print("🎯 Elite Superintelligence Paper Trading System")
    print("=" * 50)
    
    # Initialize system
    system = ElitePaperTradingSystem(
        initial_capital=100000,
        target_return=0.40
    )
    
    # Connect to IBKR
    connected = system.connect_ibkr()
    
    if connected:
        print("\n🔄 Running live paper trading cycle...")
        # Run async trading cycle
        import asyncio
        result = asyncio.run(system.run_paper_trading_cycle())
        
        if result:
            print("\n✅ Paper trading cycle completed successfully")
        else:
            print("\n❌ Paper trading cycle failed")
    else:
        print("\n📊 Running backtest mode (IBKR not connected)...")
        backtest_result = system.run_backtest_with_real_data()
        
        if backtest_result:
            print(f"\n✅ Backtest completed: {backtest_result['annualized_return']:.1%} annual return")
    
    # Cleanup
    system.disconnect_ibkr()
    print("\n🏁 Session completed")

if __name__ == "__main__":
    main()