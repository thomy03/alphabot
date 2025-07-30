#!/usr/bin/env python3
"""
Multi-Strategy Comparison System - Fair Performance Analysis
Compare all AlphaBot strategies on identical dataset and conditions
Strategies: Optimized Daily, Enhanced Swing, Balanced Swing, Adaptive Risk
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MultiStrategyComparison:
    """
    Compare multiple trading strategies on unified dataset
    Fair comparison with identical conditions
    """
    
    def __init__(self):
        # UNIFIED DATASET - Common universe for all strategies
        self.unified_universe = [
            # Mega cap tech (high performance potential)
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'NFLX',
            
            # Large cap quality
            'V', 'MA', 'UNH', 'HD', 'JPM', 'BAC', 'WFC', 'GS',
            
            # Growth tech
            'CRM', 'ADBE', 'AMD', 'QCOM', 'ORCL', 'CSCO', 'INTC',
            
            # Defensive/Cyclical
            'KO', 'PG', 'JNJ', 'WMT', 'XOM', 'CVX', 'CAT', 'GE',
            
            # ETFs for diversification
            'SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLV', 'XLI', 'VGT', 'ARKK'
        ]
        
        # UNIFIED PARAMETERS
        self.start_date = "2019-01-01"  # Bull market start for fair comparison
        self.end_date = "2024-01-01"    # 5 years exactly
        self.initial_capital = 100000
        
        print(f"🔄 MULTI-STRATEGY COMPARISON SYSTEM")
        print(f"📊 UNIFIED DATASET: {len(self.unified_universe)} symbols")
        print(f"📅 UNIFIED PERIOD: {self.start_date} to {self.end_date} (5 years)")
        print(f"💰 UNIFIED CAPITAL: ${self.initial_capital:,}")
        print(f"🎯 OBJECTIVE: Fair comparison on identical conditions")
        
        # Results storage
        self.unified_data = {}
        self.market_data = {}
        self.strategy_results = {}
        self.benchmarks = {}
    
    def run_comprehensive_comparison(self):
        """Run comprehensive multi-strategy comparison"""
        print("\\n" + "="*80)
        print("🔄 COMPREHENSIVE MULTI-STRATEGY COMPARISON")
        print("="*80)
        
        # Step 1: Download unified dataset
        print("\\n📊 Step 1: Downloading unified dataset...")
        self.download_unified_data()
        
        # Step 2: Download market indices
        print("\\n📈 Step 2: Downloading market indices...")
        self.download_market_indices()
        
        # Step 3: Run all strategies
        print("\\n🚀 Step 3: Running all strategies on unified data...")
        self.run_all_strategies()
        
        # Step 4: Download benchmarks
        print("\\n📊 Step 4: Downloading benchmark performance...")
        self.download_benchmarks()
        
        # Step 5: Generate comparison
        print("\\n📈 Step 5: Generating comprehensive comparison...")
        comparison_results = self.generate_comparison_analysis()
        
        # Step 6: Create summary
        print("\\n📋 Step 6: Creating final comparison summary...")
        self.print_comprehensive_summary(comparison_results)
        
        return comparison_results
    
    def download_unified_data(self):
        """Download unified dataset for all strategies"""
        failed_downloads = []
        
        for i, symbol in enumerate(self.unified_universe, 1):
            try:
                print(f"  📊 ({i:2d}/{len(self.unified_universe)}) {symbol:8s}...", end=" ")
                
                ticker_data = yf.download(
                    symbol, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False
                )
                
                if len(ticker_data) > 500:
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    
                    self.unified_data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ {len(ticker_data)} days (insufficient)")
                    failed_downloads.append(symbol)
                    
            except Exception as e:
                print(f"❌ Error")
                failed_downloads.append(symbol)
        
        print(f"  📊 UNIFIED DATA: {len(self.unified_data)} symbols successfully downloaded")
        if failed_downloads:
            print(f"  ⚠️ Failed downloads: {failed_downloads}")
    
    def download_market_indices(self):
        """Download market indices for regime detection"""
        indices = ['SPY', 'QQQ', 'VIX']
        
        for symbol in indices:
            try:
                if symbol == 'VIX':
                    data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
                else:
                    data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                self.market_data[symbol] = data
                print(f"  ✅ {symbol}: {len(data)} days")
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
    
    def run_all_strategies(self):
        """Run all strategies on unified data"""
        
        # Strategy 1: Optimized Daily (our 23.5% champion)
        print("\\n🚀 Running Strategy 1: Optimized Daily System...")
        self.strategy_results['Optimized Daily'] = self.run_optimized_daily_strategy()
        
        # Strategy 2: Enhanced Swing (39.2% but questionable)
        print("\\n🚀 Running Strategy 2: Enhanced Swing System...")
        self.strategy_results['Enhanced Swing'] = self.run_enhanced_swing_strategy()
        
        # Strategy 3: Balanced Swing (11.2% too conservative)
        print("\\n🚀 Running Strategy 3: Balanced Swing System...")
        self.strategy_results['Balanced Swing'] = self.run_balanced_swing_strategy()
        
        # Strategy 4: Buy and Hold SPY (baseline)
        print("\\n🚀 Running Strategy 4: Buy & Hold SPY...")
        self.strategy_results['Buy & Hold SPY'] = self.run_buy_hold_spy()
        
        # Strategy 5: Buy and Hold QQQ (aggressive baseline)
        print("\\n🚀 Running Strategy 5: Buy & Hold QQQ...")
        self.strategy_results['Buy & Hold QQQ'] = self.run_buy_hold_qqq()
        
        print(f"\\n✅ All {len(self.strategy_results)} strategies completed on unified data")
    
    def run_optimized_daily_strategy(self):
        """Run optimized daily strategy on unified data"""
        try:
            # Import and adapt optimized daily logic
            portfolio = {
                'cash': float(self.initial_capital),
                'positions': {},
                'value': float(self.initial_capital)
            }
            
            # Generate trading dates
            trading_dates = pd.bdate_range(
                start=self.start_date, 
                end=self.end_date
            ).tolist()
            
            history = []
            current_regime = 'stable'
            
            # Optimized daily parameters
            regimes = {
                'trend_up': {
                    'score_threshold': 0.08,
                    'allocation_factor': 0.98,
                    'max_positions': 18,
                    'tech_boost': 1.2
                },
                'trend_down': {
                    'score_threshold': 0.25,
                    'allocation_factor': 0.65,
                    'max_positions': 10,
                    'tech_boost': 0.8
                },
                'volatile': {
                    'score_threshold': 0.12,
                    'allocation_factor': 0.85,
                    'max_positions': 14,
                    'tech_boost': 1.1
                },
                'stable': {
                    'score_threshold': 0.06,
                    'allocation_factor': 0.92,
                    'max_positions': 16,
                    'tech_boost': 1.0
                }
            }
            
            print(f"    📈 Simulating {len(trading_dates)} days with optimized daily logic...")
            
            for i, date in enumerate(trading_dates):
                current_date = date.strftime('%Y-%m-%d')
                
                if i % 200 == 0 or i == len(trading_dates) - 1:
                    progress = (i + 1) / len(trading_dates) * 100
                    print(f"      📅 Progress: {progress:5.1f}% - {current_date}")
                
                # Detect regime (simplified)
                current_regime = self.detect_simple_regime(date)
                
                # Update portfolio value
                portfolio_value = self.update_portfolio_value(portfolio, date)
                
                # Daily rebalancing
                if i % 1 == 0:  # Daily
                    signals = self.calculate_optimized_signals(date, current_regime)
                    self.execute_optimized_rebalancing(portfolio, date, signals, current_regime, regimes)
                
                history.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': portfolio['cash'],
                    'num_positions': len(portfolio['positions']),
                    'regime': current_regime
                })
            
            return self.calculate_strategy_metrics(history, 'Optimized Daily')
            
        except Exception as e:
            print(f"    ❌ Optimized Daily failed: {e}")
            return None
    
    def run_enhanced_swing_strategy(self):
        """Run enhanced swing strategy on unified data"""
        try:
            portfolio = {
                'cash': float(self.initial_capital),
                'positions': {},
                'position_entry_dates': {},
                'position_entry_prices': {},
                'value': float(self.initial_capital)
            }
            
            trading_dates = pd.bdate_range(
                start=self.start_date, 
                end=self.end_date
            ).tolist()
            
            history = []
            current_regime = 'moderate_trend'
            
            # Enhanced swing parameters
            regimes = {
                'strong_momentum': {
                    'score_threshold': 0.15,
                    'allocation_factor': 0.95,
                    'max_positions': 8,
                    'take_profit': 0.08,
                    'stop_loss': 0.04
                },
                'moderate_trend': {
                    'score_threshold': 0.12,
                    'allocation_factor': 0.85,
                    'max_positions': 10,
                    'take_profit': 0.06,
                    'stop_loss': 0.03
                },
                'consolidation': {
                    'score_threshold': 0.08,
                    'allocation_factor': 0.70,
                    'max_positions': 6,
                    'take_profit': 0.04,
                    'stop_loss': 0.025
                },
                'high_volatility': {
                    'score_threshold': 0.10,
                    'allocation_factor': 0.80,
                    'max_positions': 8,
                    'take_profit': 0.10,
                    'stop_loss': 0.05
                }
            }
            
            print(f"    🎯 Simulating {len(trading_dates)} days with enhanced swing logic...")
            
            for i, date in enumerate(trading_dates):
                current_date = date.strftime('%Y-%m-%d')
                
                if i % 200 == 0 or i == len(trading_dates) - 1:
                    progress = (i + 1) / len(trading_dates) * 100
                    print(f"      📅 Progress: {progress:5.1f}% - {current_date}")
                
                # Regime detection
                current_regime = self.detect_swing_regime(date)
                
                # Update portfolio value
                portfolio_value = self.update_portfolio_value(portfolio, date)
                
                # Swing rebalancing (every 2 days)
                if i % 2 == 0:
                    exits_made = self.check_swing_exits(portfolio, date, current_regime, i, regimes)
                    entries_made = self.consider_swing_entries(portfolio, date, current_regime, regimes)
                
                history.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': portfolio['cash'],
                    'num_positions': len(portfolio['positions']),
                    'regime': current_regime
                })
            
            return self.calculate_strategy_metrics(history, 'Enhanced Swing')
            
        except Exception as e:
            print(f"    ❌ Enhanced Swing failed: {e}")
            return None
    
    def run_balanced_swing_strategy(self):
        """Run balanced swing strategy on unified data"""
        try:
            portfolio = {
                'cash': float(self.initial_capital),
                'positions': {},
                'position_entry_dates': {},
                'position_entry_prices': {},
                'value': float(self.initial_capital)
            }
            
            trading_dates = pd.bdate_range(
                start=self.start_date, 
                end=self.end_date
            ).tolist()
            
            history = []
            current_regime = 'sideways'
            
            # Balanced parameters
            regimes = {
                'strong_bull': {
                    'score_threshold': 0.08,
                    'allocation_factor': 0.92,
                    'max_positions': 10
                },
                'moderate_bull': {
                    'score_threshold': 0.10,
                    'allocation_factor': 0.85,
                    'max_positions': 8
                },
                'sideways': {
                    'score_threshold': 0.12,
                    'allocation_factor': 0.75,
                    'max_positions': 6
                },
                'correction': {
                    'score_threshold': 0.15,
                    'allocation_factor': 0.65,
                    'max_positions': 5
                },
                'bear_market': {
                    'score_threshold': 0.18,
                    'allocation_factor': 0.50,
                    'max_positions': 4
                }
            }
            
            print(f"    ⚖️ Simulating {len(trading_dates)} days with balanced swing logic...")
            
            for i, date in enumerate(trading_dates):
                current_date = date.strftime('%Y-%m-%d')
                
                if i % 200 == 0 or i == len(trading_dates) - 1:
                    progress = (i + 1) / len(trading_dates) * 100
                    print(f"      📅 Progress: {progress:5.1f}% - {current_date}")
                
                # Regime detection
                current_regime = self.detect_balanced_regime(date)
                
                # Update portfolio value
                portfolio_value = self.update_portfolio_value(portfolio, date)
                
                # Balanced rebalancing (every 2 days)
                if i % 2 == 0:
                    signals = self.calculate_balanced_signals(date, current_regime)
                    self.execute_balanced_rebalancing(portfolio, date, signals, current_regime, regimes)
                
                history.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': portfolio['cash'],
                    'num_positions': len(portfolio['positions']),
                    'regime': current_regime
                })
            
            return self.calculate_strategy_metrics(history, 'Balanced Swing')
            
        except Exception as e:
            print(f"    ❌ Balanced Swing failed: {e}")
            return None
    
    def run_buy_hold_spy(self):
        """Run buy and hold SPY strategy"""
        try:
            if 'SPY' not in self.unified_data:
                return None
                
            spy_prices = self.unified_data['SPY']
            
            # Buy SPY on first day
            initial_price = float(spy_prices.iloc[0])
            shares = self.initial_capital / initial_price
            
            history = []
            for date_idx, price in spy_prices.items():
                portfolio_value = shares * float(price)
                history.append({
                    'date': date_idx.strftime('%Y-%m-%d'),
                    'portfolio_value': portfolio_value,
                    'cash': 0,
                    'num_positions': 1,
                    'regime': 'buy_hold'
                })
            
            print(f"    📊 Buy & Hold SPY: {len(history)} days simulated")
            return self.calculate_strategy_metrics(history, 'Buy & Hold SPY')
            
        except Exception as e:
            print(f"    ❌ Buy & Hold SPY failed: {e}")
            return None
    
    def run_buy_hold_qqq(self):
        """Run buy and hold QQQ strategy"""
        try:
            if 'QQQ' not in self.unified_data:
                return None
                
            qqq_prices = self.unified_data['QQQ']
            
            # Buy QQQ on first day
            initial_price = float(qqq_prices.iloc[0])
            shares = self.initial_capital / initial_price
            
            history = []
            for date_idx, price in qqq_prices.items():
                portfolio_value = shares * float(price)
                history.append({
                    'date': date_idx.strftime('%Y-%m-%d'),
                    'portfolio_value': portfolio_value,
                    'cash': 0,
                    'num_positions': 1,
                    'regime': 'buy_hold'
                })
            
            print(f"    📊 Buy & Hold QQQ: {len(history)} days simulated")
            return self.calculate_strategy_metrics(history, 'Buy & Hold QQQ')
            
        except Exception as e:
            print(f"    ❌ Buy & Hold QQQ failed: {e}")
            return None
    
    def detect_simple_regime(self, current_date):
        """Simple regime detection for optimized daily"""
        try:
            spy_data = self.market_data.get('SPY')
            if spy_data is None:
                return 'stable'
            
            historical_spy = spy_data[spy_data.index <= current_date]
            if len(historical_spy) < 20:
                return 'stable'
            
            closes = historical_spy['Close']
            ma_5 = closes.rolling(5).mean().iloc[-1]
            ma_15 = closes.rolling(15).mean().iloc[-1]
            
            returns = closes.pct_change().dropna()
            volatility_10d = returns.tail(10).std() * np.sqrt(252)
            
            if len(closes) >= 6:
                momentum_5d = (closes.iloc[-1] / closes.iloc[-6]) - 1
            else:
                momentum_5d = 0
            
            # Safe conversions
            ma_5 = float(ma_5) if not pd.isna(ma_5) else float(closes.iloc[-1])
            ma_15 = float(ma_15) if not pd.isna(ma_15) else float(closes.iloc[-1])
            volatility_10d = float(volatility_10d) if not pd.isna(volatility_10d) else 0.15
            momentum_5d = float(momentum_5d) if not pd.isna(momentum_5d) else 0
            
            is_uptrend = ma_5 > ma_15
            is_high_vol = volatility_10d > 0.12
            is_strong_momentum = abs(momentum_5d) > 0.005
            
            if is_uptrend and is_strong_momentum:
                return 'trend_up'
            elif not is_uptrend and is_strong_momentum:
                return 'trend_down'
            elif is_high_vol:
                return 'volatile'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def detect_swing_regime(self, current_date):
        """Swing regime detection"""
        try:
            spy_data = self.market_data.get('SPY')
            if spy_data is None:
                return 'moderate_trend'
            
            historical_spy = spy_data[spy_data.index <= current_date]
            if len(historical_spy) < 15:
                return 'moderate_trend'
            
            closes = historical_spy['Close']
            ma_7 = closes.rolling(7).mean().iloc[-1]
            ma_20 = closes.rolling(20).mean().iloc[-1]
            
            returns = closes.pct_change().dropna()
            volatility_15d = returns.tail(15).std() * np.sqrt(252)
            
            if len(closes) >= 8:
                momentum_7d = (closes.iloc[-1] / closes.iloc[-8]) - 1
            else:
                momentum_7d = 0
            
            # Safe conversions
            ma_7 = float(ma_7) if not pd.isna(ma_7) else float(closes.iloc[-1])
            ma_20 = float(ma_20) if not pd.isna(ma_20) else float(closes.iloc[-1])
            volatility_15d = float(volatility_15d) if not pd.isna(volatility_15d) else 0.15
            momentum_7d = float(momentum_7d) if not pd.isna(momentum_7d) else 0
            
            is_uptrend = ma_7 > ma_20
            is_strong_momentum = abs(momentum_7d) > 0.01
            is_high_vol = volatility_15d > 0.15
            
            if is_uptrend and is_strong_momentum:
                return 'strong_momentum'
            elif is_high_vol and is_strong_momentum:
                return 'high_volatility'
            elif is_uptrend or is_strong_momentum:
                return 'moderate_trend'
            else:
                return 'consolidation'
                
        except Exception:
            return 'moderate_trend'
    
    def detect_balanced_regime(self, current_date):
        """Balanced regime detection"""
        try:
            spy_data = self.market_data.get('SPY')
            vix_data = self.market_data.get('VIX')
            
            if spy_data is None:
                return 'sideways'
            
            historical_spy = spy_data[spy_data.index <= current_date]
            if len(historical_spy) < 50:
                return 'sideways'
            
            closes = historical_spy['Close']
            ma_10 = closes.rolling(10).mean().iloc[-1]
            ma_20 = closes.rolling(20).mean().iloc[-1]
            ma_50 = closes.rolling(50).mean().iloc[-1]
            
            current_price = closes.iloc[-1]
            
            returns = closes.pct_change().dropna()
            vol_20d = returns.tail(20).std() * np.sqrt(252)
            
            momentum_30d = (current_price / closes.iloc[-31]) - 1 if len(closes) >= 31 else 0
            
            # VIX level
            vix_level = 20
            if vix_data is not None:
                historical_vix = vix_data[vix_data.index <= current_date]
                if len(historical_vix) > 0:
                    vix_level = float(historical_vix['Close'].iloc[-1])
            
            # Drawdown
            rolling_max_60 = closes.rolling(60).max().iloc[-1]
            current_drawdown = (current_price / rolling_max_60) - 1
            
            # Safe conversions
            current_price = float(current_price)
            ma_10 = float(ma_10) if not pd.isna(ma_10) else current_price
            ma_20 = float(ma_20) if not pd.isna(ma_20) else current_price
            ma_50 = float(ma_50) if not pd.isna(ma_50) else current_price
            vol_20d = float(vol_20d) if not pd.isna(vol_20d) else 0.15
            momentum_30d = float(momentum_30d) if not pd.isna(momentum_30d) else 0
            current_drawdown = float(current_drawdown) if not pd.isna(current_drawdown) else 0
            
            # Regime classification
            if (current_price > ma_10 > ma_20 > ma_50 and 
                momentum_30d > 0.05 and vol_20d < 0.18 and 
                vix_level < 20 and current_drawdown > -0.03):
                return 'strong_bull'
            elif (current_drawdown < -0.15 or 
                  (current_price < ma_50 and momentum_30d < -0.10) or 
                  vix_level > 35):
                return 'bear_market'
            elif (current_drawdown < -0.05 or vol_20d > 0.22 or 
                  vix_level > 25 or momentum_30d < -0.03):
                return 'correction'
            elif (current_price > ma_20 and momentum_30d > 0.01 and vol_20d < 0.25):
                return 'moderate_bull'
            else:
                return 'sideways'
                
        except Exception:
            return 'sideways'
    
    def calculate_optimized_signals(self, date, current_regime):
        """Calculate optimized signals"""
        signals = {}
        
        for symbol, prices in self.unified_data.items():
            try:
                historical_data = prices[prices.index <= date]
                if len(historical_data) < 20:
                    continue
                
                # EMA signals
                ema_short = historical_data.ewm(span=5).mean()
                ema_long = historical_data.ewm(span=15).mean()
                
                current_ema_short = float(ema_short.iloc[-1])
                current_ema_long = float(ema_long.iloc[-1])
                ema_signal = 1 if current_ema_short > current_ema_long else 0
                
                # RSI
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/7).mean()
                avg_losses = losses.ewm(alpha=1/7).mean()
                
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                current_rsi = float(rsi.iloc[-1])
                rsi_signal = 1 if current_rsi < 70 else 0
                
                # Momentum
                current_price = float(historical_data.iloc[-1])
                if len(historical_data) >= 6:
                    momentum = (current_price / float(historical_data.iloc[-6])) - 1
                    momentum_signal = 1 if momentum > 0.008 else 0
                else:
                    momentum_signal = 0
                
                # Tech boost
                tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'NFLX', 'AMD', 'CRM']
                tech_boost = 1.2 if symbol in tech_symbols else 1.0
                
                score = (0.3 * ema_signal + 0.25 * rsi_signal + 0.25 * momentum_signal + 0.2) * tech_boost
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def calculate_balanced_signals(self, date, current_regime):
        """Calculate balanced signals"""
        signals = {}
        
        for symbol, prices in self.unified_data.items():
            try:
                historical_data = prices[prices.index <= date]
                if len(historical_data) < 30:
                    continue
                
                # EMA signals
                ema_short = historical_data.ewm(span=8).mean()
                ema_long = historical_data.ewm(span=21).mean()
                
                current_ema_short = float(ema_short.iloc[-1])
                current_ema_long = float(ema_long.iloc[-1])
                ema_signal = 1 if current_ema_short > current_ema_long else 0
                
                # RSI
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/14).mean()
                avg_losses = losses.ewm(alpha=1/14).mean()
                
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                current_rsi = float(rsi.iloc[-1])
                rsi_signal = 1 if 30 < current_rsi < 70 else 0
                
                # Momentum
                current_price = float(historical_data.iloc[-1])
                if len(historical_data) >= 10:
                    momentum = (current_price / float(historical_data.iloc[-10])) - 1
                    momentum_signal = 1 if momentum > 0.01 else 0
                else:
                    momentum_signal = 0
                
                score = 0.4 * ema_signal + 0.3 * rsi_signal + 0.3 * momentum_signal
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_optimized_rebalancing(self, portfolio, date, signals, current_regime, regimes):
        """Execute optimized rebalancing"""
        regime_config = regimes[current_regime]
        
        # Select top signals
        qualified_signals = [
            (symbol, sig['score']) for symbol, sig in signals.items() 
            if sig['score'] >= regime_config['score_threshold']
        ]
        
        qualified_signals.sort(key=lambda x: x[1], reverse=True)
        max_positions = regime_config['max_positions']
        selected_symbols = [s for s, _ in qualified_signals[:max_positions]]
        
        # Calculate target positions
        target_positions = {}
        if selected_symbols:
            investable_capital = regime_config['allocation_factor']
            weight_per_stock = investable_capital / len(selected_symbols)
            weight_per_stock = min(weight_per_stock, 0.15)  # Max 15% per position
            
            for symbol in selected_symbols:
                target_positions[symbol] = weight_per_stock
        
        # Execute trades
        self.execute_rebalancing(portfolio, date, target_positions)
    
    def execute_balanced_rebalancing(self, portfolio, date, signals, current_regime, regimes):
        """Execute balanced rebalancing"""
        regime_config = regimes[current_regime]
        
        qualified_signals = [
            (symbol, sig['score']) for symbol, sig in signals.items() 
            if sig['score'] >= regime_config['score_threshold']
        ]
        
        qualified_signals.sort(key=lambda x: x[1], reverse=True)
        max_positions = regime_config['max_positions']
        selected_symbols = [s for s, _ in qualified_signals[:max_positions]]
        
        target_positions = {}
        if selected_symbols:
            investable_capital = regime_config['allocation_factor']
            weight_per_stock = investable_capital / len(selected_symbols)
            weight_per_stock = min(weight_per_stock, 0.10)  # Max 10% per position
            
            for symbol in selected_symbols:
                target_positions[symbol] = weight_per_stock
        
        self.execute_rebalancing(portfolio, date, target_positions)
    
    def check_swing_exits(self, portfolio, date, current_regime, day_index, regimes):
        """Check swing exits"""
        regime_config = regimes[current_regime]
        exits_made = 0
        positions_to_exit = []
        
        for symbol in list(portfolio['positions'].keys()):
            if symbol not in self.unified_data:
                continue
                
            try:
                prices = self.unified_data[symbol]
                available_prices = prices[prices.index <= date]
                if len(available_prices) == 0:
                    continue
                
                current_price = float(available_prices.iloc[-1])
                entry_price = portfolio['position_entry_prices'].get(symbol, current_price)
                entry_date_index = portfolio['position_entry_dates'].get(symbol, day_index)
                
                pnl_pct = (current_price / entry_price) - 1
                hold_days = day_index - entry_date_index
                
                should_exit = False
                
                # Take profit
                if pnl_pct >= regime_config['take_profit']:
                    should_exit = True
                # Stop loss
                elif pnl_pct <= -regime_config['stop_loss']:
                    should_exit = True
                # Max hold period
                elif hold_days >= 7:  # Max 7 days
                    should_exit = True
                # Min hold period
                elif hold_days < 2:
                    should_exit = False
                
                if should_exit:
                    positions_to_exit.append(symbol)
                    
            except Exception:
                continue
        
        # Execute exits
        for symbol in positions_to_exit:
            if symbol in portfolio['positions']:
                shares = portfolio['positions'][symbol]
                try:
                    prices = self.unified_data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        price = float(available_prices.iloc[-1])
                        proceeds = shares * price
                        portfolio['cash'] += proceeds
                    
                    del portfolio['positions'][symbol]
                    if symbol in portfolio['position_entry_prices']:
                        del portfolio['position_entry_prices'][symbol]
                    if symbol in portfolio['position_entry_dates']:
                        del portfolio['position_entry_dates'][symbol]
                    exits_made += 1
                except:
                    pass
        
        return exits_made
    
    def consider_swing_entries(self, portfolio, date, current_regime, regimes):
        """Consider swing entries"""
        regime_config = regimes[current_regime]
        entries_made = 0
        
        current_positions = len(portfolio['positions'])
        max_positions = regime_config['max_positions']
        
        if current_positions >= max_positions:
            return entries_made
        
        signals = self.calculate_balanced_signals(date, current_regime)
        
        qualified_signals = [
            (symbol, sig['score']) for symbol, sig in signals.items() 
            if sig['score'] >= regime_config['score_threshold']
            and symbol not in portfolio['positions']
        ]
        
        if not qualified_signals:
            return entries_made
        
        qualified_signals.sort(key=lambda x: x[1], reverse=True)
        max_new_entries = min(3, max_positions - current_positions)
        
        for symbol, score in qualified_signals[:max_new_entries]:
            try:
                prices = self.unified_data[symbol]
                available_prices = prices[prices.index <= date]
                if len(available_prices) == 0:
                    continue
                
                price = float(available_prices.iloc[-1])
                position_size = 0.12  # 12% per position
                target_value = portfolio['value'] * position_size
                
                if portfolio['cash'] >= target_value:
                    shares = target_value / price
                    cost = shares * price
                    
                    portfolio['cash'] -= cost
                    portfolio['positions'][symbol] = shares
                    portfolio['position_entry_prices'][symbol] = price
                    portfolio['position_entry_dates'][symbol] = day_index
                    
                    entries_made += 1
            except:
                continue
        
        return entries_made
    
    def execute_rebalancing(self, portfolio, date, target_positions):
        """Execute rebalancing trades"""
        current_value = portfolio['value']
        
        # Sell unwanted positions
        positions_to_sell = [s for s in portfolio['positions'].keys() if s not in target_positions]
        
        for symbol in positions_to_sell:
            if symbol in self.unified_data:
                shares = portfolio['positions'][symbol]
                if shares > 0:
                    try:
                        prices = self.unified_data[symbol]
                        available_prices = prices[prices.index <= date]
                        if len(available_prices) > 0:
                            price = float(available_prices.iloc[-1])
                            proceeds = float(shares) * price
                            portfolio['cash'] += proceeds
                    except:
                        pass
                
                del portfolio['positions'][symbol]
        
        # Buy/adjust positions
        for symbol, target_weight in target_positions.items():
            if symbol in self.unified_data:
                try:
                    prices = self.unified_data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        price = float(available_prices.iloc[-1])
                        
                        target_value = current_value * target_weight
                        target_shares = target_value / price
                        
                        current_shares = portfolio['positions'].get(symbol, 0)
                        shares_diff = target_shares - current_shares
                        
                        if abs(shares_diff * price) > current_value * 0.002:  # 0.2% threshold
                            cost = float(shares_diff) * price
                            
                            if shares_diff > 0 and portfolio['cash'] >= cost:
                                portfolio['cash'] -= cost
                                portfolio['positions'][symbol] = target_shares
                            elif shares_diff < 0:
                                portfolio['cash'] -= cost
                                portfolio['positions'][symbol] = target_shares
                except:
                    continue
    
    def update_portfolio_value(self, portfolio, date):
        """Update portfolio value"""
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in self.unified_data and shares > 0:
                try:
                    prices = self.unified_data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        current_price = float(available_prices.iloc[-1])
                        position_value = float(shares) * current_price
                        total_value += position_value
                except:
                    continue
        
        portfolio['value'] = total_value
        return total_value
    
    def calculate_strategy_metrics(self, history, strategy_name):
        """Calculate strategy performance metrics"""
        try:
            history_df = pd.DataFrame(history)
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df.set_index('date', inplace=True)
            
            values = history_df['portfolio_value']
            daily_returns = values.pct_change().dropna()
            
            total_return = (values.iloc[-1] / values.iloc[0]) - 1
            years = len(daily_returns) / 252
            annual_return = (1 + total_return) ** (1/years) - 1
            
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
            
            cumulative = values / values.iloc[0]
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            win_rate = (daily_returns > 0).mean()
            
            print(f"    ✅ {strategy_name}: {annual_return:.1%} annual, {max_drawdown:.1%} max DD")
            
            return {
                'strategy_name': strategy_name,
                'history': history,
                'performance': {
                    'total_return': float(total_return),
                    'annual_return': float(annual_return),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'calmar_ratio': float(calmar_ratio),
                    'max_drawdown': float(max_drawdown),
                    'win_rate': float(win_rate),
                    'final_value': float(values.iloc[-1]),
                    'years_simulated': float(years)
                }
            }
            
        except Exception as e:
            print(f"    ❌ {strategy_name} metrics calculation failed: {e}")
            return None
    
    def download_benchmarks(self):
        """Download benchmark performance"""
        for symbol in ['SPY', 'QQQ', 'IWM']:
            if symbol in self.unified_data:
                prices = self.unified_data[symbol]
                total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                years = len(prices) / 252
                annual_return = (1 + total_return) ** (1/years) - 1
                
                daily_returns = prices.pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                
                cumulative = prices / prices.iloc[0]
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = drawdowns.min()
                
                self.benchmarks[symbol] = {
                    'annual_return': float(annual_return),
                    'volatility': float(volatility),
                    'max_drawdown': float(max_drawdown),
                    'sharpe_ratio': float((annual_return - 0.02) / volatility) if volatility > 0 else 0
                }
                
                print(f"  ✅ {symbol}: {annual_return:.1%} annual")
    
    def generate_comparison_analysis(self):
        """Generate comprehensive comparison analysis"""
        comparison_data = []
        
        # Add strategy results
        for strategy_name, result in self.strategy_results.items():
            if result and result['performance']:
                perf = result['performance']
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Type': 'Trading Strategy' if 'Buy & Hold' not in strategy_name else 'Benchmark',
                    'Annual Return': perf['annual_return'],
                    'Volatility': perf['volatility'],
                    'Sharpe Ratio': perf['sharpe_ratio'],
                    'Max Drawdown': perf['max_drawdown'],
                    'Calmar Ratio': perf['calmar_ratio'],
                    'Win Rate': perf.get('win_rate', 0),
                    'Final Value': perf['final_value']
                })
        
        # Add benchmarks
        for benchmark, perf in self.benchmarks.items():
            if benchmark not in [s['Strategy'] for s in comparison_data]:
                comparison_data.append({
                    'Strategy': f'{benchmark} Index',
                    'Type': 'Market Index',
                    'Annual Return': perf['annual_return'],
                    'Volatility': perf['volatility'],
                    'Sharpe Ratio': perf['sharpe_ratio'],
                    'Max Drawdown': perf['max_drawdown'],
                    'Calmar Ratio': perf['annual_return'] / abs(perf['max_drawdown']) if perf['max_drawdown'] != 0 else 0,
                    'Win Rate': 0,  # N/A for indices
                    'Final Value': 100000 * (1 + (perf['annual_return'] * 5))  # Approximate
                })
        
        return {
            'comparison_table': comparison_data,
            'best_performance': self.find_best_performers(comparison_data),
            'strategy_rankings': self.rank_strategies(comparison_data)
        }
    
    def find_best_performers(self, comparison_data):
        """Find best performers in each category"""
        trading_strategies = [d for d in comparison_data if d['Type'] == 'Trading Strategy']
        
        if not trading_strategies:
            return {}
        
        best_annual = max(trading_strategies, key=lambda x: x['Annual Return'])
        best_sharpe = max(trading_strategies, key=lambda x: x['Sharpe Ratio'])
        best_drawdown = max(trading_strategies, key=lambda x: x['Max Drawdown'])  # Closest to 0
        
        return {
            'best_annual_return': {
                'strategy': best_annual['Strategy'],
                'value': best_annual['Annual Return']
            },
            'best_sharpe_ratio': {
                'strategy': best_sharpe['Strategy'],
                'value': best_sharpe['Sharpe Ratio']
            },
            'best_risk_control': {
                'strategy': best_drawdown['Strategy'],
                'value': best_drawdown['Max Drawdown']
            }
        }
    
    def rank_strategies(self, comparison_data):
        """Rank strategies by composite score"""
        trading_strategies = [d for d in comparison_data if d['Type'] == 'Trading Strategy']
        
        for strategy in trading_strategies:
            # Composite score: 50% return, 30% risk-adjusted return, 20% risk control
            return_score = strategy['Annual Return'] * 100  # 0-100 scale
            sharpe_score = min(strategy['Sharpe Ratio'] * 50, 100)  # Cap at 100
            risk_score = max(0, (strategy['Max Drawdown'] + 0.5) * 200)  # -50% to 0% → 0 to 100
            
            composite_score = (0.5 * return_score + 0.3 * sharpe_score + 0.2 * risk_score)
            strategy['Composite Score'] = composite_score
        
        # Sort by composite score
        trading_strategies.sort(key=lambda x: x['Composite Score'], reverse=True)
        
        return trading_strategies
    
    def print_comprehensive_summary(self, comparison_results):
        """Print comprehensive comparison summary"""
        print("\\n" + "="*80)
        print("🏆 COMPREHENSIVE MULTI-STRATEGY COMPARISON RESULTS")
        print("="*80)
        
        print(f"📊 Unified Dataset: {len(self.unified_data)} symbols")
        print(f"📅 Unified Period: {self.start_date} to {self.end_date} (5 years)")
        print(f"💰 Unified Capital: ${self.initial_capital:,}")
        
        print(f"\\n📋 STRATEGY PERFORMANCE COMPARISON:")
        comparison_table = comparison_results['comparison_table']
        
        # Print header
        print(f"{'Strategy':<20} {'Type':<15} {'Annual':<8} {'Volatility':<10} {'Sharpe':<7} {'Max DD':<8} {'Final Value':<12}")
        print("-" * 80)
        
        # Sort by annual return for display
        sorted_strategies = sorted(comparison_table, key=lambda x: x['Annual Return'], reverse=True)
        
        for strategy in sorted_strategies:
            strategy_type = strategy['Type']
            color_marker = "🚀" if strategy_type == "Trading Strategy" else "📊"
            
            print(f"{color_marker} {strategy['Strategy']:<18} {strategy_type:<15} "
                  f"{strategy['Annual Return']:>6.1%} {strategy['Volatility']:>8.1%} "
                  f"{strategy['Sharpe Ratio']:>6.2f} {strategy['Max Drawdown']:>6.1%} "
                  f"${strategy['Final Value']:>10,.0f}")
        
        # Best performers
        best_performers = comparison_results['best_performance']
        print(f"\\n🏆 BEST PERFORMERS:")
        print(f"  🚀 Best Annual Return:    {best_performers['best_annual_return']['strategy']:<20} ({best_performers['best_annual_return']['value']:.1%})")
        print(f"  📈 Best Risk-Adjusted:    {best_performers['best_sharpe_ratio']['strategy']:<20} (Sharpe: {best_performers['best_sharpe_ratio']['value']:.2f})")
        print(f"  🛡️ Best Risk Control:     {best_performers['best_risk_control']['strategy']:<20} (DD: {best_performers['best_risk_control']['value']:.1%})")
        
        # Strategy rankings
        rankings = comparison_results['strategy_rankings']
        print(f"\\n🏅 STRATEGY RANKINGS (Composite Score):")
        for i, strategy in enumerate(rankings, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"  {medal} {strategy['Strategy']:<20} Score: {strategy['Composite Score']:>5.1f}")
        
        # Analysis insights
        print(f"\\n💡 KEY INSIGHTS:")
        
        trading_strategies = [s for s in comparison_table if s['Type'] == 'Trading Strategy']
        if trading_strategies:
            best_strategy = max(trading_strategies, key=lambda x: x['Annual Return'])
            spy_benchmark = next((s for s in comparison_table if 'SPY' in s['Strategy']), None)
            
            if spy_benchmark:
                outperformance = best_strategy['Annual Return'] - spy_benchmark['Annual Return']
                print(f"  📈 Best strategy outperforms SPY by {outperformance:.1%}")
            
            high_performers = [s for s in trading_strategies if s['Annual Return'] > 0.15]
            print(f"  🎯 {len(high_performers)}/{len(trading_strategies)} strategies achieve >15% annual")
            
            low_risk = [s for s in trading_strategies if s['Max Drawdown'] > -0.25]
            print(f"  🛡️ {len(low_risk)}/{len(trading_strategies)} strategies control drawdown <25%")
        
        print(f"\\n✅ COMPARISON COMPLETE: All strategies tested on identical unified dataset")


def main():
    """Execute multi-strategy comparison"""
    print("🔄 MULTI-STRATEGY COMPARISON SYSTEM")
    print("Fair comparison on unified dataset")
    print("="*80)
    
    comparison_system = MultiStrategyComparison()
    results = comparison_system.run_comprehensive_comparison()
    
    return 0


if __name__ == "__main__":
    exit_code = main()