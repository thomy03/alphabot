#!/usr/bin/env python3
"""
Goldilocks Plus System - Final Optimization
Target: 21-23% annual return while maintaining same risk level (-27% max DD)
Smart enhancements without excessive risk taking
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GoldilocksPlusSystem:
    """
    Enhanced Goldilocks with smart optimizations
    Target: 21-23% annual return with controlled risk
    """
    
    def __init__(self):
        # ENHANCED UNIVERSE (adding high-quality momentum stocks)
        self.plus_universe = [
            # Core tech mega caps (proven winners)
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN',
            
            # High-momentum growth tech
            'CRM', 'ADBE', 'AMD', 'QCOM', 'ORCL', 'TSLA', 'NFLX',
            
            # Quality large caps with momentum
            'V', 'MA', 'UNH', 'HD', 'JPM', 'COST', 'PG',
            
            # Tech ETFs (core holdings)
            'QQQ', 'XLK', 'VGT', 'SOXX',  # Added SOXX for semis
            
            # Market diversification
            'SPY', 'IWM'
        ]
        
        # OPTIMIZED CONFIGURATION
        self.start_date = "2015-01-01"
        self.end_date = "2024-12-01"
        self.initial_capital = 100000
        
        # ENHANCED PARAMETERS (small optimizations)
        self.max_single_position = 0.11      # Slightly increased from 10% to 11%
        self.max_portfolio_allocation = 0.92  # Increased from 90% to 92%
        self.cash_buffer_normal = 0.08       # Reduced from 10% to 8%
        self.cash_buffer_stress = 0.18       # Reduced from 20% to 18%
        
        # ENHANCED ALLOCATION
        self.base_trend_allocation = 0.67    # Slightly increased from 65% to 67%
        self.base_swing_allocation = 0.25    # Same as before
        
        # OPTIMIZED RISK CONTROLS (same risk profile)
        self.drawdown_alert = -0.20          # Same as Goldilocks
        self.drawdown_defensive = -0.30      # Same as Goldilocks
        self.volatility_target = 0.20        # Same target
        
        # NEW: MOMENTUM ENHANCEMENT
        self.momentum_boost_threshold = 0.05  # 5% momentum for boost
        self.quality_score_weight = 0.15      # Weight for quality metrics
        
        print(f"⚖️ GOLDILOCKS PLUS SYSTEM")
        print(f"📊 ENHANCED UNIVERSE: {len(self.plus_universe)} optimized symbols")
        print(f"🎯 TARGET: 21-23% annual, ~27% max drawdown")
        print(f"💰 MAX POSITION: {self.max_single_position:.0%}")
        print(f"🚀 ALLOCATION: {self.max_portfolio_allocation:.0%}")
        
        # Storage
        self.data = {}
        self.market_data = {}
        
    def run_plus_system(self):
        """Run the enhanced Goldilocks Plus system"""
        print("\n" + "="*80)
        print("⚖️ GOLDILOCKS PLUS SYSTEM - ENHANCED EXECUTION")
        print("="*80)
        
        # Download data
        print("\n📊 Step 1: Downloading enhanced universe data...")
        self.download_plus_data()
        
        # Download market data
        print("\n📈 Step 2: Downloading market indicators...")
        self.download_market_indicators()
        
        # Run enhanced strategy
        print("\n⚖️ Step 3: Executing enhanced strategy...")
        portfolio_history = self.execute_plus_strategy()
        
        # Calculate performance
        print("\n📊 Step 4: Calculating enhanced performance...")
        performance = self.calculate_plus_performance(portfolio_history)
        
        # Generate report
        print("\n📋 Step 5: Generating Goldilocks Plus report...")
        self.generate_plus_report(performance)
        
        return performance
    
    def download_plus_data(self):
        """Download data for Plus universe"""
        failed_downloads = []
        
        for i, symbol in enumerate(self.plus_universe, 1):
            try:
                print(f"  ⚖️ ({i:2d}/{len(self.plus_universe)}) {symbol:8s}...", end=" ")
                
                ticker_data = yf.download(
                    symbol, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False
                )
                
                if len(ticker_data) > 2000:
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    
                    self.data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ {len(ticker_data)} days (insufficient)")
                    failed_downloads.append(symbol)
                    
            except Exception as e:
                print(f"❌ Error")
                failed_downloads.append(symbol)
        
        print(f"  ⚖️ PLUS DATA: {len(self.data)} symbols loaded")
        if failed_downloads:
            print(f"  ⚠️ Failed: {failed_downloads}")
    
    def download_market_indicators(self):
        """Download market indicators"""
        indices = ['SPY', 'QQQ', '^VIX']
        
        for symbol in indices:
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                market_symbol = symbol.replace('^', '')
                self.market_data[market_symbol] = data
                print(f"  ✅ {market_symbol}: {len(data)} days")
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
    
    def execute_plus_strategy(self):
        """Execute the enhanced Plus strategy"""
        portfolio = {
            'cash': float(self.initial_capital),
            'trend_positions': {},
            'swing_positions': {},
            'value': float(self.initial_capital),
            'peak_value': float(self.initial_capital)
        }
        
        trading_dates = pd.bdate_range(start=self.start_date, end=self.end_date).tolist()
        history = []
        
        print(f"    ⚖️ Plus execution: {len(trading_dates)} days")
        
        for i, date in enumerate(trading_dates):
            if i % 500 == 0:
                print(f"      📅 Progress: {(i/len(trading_dates)*100):5.1f}% - {date.strftime('%Y-%m-%d')}")
            
            # Update portfolio value
            portfolio_value = self.update_plus_portfolio_value(portfolio, date)
            
            # Update peak and calculate drawdown
            if portfolio_value > portfolio['peak_value']:
                portfolio['peak_value'] = portfolio_value
            
            current_drawdown = (portfolio_value / portfolio['peak_value']) - 1
            
            # Enhanced regime detection
            regime = self.detect_enhanced_regime(date)
            
            # Enhanced risk adjustment
            risk_factor = self.calculate_enhanced_risk_factor(current_drawdown, regime)
            
            # Enhanced allocation
            trend_alloc, swing_alloc = self.calculate_enhanced_allocation(regime, risk_factor)
            
            # Enhanced trend rebalancing (every 12 days - slightly more frequent)
            if i % 12 == 0:
                self.execute_enhanced_trend_rebalancing(portfolio, date, trend_alloc)
            
            # Enhanced swing rebalancing (every 3 days - more frequent)
            if i % 3 == 0:
                self.execute_enhanced_swing_rebalancing(portfolio, date, swing_alloc)
            
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'current_drawdown': current_drawdown,
                'regime': regime,
                'risk_factor': risk_factor,
                'trend_allocation': trend_alloc,
                'swing_allocation': swing_alloc
            })
        
        return history
    
    def detect_enhanced_regime(self, current_date):
        """Enhanced regime detection with momentum"""
        try:
            spy_data = self.market_data.get('SPY')
            vix_data = self.market_data.get('VIX')
            qqq_data = self.market_data.get('QQQ')
            
            if spy_data is None:
                return 'neutral'
            
            historical_spy = spy_data[spy_data.index <= current_date]
            if len(historical_spy) < 50:
                return 'neutral'
            
            closes = historical_spy['Close']
            
            # Enhanced moving averages
            ma_10 = closes.rolling(10).mean().iloc[-1]
            ma_20 = closes.rolling(20).mean().iloc[-1]
            ma_50 = closes.rolling(50).mean().iloc[-1]
            ma_200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else ma_50
            
            current_price = closes.iloc[-1]
            
            # Enhanced VIX analysis
            current_vix = 20
            if vix_data is not None:
                historical_vix = vix_data[vix_data.index <= current_date]
                if len(historical_vix) > 0:
                    current_vix = historical_vix['Close'].iloc[-1]
            
            # Multi-timeframe momentum
            momentum_5d = (current_price / closes.iloc[-5]) - 1 if len(closes) >= 5 else 0
            momentum_20d = (current_price / closes.iloc[-20]) - 1 if len(closes) >= 20 else 0
            
            # NASDAQ momentum (tech leadership)
            nasdaq_momentum = 0
            if qqq_data is not None:
                historical_qqq = qqq_data[qqq_data.index <= current_date]
                if len(historical_qqq) >= 10:
                    qqq_closes = historical_qqq['Close']
                    nasdaq_momentum = (qqq_closes.iloc[-1] / qqq_closes.iloc[-10]) - 1
            
            # Enhanced regime classification
            if current_vix > 35:
                return 'crisis'
            elif current_vix > 28:
                return 'high_stress'
            elif (current_price > ma_10 > ma_20 > ma_50 > ma_200 and 
                  momentum_5d > 0.02 and nasdaq_momentum > 0.03):
                return 'super_bull'  # NEW: Strong tech momentum
            elif current_price > ma_20 > ma_50 > ma_200 and momentum_20d > 0.02:
                return 'strong_bull'
            elif current_price > ma_20 > ma_50:
                return 'bull'
            elif current_price < ma_20 < ma_50 and momentum_5d < -0.02:
                return 'bear'
            elif current_vix > 22:
                return 'volatile'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def calculate_enhanced_risk_factor(self, current_drawdown, regime):
        """Enhanced risk factor with momentum consideration"""
        # Base drawdown adjustment
        if current_drawdown > self.drawdown_alert:
            drawdown_factor = 1.0
        elif current_drawdown > self.drawdown_defensive:
            range_size = self.drawdown_defensive - self.drawdown_alert
            position_in_range = (current_drawdown - self.drawdown_alert) / range_size
            drawdown_factor = 0.65 + 0.35 * position_in_range  # Slightly more aggressive
        else:
            drawdown_factor = 0.65  # Slightly higher floor
        
        # Enhanced regime factors
        regime_factors = {
            'super_bull': 1.05,     # NEW: Slight boost for super bull
            'strong_bull': 1.0,
            'bull': 0.9,
            'neutral': 0.8,
            'volatile': 0.7,
            'bear': 0.6,
            'high_stress': 0.5,
            'crisis': 0.4
        }
        
        regime_factor = regime_factors.get(regime, 0.8)
        
        # Enhanced combination
        combined_factor = (drawdown_factor + regime_factor) / 2
        return max(combined_factor, 0.45)  # Slightly higher minimum
    
    def calculate_enhanced_allocation(self, regime, risk_factor):
        """Enhanced allocation with momentum boost"""
        # Base allocations with slight enhancement
        trend_alloc = self.base_trend_allocation * risk_factor
        swing_alloc = self.base_swing_allocation * risk_factor
        
        # Enhanced regime-specific adjustments
        if regime == 'super_bull':
            trend_alloc *= 1.15  # Boost for super bull
            swing_alloc *= 1.05
        elif regime == 'strong_bull':
            trend_alloc *= 1.1   # Slight boost
        elif regime in ['volatile', 'high_stress']:
            swing_alloc *= 1.25  # More swing in volatility
            trend_alloc *= 0.9
        elif regime in ['bear', 'crisis']:
            trend_alloc *= 0.8
            swing_alloc *= 0.8
        
        # Ensure optimized total allocation
        total_alloc = trend_alloc + swing_alloc
        max_alloc = self.max_portfolio_allocation
        
        if total_alloc > max_alloc:
            scale_factor = max_alloc / total_alloc
            trend_alloc *= scale_factor
            swing_alloc *= scale_factor
        
        return trend_alloc, swing_alloc
    
    def execute_enhanced_trend_rebalancing(self, portfolio, date, allocation):
        """Enhanced trend rebalancing"""
        signals = self.calculate_enhanced_trend_signals(date)
        
        # Slightly more aggressive signal filtering
        qualified_signals = sorted(
            [(s, sig['score']) for s, sig in signals.items() if sig['score'] > 0.62],
            key=lambda x: x[1], 
            reverse=True
        )[:9]  # Increased from 8 to 9 positions
        
        if not qualified_signals:
            return
        
        # Enhanced position sizing
        target_positions = {}
        
        # Enhanced tech preferences
        tech_mega_caps = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'QQQ', 'XLK']
        semiconductor_focus = ['NVDA', 'AMD', 'QCOM', 'SOXX']  # Semiconductor boost
        
        for symbol, score in qualified_signals:
            base_weight = allocation / len(qualified_signals)
            
            # Enhanced tech boost
            if symbol in tech_mega_caps:
                weight = base_weight * 1.18  # Increased from 1.15
            elif symbol in semiconductor_focus:
                weight = base_weight * 1.22  # NEW: Semiconductor boost
            else:
                weight = base_weight
            
            # Apply enhanced position limit
            weight = min(weight, self.max_single_position)
            target_positions[symbol] = weight
        
        self.execute_enhanced_position_changes(portfolio, 'trend_positions', date, target_positions)
    
    def execute_enhanced_swing_rebalancing(self, portfolio, date, allocation):
        """Enhanced swing rebalancing"""
        signals = self.calculate_enhanced_swing_signals(date)
        
        # More aggressive swing selection
        qualified_signals = sorted(
            [(s, sig['score']) for s, sig in signals.items() if sig['score'] > 0.52],
            key=lambda x: x[1], 
            reverse=True
        )[:6]  # Increased from 5 to 6 positions
        
        if not qualified_signals:
            return
        
        target_positions = {}
        for symbol, score in qualified_signals:
            weight = allocation / len(qualified_signals)
            weight = min(weight, self.max_single_position * 0.85)  # Slightly larger swing positions
            target_positions[symbol] = weight
        
        self.execute_enhanced_position_changes(portfolio, 'swing_positions', date, target_positions)
    
    def calculate_enhanced_trend_signals(self, date):
        """Enhanced trend signals with quality metrics"""
        signals = {}
        
        for symbol, prices in self.data.items():
            try:
                historical_data = prices[prices.index <= date]
                if len(historical_data) < 100:
                    continue
                
                # Enhanced trend analysis
                ma_20 = historical_data.rolling(20).mean().iloc[-1]
                ma_50 = historical_data.rolling(50).mean().iloc[-1]
                ma_100 = historical_data.rolling(100).mean().iloc[-1]
                ma_200 = historical_data.rolling(200).mean().iloc[-1] if len(historical_data) >= 200 else ma_100
                
                current_price = float(historical_data.iloc[-1])
                
                # Enhanced trend scoring
                if current_price > ma_20 > ma_50 > ma_100 > ma_200:
                    trend_score = 1.0
                elif current_price > ma_20 > ma_50 > ma_100:
                    trend_score = 0.85
                elif current_price > ma_20 > ma_50:
                    trend_score = 0.7
                elif current_price > ma_50:
                    trend_score = 0.5
                else:
                    trend_score = 0.0
                
                # Enhanced momentum analysis
                momentum_scores = []
                for period in [10, 20, 30]:
                    if len(historical_data) >= period:
                        momentum = (current_price / float(historical_data.iloc[-period])) - 1
                        if momentum > self.momentum_boost_threshold:
                            momentum_scores.append(min(momentum * 3, 0.5))
                        else:
                            momentum_scores.append(max(momentum * 2, -0.2))
                
                avg_momentum = sum(momentum_scores) / len(momentum_scores) if momentum_scores else 0
                
                # Quality score (consistency)
                recent_returns = historical_data.pct_change().tail(20)
                volatility = recent_returns.std()
                quality_score = max(0, 0.2 - volatility * 50)  # Reward low volatility
                
                # Enhanced final score
                score = (0.6 * trend_score + 
                        0.25 * max(avg_momentum, 0) + 
                        self.quality_score_weight * quality_score)
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def calculate_enhanced_swing_signals(self, date):
        """Enhanced swing signals with momentum focus"""
        signals = {}
        
        for symbol, prices in self.data.items():
            try:
                historical_data = prices[prices.index <= date]
                if len(historical_data) < 30:
                    continue
                
                # Enhanced swing indicators
                ema_8 = historical_data.ewm(span=8).mean().iloc[-1]
                ema_21 = historical_data.ewm(span=21).mean().iloc[-1]
                ema_50 = historical_data.ewm(span=50).mean().iloc[-1] if len(historical_data) >= 50 else ema_21
                
                current_price = float(historical_data.iloc[-1])
                
                # Enhanced EMA signal
                if ema_8 > ema_21 > ema_50:
                    ema_signal = 1.0
                elif ema_8 > ema_21:
                    ema_signal = 0.8
                else:
                    ema_signal = 0.0
                
                # Enhanced RSI
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/14).mean()
                avg_losses = losses.ewm(alpha=1/14).mean()
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                # Optimized RSI scoring
                rsi_value = rsi.iloc[-1]
                if 45 <= rsi_value <= 55:
                    rsi_signal = 1.0
                elif 35 <= rsi_value <= 65:
                    rsi_signal = 0.8
                elif 30 <= rsi_value <= 70:
                    rsi_signal = 0.6
                else:
                    rsi_signal = 0.2
                
                # Enhanced momentum
                momentum_signals = []
                for period in [3, 5, 7]:
                    if len(historical_data) >= period:
                        momentum = (current_price / float(historical_data.iloc[-period])) - 1
                        if 0.003 < momentum < 0.08:  # Sweet spot for swing
                            momentum_signals.append(1.0)
                        elif 0.001 < momentum < 0.12:
                            momentum_signals.append(0.7)
                        else:
                            momentum_signals.append(0.3)
                
                avg_momentum_signal = sum(momentum_signals) / len(momentum_signals) if momentum_signals else 0.5
                
                # Enhanced swing score
                score = (0.4 * ema_signal + 
                        0.3 * rsi_signal + 
                        0.3 * avg_momentum_signal)
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_enhanced_position_changes(self, portfolio, position_type, date, target_positions):
        """Enhanced position changes with improved execution"""
        current_positions = portfolio[position_type]
        current_value = portfolio['value']
        
        # Sell unwanted positions
        positions_to_sell = [s for s in current_positions.keys() if s not in target_positions]
        for symbol in positions_to_sell:
            if symbol in self.data:
                shares = current_positions[symbol]
                if shares > 0:
                    try:
                        prices = self.data[symbol]
                        available_prices = prices[prices.index <= date]
                        if len(available_prices) > 0:
                            price = float(available_prices.iloc[-1])
                            proceeds = float(shares) * price * 0.9998  # Minimal transaction cost
                            portfolio['cash'] += proceeds
                    except:
                        pass
                del current_positions[symbol]
        
        # Buy/adjust positions
        for symbol, target_weight in target_positions.items():
            if symbol in self.data:
                try:
                    prices = self.data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        price = float(available_prices.iloc[-1])
                        target_value = current_value * target_weight
                        target_shares = target_value / price
                        
                        current_shares = current_positions.get(symbol, 0)
                        shares_diff = target_shares - current_shares
                        
                        if abs(shares_diff * price) > current_value * 0.008:  # Slightly higher threshold
                            cost = float(shares_diff) * price
                            if shares_diff > 0 and portfolio['cash'] >= cost * 1.0002:
                                portfolio['cash'] -= cost * 1.0002
                                current_positions[symbol] = target_shares
                            elif shares_diff < 0:
                                portfolio['cash'] -= cost * 0.9998
                                current_positions[symbol] = target_shares if target_shares > 0 else 0
                                if current_positions[symbol] <= 0:
                                    current_positions.pop(symbol, None)
                except:
                    continue
    
    def update_plus_portfolio_value(self, portfolio, date):
        """Update Plus portfolio value"""
        total_value = portfolio['cash']
        
        # Add all positions
        for position_type in ['trend_positions', 'swing_positions']:
            for symbol, shares in portfolio[position_type].items():
                if symbol in self.data and shares > 0:
                    try:
                        prices = self.data[symbol]
                        available_prices = prices[prices.index <= date]
                        if len(available_prices) > 0:
                            current_price = float(available_prices.iloc[-1])
                            position_value = float(shares) * current_price
                            total_value += position_value
                    except:
                        continue
        
        portfolio['value'] = total_value
        return total_value
    
    def calculate_plus_performance(self, history):
        """Calculate Plus performance metrics"""
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
            
            return {
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
            
        except Exception as e:
            print(f"❌ Plus performance calculation failed: {e}")
            return None
    
    def generate_plus_report(self, performance):
        """Generate Plus performance report"""
        if not performance:
            print("❌ No performance data to report")
            return
        
        print("\n" + "="*80)
        print("⚖️ GOLDILOCKS PLUS SYSTEM - PERFORMANCE REPORT")
        print("="*80)
        
        print(f"📊 ENHANCED UNIVERSE: {len(self.data)} optimized symbols")
        print(f"📅 TESTING PERIOD: {self.start_date} to {self.end_date}")
        print(f"💰 INITIAL CAPITAL: ${self.initial_capital:,}")
        
        print(f"\n⚖️ GOLDILOCKS PLUS PERFORMANCE:")
        print(f"  📈 Annual Return:     {performance['annual_return']:>8.1%}")
        print(f"  📊 Total Return:      {performance['total_return']:>8.1%}")
        print(f"  💰 Final Value:       ${performance['final_value']:>10,.0f}")
        print(f"  📉 Max Drawdown:      {performance['max_drawdown']:>8.1%}")
        print(f"  ⚡ Volatility:        {performance['volatility']:>8.1%}")
        print(f"  🎯 Sharpe Ratio:      {performance['sharpe_ratio']:>8.2f}")
        print(f"  📊 Calmar Ratio:      {performance['calmar_ratio']:>8.2f}")
        print(f"  ✅ Win Rate:          {performance['win_rate']:>8.1%}")
        
        print(f"\n🎯 VS BENCHMARKS:")
        nasdaq_annual = 0.184
        spy_annual = 0.134
        
        nasdaq_gap = performance['annual_return'] - nasdaq_annual
        spy_gap = performance['annual_return'] - spy_annual
        
        print(f"  📊 vs NASDAQ (18.4%): {nasdaq_gap:>8.1%} ({'BEATS' if nasdaq_gap > 0 else 'LAGS'})")
        print(f"  📊 vs S&P 500 (13.4%): {spy_gap:>8.1%} ({'BEATS' if spy_gap > 0 else 'LAGS'})")
        
        # Plus vs Goldilocks comparison
        goldilocks_return = 0.198
        goldilocks_dd = -0.268
        
        return_improvement = performance['annual_return'] - goldilocks_return
        dd_change = performance['max_drawdown'] - goldilocks_dd
        
        print(f"\n⚖️ VS GOLDILOCKS ORIGINAL:")
        print(f"  📈 Return Improvement: {return_improvement:>8.1%}")
        print(f"  📉 Drawdown Change:    {dd_change:>8.1%}")
        
        # Assessment
        target_return_achieved = performance['annual_return'] >= 0.21
        risk_controlled = performance['max_drawdown'] > -0.30
        improvement_achieved = return_improvement > 0.005  # 0.5% improvement target
        
        print(f"\n⚖️ PLUS ASSESSMENT:")
        print(f"  📈 Target Return (21%+):    {'✅ ACHIEVED' if target_return_achieved else '🔧 NEEDS WORK'}")
        print(f"  📉 Risk Controlled (<30%):  {'✅ CONTROLLED' if risk_controlled else '⚠️ ELEVATED'}")
        print(f"  🚀 Improvement (>0.5%):     {'✅ ENHANCED' if improvement_achieved else '➡️ MARGINAL'}")
        
        success_count = sum([target_return_achieved, risk_controlled, improvement_achieved])
        
        if success_count == 3:
            rating = "🌟 PERFECT PLUS"
        elif success_count == 2:
            rating = "🏆 EXCELLENT PLUS"
        elif success_count == 1:
            rating = "✅ GOOD PLUS"
        else:
            rating = "➡️ MARGINAL IMPROVEMENT"
        
        print(f"\n{rating}")
        
        # Final recommendation
        if target_return_achieved and risk_controlled:
            print(f"\n🎉 SUCCESS: Goldilocks Plus achieves enhanced performance!")
            print(f"🚀 RECOMMENDATION: This is our optimal final system")
        elif improvement_achieved:
            print(f"\n✅ IMPROVEMENT: Plus system shows meaningful enhancement")
            print(f"🚀 RECOMMENDATION: Consider this enhanced version")
        else:
            print(f"\n➡️ MARGINAL: Improvement not significant enough")
            print(f"🚀 RECOMMENDATION: Stick with original Goldilocks")


def main():
    """Execute Goldilocks Plus System"""
    print("⚖️ GOLDILOCKS PLUS SYSTEM")
    print("Enhanced Performance Optimization")
    print("="*80)
    
    system = GoldilocksPlusSystem()
    performance = system.run_plus_system()
    
    return 0


if __name__ == "__main__":
    exit_code = main()