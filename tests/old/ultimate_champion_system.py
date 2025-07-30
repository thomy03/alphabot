#!/usr/bin/env python3
"""
Ultimate Champion System - Hybrid Best-of-Best Strategy
Combines Simple Trend Following (19.7%) + Enhanced Swing (19.5%)
Target: 22-25% annual return with optimal risk management
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UltimateChampionSystem:
    """
    Ultimate hybrid system combining our 2 best strategies:
    - Simple Trend Following (monthly rebalancing, 19.7% annual)
    - Enhanced Swing Trading (3-day rebalancing, 19.5% annual)
    Dynamic switching based on market conditions
    """
    
    def __init__(self):
        # CHAMPION UNIVERSE (proven performers)
        self.champion_universe = [
            # Mega tech champions (highest weights)
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA',
            
            # Growth tech leaders
            'CRM', 'ADBE', 'AMD', 'QCOM', 'ORCL',
            
            # Quality large caps
            'V', 'MA', 'UNH', 'HD', 'JPM',
            
            # Tech ETFs (core holdings)
            'QQQ', 'XLK', 'VGT',
            
            # Market ETFs (diversification)
            'SPY', 'IWM'
        ]
        
        # ULTIMATE CONFIGURATION
        self.start_date = "2015-01-01"
        self.end_date = "2024-12-01"
        self.initial_capital = 100000
        
        # DYNAMIC STRATEGY ALLOCATION
        self.base_trend_allocation = 0.70    # 70% Simple Trend (base)
        self.base_swing_allocation = 0.30    # 30% Enhanced Swing (opportunistic)
        
        # CHAMPION PARAMETERS
        self.max_positions = 12              # Concentrated excellence
        self.tech_mega_weight = 0.50         # 50% in tech mega caps
        self.cash_buffer = 0.05              # 5% cash minimum
        
        # ADAPTIVE THRESHOLDS
        self.volatility_threshold = 20       # Switch to swing if VIX > 20
        self.momentum_threshold = 0.02       # 2% momentum for trend confirmation
        
        print(f"🏆 ULTIMATE CHAMPION SYSTEM")
        print(f"📊 CHAMPION UNIVERSE: {len(self.champion_universe)} elite symbols")
        print(f"🎯 TARGET: 22-25% annual return")
        print(f"⚖️ HYBRID: 70% Trend + 30% Swing (adaptive)")
        
        # Storage
        self.data = {}
        self.market_data = {}
        self.performance_history = []
        
    def run_ultimate_champion(self):
        """Run the ultimate champion hybrid system"""
        print("\n" + "="*80)
        print("🏆 ULTIMATE CHAMPION SYSTEM - HYBRID EXECUTION")
        print("="*80)
        
        # Download champion data
        print("\n📊 Step 1: Downloading champion universe data...")
        self.download_champion_data()
        
        # Download market indices
        print("\n📈 Step 2: Downloading market regime indicators...")
        self.download_market_data()
        
        # Run hybrid strategy
        print("\n🚀 Step 3: Executing ultimate hybrid strategy...")
        portfolio_history = self.execute_hybrid_strategy()
        
        # Calculate performance
        print("\n📊 Step 4: Calculating ultimate performance...")
        performance = self.calculate_ultimate_performance(portfolio_history)
        
        # Generate report
        print("\n📋 Step 5: Generating champion performance report...")
        self.generate_champion_report(performance)
        
        return performance
    
    def download_champion_data(self):
        """Download data for champion universe"""
        failed_downloads = []
        
        for i, symbol in enumerate(self.champion_universe, 1):
            try:
                print(f"  🏆 ({i:2d}/{len(self.champion_universe)}) {symbol:8s}...", end=" ")
                
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
        
        print(f"  🏆 CHAMPION DATA: {len(self.data)} symbols loaded")
        if failed_downloads:
            print(f"  ⚠️ Failed: {failed_downloads}")
    
    def download_market_data(self):
        """Download market regime indicators"""
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
    
    def execute_hybrid_strategy(self):
        """Execute the hybrid trading strategy"""
        portfolio = {
            'cash': float(self.initial_capital),
            'trend_positions': {},      # Simple Trend positions
            'swing_positions': {},      # Enhanced Swing positions
            'value': float(self.initial_capital)
        }
        
        trading_dates = pd.bdate_range(start=self.start_date, end=self.end_date).tolist()
        history = []
        
        print(f"    🏆 Ultimate hybrid execution: {len(trading_dates)} days")
        
        for i, date in enumerate(trading_dates):
            if i % 500 == 0:
                print(f"      📅 Progress: {(i/len(trading_dates)*100):5.1f}% - {date.strftime('%Y-%m-%d')}")
            
            # Market regime detection
            regime = self.detect_market_regime(date)
            
            # Dynamic allocation based on regime
            trend_allocation, swing_allocation = self.calculate_dynamic_allocation(regime)
            
            # Update portfolio value
            portfolio_value = self.update_hybrid_portfolio_value(portfolio, date)
            
            # Trend following rebalancing (monthly)
            if i % 20 == 0:  # Every 20 trading days ≈ monthly
                self.execute_trend_rebalancing(portfolio, date, trend_allocation)
            
            # Swing trading rebalancing (every 3 days)
            if i % 3 == 0:
                self.execute_swing_rebalancing(portfolio, date, swing_allocation)
            
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'regime': regime,
                'trend_allocation': trend_allocation,
                'swing_allocation': swing_allocation
            })
        
        return history
    
    def detect_market_regime(self, current_date):
        """Advanced market regime detection"""
        try:
            spy_data = self.market_data.get('SPY')
            vix_data = self.market_data.get('VIX')
            
            if spy_data is None:
                return 'neutral'
            
            historical_spy = spy_data[spy_data.index <= current_date]
            if len(historical_spy) < 50:
                return 'neutral'
            
            closes = historical_spy['Close']
            
            # Moving averages
            ma_20 = closes.rolling(20).mean().iloc[-1]
            ma_50 = closes.rolling(50).mean().iloc[-1]
            ma_200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else ma_50
            
            current_price = closes.iloc[-1]
            
            # Volatility (VIX)
            current_vix = 20  # Default
            if vix_data is not None:
                historical_vix = vix_data[vix_data.index <= current_date]
                if len(historical_vix) > 0:
                    current_vix = historical_vix['Close'].iloc[-1]
            
            # Momentum
            momentum = (current_price / closes.iloc[-20]) - 1 if len(closes) >= 20 else 0
            
            # Advanced regime classification
            if current_vix > 30:
                return 'crisis'
            elif current_vix > 25:
                return 'high_volatility'
            elif current_price > ma_20 > ma_50 > ma_200 and momentum > 0.02:
                return 'strong_bull'
            elif current_price > ma_20 > ma_50:
                return 'bull'
            elif current_price < ma_20 < ma_50 and momentum < -0.02:
                return 'bear'
            elif current_vix > 20:
                return 'volatile'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def calculate_dynamic_allocation(self, regime):
        """Calculate dynamic allocation between trend and swing"""
        regime_allocations = {
            'strong_bull': (0.80, 0.20),    # More trend following in strong bull
            'bull': (0.70, 0.30),           # Base allocation
            'neutral': (0.60, 0.40),        # More swing in neutral
            'volatile': (0.40, 0.60),       # More swing in volatility
            'high_volatility': (0.30, 0.70), # Heavy swing trading
            'bear': (0.50, 0.50),           # Balanced approach
            'crisis': (0.20, 0.80)          # Maximum swing flexibility
        }
        
        return regime_allocations.get(regime, (0.70, 0.30))
    
    def execute_trend_rebalancing(self, portfolio, date, allocation):
        """Execute trend following rebalancing (monthly)"""
        signals = self.calculate_trend_signals(date)
        
        # Select top trend signals
        qualified_signals = sorted(
            [(s, sig['score']) for s, sig in signals.items() if sig['score'] > 0.7],
            key=lambda x: x[1], 
            reverse=True
        )[:8]  # Top 8 for trend
        
        if not qualified_signals:
            return
        
        # Calculate target positions
        target_positions = {}
        total_trend_capital = portfolio['value'] * allocation
        
        # Tech mega cap boost
        tech_mega_caps = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'QQQ']
        
        for symbol, score in qualified_signals:
            base_weight = allocation / len(qualified_signals)
            
            # Boost tech mega caps
            if symbol in tech_mega_caps:
                weight = base_weight * 1.3  # 30% boost
            else:
                weight = base_weight
            
            # Cap individual positions
            weight = min(weight, 0.12)  # Max 12% per position
            target_positions[symbol] = weight
        
        # Execute trend trades
        self.execute_position_changes(portfolio, 'trend_positions', date, target_positions)
    
    def execute_swing_rebalancing(self, portfolio, date, allocation):
        """Execute swing trading rebalancing (every 3 days)"""
        signals = self.calculate_swing_signals(date)
        
        # Select top swing signals
        qualified_signals = sorted(
            [(s, sig['score']) for s, sig in signals.items() if sig['score'] > 0.6],
            key=lambda x: x[1], 
            reverse=True
        )[:6]  # Top 6 for swing
        
        if not qualified_signals:
            return
        
        # Calculate target positions
        target_positions = {}
        
        for symbol, score in qualified_signals:
            weight = allocation / len(qualified_signals)
            weight = min(weight, 0.08)  # Max 8% per swing position
            target_positions[symbol] = weight
        
        # Execute swing trades
        self.execute_position_changes(portfolio, 'swing_positions', date, target_positions)
    
    def calculate_trend_signals(self, date):
        """Calculate trend following signals (long-term)"""
        signals = {}
        
        for symbol, prices in self.data.items():
            try:
                historical_data = prices[prices.index <= date]
                if len(historical_data) < 100:
                    continue
                
                # Long-term trend indicators
                ma_50 = historical_data.rolling(50).mean().iloc[-1]
                ma_100 = historical_data.rolling(100).mean().iloc[-1]
                ma_200 = historical_data.rolling(200).mean().iloc[-1] if len(historical_data) >= 200 else ma_100
                
                current_price = float(historical_data.iloc[-1])
                
                # Trend strength
                if current_price > ma_50 > ma_100 > ma_200:
                    trend_score = 1.0
                elif current_price > ma_50 > ma_100:
                    trend_score = 0.8
                elif current_price > ma_50:
                    trend_score = 0.6
                else:
                    trend_score = 0.0
                
                # Momentum confirmation
                if len(historical_data) >= 50:
                    momentum = (current_price / float(historical_data.iloc[-50])) - 1
                    momentum_score = min(momentum * 5, 0.5) if momentum > 0 else 0
                else:
                    momentum_score = 0
                
                # Final score
                score = 0.7 * trend_score + 0.3 * momentum_score
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def calculate_swing_signals(self, date):
        """Calculate swing trading signals (short-term)"""
        signals = {}
        
        for symbol, prices in self.data.items():
            try:
                historical_data = prices[prices.index <= date]
                if len(historical_data) < 30:
                    continue
                
                # Short-term indicators
                ema_8 = historical_data.ewm(span=8).mean().iloc[-1]
                ema_21 = historical_data.ewm(span=21).mean().iloc[-1]
                
                current_price = float(historical_data.iloc[-1])
                
                # EMA crossover
                ema_signal = 1.0 if ema_8 > ema_21 else 0.0
                
                # RSI
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/14).mean()
                avg_losses = losses.ewm(alpha=1/14).mean()
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                rsi_signal = 1.0 if 30 < rsi.iloc[-1] < 70 else 0.5
                
                # Short-term momentum
                if len(historical_data) >= 5:
                    momentum = (current_price / float(historical_data.iloc[-5])) - 1
                    momentum_signal = 1.0 if momentum > 0.005 else 0.0
                else:
                    momentum_signal = 0.0
                
                # Volatility boost
                volatility = historical_data.pct_change().rolling(10).std().iloc[-1]
                vol_boost = min(volatility * 10, 0.3) if volatility > 0.02 else 0
                
                # Final swing score
                score = 0.4 * ema_signal + 0.3 * rsi_signal + 0.3 * momentum_signal + vol_boost
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_position_changes(self, portfolio, position_type, date, target_positions):
        """Execute position changes for trend or swing positions"""
        current_positions = portfolio[position_type]
        current_value = portfolio['value']
        
        # Sell positions not in targets
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
                            proceeds = float(shares) * price
                            portfolio['cash'] += proceeds
                    except:
                        pass
                del current_positions[symbol]
        
        # Buy/adjust target positions
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
                        
                        if abs(shares_diff * price) > current_value * 0.005:  # 0.5% threshold
                            cost = float(shares_diff) * price
                            if shares_diff > 0 and portfolio['cash'] >= cost:
                                portfolio['cash'] -= cost
                                current_positions[symbol] = target_shares
                            elif shares_diff < 0:
                                portfolio['cash'] -= cost
                                current_positions[symbol] = target_shares if target_shares > 0 else 0
                                if current_positions[symbol] <= 0:
                                    current_positions.pop(symbol, None)
                except:
                    continue
    
    def update_hybrid_portfolio_value(self, portfolio, date):
        """Update total portfolio value"""
        total_value = portfolio['cash']
        
        # Add trend positions value
        for symbol, shares in portfolio['trend_positions'].items():
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
        
        # Add swing positions value
        for symbol, shares in portfolio['swing_positions'].items():
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
    
    def calculate_ultimate_performance(self, history):
        """Calculate ultimate system performance"""
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
            print(f"❌ Performance calculation failed: {e}")
            return None
    
    def generate_champion_report(self, performance):
        """Generate ultimate champion performance report"""
        if not performance:
            print("❌ No performance data to report")
            return
        
        print("\n" + "="*80)
        print("🏆 ULTIMATE CHAMPION SYSTEM - PERFORMANCE REPORT")
        print("="*80)
        
        print(f"📊 CHAMPION UNIVERSE: {len(self.data)} elite symbols")
        print(f"📅 TESTING PERIOD: {self.start_date} to {self.end_date}")
        print(f"💰 INITIAL CAPITAL: ${self.initial_capital:,}")
        
        print(f"\n🏆 ULTIMATE PERFORMANCE:")
        print(f"  📈 Annual Return:     {performance['annual_return']:>8.1%}")
        print(f"  📊 Total Return:      {performance['total_return']:>8.1%}")
        print(f"  💰 Final Value:       ${performance['final_value']:>10,.0f}")
        print(f"  📉 Max Drawdown:      {performance['max_drawdown']:>8.1%}")
        print(f"  ⚡ Volatility:        {performance['volatility']:>8.1%}")
        print(f"  🎯 Sharpe Ratio:      {performance['sharpe_ratio']:>8.2f}")
        print(f"  📊 Calmar Ratio:      {performance['calmar_ratio']:>8.2f}")
        print(f"  ✅ Win Rate:          {performance['win_rate']:>8.1%}")
        
        print(f"\n🎯 VS BENCHMARKS:")
        nasdaq_annual = 0.184  # 18.4% from our previous test
        spy_annual = 0.134     # 13.4% from our previous test
        
        nasdaq_gap = performance['annual_return'] - nasdaq_annual
        spy_gap = performance['annual_return'] - spy_annual
        
        print(f"  📊 vs NASDAQ (18.4%): {nasdaq_gap:>8.1%} ({'BEATS' if nasdaq_gap > 0 else 'LAGS'})")
        print(f"  📊 vs S&P 500 (13.4%): {spy_gap:>8.1%} ({'BEATS' if spy_gap > 0 else 'LAGS'})")
        
        # Performance classification
        if performance['annual_return'] > 0.22:
            rating = "🌟 EXCEPTIONAL"
        elif performance['annual_return'] > 0.20:
            rating = "🏆 EXCELLENT"
        elif performance['annual_return'] > 0.18:
            rating = "✅ VERY GOOD"
        else:
            rating = "📊 GOOD"
        
        print(f"\n{rating}")
        print(f"🎯 TARGET ACHIEVED: {'YES' if performance['annual_return'] > 0.22 else 'PARTIAL'}")
        
        # Risk assessment
        if performance['max_drawdown'] > -0.35:
            risk_rating = "🔴 HIGH RISK"
        elif performance['max_drawdown'] > -0.25:
            risk_rating = "🟡 MODERATE RISK"
        else:
            risk_rating = "🟢 CONTROLLED RISK"
        
        print(f"🛡️ RISK LEVEL: {risk_rating}")
        
        print(f"\n💡 CHAMPION INSIGHTS:")
        print(f"  🔄 Hybrid Strategy: Dynamic trend + swing allocation")
        print(f"  🎯 Elite Universe: {len(self.data)} top performers only")
        print(f"  ⚖️ Risk Management: Adaptive position sizing")
        print(f"  📊 Regime Detection: Advanced market condition analysis")
        
        if performance['annual_return'] > nasdaq_annual:
            print(f"\n🎉 SUCCESS: Ultimate Champion BEATS the market!")
            print(f"🚀 RECOMMENDATION: Deploy this system for live trading")
        else:
            print(f"\n🔧 NEEDS OPTIMIZATION: Target not fully achieved")


def main():
    """Execute Ultimate Champion System"""
    print("🏆 ULTIMATE CHAMPION SYSTEM")
    print("Hybrid Best-of-Best Strategy")
    print("="*80)
    
    champion = UltimateChampionSystem()
    performance = champion.run_ultimate_champion()
    
    return 0


if __name__ == "__main__":
    exit_code = main()