#!/usr/bin/env python3
"""
Optimized Daily Trading System - Allocation optimization to beat NASDAQ
Target: >20% annual return with aggressive but controlled allocations
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OptimizedDailySystem:
    """
    Daily system with optimized allocations targeting NASDAQ+ performance
    """
    
    def __init__(self):
        # Expanded universe for better opportunities
        self.usa_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
            'AMZN', 'TSLA', 'NFLX', 'AMD', 'CRM',  # Added high-growth tech
            'JPM', 'WFC', 'JNJ', 'UNH', 'KO',
            'PG', 'XOM', 'CVX', 'CAT', 'GE'
        ]
        
        self.europe_stocks = [
            'EWG', 'EWQ', 'EWI', 'EWP', 'EWU',
            'EWN', 'EWO', 'EWK', 'ASML', 'SAP', 'NVO', 'NESN.SW'
        ]
        
        # Add tech-focused ETFs
        self.tech_etfs = [
            'QQQ',   # NASDAQ 100
            'XLK',   # Technology sector
            'VGT'    # Vanguard tech
        ]
        
        # Configuration
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        self.initial_capital = 100000
        self.rebalance_frequency = 1  # DAILY
        
        # Optimized allocation
        self.usa_allocation = 0.65      # Slightly reduced for tech ETFs
        self.europe_allocation = 0.25   # Reduced
        self.tech_etf_allocation = 0.10 # NEW: Dedicated tech allocation
        self.max_position = 0.07        # Slightly reduced for diversification
        
        # OPTIMIZED REGIMES for aggressive performance
        self.daily_regimes = {
            'trend_up': {
                'name': 'Daily Strong Uptrend',
                'score_threshold': 0.08,     # LOWERED: More opportunities
                'allocation_factor': 0.98,   # INCREASED: 95% → 98%
                'max_positions': 18,         # INCREASED: 15 → 18
                'tech_boost': 1.2            # NEW: 20% boost to tech in uptrend
            },
            'trend_down': {
                'name': 'Daily Downtrend',
                'score_threshold': 0.25,     # LOWERED: 30% → 25%
                'allocation_factor': 0.65,   # INCREASED: 60% → 65%
                'max_positions': 10,         # INCREASED: 8 → 10
                'tech_boost': 0.8            # NEW: Reduce tech in downtrend
            },
            'volatile': {
                'name': 'High Daily Volatility',
                'score_threshold': 0.12,     # LOWERED: 20% → 12%
                'allocation_factor': 0.85,   # INCREASED: 75% → 85%
                'max_positions': 14,         # INCREASED: 10 → 14
                'tech_boost': 1.1            # NEW: Slight tech boost in volatility
            },
            'stable': {
                'name': 'Stable Market',
                'score_threshold': 0.06,     # LOWERED: 5% → 6%
                'allocation_factor': 0.92,   # INCREASED: 90% → 92%
                'max_positions': 16,         # INCREASED: 12 → 16
                'tech_boost': 1.0            # NEW: Neutral tech allocation
            }
        }
        
        print(f"🚀 OPTIMIZED DAILY TRADING SYSTEM")
        print(f"🎯 TARGET: Beat NASDAQ (>20% annual)")
        print(f"🌍 Expanded Universe: {len(self.usa_stocks)} USA + {len(self.europe_stocks)} Europe + {len(self.tech_etfs)} Tech ETFs")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"⚡ OPTIMIZED: More aggressive allocations + tech focus")
        print(f"💰 DeGiro fees: <0.1% → Daily trading viable")
    
    def run_optimized_backtest(self):
        """Execute optimized daily backtest"""
        print("\n" + "="*80)
        print("🚀 OPTIMIZED DAILY SYSTEM - Targeting NASDAQ+ Performance")
        print("="*80)
        
        # Download data
        print("\n📊 Step 1: Downloading expanded universe...")
        all_data = self.download_all_data()
        
        # Download SPY
        print("\n📈 Step 2: Downloading SPY for regime detection...")
        spy_data = self.download_spy()
        
        # Run simulation
        print("\n🚀 Step 3: Running optimized daily simulation...")
        portfolio_results = self.simulate_optimized_trading(all_data, spy_data)
        
        # Calculate performance
        print("\n📊 Step 4: Calculating performance...")
        performance = self.calculate_performance_metrics(portfolio_results)
        
        # Download NASDAQ for comparison
        print("\n📈 Step 5: Downloading NASDAQ (QQQ) for comparison...")
        nasdaq_performance = self.get_nasdaq_performance()
        
        # Results
        results = {
            'config': {
                'system_type': 'optimized_daily',
                'target_performance': '>20% annual',
                'universe_size': len(all_data),
                'optimizations': ['aggressive_allocations', 'tech_focus', 'lower_thresholds']
            },
            'performance': performance,
            'nasdaq_comparison': nasdaq_performance,
            'trades_summary': portfolio_results['trades_summary'],
            'regime_history': portfolio_results['regime_history']
        }
        
        self.print_optimized_summary(results)
        return results
    
    def download_all_data(self):
        """Download expanded universe data"""
        all_symbols = self.usa_stocks + self.europe_stocks + self.tech_etfs
        data = {}
        
        for i, symbol in enumerate(all_symbols, 1):
            try:
                print(f"  📊 ({i:2d}/{len(all_symbols)}) {symbol:8s}...", end=" ")
                
                ticker_data = yf.download(
                    symbol, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False
                )
                
                if len(ticker_data) > 500:
                    # Fix MultiIndex if present
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    
                    data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ {len(ticker_data)} days")
                    
            except Exception as e:
                print(f"❌ Error")
        
        print(f"  Downloaded: {len(data)} symbols")
        return data
    
    def download_spy(self):
        """Download SPY with MultiIndex fix"""
        try:
            spy_data = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
            
            # FIX: Remove MultiIndex from yfinance
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.droplevel(1)
            
            print(f"  ✅ SPY downloaded: {len(spy_data)} days (MultiIndex fixed)")
            return spy_data
        except Exception as e:
            print(f"  ❌ SPY failed: {e}")
            return None
    
    def get_nasdaq_performance(self):
        """Get NASDAQ performance for comparison"""
        try:
            qqq_data = yf.download('QQQ', start=self.start_date, end=self.end_date, progress=False)
            
            if isinstance(qqq_data.columns, pd.MultiIndex):
                qqq_data.columns = qqq_data.columns.droplevel(1)
            
            closes = qqq_data['Close']
            total_return = (closes.iloc[-1] / closes.iloc[0]) - 1
            trading_days = len(closes)
            years = trading_days / 252
            annual_return = (1 + total_return) ** (1/years) - 1
            
            print(f"  ✅ NASDAQ (QQQ): {annual_return:.1%} annual")
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'years': float(years)
            }
        except Exception as e:
            print(f"  ❌ NASDAQ download failed: {e}")
            return {
                'total_return': 0.18,
                'annual_return': 0.18,
                'years': 5.0
            }
    
    def detect_daily_regime(self, spy_data, current_date):
        """Enhanced daily regime detection"""
        
        if spy_data is None:
            return 'stable'
        
        historical_spy = spy_data[spy_data.index <= current_date]
        
        if len(historical_spy) < 20:
            return 'stable'
        
        try:
            closes = historical_spy['Close']
            
            # Short-term trend (5-day vs 15-day)
            ma_5 = closes.rolling(5).mean().iloc[-1]
            ma_15 = closes.rolling(15).mean().iloc[-1]
            
            # Short-term volatility (10-day)
            returns = closes.pct_change().dropna()
            volatility_10d = returns.tail(10).std() * np.sqrt(252)
            
            # Short-term momentum (5-day)
            if len(closes) >= 6:
                momentum_5d = (closes.iloc[-1] / closes.iloc[-6]) - 1
            else:
                momentum_5d = 0
            
            # Crisis detection (20-day drawdown)
            if len(closes) >= 20:
                rolling_max = closes.rolling(20).max().iloc[-1]
                drawdown = (closes.iloc[-1] / rolling_max) - 1
                drawdown = float(drawdown) if not pd.isna(drawdown) else 0
            else:
                drawdown = 0
            
            # Safe conversions
            ma_5 = float(ma_5) if not pd.isna(ma_5) else float(closes.iloc[-1])
            ma_15 = float(ma_15) if not pd.isna(ma_15) else float(closes.iloc[-1])
            volatility_10d = float(volatility_10d) if not pd.isna(volatility_10d) else 0.15
            momentum_5d = float(momentum_5d) if not pd.isna(momentum_5d) else 0
            
            # OPTIMIZED regime logic (more sensitive for opportunities)
            is_uptrend = ma_5 > ma_15
            is_high_vol = volatility_10d > 0.12
            is_strong_momentum = abs(momentum_5d) > 0.005
            is_crisis = drawdown < -0.05
            
            # Determine regime with enhanced logic
            if is_crisis:
                return 'trend_down'
            elif is_uptrend and is_strong_momentum:
                return 'trend_up'
            elif not is_uptrend and is_strong_momentum:
                return 'trend_down'
            elif is_high_vol:
                return 'volatile'
            else:
                return 'stable'
                
        except Exception as e:
            print(f"    ⚠️ Regime detection error: {e}")
            return 'stable'
    
    def simulate_optimized_trading(self, data, spy_data):
        """Optimized trading simulation"""
        
        # Initialize
        portfolio = {
            'cash': float(self.initial_capital),
            'positions': {},
            'value': float(self.initial_capital)
        }
        
        # Generate dates
        trading_dates = pd.bdate_range(
            start=self.start_date, 
            end=self.end_date
        ).tolist()
        
        # Track results
        history = []
        trades = []
        regime_history = []
        
        current_regime = 'stable'
        
        print(f"  🚀 Simulating {len(trading_dates)} days with OPTIMIZED daily rebalancing...")
        
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            
            # Progress
            if i % 50 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                print(f"    📅 Progress: {progress:5.1f}% - {current_date} - Regime: {current_regime}")
            
            # Detect regime daily
            new_regime = self.detect_daily_regime(spy_data, date)
            if new_regime != current_regime:
                current_regime = new_regime
                regime_history.append({
                    'date': current_date,
                    'regime': current_regime
                })
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(portfolio, data, date)
            
            # DAILY rebalancing with OPTIMIZED parameters
            if i == 0 or i % self.rebalance_frequency == 0:
                signals = self.calculate_optimized_signals(data, date, current_regime)
                trade_summary = self.execute_optimized_rebalancing(
                    portfolio, data, date, signals, current_regime
                )
                
                if trade_summary['trades_made'] > 0:
                    trades.append({
                        'date': current_date,
                        'regime': current_regime,
                        **trade_summary
                    })
            
            # Record
            history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'num_positions': len(portfolio['positions']),
                'regime': current_regime
            })
        
        return {
            'history': history,
            'regime_history': regime_history,
            'trades_summary': {
                'total_trades': sum(t.get('trades_made', 0) for t in trades),
                'regime_changes': len(regime_history),
                'final_value': portfolio_value,
                'final_positions': len(portfolio['positions'])
            }
        }
    
    def calculate_optimized_signals(self, data, date, current_regime):
        """Calculate signals with regime-specific optimizations"""
        signals = {}
        regime_config = self.daily_regimes[current_regime]
        
        for symbol, prices in data.items():
            try:
                historical_data = prices[prices.index <= date]
                
                if len(historical_data) < 20:
                    continue
                
                # Enhanced EMA signals
                ema_short = historical_data.ewm(span=5).mean()
                ema_long = historical_data.ewm(span=15).mean()
                
                current_ema_short = float(ema_short.iloc[-1])
                current_ema_long = float(ema_long.iloc[-1])
                ema_signal = 1 if current_ema_short > current_ema_long else 0
                
                # EMA acceleration (trend strength)
                if len(ema_short) >= 3:
                    ema_acceleration = (ema_short.iloc[-1] - ema_short.iloc[-3]) / ema_short.iloc[-3]
                    acceleration_signal = 1 if ema_acceleration > 0.001 else 0  # 0.1% acceleration
                else:
                    acceleration_signal = 0
                
                # Enhanced RSI
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/7).mean()
                avg_losses = losses.ewm(alpha=1/7).mean()
                
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                current_rsi = float(rsi.iloc[-1])
                rsi_signal = 1 if current_rsi < 70 else 0  # Slightly more aggressive
                
                # Enhanced momentum
                current_price = float(historical_data.iloc[-1])
                if len(historical_data) >= 6:
                    momentum = (current_price / float(historical_data.iloc[-6])) - 1
                    momentum_signal = 1 if momentum > 0.008 else 0  # 0.8% in 5 days
                else:
                    momentum_signal = 0
                
                # Tech boost for tech symbols
                tech_boost = 1.0
                if symbol in self.tech_etfs:
                    tech_boost = regime_config['tech_boost']
                elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'NFLX', 'AMD', 'CRM']:
                    tech_boost = regime_config['tech_boost']
                
                # OPTIMIZED scoring with tech boost
                base_score = (0.3 * ema_signal + 
                             0.25 * rsi_signal + 
                             0.25 * momentum_signal + 
                             0.2 * acceleration_signal)
                
                score = base_score * tech_boost
                
                signals[symbol] = {
                    'score': score,
                    'ema_signal': ema_signal,
                    'rsi_signal': rsi_signal,
                    'momentum_signal': momentum_signal,
                    'acceleration_signal': acceleration_signal,
                    'tech_boost': tech_boost,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_optimized_rebalancing(self, portfolio, data, date, signals, current_regime):
        """Execute optimized rebalancing"""
        
        regime_config = self.daily_regimes[current_regime]
        
        # Separate by category
        usa_signals = {s: sig for s, sig in signals.items() if s in self.usa_stocks}
        europe_signals = {s: sig for s, sig in signals.items() if s in self.europe_stocks}
        tech_signals = {s: sig for s, sig in signals.items() if s in self.tech_etfs}
        
        # Select with OPTIMIZED threshold
        score_threshold = regime_config['score_threshold']
        max_positions = regime_config['max_positions']
        
        # USA selection
        usa_qualified = sorted(
            [(s, sig['score']) for s, sig in usa_signals.items() if sig['score'] >= score_threshold],
            key=lambda x: x[1], reverse=True
        )[:int(max_positions * 0.6)]
        
        # Europe selection
        europe_qualified = sorted(
            [(s, sig['score']) for s, sig in europe_signals.items() if sig['score'] >= score_threshold],
            key=lambda x: x[1], reverse=True
        )[:int(max_positions * 0.25)]
        
        # Tech ETF selection (NEW)
        tech_qualified = sorted(
            [(s, sig['score']) for s, sig in tech_signals.items() if sig['score'] >= score_threshold],
            key=lambda x: x[1], reverse=True
        )[:int(max_positions * 0.15)]
        
        selected_usa = [s for s, _ in usa_qualified]
        selected_europe = [s for s, _ in europe_qualified]
        selected_tech = [s for s, _ in tech_qualified]
        
        # Calculate OPTIMIZED allocations
        target_positions = {}
        total_selected = len(selected_usa) + len(selected_europe) + len(selected_tech)
        
        if total_selected > 0:
            investable_capital = regime_config['allocation_factor']
            
            # USA allocation
            if selected_usa:
                usa_weight_per_stock = (investable_capital * self.usa_allocation) / len(selected_usa)
                for symbol in selected_usa:
                    target_positions[symbol] = min(usa_weight_per_stock, self.max_position)
            
            # Europe allocation
            if selected_europe:
                europe_weight_per_stock = (investable_capital * self.europe_allocation) / len(selected_europe)
                for symbol in selected_europe:
                    target_positions[symbol] = min(europe_weight_per_stock, self.max_position)
            
            # Tech ETF allocation (NEW)
            if selected_tech:
                tech_weight_per_stock = (investable_capital * self.tech_etf_allocation) / len(selected_tech)
                for symbol in selected_tech:
                    target_positions[symbol] = min(tech_weight_per_stock, self.max_position)
        
        # Execute trades with low threshold for daily
        trades_made = self.execute_trades(portfolio, data, date, target_positions, threshold=0.002)
        
        return {
            'trades_made': trades_made,
            'selected_usa': selected_usa,
            'selected_europe': selected_europe,
            'selected_tech': selected_tech,
            'regime': current_regime,
            'threshold_used': score_threshold,
            'total_positions': len(target_positions)
        }
    
    def execute_trades(self, portfolio, data, date, target_positions, threshold=0.002):
        """Execute trades with optimized thresholds"""
        trades_count = 0
        current_value = portfolio['value']
        
        # Sell unwanted positions
        positions_to_sell = [s for s in portfolio['positions'].keys() if s not in target_positions]
        
        for symbol in positions_to_sell:
            if symbol in data:
                shares = portfolio['positions'][symbol]
                if shares > 0:
                    try:
                        prices = data[symbol]
                        available_prices = prices[prices.index <= date]
                        if len(available_prices) > 0:
                            price = float(available_prices.iloc[-1])
                            proceeds = float(shares) * price
                            portfolio['cash'] += proceeds
                            trades_count += 1
                    except:
                        pass
                
                del portfolio['positions'][symbol]
        
        # Buy/adjust positions
        for symbol, target_weight in target_positions.items():
            if symbol in data:
                try:
                    prices = data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        price = float(available_prices.iloc[-1])
                        
                        target_value = current_value * target_weight
                        target_shares = target_value / price
                        
                        current_shares = portfolio['positions'].get(symbol, 0)
                        shares_diff = target_shares - current_shares
                        
                        trade_value = abs(float(shares_diff) * price)
                        threshold_value = current_value * threshold
                        
                        if trade_value > threshold_value:
                            cost = float(shares_diff) * price
                            
                            if shares_diff > 0 and portfolio['cash'] >= cost:
                                portfolio['cash'] -= cost
                                portfolio['positions'][symbol] = target_shares
                                trades_count += 1
                            elif shares_diff < 0:
                                portfolio['cash'] -= cost
                                portfolio['positions'][symbol] = target_shares
                                trades_count += 1
                except:
                    continue
        
        return trades_count
    
    def update_portfolio_value(self, portfolio, data, date):
        """Update portfolio value"""
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in data and shares > 0:
                try:
                    prices = data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        current_price = float(available_prices.iloc[-1])
                        position_value = float(shares) * current_price
                        total_value += position_value
                except:
                    continue
        
        portfolio['value'] = total_value
        return total_value
    
    def calculate_performance_metrics(self, portfolio_results):
        """Calculate performance metrics"""
        history = pd.DataFrame(portfolio_results['history'])
        history['date'] = pd.to_datetime(history['date'])
        history.set_index('date', inplace=True)
        
        values = history['portfolio_value']
        daily_returns = values.pct_change().dropna()
        
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        trading_days = len(daily_returns)
        years = trading_days / 252
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
            'trading_days': int(trading_days),
            'final_value': float(values.iloc[-1]),
            'years_simulated': float(years)
        }
    
    def print_optimized_summary(self, results):
        """Print optimized summary with NASDAQ comparison"""
        print("\n" + "="*80)
        print("🚀 OPTIMIZED DAILY SYSTEM SUMMARY")
        print("="*80)
        
        config = results['config']
        perf = results['performance']
        trades = results['trades_summary']
        nasdaq = results['nasdaq_comparison']
        
        print(f"🚀 Optimized Configuration:")
        print(f"  System Type:           {config['system_type']}")
        print(f"  Target Performance:    {config['target_performance']}")
        print(f"  Assets analyzed:       {config['universe_size']:>8}")
        print(f"  Optimizations:         {', '.join(config['optimizations'])}")
        
        print(f"\n🎯 OPTIMIZED Performance:")
        print(f"  Total Return (5Y):     {perf['total_return']:>8.1%}")
        print(f"  Annual Return:         {perf['annual_return']:>8.1%}")
        print(f"  Volatility:            {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:          {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:              {perf['win_rate']:>8.1%}")
        print(f"  Final Value:           ${perf['final_value']:>8,.0f}")
        
        print(f"\n📈 vs NASDAQ Comparison:")
        print(f"  NASDAQ (QQQ) Annual:   {nasdaq['annual_return']:>8.1%}")
        print(f"  Our System Annual:     {perf['annual_return']:>8.1%}")
        
        outperformance = perf['annual_return'] - nasdaq['annual_return']
        print(f"  Outperformance:        {outperformance:>8.1%} ({outperformance/nasdaq['annual_return']*100:+.0f}%)")
        
        print(f"\n🔄 Optimized Trading Activity:")
        print(f"  Total Trades:          {trades['total_trades']:>8,}")
        print(f"  Regime Changes:        {trades['regime_changes']:>8,}")
        print(f"  Final Positions:       {trades['final_positions']:>8,}")
        
        # Target achievement
        target_achieved = perf['annual_return'] > 0.20
        nasdaq_beaten = perf['annual_return'] > nasdaq['annual_return']
        
        print(f"\n🏆 OPTIMIZATION RESULTS:")
        if target_achieved and nasdaq_beaten:
            print("🎉 SUCCESS: Target >20% achieved AND NASDAQ beaten!")
        elif target_achieved:
            print("✅ GOOD: Target >20% achieved!")
        elif nasdaq_beaten:
            print("👍 PROGRESS: NASDAQ beaten, approaching 20% target!")
        else:
            print("⚠️ Need further optimization")
        
        if outperformance > 0.02:
            print(f"🚀 EXCELLENT: Strong outperformance of NASDAQ (+{outperformance:.1%})")
        elif outperformance > 0:
            print(f"✅ POSITIVE: Outperforming NASDAQ (+{outperformance:.1%})")


def main():
    """Execute optimized daily system"""
    print("🚀 OPTIMIZED DAILY TRADING SYSTEM")
    print("Targeting >20% annual to beat NASDAQ")
    print("="*80)
    
    system = OptimizedDailySystem()
    results = system.run_optimized_backtest()
    
    return 0


if __name__ == "__main__":
    exit_code = main()