#!/usr/bin/env python3
"""
Fixed Adaptive System - Système adaptatif avec seuils corrigés et calculs robustes
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FixedAdaptiveSystem:
    """
    Système adaptatif corrigé avec seuils réalistes
    """
    
    def __init__(self):
        # Universe
        self.usa_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
            'JPM', 'BAC', 'WFC', 'JNJ', 'UNH',
            'KO', 'PG', 'XOM', 'CVX', 'CAT', 'GE'
        ]
        
        self.europe_stocks = [
            'EWG', 'EWQ', 'EWI', 'EWP', 'EWU',
            'EWN', 'EWO', 'EWK', 'ASML', 'SAP', 'NVO', 'NESN.SW'
        ]
        
        # Configuration
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        self.initial_capital = 100000
        self.rebalance_frequency = 7
        
        # Allocation
        self.usa_allocation = 0.70
        self.europe_allocation = 0.30
        self.max_position = 0.08
        
        # RÉGIMES CORRIGÉS avec seuils RÉALISTES
        self.market_regimes = {
            'bull_low_vol': {
                'name': 'Bull Market Low Volatility',
                'ema_short': 12, 'ema_long': 26,
                'rsi_period': 14, 'rsi_threshold': 80,
                'score_threshold': 0.1,  # RÉDUIT de 0.3 → 0.1
                'max_positions': 15,
                'allocation_factor': 0.95  # 95% invested
            },
            'bull_high_vol': {
                'name': 'Bull Market High Volatility', 
                'ema_short': 8, 'ema_long': 21,
                'rsi_period': 10, 'rsi_threshold': 75,
                'score_threshold': 0.15,  # RÉDUIT de 0.4 → 0.15
                'max_positions': 12,
                'allocation_factor': 0.90  # 90% invested
            },
            'bear_market': {
                'name': 'Bear Market',
                'ema_short': 5, 'ema_long': 15,
                'rsi_period': 8, 'rsi_threshold': 65,
                'score_threshold': 0.25,  # RÉDUIT de 0.6 → 0.25
                'max_positions': 8,
                'allocation_factor': 0.70  # 70% invested
            },
            'sideways_low_vol': {
                'name': 'Sideways Low Volatility',
                'ema_short': 20, 'ema_long': 50,
                'rsi_period': 18, 'rsi_threshold': 78,
                'score_threshold': 0.05,  # RÉDUIT de 0.35 → 0.05
                'max_positions': 10,
                'allocation_factor': 0.85  # 85% invested
            },
            'sideways_high_vol': {
                'name': 'Sideways High Volatility',
                'ema_short': 6, 'ema_long': 18,
                'rsi_period': 12, 'rsi_threshold': 70,
                'score_threshold': 0.2,  # RÉDUIT de 0.5 → 0.2
                'max_positions': 9,
                'allocation_factor': 0.75  # 75% invested
            },
            'crisis': {
                'name': 'Crisis Mode',
                'ema_short': 3, 'ema_long': 10,
                'rsi_period': 6, 'rsi_threshold': 60,
                'score_threshold': 0.3,  # RÉDUIT de 0.7 → 0.3
                'max_positions': 5,
                'allocation_factor': 0.50  # 50% invested
            }
        }
        
        print(f"🧠 FIXED ADAPTIVE SYSTEM")
        print(f"🌍 Universe: {len(self.usa_stocks)} USA + {len(self.europe_stocks)} Europe")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"⚡ CORRECTED thresholds: 0.05-0.3 (was 0.3-0.7)")
        print(f"🔧 Robust calculations with error handling")
    
    def run_fixed_backtest(self):
        """Execute le backtest adaptatif corrigé"""
        print("\n" + "="*80)
        print("🔧 FIXED ADAPTIVE SYSTEM - Corrected Thresholds & Calculations")
        print("="*80)
        
        # Download data
        print("\n📊 Step 1: Downloading data...")
        all_data = self.download_all_data()
        
        # Download SPY
        print("\n📈 Step 2: Downloading SPY benchmark...")
        spy_data = self.download_spy()
        
        # Run simulation
        print("\n🧠 Step 3: Running fixed adaptive simulation...")
        portfolio_results = self.simulate_fixed_adaptive(all_data, spy_data)
        
        # Calculate performance
        print("\n📊 Step 4: Calculating performance...")
        performance = self.calculate_performance_metrics(portfolio_results)
        
        # Results
        results = {
            'config': {
                'system_type': 'fixed_adaptive',
                'corrected_thresholds': True,
                'universe_size': len(all_data)
            },
            'performance': performance,
            'trades_summary': portfolio_results['trades_summary'],
            'regime_history': portfolio_results['regime_history']
        }
        
        self.print_fixed_summary(results)
        return results
    
    def download_all_data(self):
        """Download data"""
        all_symbols = self.usa_stocks + self.europe_stocks
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
                
                if len(ticker_data) > 1000:
                    data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ {len(ticker_data)} days")
                    
            except Exception as e:
                print(f"❌ Error")
        
        print(f"  Downloaded: {len(data)} symbols")
        return data
    
    def download_spy(self):
        """Download SPY"""
        try:
            spy_data = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
            print(f"  ✅ SPY downloaded: {len(spy_data)} days")
            return spy_data
        except Exception as e:
            print(f"  ❌ SPY failed: {e}")
            return None
    
    def detect_market_regime(self, spy_data, current_date):
        """Détection de régime robuste"""
        
        if spy_data is None:
            return 'sideways_low_vol'
        
        # Get data up to current date
        historical_spy = spy_data[spy_data.index <= current_date]
        
        if len(historical_spy) < 60:
            return 'sideways_low_vol'
        
        try:
            closes = historical_spy['Close']
            
            # Trend detection
            ma_20 = closes.rolling(20).mean().iloc[-1]
            ma_60 = closes.rolling(60).mean().iloc[-1]
            
            # Volatility
            returns = closes.pct_change().dropna()
            volatility_20d = returns.tail(20).std() * np.sqrt(252)
            
            # Momentum
            if len(closes) >= 21:
                momentum_20d = (closes.iloc[-1] / closes.iloc[-21]) - 1
            else:
                momentum_20d = 0
            
            # Drawdown
            rolling_max = closes.rolling(60).max()
            current_drawdown = (closes.iloc[-1] / rolling_max.iloc[-1]) - 1
            
            # Safe conversions
            ma_20 = float(ma_20) if not pd.isna(ma_20) else float(closes.iloc[-1])
            ma_60 = float(ma_60) if not pd.isna(ma_60) else float(closes.iloc[-1])
            volatility_20d = float(volatility_20d) if not pd.isna(volatility_20d) else 0.15
            momentum_20d = float(momentum_20d) if not pd.isna(momentum_20d) else 0
            current_drawdown = float(current_drawdown) if not pd.isna(current_drawdown) else 0
            
            # Regime logic
            is_uptrend = ma_20 > ma_60
            is_high_vol = volatility_20d > 0.25
            is_strong_momentum = abs(momentum_20d) > 0.05
            is_crisis = current_drawdown < -0.15
            
            # Determine regime
            if is_crisis:
                return 'crisis'
            elif is_uptrend and is_strong_momentum:
                return 'bull_high_vol' if is_high_vol else 'bull_low_vol'
            elif not is_uptrend and is_strong_momentum:
                return 'bear_market'
            elif is_high_vol:
                return 'sideways_high_vol'
            else:
                return 'sideways_low_vol'
                
        except Exception as e:
            return 'sideways_low_vol'  # Safe default
    
    def simulate_fixed_adaptive(self, data, spy_data):
        """Simulation adaptative corrigée"""
        
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
        regime_history = []
        trades = []
        
        current_regime = 'sideways_low_vol'
        
        print(f"  🎯 Simulating {len(trading_dates)} days with FIXED adaptive logic...")
        
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            
            # Progress
            if i % 250 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                print(f"    📅 Progress: {progress:5.1f}% - {current_date} - Regime: {current_regime}")
            
            # Detect regime weekly
            if i % 7 == 0:
                new_regime = self.detect_market_regime(spy_data, date)
                if new_regime != current_regime:
                    current_regime = new_regime
                    regime_history.append({
                        'date': current_date,
                        'regime': current_regime
                    })
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(portfolio, data, date)
            
            # Rebalancing with FIXED adaptive parameters
            if i == 0 or i % self.rebalance_frequency == 0:
                signals = self.calculate_fixed_signals(data, date, current_regime)
                trade_summary = self.execute_fixed_rebalancing(
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
    
    def calculate_fixed_signals(self, data, date, current_regime):
        """Calcul de signaux robuste et corrigé"""
        signals = {}
        regime_params = self.market_regimes[current_regime]
        
        for symbol, prices in data.items():
            try:
                # Get historical data
                historical_data = prices[prices.index <= date]
                
                min_data_needed = max(regime_params['ema_long'], regime_params['rsi_period']) + 10
                if len(historical_data) < min_data_needed:
                    continue
                
                # EMA calculation - ROBUST
                try:
                    ema_short = historical_data.ewm(span=regime_params['ema_short']).mean()
                    ema_long = historical_data.ewm(span=regime_params['ema_long']).mean()
                    
                    current_ema_short = float(ema_short.iloc[-1])
                    current_ema_long = float(ema_long.iloc[-1])
                    
                    ema_signal = 1 if current_ema_short > current_ema_long else 0
                except:
                    ema_signal = 0
                
                # RSI calculation - ROBUST
                try:
                    delta = historical_data.diff()
                    gains = delta.where(delta > 0, 0.0)
                    losses = -delta.where(delta < 0, 0.0)
                    avg_gains = gains.ewm(alpha=1/regime_params['rsi_period']).mean()
                    avg_losses = losses.ewm(alpha=1/regime_params['rsi_period']).mean()
                    
                    # Handle division by zero
                    rs = avg_gains / avg_losses.replace(0, 0.001)
                    rsi = 100 - (100 / (1 + rs))
                    
                    current_rsi = float(rsi.iloc[-1])
                    rsi_signal = 1 if current_rsi < regime_params['rsi_threshold'] else 0
                except:
                    current_rsi = 50
                    rsi_signal = 1
                
                # Momentum - ROBUST
                try:
                    current_price = float(historical_data.iloc[-1])
                    if len(historical_data) >= 21:
                        momentum_20d = (current_price / float(historical_data.iloc[-21])) - 1
                        momentum_signal = 1 if momentum_20d > 0.02 else 0
                    else:
                        momentum_signal = 0
                except:
                    momentum_signal = 0
                    current_price = float(historical_data.iloc[-1])
                
                # SIMPLE scoring
                score = 0.6 * ema_signal + 0.4 * rsi_signal
                
                signals[symbol] = {
                    'score': score,
                    'ema_signal': ema_signal,
                    'rsi_signal': rsi_signal,
                    'price': current_price
                }
                
            except Exception as e:
                # Skip problematic symbols
                continue
        
        return signals
    
    def execute_fixed_rebalancing(self, portfolio, data, date, signals, current_regime):
        """Rebalancing avec paramètres corrigés"""
        
        regime_params = self.market_regimes[current_regime]
        
        # Separate by region
        usa_signals = {s: sig for s, sig in signals.items() if s in self.usa_stocks}
        europe_signals = {s: sig for s, sig in signals.items() if s in self.europe_stocks}
        
        # Select with CORRECTED threshold
        score_threshold = regime_params['score_threshold']
        max_positions = regime_params['max_positions']
        
        usa_qualified = sorted(
            [(s, sig['score']) for s, sig in usa_signals.items() if sig['score'] >= score_threshold],
            key=lambda x: x[1], reverse=True
        )[:int(max_positions * 0.6)]
        
        europe_qualified = sorted(
            [(s, sig['score']) for s, sig in europe_signals.items() if sig['score'] >= score_threshold],
            key=lambda x: x[1], reverse=True
        )[:int(max_positions * 0.4)]
        
        selected_usa = [s for s, _ in usa_qualified]
        selected_europe = [s for s, _ in europe_qualified]
        
        # Calculate allocations
        target_positions = {}
        total_selected = len(selected_usa) + len(selected_europe)
        
        if total_selected > 0:
            investable_capital = regime_params['allocation_factor']
            
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
        
        # Execute trades
        trades_made = self.execute_trades(portfolio, data, date, target_positions)
        
        return {
            'trades_made': trades_made,
            'selected_usa': selected_usa,
            'selected_europe': selected_europe,
            'regime': current_regime,
            'threshold_used': score_threshold,
            'total_positions': len(target_positions)
        }
    
    def execute_trades(self, portfolio, data, date, target_positions):
        """Execute trades"""
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
                        threshold = current_value * 0.005
                        
                        if trade_value > threshold:
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
        """Calculate performance"""
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
    
    def print_fixed_summary(self, results):
        """Print summary"""
        print("\n" + "="*80)
        print("🔧 FIXED ADAPTIVE SYSTEM SUMMARY")
        print("="*80)
        
        config = results['config']
        perf = results['performance']
        trades = results['trades_summary']
        
        print(f"🔧 Fixed Configuration:")
        print(f"  System Type:           {config['system_type']}")
        print(f"  Corrected Thresholds:  {config['corrected_thresholds']}")
        print(f"  Assets analyzed:       {config['universe_size']:>8}")
        
        print(f"\n🚀 FIXED Performance:")
        print(f"  Total Return (5Y):     {perf['total_return']:>8.1%}")
        print(f"  Annual Return:         {perf['annual_return']:>8.1%}")
        print(f"  Volatility:            {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:          {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:              {perf['win_rate']:>8.1%}")
        print(f"  Final Value:           ${perf['final_value']:>8,.0f}")
        
        print(f"\n🔄 Fixed Trading Activity:")
        print(f"  Total Trades:          {trades['total_trades']:>8,}")
        print(f"  Regime Changes:        {trades['regime_changes']:>8,}")
        print(f"  Final Positions:       {trades['final_positions']:>8,}")
        
        baseline_annual = 0.097
        improvement = perf['annual_return'] - baseline_annual
        
        print(f"\n🎖️ FIXED RESULTS:")
        print(f"  Baseline Performance:  {baseline_annual:>8.1%}")
        print(f"  Fixed Performance:     {perf['annual_return']:>8.1%}")
        print(f"  Improvement:           {improvement:>8.1%} ({improvement/baseline_annual*100:+.0f}%)")
        
        if trades['total_trades'] > 0:
            print("✅ FIXED: Trades are now being executed!")
            if perf['annual_return'] > 0.12:
                print("🎉 EXCELLENT: >12% annual with adaptive intelligence!")
            elif perf['annual_return'] > baseline_annual:
                print("👍 IMPROVED: Adaptive system working!")
        else:
            print("⚠️ Still no trades - need further investigation")


def main():
    """Execute fixed adaptive system"""
    print("🔧 FIXED ADAPTIVE TRADING SYSTEM")
    print("Corrected thresholds and robust calculations")
    print("="*80)
    
    system = FixedAdaptiveSystem()
    results = system.run_fixed_backtest()
    
    return 0


if __name__ == "__main__":
    exit_code = main()