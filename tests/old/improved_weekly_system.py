#!/usr/bin/env python3
"""
Improved Weekly Trading System - Détection de régime CORRIGÉE
Seuils réalistes pour capturer les vrais changements de marché
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ImprovedWeeklySystem:
    """
    Système hebdomadaire avec détection de régime CORRIGÉE
    """
    
    def __init__(self):
        # Universe
        self.usa_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
            'JPM', 'WFC', 'JNJ', 'UNH', 'KO',
            'PG', 'XOM', 'CVX', 'CAT', 'GE'
        ]
        
        self.europe_stocks = [
            'EWG', 'EWQ', 'EWI', 'EWP', 'EWU',
            'EWN', 'EWO', 'EWK', 'ASML', 'SAP', 'NVO', 'NESN.SW'
        ]
        
        # Configuration
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        self.initial_capital = 100000
        self.rebalance_frequency = 5  # WEEKLY
        
        # Allocation
        self.usa_allocation = 0.70
        self.europe_allocation = 0.30
        self.max_position = 0.08
        
        # RÉGIMES CORRIGÉS avec seuils RÉALISTES
        self.market_regimes = {
            'bull_trend': {
                'name': 'Bull Market Trend',
                'score_threshold': 0.1,    # RÉDUIT de 0.15
                'allocation_factor': 0.95,
                'max_positions': 15
            },
            'bear_trend': {
                'name': 'Bear Market Trend',
                'score_threshold': 0.25,   # RÉDUIT de 0.35
                'allocation_factor': 0.70,
                'max_positions': 8
            },
            'high_volatility': {
                'name': 'High Volatility Period',
                'score_threshold': 0.15,   # RÉDUIT de 0.25
                'allocation_factor': 0.80,
                'max_positions': 10
            },
            'recovery': {
                'name': 'Market Recovery',
                'score_threshold': 0.1,    # RÉDUIT de 0.2
                'allocation_factor': 0.90,
                'max_positions': 12
            },
            'consolidation': {
                'name': 'Market Consolidation',
                'score_threshold': 0.05,   # RÉDUIT de 0.1
                'allocation_factor': 0.85,
                'max_positions': 12
            }
        }
        
        print(f"📊 IMPROVED WEEKLY SYSTEM")
        print(f"🌍 Universe: {len(self.usa_stocks)} USA + {len(self.europe_stocks)} Europe")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"🔧 FIXED regime detection with realistic thresholds")
        print(f"⚡ Optimized for weekly trading with transaction cost efficiency")
    
    def run_improved_backtest(self):
        """Execute backtest avec détection de régime corrigée"""
        print("\n" + "="*80)
        print("📊 IMPROVED WEEKLY SYSTEM - Fixed Regime Detection")
        print("="*80)
        
        # Download data
        print("\n📊 Step 1: Downloading data...")
        all_data = self.download_all_data()
        
        # Download SPY
        print("\n📈 Step 2: Downloading SPY for IMPROVED regime detection...")
        spy_data = self.download_spy()
        
        # Run simulation
        print("\n📊 Step 3: Running improved weekly simulation...")
        portfolio_results = self.simulate_improved_trading(all_data, spy_data)
        
        # Calculate performance
        print("\n📊 Step 4: Calculating performance...")
        performance = self.calculate_performance_metrics(portfolio_results)
        
        # Results
        results = {
            'config': {
                'system_type': 'improved_weekly',
                'fixed_regime_detection': True,
                'universe_size': len(all_data)
            },
            'performance': performance,
            'trades_summary': portfolio_results['trades_summary'],
            'regime_history': portfolio_results['regime_history']
        }
        
        self.print_improved_summary(results)
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
                
                if len(ticker_data) > 500:
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
    
    def detect_improved_regime(self, spy_data, current_date):
        """Détection de régime CORRIGÉE avec seuils réalistes"""
        
        if spy_data is None:
            return 'consolidation'
        
        historical_spy = spy_data[spy_data.index <= current_date]
        
        if len(historical_spy) < 40:
            return 'consolidation'
        
        try:
            closes = historical_spy['Close']
            
            # Trend detection (PLUS SENSIBLE)
            ma_10 = closes.rolling(10).mean().iloc[-1]
            ma_30 = closes.rolling(30).mean().iloc[-1]
            
            # Volatility (SEUILS RÉALISTES)
            returns = closes.pct_change().dropna()
            volatility_20d = returns.tail(20).std() * np.sqrt(252)
            
            # Momentum (SEUILS RÉALISTES)
            if len(closes) >= 11:
                momentum_10d = (closes.iloc[-1] / closes.iloc[-11]) - 1
            else:
                momentum_10d = 0
            
            # Long-term trend
            if len(closes) >= 60:
                ma_20 = closes.rolling(20).mean().iloc[-1]
                ma_60 = closes.rolling(60).mean().iloc[-1]
                long_term_trend = ma_20 > ma_60
            else:
                long_term_trend = True
            
            # Drawdown (PLUS SENSIBLE)
            if len(closes) >= 30:
                rolling_max = closes.rolling(30).max()
                current_drawdown = (closes.iloc[-1] / rolling_max.iloc[-1]) - 1
            else:
                current_drawdown = 0
            
            # Safe conversions
            ma_10 = float(ma_10) if not pd.isna(ma_10) else float(closes.iloc[-1])
            ma_30 = float(ma_30) if not pd.isna(ma_30) else float(closes.iloc[-1])
            volatility_20d = float(volatility_20d) if not pd.isna(volatility_20d) else 0.15
            momentum_10d = float(momentum_10d) if not pd.isna(momentum_10d) else 0
            current_drawdown = float(current_drawdown) if not pd.isna(current_drawdown) else 0
            
            # IMPROVED regime logic avec seuils RÉALISTES
            is_uptrend = ma_10 > ma_30
            is_high_vol = volatility_20d > 0.18       # RÉDUIT de 0.22 → 0.18
            is_strong_momentum = abs(momentum_10d) > 0.015  # RÉDUIT de 0.025 → 0.015 
            is_recovering = current_drawdown > -0.08 and current_drawdown < -0.02  # ÉLARGI
            is_crisis = current_drawdown < -0.10      # RÉDUIT de -0.15 → -0.10
            
            # Determine regime avec NOUVELLES conditions
            if is_crisis:
                return 'bear_trend'  # Crisis = Bear
            elif is_recovering and is_uptrend:
                return 'recovery'
            elif is_uptrend and long_term_trend and is_strong_momentum:
                return 'bull_trend'
            elif not is_uptrend and is_strong_momentum:
                return 'bear_trend'
            elif is_high_vol:
                return 'high_volatility'
            else:
                return 'consolidation'
                
        except Exception as e:
            return 'consolidation'
    
    def simulate_improved_trading(self, data, spy_data):
        """Simulation avec détection de régime CORRIGÉE"""
        
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
        
        current_regime = 'consolidation'
        
        print(f"  🎯 Simulating {len(trading_dates)} days with IMPROVED regime detection...")
        
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            
            # Progress
            if i % 100 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                print(f"    📅 Progress: {progress:5.1f}% - {current_date} - Regime: {current_regime}")
            
            # Detect regime weekly (every 5 days) - IMPROVED
            if i % 5 == 0:
                new_regime = self.detect_improved_regime(spy_data, date)
                if new_regime != current_regime:
                    print(f"    🔄 REGIME CHANGE: {current_regime} → {new_regime}")
                    current_regime = new_regime
                    regime_history.append({
                        'date': current_date,
                        'regime': current_regime
                    })
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(portfolio, data, date)
            
            # WEEKLY rebalancing with IMPROVED parameters
            if i == 0 or i % self.rebalance_frequency == 0:
                signals = self.calculate_improved_signals(data, date, current_regime)
                trade_summary = self.execute_improved_rebalancing(
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
                'final_positions': len(portfolio['positions']),
                'rebalance_events': len([t for t in trades if len(t) > 0])
            }
        }
    
    def calculate_improved_signals(self, data, date, current_regime):
        """Calcul de signaux avec paramètres de régime"""
        signals = {}
        
        for symbol, prices in data.items():
            try:
                historical_data = prices[prices.index <= date]
                
                if len(historical_data) < 35:
                    continue
                
                # EMA signals (optimized for weekly)
                ema_short = historical_data.ewm(span=10).mean()
                ema_long = historical_data.ewm(span=30).mean()
                
                current_ema_short = float(ema_short.iloc[-1])
                current_ema_long = float(ema_long.iloc[-1])
                ema_signal = 1 if current_ema_short > current_ema_long else 0
                
                # RSI signals (optimized for weekly)
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/14).mean()
                avg_losses = losses.ewm(alpha=1/14).mean()
                
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                current_rsi = float(rsi.iloc[-1])
                rsi_signal = 1 if current_rsi < 75 else 0
                
                # Momentum signal
                current_price = float(historical_data.iloc[-1])
                if len(historical_data) >= 11:
                    momentum = (current_price / float(historical_data.iloc[-11])) - 1
                    momentum_signal = 1 if momentum > 0.015 else 0  # 1.5% in 10 days
                else:
                    momentum_signal = 0
                
                # Weighted scoring
                score = 0.4 * ema_signal + 0.3 * rsi_signal + 0.3 * momentum_signal
                
                signals[symbol] = {
                    'score': score,
                    'ema_signal': ema_signal,
                    'rsi_signal': rsi_signal,
                    'momentum_signal': momentum_signal,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_improved_rebalancing(self, portfolio, data, date, signals, current_regime):
        """Rebalancing avec NOUVEAUX seuils"""
        
        regime_config = self.market_regimes[current_regime]
        
        # Separate by region
        usa_signals = {s: sig for s, sig in signals.items() if s in self.usa_stocks}
        europe_signals = {s: sig for s, sig in signals.items() if s in self.europe_stocks}
        
        # Select with CORRECTED threshold
        score_threshold = regime_config['score_threshold']
        max_positions = regime_config['max_positions']
        
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
        
        # Execute trades
        trades_made = self.execute_trades(portfolio, data, date, target_positions, threshold=0.005)
        
        return {
            'trades_made': trades_made,
            'selected_usa': selected_usa,
            'selected_europe': selected_europe,
            'regime': current_regime,
            'threshold_used': score_threshold,
            'total_positions': len(target_positions)
        }
    
    def execute_trades(self, portfolio, data, date, target_positions, threshold=0.005):
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
    
    def print_improved_summary(self, results):
        """Print summary"""
        print("\n" + "="*80)
        print("📊 IMPROVED WEEKLY SYSTEM SUMMARY")
        print("="*80)
        
        config = results['config']
        perf = results['performance']
        trades = results['trades_summary']
        regime_hist = results['regime_history']
        
        print(f"📊 Improved Configuration:")
        print(f"  System Type:           {config['system_type']}")
        print(f"  Fixed Regime Detection: {config['fixed_regime_detection']}")
        print(f"  Assets analyzed:       {config['universe_size']:>8}")
        
        print(f"\n🚀 IMPROVED Performance:")
        print(f"  Total Return (5Y):     {perf['total_return']:>8.1%}")
        print(f"  Annual Return:         {perf['annual_return']:>8.1%}")
        print(f"  Volatility:            {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:          {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:              {perf['win_rate']:>8.1%}")
        print(f"  Final Value:           ${perf['final_value']:>8,.0f}")
        
        print(f"\n🔄 Improved Trading Activity:")
        print(f"  Total Trades:          {trades['total_trades']:>8,}")
        print(f"  Regime Changes:        {trades['regime_changes']:>8,}")
        print(f"  Final Positions:       {trades['final_positions']:>8,}")
        print(f"  Rebalance Events:      {trades['rebalance_events']:>8,}")
        
        # Regime breakdown
        if regime_hist:
            print(f"\n📊 Regime Detection Results:")
            regime_counts = {}
            for entry in regime_hist:
                regime = entry['regime']
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            for regime, count in regime_counts.items():
                regime_name = self.market_regimes[regime]['name']
                print(f"  {regime_name:25s} {count:>3d} periods")
        
        # Performance comparison
        baseline_annual = 0.173  # Original weekly system
        improvement = perf['annual_return'] - baseline_annual
        
        print(f"\n🎖️ IMPROVED RESULTS:")
        print(f"  Original Weekly:       {baseline_annual:>8.1%}")
        print(f"  Improved Weekly:       {perf['annual_return']:>8.1%}")
        print(f"  Improvement:           {improvement:>8.1%} ({improvement/baseline_annual*100:+.0f}%)")
        
        if trades['regime_changes'] > 0:
            print("✅ SUCCESS: Regime detection is now working!")
            if perf['annual_return'] > baseline_annual:
                print("🎉 EXCELLENT: Improved regime detection boosting performance!")
            else:
                print("👍 GOOD: Regime detection functional, further optimization possible")
        else:
            print("⚠️ Still need regime detection refinement")


def main():
    """Execute improved weekly system"""
    print("📊 IMPROVED WEEKLY TRADING SYSTEM")
    print("Fixed regime detection with realistic thresholds")
    print("="*80)
    
    system = ImprovedWeeklySystem()
    results = system.run_improved_backtest()
    
    return 0


if __name__ == "__main__":
    exit_code = main()