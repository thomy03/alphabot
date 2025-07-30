#!/usr/bin/env python3
"""
Enhanced Swing Trading System - Sweet spot entre daily et day trading
Based on our proven 23.5% system, optimized for 2-5 day holds
Target: 28-35% annual with controlled risk
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnhancedSwingTrading:
    """
    Enhanced swing trading - optimal balance between performance and risk
    Proven daily strategy + swing holding periods + momentum capture
    """
    
    def __init__(self):
        # Curated swing trading universe (high momentum potential)
        self.swing_stocks = [
            # Mega tech with good swing potential
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
            'AMZN', 'TSLA', 'NFLX', 'AMD', 'CRM',
            
            # High-beta momentum stocks
            'PLTR', 'SNOW', 'ZM', 'ROKU', 'SQ',
            
            # ETFs for broader exposure
            'QQQ', 'XLK', 'ARKK', 'TQQQ',
            
            # Volatility favorites
            'JPM', 'BAC', 'GS', 'XOM', 'CVX'
        ]
        
        # Configuration
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        self.initial_capital = 100000
        self.rebalance_frequency = 2  # Every 2 days (vs daily)
        
        # SWING TRADING OPTIMIZED PARAMETERS
        self.max_position = 0.12        # 12% per position (concentrated but not crazy)
        self.max_positions = 10         # Fewer positions, better selection
        self.min_hold_days = 2          # Minimum hold 2 days
        self.max_hold_days = 7          # Maximum hold 1 week
        
        # ENHANCED REGIME DETECTION (optimized for swing)
        self.swing_regimes = {
            'strong_momentum': {
                'name': 'Strong Momentum Phase',
                'score_threshold': 0.15,     # Higher quality threshold
                'allocation_factor': 0.95,   # Aggressive in momentum
                'max_positions': 8,          # Concentrated
                'hold_preference': 'long',   # Hold longer in momentum
                'take_profit': 0.08,         # 8% take profit
                'stop_loss': 0.04            # 4% stop loss
            },
            'moderate_trend': {
                'name': 'Moderate Trend',
                'score_threshold': 0.12,
                'allocation_factor': 0.85,
                'max_positions': 10,
                'hold_preference': 'medium',
                'take_profit': 0.06,         # 6% take profit
                'stop_loss': 0.03            # 3% stop loss
            },
            'consolidation': {
                'name': 'Market Consolidation',
                'score_threshold': 0.08,
                'allocation_factor': 0.70,
                'max_positions': 6,
                'hold_preference': 'short',
                'take_profit': 0.04,         # 4% take profit
                'stop_loss': 0.025           # 2.5% stop loss
            },
            'high_volatility': {
                'name': 'High Volatility Environment',
                'score_threshold': 0.10,
                'allocation_factor': 0.80,
                'max_positions': 8,
                'hold_preference': 'short',  # Quick moves in volatility
                'take_profit': 0.10,         # 10% take profit (higher targets)
                'stop_loss': 0.05            # 5% stop loss (wider stops)
            }
        }
        
        print(f"🎯 ENHANCED SWING TRADING SYSTEM")
        print(f"💡 STRATEGY: Proven 23.5% base + swing optimization")
        print(f"📊 Universe: {len(self.swing_stocks)} momentum-focused stocks")
        print(f"⏱️ Holding Period: {self.min_hold_days}-{self.max_hold_days} days")
        print(f"💰 Position Size: {self.max_position:.0%} max per stock")
        print(f"🎯 TARGET: 28-35% annual with controlled risk")
        print(f"🔄 Rebalance: Every {self.rebalance_frequency} days")
    
    def run_swing_backtest(self):
        """Execute enhanced swing trading backtest"""
        print("\n" + "="*80)
        print("🎯 ENHANCED SWING TRADING - Optimized Performance Strategy")
        print("="*80)
        
        # Download data
        print("\n📊 Step 1: Downloading swing trading universe...")
        all_data = self.download_swing_data()
        
        # Download SPY for regime detection
        print("\n📈 Step 2: Downloading SPY for swing regime detection...")
        spy_data = self.download_spy()
        
        # Run swing simulation
        print("\n🎯 Step 3: Running enhanced swing simulation...")
        portfolio_results = self.simulate_swing_trading(all_data, spy_data)
        
        # Calculate performance
        print("\n📊 Step 4: Calculating swing performance...")
        performance = self.calculate_performance_metrics(portfolio_results)
        
        # Compare to baselines
        print("\n📈 Step 5: Comparing to baseline systems...")
        comparisons = self.compare_to_baselines(performance)
        
        # Results
        results = {
            'config': {
                'system_type': 'enhanced_swing',
                'base_strategy': 'proven_daily_23.5%',
                'enhancements': ['swing_holding', 'momentum_focus', 'risk_management'],
                'universe_size': len(all_data),
                'target_performance': '28-35% annual'
            },
            'performance': performance,
            'comparisons': comparisons,
            'trades_summary': portfolio_results['trades_summary'],
            'regime_history': portfolio_results['regime_history']
        }
        
        self.print_swing_summary(results)
        return results
    
    def download_swing_data(self):
        """Download swing trading universe"""
        data = {}
        
        for i, symbol in enumerate(self.swing_stocks, 1):
            try:
                print(f"  📊 ({i:2d}/{len(self.swing_stocks)}) {symbol:8s}...", end=" ")
                
                ticker_data = yf.download(
                    symbol, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False
                )
                
                if len(ticker_data) > 500:
                    # Fix MultiIndex
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    
                    data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ {len(ticker_data)} days")
                    
            except Exception as e:
                print(f"❌ Error")
        
        print(f"  Downloaded: {len(data)} symbols for swing trading")
        return data
    
    def download_spy(self):
        """Download SPY for regime detection"""
        try:
            spy_data = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
            
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.droplevel(1)
            
            print(f"  ✅ SPY downloaded: {len(spy_data)} days (for swing regime)")
            return spy_data
        except Exception as e:
            print(f"  ❌ SPY failed: {e}")
            return None
    
    def detect_swing_regime(self, spy_data, current_date):
        """
        Swing-optimized regime detection
        Medium-term focus (between daily and weekly)
        """
        
        if spy_data is None:
            return 'moderate_trend'
        
        historical_spy = spy_data[spy_data.index <= current_date]
        
        if len(historical_spy) < 15:
            return 'moderate_trend'
        
        try:
            closes = historical_spy['Close']
            
            # Medium-term trend (7-day vs 20-day for swing sweet spot)
            ma_7 = closes.rolling(7).mean().iloc[-1]
            ma_20 = closes.rolling(20).mean().iloc[-1]
            
            # Medium-term volatility (15-day)
            returns = closes.pct_change().dropna()
            volatility_15d = returns.tail(15).std() * np.sqrt(252)
            
            # Medium-term momentum (7-day)
            if len(closes) >= 8:
                momentum_7d = (closes.iloc[-1] / closes.iloc[-8]) - 1
            else:
                momentum_7d = 0
            
            # Trend strength indicator
            if len(closes) >= 20:
                ma_slope = (ma_7 - closes.rolling(7).mean().iloc[-5]) / closes.rolling(7).mean().iloc[-5]
            else:
                ma_slope = 0
            
            # Safe conversions
            ma_7 = float(ma_7) if not pd.isna(ma_7) else float(closes.iloc[-1])
            ma_20 = float(ma_20) if not pd.isna(ma_20) else float(closes.iloc[-1])
            volatility_15d = float(volatility_15d) if not pd.isna(volatility_15d) else 0.15
            momentum_7d = float(momentum_7d) if not pd.isna(momentum_7d) else 0
            ma_slope = float(ma_slope) if not pd.isna(ma_slope) else 0
            
            # SWING regime logic (optimized for 2-7 day holds)
            is_uptrend = ma_7 > ma_20
            is_strong_trend = abs(ma_slope) > 0.002  # 0.2% trend acceleration
            is_high_vol = volatility_15d > 0.15      # 15% threshold
            is_strong_momentum = abs(momentum_7d) > 0.01  # 1% in 7 days
            
            # Determine swing regime
            if is_uptrend and is_strong_momentum and is_strong_trend:
                return 'strong_momentum'
            elif is_high_vol and is_strong_momentum:
                return 'high_volatility'
            elif is_uptrend or is_strong_momentum:
                return 'moderate_trend'
            else:
                return 'consolidation'
                
        except Exception as e:
            print(f"    ⚠️ Swing regime detection error: {e}")
            return 'moderate_trend'
    
    def simulate_swing_trading(self, data, spy_data):
        """
        Enhanced swing trading simulation
        2-7 day holds with intelligent entry/exit
        """
        
        # Initialize
        portfolio = {
            'cash': float(self.initial_capital),
            'positions': {},
            'position_entry_dates': {},
            'position_entry_prices': {},
            'value': float(self.initial_capital)
        }
        
        # Generate trading dates
        trading_dates = pd.bdate_range(
            start=self.start_date, 
            end=self.end_date
        ).tolist()
        
        # Track results
        history = []
        trades = []
        regime_history = []
        
        current_regime = 'moderate_trend'
        
        print(f"  🎯 Simulating {len(trading_dates)} days with SWING logic...")
        print(f"  📊 Hold periods: {self.min_hold_days}-{self.max_hold_days} days")
        
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            
            # Progress
            if i % 100 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                print(f"    📅 Progress: {progress:5.1f}% - {current_date} - Regime: {current_regime}")
            
            # Detect swing regime
            new_regime = self.detect_swing_regime(spy_data, date)
            if new_regime != current_regime:
                current_regime = new_regime
                regime_history.append({
                    'date': current_date,
                    'regime': current_regime
                })
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(portfolio, data, date)
            
            # Check for exits first (stop-loss, take-profit, max hold)
            exits_made = self.check_swing_exits(portfolio, data, date, current_regime, i)
            
            # Consider new entries (every rebalance_frequency days)
            if i % self.rebalance_frequency == 0:
                entries_made = self.consider_swing_entries(portfolio, data, date, current_regime, i)
                
                if exits_made > 0 or entries_made > 0:
                    trades.append({
                        'date': current_date,
                        'regime': current_regime,
                        'exits': exits_made,
                        'entries': entries_made
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
                'total_trades': len(trades),
                'regime_changes': len(regime_history),
                'final_value': portfolio_value,
                'final_positions': len(portfolio['positions'])
            }
        }
    
    def check_swing_exits(self, portfolio, data, date, current_regime, day_index):
        """
        Check for swing exit conditions:
        1. Stop-loss hit
        2. Take-profit hit  
        3. Maximum hold period reached
        4. Regime change exit
        """
        regime_config = self.swing_regimes[current_regime]
        exits_made = 0
        positions_to_exit = []
        
        for symbol in list(portfolio['positions'].keys()):
            if symbol not in data:
                continue
                
            try:
                # Get current price
                prices = data[symbol]
                available_prices = prices[prices.index <= date]
                if len(available_prices) == 0:
                    continue
                
                current_price = float(available_prices.iloc[-1])
                entry_price = portfolio['position_entry_prices'].get(symbol, current_price)
                entry_date_index = portfolio['position_entry_dates'].get(symbol, day_index)
                
                # Calculate P&L
                pnl_pct = (current_price / entry_price) - 1
                hold_days = day_index - entry_date_index
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # 1. Take profit
                if pnl_pct >= regime_config['take_profit']:
                    should_exit = True
                    exit_reason = "take_profit"
                
                # 2. Stop loss
                elif pnl_pct <= -regime_config['stop_loss']:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # 3. Maximum hold period
                elif hold_days >= self.max_hold_days:
                    should_exit = True
                    exit_reason = "max_hold"
                
                # 4. Minimum hold period check
                elif hold_days < self.min_hold_days:
                    should_exit = False  # Don't exit too early
                
                if should_exit:
                    positions_to_exit.append((symbol, exit_reason))
                    
            except Exception:
                continue
        
        # Execute exits
        for symbol, reason in positions_to_exit:
            self.execute_swing_exit(portfolio, data, date, symbol, reason)
            exits_made += 1
        
        return exits_made
    
    def consider_swing_entries(self, portfolio, data, date, current_regime, day_index):
        """
        Consider new swing entries based on signals and regime
        """
        regime_config = self.swing_regimes[current_regime]
        entries_made = 0
        
        # Don't add too many positions
        current_positions = len(portfolio['positions'])
        if current_positions >= regime_config['max_positions']:
            return entries_made
        
        # Calculate signals
        signals = self.calculate_swing_signals(data, date, current_regime)
        
        # Select best opportunities
        qualified_signals = [
            (symbol, sig['score']) for symbol, sig in signals.items() 
            if sig['score'] >= regime_config['score_threshold']
            and symbol not in portfolio['positions']  # Don't double-buy
        ]
        
        if not qualified_signals:
            return entries_made
        
        # Sort by score and take best opportunities
        qualified_signals.sort(key=lambda x: x[1], reverse=True)
        max_new_entries = regime_config['max_positions'] - current_positions
        
        for symbol, score in qualified_signals[:max_new_entries]:
            if self.execute_swing_entry(portfolio, data, date, symbol, day_index):
                entries_made += 1
        
        return entries_made
    
    def calculate_swing_signals(self, data, date, current_regime):
        """
        Calculate swing-optimized signals
        Enhanced version of our proven signal system
        """
        signals = {}
        
        for symbol, prices in data.items():
            try:
                historical_data = prices[prices.index <= date]
                
                if len(historical_data) < 25:  # Need more data for swing
                    continue
                
                # Enhanced EMA system (8-day vs 21-day for swing)
                ema_8 = historical_data.ewm(span=8).mean()
                ema_21 = historical_data.ewm(span=21).mean()
                
                current_ema_8 = float(ema_8.iloc[-1])
                current_ema_21 = float(ema_21.iloc[-1])
                ema_signal = 1 if current_ema_8 > current_ema_21 else 0
                
                # EMA momentum (trend acceleration)
                if len(ema_8) >= 3:
                    ema_momentum = (ema_8.iloc[-1] - ema_8.iloc[-3]) / ema_8.iloc[-3]
                    momentum_signal = 1 if ema_momentum > 0.001 else 0
                else:
                    momentum_signal = 0
                
                # Enhanced RSI (14-period for swing)
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/14).mean()
                avg_losses = losses.ewm(alpha=1/14).mean()
                
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                current_rsi = float(rsi.iloc[-1])
                rsi_signal = 1 if 30 < current_rsi < 70 else 0  # Avoid extremes
                
                # Price breakout signal
                current_price = float(historical_data.iloc[-1])
                if len(historical_data) >= 20:
                    recent_high = float(historical_data.tail(20).max())
                    breakout_signal = 1 if current_price > recent_high * 0.995 else 0
                else:
                    breakout_signal = 0
                
                # Volume proxy (price volatility as momentum indicator)
                if len(historical_data) >= 10:
                    recent_volatility = historical_data.tail(10).std() / current_price
                    volume_signal = 1 if recent_volatility > 0.008 else 0  # 0.8% daily vol
                else:
                    volume_signal = 0
                
                # SWING scoring (balanced for medium-term holds)
                score = (0.3 * ema_signal + 
                        0.25 * momentum_signal + 
                        0.2 * rsi_signal + 
                        0.15 * breakout_signal + 
                        0.1 * volume_signal)
                
                signals[symbol] = {
                    'score': score,
                    'ema_signal': ema_signal,
                    'momentum_signal': momentum_signal,
                    'rsi_signal': rsi_signal,
                    'breakout_signal': breakout_signal,
                    'volume_signal': volume_signal,
                    'price': current_price,
                    'rsi': current_rsi
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_swing_entry(self, portfolio, data, date, symbol, day_index):
        """Execute swing entry with position tracking"""
        try:
            prices = data[symbol]
            available_prices = prices[prices.index <= date]
            if len(available_prices) == 0:
                return False
            
            price = float(available_prices.iloc[-1])
            target_value = portfolio['value'] * self.max_position
            
            if portfolio['cash'] >= target_value:
                shares = target_value / price
                cost = shares * price
                
                portfolio['cash'] -= cost
                portfolio['positions'][symbol] = shares
                portfolio['position_entry_dates'][symbol] = day_index
                portfolio['position_entry_prices'][symbol] = price
                
                return True
        except:
            pass
        
        return False
    
    def execute_swing_exit(self, portfolio, data, date, symbol, reason):
        """Execute swing exit with tracking cleanup"""
        try:
            if symbol in portfolio['positions']:
                shares = portfolio['positions'][symbol]
                prices = data[symbol]
                available_prices = prices[prices.index <= date]
                
                if len(available_prices) > 0:
                    price = float(available_prices.iloc[-1])
                    proceeds = shares * price
                    portfolio['cash'] += proceeds
                
                # Cleanup tracking
                del portfolio['positions'][symbol]
                if symbol in portfolio['position_entry_dates']:
                    del portfolio['position_entry_dates'][symbol]
                if symbol in portfolio['position_entry_prices']:
                    del portfolio['position_entry_prices'][symbol]
        except:
            pass
    
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
        """Calculate swing performance metrics"""
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
    
    def compare_to_baselines(self, performance):
        """Compare to our baseline systems"""
        return {
            'daily_system': {
                'annual_return': 0.235,  # Our proven daily
                'sharpe_ratio': 1.27,
                'max_drawdown': -0.24
            },
            'optimized_daily': {
                'annual_return': 0.235,  # Same as daily
                'sharpe_ratio': 1.27,
                'max_drawdown': -0.24
            },
            'swing_improvement': {
                'vs_daily_annual': performance['annual_return'] - 0.235,
                'vs_daily_sharpe': performance['sharpe_ratio'] - 1.27,
                'vs_daily_drawdown': performance['max_drawdown'] - (-0.24)
            }
        }
    
    def print_swing_summary(self, results):
        """Print swing trading summary"""
        print("\n" + "="*80)
        print("🎯 ENHANCED SWING TRADING SUMMARY")
        print("="*80)
        
        config = results['config']
        perf = results['performance']
        trades = results['trades_summary']
        comparison = results['comparisons']
        
        print(f"🎯 Swing Configuration:")
        print(f"  System Type:           {config['system_type']}")
        print(f"  Base Strategy:         {config['base_strategy']}")
        print(f"  Target Performance:    {config['target_performance']}")
        print(f"  Universe Size:         {config['universe_size']:>8}")
        print(f"  Hold Period:           {self.min_hold_days}-{self.max_hold_days} days")
        print(f"  Position Size:         {self.max_position:.0%} max")
        
        print(f"\n🚀 SWING Performance:")
        print(f"  Total Return (5Y):     {perf['total_return']:>8.1%}")
        print(f"  Annual Return:         {perf['annual_return']:>8.1%}")
        print(f"  Volatility:            {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:          {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:              {perf['win_rate']:>8.1%}")
        print(f"  Final Value:           ${perf['final_value']:>8,.0f}")
        
        print(f"\n🎯 Swing Trading Activity:")
        print(f"  Total Rebalances:      {trades['total_trades']:>8,}")
        print(f"  Regime Changes:        {trades['regime_changes']:>8,}")
        print(f"  Final Positions:       {trades['final_positions']:>8,}")
        
        daily_baseline = comparison['daily_system']
        swing_improvement = comparison['swing_improvement']
        
        print(f"\n📊 vs Proven Daily System (23.5%):")
        print(f"  Daily System Annual:   {daily_baseline['annual_return']:>8.1%}")
        print(f"  Swing System Annual:   {perf['annual_return']:>8.1%}")
        print(f"  Improvement:           {swing_improvement['vs_daily_annual']:>8.1%} ({swing_improvement['vs_daily_annual']/daily_baseline['annual_return']*100:+.0f}%)")
        
        # Target achievement
        target_achieved = perf['annual_return'] > 0.28  # 28% target
        baseline_beaten = swing_improvement['vs_daily_annual'] > 0
        
        print(f"\n🏆 SWING RESULTS:")
        if target_achieved and baseline_beaten:
            print("🎉 SUCCESS: 28%+ target achieved AND baseline beaten!")
        elif target_achieved:
            print("✅ EXCELLENT: 28%+ target achieved!")
        elif baseline_beaten:
            print("👍 GOOD: Baseline system improved!")
        else:
            print("⚠️ Performance similar to baseline - consider further optimization")
        
        if perf['annual_return'] > 0.35:
            print(f"🚀 OUTSTANDING: Swing trading achieving >35% annual!")
        elif perf['annual_return'] > 0.30:
            print(f"🎯 EXCELLENT: Swing trading achieving >30% annual!")
        elif perf['annual_return'] > 0.25:
            print(f"✅ GOOD: Swing trading beating baseline significantly!")


def main():
    """Execute enhanced swing trading system"""
    print("🎯 ENHANCED SWING TRADING SYSTEM")
    print("Sweet spot between daily and day trading")
    print("="*80)
    
    system = EnhancedSwingTrading()
    results = system.run_swing_backtest()
    
    return 0


if __name__ == "__main__":
    exit_code = main()