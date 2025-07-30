#!/usr/bin/env python3
"""
Weekly Trading System - Système optimisé pour trading hebdomadaire
Timeframe cohérent : rebalancing hebdomadaire + indicateurs moyens
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class WeeklyTradingSystem:
    """
    Système de trading hebdomadaire avec timeframes cohérents
    """
    
    def __init__(self):
        # Universe complet
        self.usa_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
            'JPM', 'WFC', 'JNJ', 'UNH', 'KO',
            'PG', 'XOM', 'CVX', 'CAT', 'GE'
        ]
        
        self.europe_stocks = [
            'EWG', 'EWQ', 'EWI', 'EWP', 'EWU',
            'EWN', 'EWO', 'EWK', 'ASML', 'SAP', 'NVO', 'NESN.SW'
        ]
        
        # Configuration WEEKLY
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        self.initial_capital = 100000
        self.rebalance_frequency = 5  # WEEKLY rebalancing (every 5 business days)
        
        # Allocation
        self.usa_allocation = 0.70
        self.europe_allocation = 0.30
        self.max_position = 0.08
        
        # PARAMÈTRES OPTIMISÉS POUR WEEKLY TRADING
        self.weekly_params = {
            'ema_short': 10,     # 10-day EMA (2 weeks)
            'ema_long': 30,      # 30-day EMA (6 weeks)
            'rsi_period': 14,    # 14-day RSI (3 weeks)
            'rsi_threshold': 75,
            'momentum_period': 10, # 10-day momentum (2 weeks)
            'volatility_period': 20,  # 20-day volatility (1 month)
            'score_threshold': 0.2,
            'max_positions': 15
        }
        
        # Market regime detection for WEEKLY (medium periods)
        self.weekly_regimes = {
            'bull_trend': {
                'name': 'Weekly Bull Trend',
                'score_threshold': 0.15,
                'allocation_factor': 0.95,
                'max_positions': 15
            },
            'bear_trend': {
                'name': 'Weekly Bear Trend',
                'score_threshold': 0.35,
                'allocation_factor': 0.65,
                'max_positions': 8
            },
            'consolidation': {
                'name': 'Consolidation Phase',
                'score_threshold': 0.1,
                'allocation_factor': 0.85,
                'max_positions': 12
            },
            'high_volatility': {
                'name': 'High Volatility Period',
                'score_threshold': 0.25,
                'allocation_factor': 0.75,
                'max_positions': 10
            },
            'recovery': {
                'name': 'Recovery Phase',
                'score_threshold': 0.2,
                'allocation_factor': 0.90,
                'max_positions': 14
            }
        }
        
        print(f"📊 WEEKLY TRADING SYSTEM")
        print(f"🌍 Universe: {len(self.usa_stocks)} USA + {len(self.europe_stocks)} Europe")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"🔄 Rebalancing: WEEKLY (every 5 business days)")
        print(f"⚡ Optimized for: Medium-term signals & weekly execution")
        print(f"📊 Parameters: EMA({self.weekly_params['ema_short']}/{self.weekly_params['ema_long']}), RSI({self.weekly_params['rsi_period']})")
    
    def run_weekly_backtest(self):
        """Execute weekly trading backtest"""
        print("\n" + "="*80)
        print("📊 WEEKLY TRADING SYSTEM - Optimized Weekly Execution")
        print("="*80)
        
        # Download data
        print("\n📊 Step 1: Downloading data...")
        all_data = self.download_all_data()
        
        # Download SPY for regime detection
        print("\n📈 Step 2: Downloading SPY for weekly regime detection...")
        spy_data = self.download_spy()
        
        # Run weekly simulation
        print("\n📊 Step 3: Running weekly trading simulation...")
        portfolio_results = self.simulate_weekly_trading(all_data, spy_data)
        
        # Calculate performance
        print("\n📊 Step 4: Calculating performance...")
        performance = self.calculate_performance_metrics(portfolio_results)
        
        # Results
        results = {
            'config': {
                'system_type': 'weekly_trading',
                'rebalance_frequency': 'weekly',
                'parameters': self.weekly_params,
                'universe_size': len(all_data)
            },
            'performance': performance,
            'trades_summary': portfolio_results['trades_summary']
        }
        
        self.print_weekly_summary(results)
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
    
    def detect_weekly_regime(self, spy_data, current_date):
        """Détection de régime pour trading hebdomadaire (périodes moyennes)"""
        
        if spy_data is None:
            return 'consolidation'
        
        historical_spy = spy_data[spy_data.index <= current_date]
        
        if len(historical_spy) < 40:
            return 'consolidation'
        
        try:
            closes = historical_spy['Close']
            
            # Medium-term trend (10-day vs 30-day)
            ma_10 = closes.rolling(10).mean().iloc[-1]
            ma_30 = closes.rolling(30).mean().iloc[-1]
            
            # Medium-term volatility (20-day)
            returns = closes.pct_change().dropna()
            volatility_20d = returns.tail(20).std() * np.sqrt(252)
            
            # Medium-term momentum (10-day)
            if len(closes) >= 11:
                momentum_10d = (closes.iloc[-1] / closes.iloc[-11]) - 1
            else:
                momentum_10d = 0
            
            # Longer-term trend confirmation (20-day vs 60-day)
            if len(closes) >= 60:
                ma_20 = closes.rolling(20).mean().iloc[-1]
                ma_60 = closes.rolling(60).mean().iloc[-1]
                long_term_trend = ma_20 > ma_60
            else:
                long_term_trend = True
            
            # Drawdown for recovery detection
            if len(closes) >= 30:
                rolling_max = closes.rolling(30).max()
                current_drawdown = (closes.iloc[-1] / rolling_max.iloc[-1]) - 1
            else:
                current_drawdown = 0
            
            # Safe conversions - FIXED pandas Series issues
            ma_10 = float(ma_10) if not pd.isna(ma_10) else float(closes.iloc[-1])
            ma_30 = float(ma_30) if not pd.isna(ma_30) else float(closes.iloc[-1])
            volatility_20d = float(volatility_20d) if not pd.isna(volatility_20d) else 0.15
            momentum_10d = float(momentum_10d) if not pd.isna(momentum_10d) else 0
            current_drawdown = float(current_drawdown) if not pd.isna(current_drawdown) else 0
            
            # Convert ma_20 and ma_60 if they exist
            if len(closes) >= 60:
                ma_20 = float(ma_20) if not pd.isna(ma_20) else float(closes.iloc[-1])
                ma_60 = float(ma_60) if not pd.isna(ma_60) else float(closes.iloc[-1])
                long_term_trend = ma_20 > ma_60
            
            # Weekly regime logic - CORRECTED THRESHOLDS from debug
            is_uptrend = ma_10 > ma_30
            is_high_vol = volatility_20d > 0.12  # FIXED: 22% → 12% (from debug analysis)
            is_strong_momentum = abs(momentum_10d) > 0.005  # FIXED: 2.5% → 0.5% (from debug analysis)  
            is_crisis = current_drawdown < -0.05  # NEW: Crisis detection at -5% (from debug analysis)
            is_recovering = current_drawdown > -0.08 and current_drawdown < -0.02  # EXPANDED recovery range
            
            # Determine weekly regime - IMPROVED LOGIC with crisis detection
            if is_crisis:
                return 'bear_trend'  # Crisis = Bear market
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
            print(f"    ⚠️ Regime detection error: {e}")  # DEBUG: Show what's failing
            return 'consolidation'
    
    def simulate_weekly_trading(self, data, spy_data):
        """Simulation de trading hebdomadaire"""
        
        # Initialize
        portfolio = {
            'cash': float(self.initial_capital),
            'positions': {},
            'value': float(self.initial_capital)
        }
        
        # Generate trading dates (business days only)
        trading_dates = pd.bdate_range(
            start=self.start_date, 
            end=self.end_date
        ).tolist()
        
        # Track results
        history = []
        trades = []
        regime_history = []
        
        current_regime = 'consolidation'
        
        print(f"  📊 Simulating {len(trading_dates)} days with WEEKLY rebalancing...")
        
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            
            # Progress every 100 days
            if i % 100 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                print(f"    📅 Progress: {progress:5.1f}% - {current_date} - Regime: {current_regime}")
            
            # Detect regime weekly (every 5 days)
            if i % 5 == 0:
                new_regime = self.detect_weekly_regime(spy_data, date)
                if new_regime != current_regime:
                    current_regime = new_regime
                    regime_history.append({
                        'date': current_date,
                        'regime': current_regime
                    })
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(portfolio, data, date)
            
            # WEEKLY rebalancing
            if i == 0 or i % self.rebalance_frequency == 0:  # Every 5 business days
                signals = self.calculate_weekly_signals(data, date)
                trade_summary = self.execute_weekly_rebalancing(
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
    
    def calculate_weekly_signals(self, data, date):
        """Calcul de signaux optimisés pour trading hebdomadaire"""
        signals = {}
        
        for symbol, prices in data.items():
            try:
                historical_data = prices[prices.index <= date]
                
                if len(historical_data) < 35:  # Minimum for weekly signals
                    continue
                
                # Medium-term EMA (weekly optimized)
                ema_short = historical_data.ewm(span=self.weekly_params['ema_short']).mean()
                ema_long = historical_data.ewm(span=self.weekly_params['ema_long']).mean()
                
                current_ema_short = float(ema_short.iloc[-1])
                current_ema_long = float(ema_long.iloc[-1])
                ema_signal = 1 if current_ema_short > current_ema_long else 0
                
                # EMA strength signal (how strong is the crossover)
                ema_strength = (current_ema_short - current_ema_long) / current_ema_long
                ema_strength_signal = 1 if abs(ema_strength) > 0.02 else 0  # 2% separation
                
                # Medium-term RSI (weekly optimized)
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/self.weekly_params['rsi_period']).mean()
                avg_losses = losses.ewm(alpha=1/self.weekly_params['rsi_period']).mean()
                
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                current_rsi = float(rsi.iloc[-1])
                rsi_signal = 1 if current_rsi < self.weekly_params['rsi_threshold'] else 0
                
                # Medium-term momentum (weekly optimized)
                current_price = float(historical_data.iloc[-1])
                if len(historical_data) >= self.weekly_params['momentum_period'] + 1:
                    momentum = (current_price / float(historical_data.iloc[-(self.weekly_params['momentum_period']+1)])) - 1
                    momentum_signal = 1 if momentum > 0.02 else 0  # 2% in 10 days
                else:
                    momentum_signal = 0
                
                # Volume confirmation (if available)
                volume_signal = 0  # Simplified for now
                
                # Price position relative to recent range
                if len(historical_data) >= 20:
                    recent_high = float(historical_data.tail(20).max())
                    recent_low = float(historical_data.tail(20).min())
                    price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
                    position_signal = 1 if price_position > 0.6 else 0  # In upper 40% of range
                else:
                    position_signal = 0
                
                # Weekly scoring (balanced approach for medium-term)
                score = (0.35 * ema_signal + 
                        0.2 * rsi_signal + 
                        0.25 * momentum_signal + 
                        0.1 * ema_strength_signal + 
                        0.1 * position_signal)
                
                signals[symbol] = {
                    'score': score,
                    'ema_signal': ema_signal,
                    'rsi_signal': rsi_signal,
                    'momentum_signal': momentum_signal,
                    'ema_strength_signal': ema_strength_signal,
                    'position_signal': position_signal,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_weekly_rebalancing(self, portfolio, data, date, signals, current_regime):
        """Rebalancing hebdomadaire"""
        
        regime_config = self.weekly_regimes[current_regime]
        
        # Separate by region
        usa_signals = {s: sig for s, sig in signals.items() if s in self.usa_stocks}
        europe_signals = {s: sig for s, sig in signals.items() if s in self.europe_stocks}
        
        # Select with regime-adjusted threshold
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
        
        # Execute trades (medium threshold for weekly trading)
        trades_made = self.execute_trades(portfolio, data, date, target_positions, threshold=0.005)  # 0.5% threshold
        
        return {
            'trades_made': trades_made,
            'selected_usa': selected_usa,
            'selected_europe': selected_europe,
            'regime': current_regime,
            'threshold_used': score_threshold,
            'total_positions': len(target_positions)
        }
    
    def execute_trades(self, portfolio, data, date, target_positions, threshold=0.005):
        """Execute trades optimisé pour weekly"""
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
    
    def print_weekly_summary(self, results):
        """Print summary optimisé pour weekly"""
        print("\n" + "="*80)
        print("📊 WEEKLY TRADING SYSTEM SUMMARY")
        print("="*80)
        
        config = results['config']
        perf = results['performance']
        trades = results['trades_summary']
        
        print(f"📊 Weekly Configuration:")
        print(f"  System Type:           {config['system_type']}")
        print(f"  Rebalance Frequency:   {config['rebalance_frequency']}")
        print(f"  EMA Periods:           {config['parameters']['ema_short']}/{config['parameters']['ema_long']}")
        print(f"  RSI Period:            {config['parameters']['rsi_period']}")
        print(f"  Assets analyzed:       {config['universe_size']:>8}")
        
        print(f"\n🚀 WEEKLY Performance:")
        print(f"  Total Return (5Y):     {perf['total_return']:>8.1%}")
        print(f"  Annual Return:         {perf['annual_return']:>8.1%}")
        print(f"  Volatility:            {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:          {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:              {perf['win_rate']:>8.1%}")
        print(f"  Final Value:           ${perf['final_value']:>8,.0f}")
        
        print(f"\n🔄 Weekly Trading Activity:")
        print(f"  Total Trades:          {trades['total_trades']:>8,}")
        print(f"  Regime Changes:        {trades['regime_changes']:>8,}")
        print(f"  Final Positions:       {trades['final_positions']:>8,}")
        print(f"  Rebalance Events:      {trades['rebalance_events']:>8,}")
        print(f"  Avg Trades/Rebalance:  {trades['total_trades']/max(trades['rebalance_events'],1):>8.1f}")
        
        baseline_annual = 0.097
        improvement = perf['annual_return'] - baseline_annual
        
        print(f"\n🎖️ WEEKLY RESULTS:")
        print(f"  Baseline Performance:  {baseline_annual:>8.1%}")
        print(f"  Weekly Performance:    {perf['annual_return']:>8.1%}")
        print(f"  Improvement:           {improvement:>8.1%} ({improvement/baseline_annual*100:+.0f}%)")
        
        if perf['annual_return'] > 0.15:
            print("🎉 EXCELLENT: Weekly trading achieving >15% annual!")
        elif perf['annual_return'] > 0.12:
            print("✅ VERY GOOD: Weekly trading achieving >12% annual!")
        elif improvement > 0:
            print("👍 IMPROVED: Weekly trading outperforming baseline!")
        else:
            print("⚠️ Weekly trading needs optimization")


def main():
    """Execute weekly trading system"""
    print("📊 WEEKLY TRADING SYSTEM")
    print("Optimized for medium-term signals & weekly execution")
    print("="*80)
    
    system = WeeklyTradingSystem()
    results = system.run_weekly_backtest()
    
    return 0


if __name__ == "__main__":
    exit_code = main()