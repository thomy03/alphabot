#!/usr/bin/env python3
"""
Fixed 5-Year Backtest - Correction des bugs identifiés
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class Fixed5YearBacktester:
    """
    Backtest 5 ans avec corrections des bugs
    """
    
    def __init__(self):
        # Univers simplifié pour debug
        self.usa_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
            'JPM', 'BAC', 'JNJ', 'UNH', 'KO'
        ]
        
        self.europe_stocks = [
            'EWG', 'EWQ', 'EWU', 'EWI', 'EWP'
        ]
        
        # Configuration
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        self.initial_capital = 100000
        self.rebalance_frequency = 21  # Monthly instead of weekly
        
        # Allocation target
        self.usa_allocation = 0.70
        self.europe_allocation = 0.30
        self.max_position = 0.08  # 8% max per asset (more flexible)
        
        # Seuils plus permissifs
        self.min_score_threshold = 0.4  # Reduced from 0.6
        self.min_trade_threshold = 0.005  # 0.5% instead of 1%
        
        print(f"🌍 Universe: {len(self.usa_stocks)} USA + {len(self.europe_stocks)} Europe")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"💰 Initial capital: ${self.initial_capital:,}")
        print(f"🔄 Rebalancing: Every {self.rebalance_frequency} days")
        print(f"🎯 Score threshold: {self.min_score_threshold}")
    
    def run_fixed_backtest(self):
        """Execute le backtest avec corrections"""
        print("\n" + "="*80)
        print("📈 FIXED 5-YEAR BACKTEST")
        print("="*80)
        
        # Step 1: Download data
        print("\n📊 Step 1: Downloading data...")
        all_data = self.download_all_data()
        
        if len(all_data) < 10:
            print("❌ Insufficient data downloaded")
            return None
        
        # Step 2: Create trading calendar
        print("\n📅 Step 2: Creating trading calendar...")
        trading_dates = self.create_trading_calendar()
        
        # Step 3: Run simulation with debugging
        print("\n💼 Step 3: Running simulation with debug...")
        portfolio_results = self.simulate_trading_with_debug(all_data, trading_dates)
        
        # Step 4: Calculate metrics
        print("\n📊 Step 4: Calculating metrics...")
        performance = self.calculate_performance_metrics(portfolio_results)
        
        # Step 5: Generate results
        results = {
            'config': {
                'universe_size': len(all_data),
                'trading_days': len(trading_dates),
                'rebalance_frequency': self.rebalance_frequency,
                'score_threshold': self.min_score_threshold,
                'trade_threshold': self.min_trade_threshold
            },
            'performance': performance,
            'portfolio_history': portfolio_results['history'][-10:],
            'trades_summary': portfolio_results['trades_summary'],
            'debug_info': portfolio_results['debug_info']
        }
        
        self.print_detailed_summary(results)
        return results
    
    def download_all_data(self):
        """Download avec gestion d'erreurs améliorée"""
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
                
                if len(ticker_data) > 800:  # ~3+ years minimum
                    data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data):4d} days")
                else:
                    print(f"❌ {len(ticker_data):4d} days (insufficient)")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:15]}...")
        
        print(f"\n  ✅ Downloaded: {len(data)}/{len(all_symbols)} symbols")
        return data
    
    def create_trading_calendar(self):
        """Create business days calendar"""
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        trading_dates = pd.bdate_range(start=start, end=end).tolist()
        return trading_dates
    
    def simulate_trading_with_debug(self, data, trading_dates):
        """Simulation avec debugging détaillé"""
        
        # Initialize portfolio
        portfolio = {
            'cash': float(self.initial_capital),
            'positions': {},
            'value': float(self.initial_capital)
        }
        
        # Track results
        history = []
        trades = []
        rebalance_dates = []
        debug_info = {
            'signal_generations': 0,
            'successful_signals': 0,
            'assets_selected': 0,
            'trades_attempted': 0,
            'trades_executed': 0,
            'rebalance_events': 0
        }
        
        print(f"  🎯 Simulating {len(trading_dates)} trading days...")
        
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            
            # Progress indicator
            if i % 125 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                print(f"    📅 Progress: {progress:5.1f}% - {current_date}")
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(portfolio, data, date)
            
            # Check if rebalancing needed
            if i == 0 or i % self.rebalance_frequency == 0:
                debug_info['rebalance_events'] += 1
                rebalance_dates.append(current_date)
                
                # Generate signals with debug
                signals = self.calculate_signals_with_debug(data, date, debug_info)
                
                # Execute rebalancing with debug
                trade_summary = self.execute_rebalancing_with_debug(
                    portfolio, data, date, signals, debug_info
                )
                
                if trade_summary['trades_made'] > 0:
                    trades.append({
                        'date': current_date,
                        **trade_summary
                    })
            
            # Record daily state
            history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'num_positions': len(portfolio['positions']),
                'rebalanced': current_date in rebalance_dates
            })
        
        return {
            'history': history,
            'trades_summary': {
                'total_rebalances': len(rebalance_dates),
                'total_trades': sum(t.get('trades_made', 0) for t in trades),
                'final_value': portfolio_value,
                'final_positions': len(portfolio['positions'])
            },
            'debug_info': debug_info
        }
    
    def calculate_signals_with_debug(self, data, date, debug_info):
        """Calculate signals avec debugging"""
        signals = {}
        
        for symbol, prices in data.items():
            debug_info['signal_generations'] += 1
            
            try:
                # Get data up to this date
                historical_data = prices[prices.index <= date]
                
                if len(historical_data) >= 60:
                    # Calculate indicators
                    ema_20 = historical_data.ewm(span=20).mean()
                    ema_50 = historical_data.ewm(span=50).mean()
                    
                    # RSI
                    delta = historical_data.diff()
                    gains = delta.where(delta > 0, 0.0)
                    losses = -delta.where(delta < 0, 0.0)
                    avg_gains = gains.ewm(alpha=1/14).mean()
                    avg_losses = losses.ewm(alpha=1/14).mean()
                    rs = avg_gains / avg_losses
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Latest values
                    current_ema_20 = float(ema_20.iloc[-1])
                    current_ema_50 = float(ema_50.iloc[-1])
                    current_rsi = float(rsi.iloc[-1])
                    current_price = float(historical_data.iloc[-1])
                    
                    # Signal calculation (plus permissif)
                    ema_signal = 1 if current_ema_20 > current_ema_50 else 0
                    rsi_signal = 1 if current_rsi < 75 else 0  # 75 au lieu de 70
                    
                    # Combined score
                    score = 0.6 * ema_signal + 0.4 * rsi_signal
                    
                    signals[symbol] = {
                        'score': score,
                        'ema_bullish': ema_signal,
                        'rsi_ok': rsi_signal,
                        'price': current_price,
                        'rsi_value': current_rsi
                    }
                    
                    debug_info['successful_signals'] += 1
                    
            except Exception as e:
                continue
        
        return signals
    
    def execute_rebalancing_with_debug(self, portfolio, data, date, signals, debug_info):
        """Execute rebalancing avec debugging"""
        
        # Separate signals by region
        usa_signals = {s: sig for s, sig in signals.items() if s in self.usa_stocks}
        europe_signals = {s: sig for s, sig in signals.items() if s in self.europe_stocks}
        
        # Select top performers with lower threshold
        usa_top = sorted(
            [(s, sig['score']) for s, sig in usa_signals.items() if sig['score'] >= self.min_score_threshold],
            key=lambda x: x[1], reverse=True
        )[:6]  # Top 6 USA
        
        europe_top = sorted(
            [(s, sig['score']) for s, sig in europe_signals.items() if sig['score'] >= self.min_score_threshold],
            key=lambda x: x[1], reverse=True
        )[:3]  # Top 3 Europe
        
        # Debug asset selection
        debug_info['assets_selected'] += len(usa_top) + len(europe_top)
        
        # Calculate target allocations
        selected_usa = [s for s, _ in usa_top]
        selected_europe = [s for s, _ in europe_top]
        
        target_positions = {}
        total_selected = len(selected_usa) + len(selected_europe)
        
        if total_selected > 0:
            if selected_usa:
                usa_weight_per_stock = self.usa_allocation / len(selected_usa)
                for symbol in selected_usa:
                    target_positions[symbol] = min(usa_weight_per_stock, self.max_position)
            
            if selected_europe:
                europe_weight_per_stock = self.europe_allocation / len(selected_europe)
                for symbol in selected_europe:
                    target_positions[symbol] = min(europe_weight_per_stock, self.max_position)
        
        # Execute trades with debug
        trades_made = self.execute_trades_with_debug(portfolio, data, date, target_positions, debug_info)
        
        return {
            'trades_made': trades_made,
            'selected_usa': selected_usa,
            'selected_europe': selected_europe,
            'total_positions': len(target_positions),
            'usa_signals': len(usa_signals),
            'europe_signals': len(europe_signals),
            'usa_above_threshold': len([s for s, sig in usa_signals.items() if sig['score'] >= self.min_score_threshold]),
            'europe_above_threshold': len([s for s, sig in europe_signals.items() if sig['score'] >= self.min_score_threshold])
        }
    
    def execute_trades_with_debug(self, portfolio, data, date, target_positions, debug_info):
        """Execute trades avec debugging détaillé"""
        trades_count = 0
        current_value = portfolio['value']
        
        # Sell positions not in target
        positions_to_sell = []
        for symbol in list(portfolio['positions'].keys()):
            if symbol not in target_positions:
                positions_to_sell.append(symbol)
        
        for symbol in positions_to_sell:
            debug_info['trades_attempted'] += 1
            if symbol in data:
                shares = portfolio['positions'][symbol]
                if shares > 0:
                    prices = data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        price = float(available_prices.iloc[-1])
                        proceeds = shares * price
                        portfolio['cash'] += proceeds
                        trades_count += 1
                        debug_info['trades_executed'] += 1
                
                del portfolio['positions'][symbol]
        
        # Execute buys/adjustments with more permissive threshold
        for symbol, target_weight in target_positions.items():
            debug_info['trades_attempted'] += 1
            if symbol in data:
                prices = data[symbol]
                available_prices = prices[prices.index <= date]
                if len(available_prices) > 0:
                    price = float(available_prices.iloc[-1])
                    target_value = current_value * target_weight
                    target_shares = target_value / price
                    
                    current_shares = portfolio['positions'].get(symbol, 0)
                    shares_diff = target_shares - current_shares
                    
                    # Lower threshold for trade execution
                    if abs(shares_diff * price) > current_value * self.min_trade_threshold:
                        cost = shares_diff * price
                        
                        if shares_diff > 0 and portfolio['cash'] >= cost:  # Buy
                            portfolio['cash'] -= cost
                            portfolio['positions'][symbol] = target_shares
                            trades_count += 1
                            debug_info['trades_executed'] += 1
                        elif shares_diff < 0:  # Sell partial
                            portfolio['cash'] -= cost  # cost is negative
                            portfolio['positions'][symbol] = target_shares
                            trades_count += 1
                            debug_info['trades_executed'] += 1
        
        return trades_count
    
    def update_portfolio_value(self, portfolio, data, date):
        """Update portfolio value"""
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in data and shares > 0:
                prices = data[symbol]
                available_prices = prices[prices.index <= date]
                if len(available_prices) > 0:
                    current_price = float(available_prices.iloc[-1])
                    position_value = shares * current_price
                    total_value += position_value
        
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
    
    def print_detailed_summary(self, results):
        """Print summary with debug info"""
        print("\n" + "="*80)
        print("📋 FIXED 5-YEAR BACKTEST SUMMARY")
        print("="*80)
        
        config = results['config']
        perf = results['performance']
        trades = results['trades_summary']
        debug = results['debug_info']
        
        print(f"🌍 Configuration:")
        print(f"  Universe size:         {config['universe_size']:>8}")
        print(f"  Trading days:          {config['trading_days']:>8}")
        print(f"  Rebalance frequency:   {config['rebalance_frequency']:>8}")
        print(f"  Score threshold:       {config['score_threshold']:>8.1f}")
        print(f"  Trade threshold:       {config['trade_threshold']:>8.1%}")
        
        print(f"\n📊 Performance:")
        print(f"  Total Return:          {perf['total_return']:>8.1%}")
        print(f"  Annual Return:         {perf['annual_return']:>8.1%}")
        print(f"  Volatility:            {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:          {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:              {perf['win_rate']:>8.1%}")
        print(f"  Final Value:           ${perf['final_value']:>8,.0f}")
        
        print(f"\n🔧 Debug Info:")
        print(f"  Signal generations:    {debug['signal_generations']:>8}")
        print(f"  Successful signals:    {debug['successful_signals']:>8}")
        print(f"  Assets selected:       {debug['assets_selected']:>8}")
        print(f"  Trades attempted:      {debug['trades_attempted']:>8}")
        print(f"  Trades executed:       {debug['trades_executed']:>8}")
        print(f"  Rebalance events:      {debug['rebalance_events']:>8}")
        
        print(f"\n📈 Trading Activity:")
        print(f"  Total Rebalances:      {trades['total_rebalances']:>8}")
        print(f"  Total Trades:          {trades['total_trades']:>8}")
        print(f"  Final Positions:       {trades['final_positions']:>8}")
        print(f"  Success Rate:          {(debug['trades_executed']/max(debug['trades_attempted'],1)*100):>7.1f}%")
        
        # Diagnostic
        print(f"\n💡 Diagnostic:")
        if debug['trades_executed'] > 0:
            print("  ✅ Trading logic is working")
            if perf['annual_return'] > 0.05:
                print("  ✅ Positive returns generated")
            else:
                print("  ⚠️ Low returns - need strategy optimization")
        else:
            print("  ❌ No trades executed - check thresholds")


def main():
    """Execute fixed backtest"""
    print("📈 FIXED 5-YEAR BACKTEST")
    print("Debugging and fixing trading logic issues")
    print("="*80)
    
    backtester = Fixed5YearBacktester()
    results = backtester.run_fixed_backtest()
    
    if results and results['debug_info']['trades_executed'] > 0:
        print("\n🎉 FIXED BACKTEST SUCCESS!")
        print("✅ Trading logic now functional")
        return 0
    else:
        print("\n❌ ISSUES STILL PRESENT")
        print("🔧 Further debugging needed")
        return 1


if __name__ == "__main__":
    exit_code = main()