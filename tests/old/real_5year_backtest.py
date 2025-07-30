#!/usr/bin/env python3
"""
Real 5-Year Backtest - Conditions réelles complètes
Simulation rigoureuse sur 28 actifs, 5 ans, rebalancing hebdomadaire
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class Real5YearBacktester:
    """
    Backtest rigoureux 5 ans en conditions réelles
    """
    
    def __init__(self):
        # Univers complet 70% USA / 30% Europe
        self.usa_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',  # Tech
            'JPM', 'BAC', 'WFC',                       # Finance  
            'JNJ', 'UNH',                              # Healthcare
            'KO', 'PG',                                # Consumer
            'XOM', 'CVX',                              # Energy
            'CAT', 'GE'                                # Industrial
        ]
        
        self.europe_stocks = [
            'EWG', 'EWQ', 'EWI', 'EWP', 'EWU',        # Europe ETFs
            'EWN', 'EWO', 'EWK',                       # More Europe ETFs
            'ASML', 'SAP', 'NVO', 'NESN.SW'           # European ADRs
        ]
        
        # Configuration
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        self.initial_capital = 100000
        self.rebalance_frequency = 7  # Weekly
        
        # Allocation target
        self.usa_allocation = 0.70
        self.europe_allocation = 0.30
        self.max_position = 0.05  # 5% max per asset
        
        print(f"🌍 Universe: {len(self.usa_stocks)} USA + {len(self.europe_stocks)} Europe = {len(self.usa_stocks) + len(self.europe_stocks)} total")
        print(f"📅 Period: {self.start_date} to {self.end_date} (5 years)")
        print(f"💰 Initial capital: ${self.initial_capital:,}")
        print(f"🔄 Rebalancing: Every {self.rebalance_frequency} days")
        print(f"🎯 FIXED: Score threshold 0.4, Trade threshold 0.5%, RSI threshold 75")
    
    def run_complete_backtest(self):
        """Execute le backtest complet"""
        print("\n" + "="*80)
        print("📈 REAL 5-YEAR BACKTEST - Complete Conditions")
        print("="*80)
        
        # Step 1: Download all data
        print("\n📊 Step 1: Downloading 5-year data...")
        all_data = self.download_all_data()
        
        if len(all_data) < 15:  # Minimum viable
            print("❌ Insufficient data downloaded")
            return None
        
        # Step 2: Generate trading calendar
        print("\n📅 Step 2: Creating trading calendar...")
        trading_dates = self.create_trading_calendar()
        print(f"  📅 Trading days: {len(trading_dates)}")
        
        # Step 3: Run day-by-day simulation
        print("\n💼 Step 3: Running day-by-day simulation...")
        portfolio_results = self.simulate_trading(all_data, trading_dates)
        
        # Step 4: Calculate performance metrics
        print("\n📊 Step 4: Calculating performance metrics...")
        performance = self.calculate_performance_metrics(portfolio_results)
        
        # Step 5: Benchmark comparison
        print("\n🎯 Step 5: Benchmark comparison...")
        benchmarks = self.compare_benchmarks()
        
        # Step 6: Generate results
        results = {
            'config': {
                'universe_size': len(all_data),
                'trading_days': len(trading_dates),
                'start_date': self.start_date,
                'end_date': self.end_date,
                'usa_allocation': self.usa_allocation,
                'europe_allocation': self.europe_allocation
            },
            'performance': performance,
            'benchmarks': benchmarks,
            'portfolio_history': portfolio_results['history'][:10],  # Sample only for JSON
            'trades_summary': portfolio_results['trades_summary']
        }
        
        self.save_results(results)
        self.print_complete_summary(results)
        
        return results
    
    def download_all_data(self):
        """Download historical data for all assets"""
        all_symbols = self.usa_stocks + self.europe_stocks
        data = {}
        failed = []
        
        for i, symbol in enumerate(all_symbols, 1):
            try:
                print(f"  📊 ({i:2d}/{len(all_symbols)}) {symbol:10s}...", end=" ")
                
                # Download with error handling
                ticker_data = yf.download(
                    symbol, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False,
                    threads=False
                )
                
                if len(ticker_data) > 1000:  # ~4+ years of data
                    # Clean data
                    prices = ticker_data['Close'].dropna()
                    data[symbol] = prices
                    print(f"✅ {len(prices):4d} days")
                else:
                    failed.append(symbol)
                    print(f"❌ Only {len(ticker_data)} days")
                    
            except Exception as e:
                failed.append(symbol)
                print(f"❌ Error: {str(e)[:20]}...")
        
        print(f"\n  ✅ Successfully downloaded: {len(data)}/{len(all_symbols)} symbols")
        if failed:
            print(f"  ❌ Failed symbols: {failed}")
        
        return data
    
    def create_trading_calendar(self):
        """Create business days calendar"""
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        
        # Business days only
        trading_dates = pd.bdate_range(start=start, end=end).tolist()
        return trading_dates
    
    def simulate_trading(self, data, trading_dates):
        """Main trading simulation"""
        
        # Initialize portfolio
        portfolio = {
            'cash': float(self.initial_capital),
            'positions': {},  # {symbol: shares}
            'value': float(self.initial_capital)
        }
        
        # Track results
        history = []
        trades = []
        rebalance_dates = []
        
        print(f"  🎯 Simulating {len(trading_dates)} trading days...")
        
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            
            # Progress indicator
            if i % 250 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                print(f"    📅 Progress: {progress:5.1f}% - {current_date}")
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(portfolio, data, date)
            
            # Check if rebalancing needed
            if i == 0 or i % self.rebalance_frequency == 0:
                rebalance_dates.append(current_date)
                
                # Generate signals for this date
                signals = self.calculate_signals_for_date(data, date)
                
                # Execute rebalancing
                trade_summary = self.execute_rebalancing(portfolio, data, date, signals)
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
                'rebalance_dates': rebalance_dates[:10],  # Sample
                'final_value': portfolio_value,
                'final_positions': len(portfolio['positions'])
            }
        }
    
    def update_portfolio_value(self, portfolio, data, date):
        """Update portfolio value based on current prices"""
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in data and shares > 0:
                # Get price for this date (or closest available)
                prices = data[symbol]
                
                # Find price on or before this date
                available_prices = prices[prices.index <= date]
                if len(available_prices) > 0:
                    current_price = available_prices.iloc[-1]
                    if hasattr(current_price, 'item'):
                        current_price = current_price.item()
                    position_value = float(shares) * float(current_price)
                    total_value += position_value
        
        portfolio['value'] = total_value
        return total_value
    
    def calculate_signals_for_date(self, data, date):
        """Calculate EMA+RSI signals for all assets at given date"""
        signals = {}
        
        for symbol, prices in data.items():
            try:
                # Get data up to this date only (no forward-looking bias)
                historical_data = prices[prices.index <= date]
                
                if len(historical_data) >= 60:  # Minimum for stable indicators
                    # Calculate EMA
                    ema_20 = historical_data.ewm(span=20).mean()
                    ema_50 = historical_data.ewm(span=50).mean()
                    
                    # Calculate RSI
                    delta = historical_data.diff()
                    gains = delta.where(delta > 0, 0.0)
                    losses = -delta.where(delta < 0, 0.0)
                    avg_gains = gains.ewm(alpha=1/14).mean()
                    avg_losses = losses.ewm(alpha=1/14).mean()
                    rs = avg_gains / avg_losses
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Latest values - FIXED with proper conversion
                    current_ema_20 = ema_20.iloc[-1]
                    current_ema_50 = ema_50.iloc[-1]
                    current_rsi = rsi.iloc[-1]
                    current_price = historical_data.iloc[-1]
                    
                    # Convert pandas Series to float if needed
                    if hasattr(current_ema_20, 'item'):
                        current_ema_20 = current_ema_20.item()
                    if hasattr(current_ema_50, 'item'):
                        current_ema_50 = current_ema_50.item()
                    if hasattr(current_rsi, 'item'):
                        current_rsi = current_rsi.item()
                    if hasattr(current_price, 'item'):
                        current_price = current_price.item()
                    
                    # Signal calculation
                    ema_signal = 1 if current_ema_20 > current_ema_50 else 0
                    rsi_signal = 1 if current_rsi < 75 else 0  # Not overbought - FIXED
                    
                    # Combined score
                    score = 0.6 * ema_signal + 0.4 * rsi_signal
                    
                    signals[symbol] = {
                        'score': score,
                        'ema_bullish': ema_signal,
                        'rsi_ok': rsi_signal,
                        'price': current_price,
                        'rsi_value': current_rsi
                    }
                    
            except Exception:
                continue  # Skip problematic symbols
        
        return signals
    
    def execute_rebalancing(self, portfolio, data, date, signals):
        """Execute portfolio rebalancing"""
        
        # Separate USA and Europe signals
        usa_signals = {s: sig for s, sig in signals.items() if s in self.usa_stocks}
        europe_signals = {s: sig for s, sig in signals.items() if s in self.europe_stocks}
        
        # Select top performers (score > 0.4 - FIXED)
        usa_top = sorted(
            [(s, sig['score']) for s, sig in usa_signals.items() if sig['score'] >= 0.4],
            key=lambda x: x[1], reverse=True
        )[:8]  # Top 8 USA
        
        europe_top = sorted(
            [(s, sig['score']) for s, sig in europe_signals.items() if sig['score'] >= 0.4],
            key=lambda x: x[1], reverse=True
        )[:4]  # Top 4 Europe
        
        # Calculate target allocations
        selected_usa = [s for s, _ in usa_top]
        selected_europe = [s for s, _ in europe_top]
        
        target_positions = {}
        total_selected = len(selected_usa) + len(selected_europe)
        
        if total_selected > 0:
            # USA allocation
            if selected_usa:
                usa_weight_per_stock = self.usa_allocation / len(selected_usa)
                for symbol in selected_usa:
                    target_positions[symbol] = min(usa_weight_per_stock, self.max_position)
            
            # Europe allocation
            if selected_europe:
                europe_weight_per_stock = self.europe_allocation / len(selected_europe)
                for symbol in selected_europe:
                    target_positions[symbol] = min(europe_weight_per_stock, self.max_position)
        
        # Execute trades
        trades_made = self.execute_trades(portfolio, data, date, target_positions)
        
        return {
            'trades_made': trades_made,
            'selected_usa': selected_usa,
            'selected_europe': selected_europe,
            'total_positions': len(target_positions)
        }
    
    def execute_trades(self, portfolio, data, date, target_positions):
        """Execute actual trades"""
        trades_count = 0
        current_value = portfolio['value']
        
        # First, sell positions not in target
        positions_to_sell = []
        for symbol in list(portfolio['positions'].keys()):
            if symbol not in target_positions:
                positions_to_sell.append(symbol)
        
        # Execute sells
        for symbol in positions_to_sell:
            if symbol in data:
                shares = portfolio['positions'][symbol]
                if shares > 0:
                    # Get current price
                    prices = data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        price = available_prices.iloc[-1]
                        if hasattr(price, 'item'):
                            price = price.item()
                        proceeds = shares * price
                        portfolio['cash'] += proceeds
                        trades_count += 1
                
                del portfolio['positions'][symbol]
        
        # Execute buys/adjustments
        for symbol, target_weight in target_positions.items():
            if symbol in data:
                # Get current price
                prices = data[symbol]
                available_prices = prices[prices.index <= date]
                if len(available_prices) > 0:
                    price = available_prices.iloc[-1]
                    if hasattr(price, 'item'):
                        price = price.item()
                    target_value = current_value * target_weight
                    target_shares = target_value / price
                    
                    current_shares = portfolio['positions'].get(symbol, 0)
                    shares_diff = target_shares - current_shares
                    
                    # Convert to float to avoid pandas Series issues
                    trade_value = abs(float(shares_diff) * float(price))
                    threshold_value = current_value * 0.005
                    
                    if trade_value > threshold_value:  # 0.5% threshold - FIXED
                        cost = float(shares_diff) * float(price)
                        
                        if shares_diff > 0 and portfolio['cash'] >= cost:  # Buy
                            portfolio['cash'] -= cost
                            portfolio['positions'][symbol] = target_shares
                            trades_count += 1
                        elif shares_diff < 0:  # Sell partial
                            portfolio['cash'] -= cost  # cost is negative
                            portfolio['positions'][symbol] = target_shares
                            trades_count += 1
        
        return trades_count
    
    def calculate_performance_metrics(self, portfolio_results):
        """Calculate comprehensive performance metrics"""
        
        history = pd.DataFrame(portfolio_results['history'])
        history['date'] = pd.to_datetime(history['date'])
        history.set_index('date', inplace=True)
        
        # Portfolio values
        values = history['portfolio_value']
        daily_returns = values.pct_change().dropna()
        
        # Basic metrics
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        trading_days = len(daily_returns)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1/years) - 1
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = values / values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # CVaR (5% worst returns)
        var_95 = daily_returns.quantile(0.05)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        # Ulcer Index
        ulcer_index = np.sqrt((drawdowns ** 2).mean())
        
        # Win metrics
        win_rate = (daily_returns > 0).mean()
        
        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': float(max_drawdown),
            'cvar_95': float(cvar_95),
            'ulcer_index': float(ulcer_index),
            'win_rate': float(win_rate),
            'trading_days': int(trading_days),
            'final_value': float(values.iloc[-1]),
            'years_simulated': float(years)
        }
    
    def compare_benchmarks(self):
        """Compare against benchmarks"""
        
        try:
            # Download benchmark data
            spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)['Close']
            ezu = yf.download('EZU', start=self.start_date, end=self.end_date, progress=False)['Close']  # Europe
            
            # Calculate returns
            spy_return = (spy.iloc[-1] / spy.iloc[0]) - 1
            ezu_return = (ezu.iloc[-1] / ezu.iloc[0]) - 1
            
            # Mixed benchmark (70% SPY + 30% EZU)
            mixed_return = 0.7 * spy_return + 0.3 * ezu_return
            
            # Annualized
            years = 5.0
            spy_annual = (1 + spy_return) ** (1/years) - 1
            ezu_annual = (1 + ezu_return) ** (1/years) - 1
            mixed_annual = (1 + mixed_return) ** (1/years) - 1
            
            return {
                'spy_total': float(spy_return),
                'spy_annual': float(spy_annual),
                'europe_total': float(ezu_return),
                'europe_annual': float(ezu_annual),
                'mixed_total': float(mixed_return),
                'mixed_annual': float(mixed_annual)
            }
            
        except Exception as e:
            print(f"    ⚠️ Benchmark download failed: {e}")
            return {'error': 'Benchmark data unavailable'}
    
    def save_results(self, results):
        """Save comprehensive results"""
        output_dir = Path("backtests/real_5year")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"real_5year_backtest_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Complete results saved to {output_dir / filename}")
    
    def print_complete_summary(self, results):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("📋 REAL 5-YEAR BACKTEST COMPLETE SUMMARY")
        print("="*80)
        
        config = results['config']
        perf = results['performance']
        bench = results['benchmarks']
        trades = results['trades_summary']
        
        print(f"🌍 Universe & Configuration:")
        print(f"  Assets analyzed:       {config['universe_size']:>8,}")
        print(f"  Trading days:          {config['trading_days']:>8,}")
        print(f"  Years simulated:       {perf['years_simulated']:>8.1f}")
        print(f"  USA allocation:        {config['usa_allocation']:>8.1%}")
        print(f"  Europe allocation:     {config['europe_allocation']:>8.1%}")
        
        print(f"\n📊 AlphaBot Performance:")
        print(f"  Total Return (5Y):     {perf['total_return']:>8.1%}")
        print(f"  Annual Return:         {perf['annual_return']:>8.1%}")
        print(f"  Volatility:            {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:          {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {perf['max_drawdown']:>8.1%}")
        print(f"  CVaR 95%:              {perf['cvar_95']:>8.1%}")
        print(f"  Ulcer Index:           {perf['ulcer_index']:>8.2f}")
        print(f"  Win Rate:              {perf['win_rate']:>8.1%}")
        print(f"  Final Value:           ${perf['final_value']:>8,.0f}")
        
        if 'error' not in bench:
            print(f"\n🎯 Benchmark Comparison:")
            print(f"  AlphaBot Annual:       {perf['annual_return']:>8.1%}")
            print(f"  S&P 500 Annual:        {bench['spy_annual']:>8.1%}")
            print(f"  Europe Annual:         {bench['europe_annual']:>8.1%}")
            print(f"  Mixed (70/30) Annual:  {bench['mixed_annual']:>8.1%}")
            
            alpha_vs_spy = perf['annual_return'] - bench['spy_annual']
            alpha_vs_mixed = perf['annual_return'] - bench['mixed_annual']
            
            print(f"  Alpha vs S&P 500:      {alpha_vs_spy:>8.1%}")
            print(f"  Alpha vs Mixed:        {alpha_vs_mixed:>8.1%}")
            print(f"  Outperformance SPY:    {'✅ YES' if alpha_vs_spy > 0 else '❌ NO'}")
            print(f"  Outperformance Mixed:  {'✅ YES' if alpha_vs_mixed > 0 else '❌ NO'}")
        
        print(f"\n📈 Trading Activity:")
        print(f"  Total Rebalances:      {trades['total_rebalances']:>8,}")
        print(f"  Total Trades:          {trades['total_trades']:>8,}")
        print(f"  Final Positions:       {trades['final_positions']:>8,}")
        print(f"  Avg Trades/Rebalance:  {trades['total_trades']/max(trades['total_rebalances'],1):>8.1f}")
        
        print(f"\n💡 Expert Validation:")
        
        # Target validation
        targets_met = []
        if perf['annual_return'] > 0.10:  # >10%
            targets_met.append("✅ Return target exceeded")
        else:
            targets_met.append("⚠️ Return below 10% target")
        
        if perf['sharpe_ratio'] > 1.0:
            targets_met.append("✅ Sharpe ratio >1.0")
        else:
            targets_met.append("⚠️ Sharpe ratio <1.0")
        
        if abs(perf['max_drawdown']) < 0.15:
            targets_met.append("✅ Drawdown control <15%")
        else:
            targets_met.append("⚠️ Drawdown >15%")
        
        if 'error' not in bench and alpha_vs_mixed > 0:
            targets_met.append("✅ Benchmark outperformance")
        else:
            targets_met.append("⚠️ Benchmark underperformance")
        
        for target in targets_met:
            print(f"  {target}")
        
        # Final verdict
        success_count = sum(1 for t in targets_met if t.startswith("✅"))
        total_targets = len(targets_met)
        
        print(f"\n🎯 FINAL VERDICT ({success_count}/{total_targets} targets met):")
        if success_count >= 3:
            print("  🎉 SYSTÈME VALIDÉ EN CONDITIONS RÉELLES!")
            print("  ✅ Prêt pour production avec capital limité")
        elif success_count >= 2:
            print("  🟡 Système fonctionnel, optimisations recommandées")
            print("  🔧 Procéder aux améliorations Sprint 35-36")
        else:
            print("  🔴 Système nécessite révision majeure")
            print("  🛠️ Retour aux optimisations critiques")


def main():
    """Execute real 5-year backtest"""
    print("📈 REAL 5-YEAR BACKTEST")
    print("Complete validation in real market conditions")
    print("="*80)
    
    backtester = Real5YearBacktester()
    results = backtester.run_complete_backtest()
    
    if results:
        print("\n🎉 REAL 5-YEAR BACKTEST COMPLETED!")
        return 0
    else:
        print("\n❌ BACKTEST FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()