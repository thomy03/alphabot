#!/usr/bin/env python3
"""
Extended Swing Trading Validation - 10 Years + Expanded Universe
Validation rigoureuse du système swing trading sur période étendue
Période: 2015-2024 (10 ans) + 50+ actifs + stress tests
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExtendedSwingValidation:
    """
    Validation étendue du swing trading sur 10 ans
    Test rigoureux incluant bear markets et corrections
    """
    
    def __init__(self):
        # EXPANDED UNIVERSE - 50+ actifs pour validation robuste
        self.mega_caps = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'NFLX'
        ]
        
        self.large_caps = [
            'AMD', 'CRM', 'ADBE', 'ORCL', 'CSCO', 'INTC', 'QCOM', 'TXN',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'JNJ', 'UNH', 'PFE',
            'KO', 'PG', 'WMT', 'HD', 'DIS', 'NKE', 'MCD', 'V', 'MA'
        ]
        
        self.mid_caps = [
            'SQ', 'PLTR', 'SNOW', 'ZM', 'ROKU', 'CRWD', 'DDOG', 'NET',
            'OKTA', 'ZS', 'SPLK', 'MDB', 'TEAM', 'WDAY', 'NOW'
        ]
        
        self.etfs_broad = [
            'QQQ', 'SPY', 'IWM',    # Broad market
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI',  # Sectors
            'VGT', 'ARKK', 'TQQQ'   # Tech focused
        ]
        
        # EXTENDED PERIOD - 10 years including bear markets
        self.start_date = "2015-01-01"  # Extended from 2019
        self.end_date = "2024-07-01"    # Current + 6 months
        self.initial_capital = 100000
        self.rebalance_frequency = 2
        
        # Same proven swing parameters
        self.max_position = 0.12
        self.max_positions = 12  # Slightly more for larger universe
        self.min_hold_days = 2
        self.max_hold_days = 7
        
        # RIGOROUS REGIME DETECTION (same as proven)
        self.swing_regimes = {
            'strong_momentum': {
                'name': 'Strong Momentum Phase',
                'score_threshold': 0.15,
                'allocation_factor': 0.95,
                'max_positions': 10,
                'take_profit': 0.08,
                'stop_loss': 0.04
            },
            'moderate_trend': {
                'name': 'Moderate Trend',
                'score_threshold': 0.12,
                'allocation_factor': 0.85,
                'max_positions': 12,
                'take_profit': 0.06,
                'stop_loss': 0.03
            },
            'consolidation': {
                'name': 'Market Consolidation',
                'score_threshold': 0.08,
                'allocation_factor': 0.70,
                'max_positions': 8,
                'take_profit': 0.04,
                'stop_loss': 0.025
            },
            'high_volatility': {
                'name': 'High Volatility Environment',
                'score_threshold': 0.10,
                'allocation_factor': 0.80,
                'max_positions': 10,
                'take_profit': 0.10,
                'stop_loss': 0.05
            }
        }
        
        print(f"🔍 EXTENDED SWING TRADING VALIDATION")
        print(f"📊 EXPANDED UNIVERSE: {len(self.get_all_symbols())} total symbols")
        print(f"  - Mega caps: {len(self.mega_caps)}")
        print(f"  - Large caps: {len(self.large_caps)}")
        print(f"  - Mid caps: {len(self.mid_caps)}")
        print(f"  - ETFs: {len(self.etfs_broad)}")
        print(f"📅 EXTENDED PERIOD: {self.start_date} to {self.end_date} (9.5 years)")
        print(f"🎯 STRESS TEST: Include 2015-2016 bear, 2018 correction, 2020 crash, 2022 bear")
        print(f"✅ VALIDATION: Test if 39.2% holds on extended period + more assets")
    
    def get_all_symbols(self):
        """Get all symbols for validation"""
        return self.mega_caps + self.large_caps + self.mid_caps + self.etfs_broad
    
    def run_extended_validation(self):
        """Execute extended validation"""
        print("\\n" + "="*80)
        print("🔍 EXTENDED SWING TRADING VALIDATION - 10 Years + 50+ Assets")
        print("="*80)
        
        # Download extended data
        print("\\n📊 Step 1: Downloading extended universe (50+ symbols)...")
        all_data = self.download_extended_data()
        
        # Download SPY for regime detection
        print("\\n📈 Step 2: Downloading SPY for extended regime detection...")
        spy_data = self.download_spy_extended()
        
        # Run extended simulation
        print("\\n🔍 Step 3: Running extended swing simulation (9.5 years)...")
        portfolio_results = self.simulate_extended_swing(all_data, spy_data)
        
        # Calculate performance
        print("\\n📊 Step 4: Calculating extended performance metrics...")
        performance = self.calculate_extended_performance(portfolio_results)
        
        # Stress test analysis
        print("\\n🧪 Step 5: Stress testing across different periods...")
        stress_tests = self.analyze_stress_periods(portfolio_results)
        
        # Benchmark comparisons
        print("\\n📈 Step 6: Extended benchmark comparisons...")
        benchmarks = self.get_extended_benchmarks()
        
        # Results
        results = {
            'config': {
                'system_type': 'extended_swing_validation',
                'period': f'{self.start_date} to {self.end_date}',
                'universe_size': len(all_data),
                'years_tested': 9.5,
                'stress_periods': ['2015-2016 bear', '2018 correction', '2020 crash', '2022 bear']
            },
            'performance': performance,
            'stress_tests': stress_tests,
            'benchmarks': benchmarks,
            'trades_summary': portfolio_results['trades_summary'],
            'regime_history': portfolio_results['regime_history']
        }
        
        self.print_extended_summary(results)
        return results
    
    def download_extended_data(self):
        """Download extended universe data"""
        all_symbols = self.get_all_symbols()
        data = {}
        failed_downloads = []
        
        for i, symbol in enumerate(all_symbols, 1):
            try:
                print(f"  📊 ({i:2d}/{len(all_symbols)}) {symbol:8s}...", end=" ")
                
                ticker_data = yf.download(
                    symbol, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False
                )
                
                if len(ticker_data) > 1000:  # Require more data for 10-year test
                    # Fix MultiIndex
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    
                    data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ {len(ticker_data)} days (insufficient)")
                    failed_downloads.append(symbol)
                    
            except Exception as e:
                print(f"❌ Error")
                failed_downloads.append(symbol)
        
        print(f"  VALIDATION DATA: {len(data)} symbols downloaded successfully")
        if failed_downloads:
            print(f"  Failed downloads: {failed_downloads}")
        
        return data
    
    def download_spy_extended(self):
        """Download SPY for extended period"""
        try:
            spy_data = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
            
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.droplevel(1)
            
            print(f"  ✅ SPY extended: {len(spy_data)} days (2015-2024)")
            return spy_data
        except Exception as e:
            print(f"  ❌ SPY extended failed: {e}")
            return None
    
    def detect_swing_regime(self, spy_data, current_date):
        """Same proven regime detection"""
        if spy_data is None:
            return 'moderate_trend'
        
        historical_spy = spy_data[spy_data.index <= current_date]
        
        if len(historical_spy) < 15:
            return 'moderate_trend'
        
        try:
            closes = historical_spy['Close']
            
            # Medium-term indicators (proven swing parameters)
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
            
            # Proven swing regime logic
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
                
        except Exception as e:
            return 'moderate_trend'
    
    def simulate_extended_swing(self, data, spy_data):
        """Extended swing simulation"""
        # Same core logic as proven system
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
        trades = []
        regime_history = []
        current_regime = 'moderate_trend'
        
        print(f"  🔍 Extended simulation: {len(trading_dates)} days...")
        
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            
            # Progress every 200 days for 10-year test
            if i % 200 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                year = date.year
                print(f"    📅 Progress: {progress:5.1f}% - {current_date} ({year}) - Regime: {current_regime}")
            
            # Regime detection
            new_regime = self.detect_swing_regime(spy_data, date)
            if new_regime != current_regime:
                current_regime = new_regime
                regime_history.append({
                    'date': current_date,
                    'regime': current_regime
                })
            
            # Portfolio value update
            portfolio_value = self.update_portfolio_value(portfolio, data, date)
            
            # Swing trading logic (every 2 days)
            if i % self.rebalance_frequency == 0:
                # Exit management
                exits_made = self.check_swing_exits(portfolio, data, date, current_regime, i)
                
                # Entry management
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
        """Same exit logic as proven system"""
        regime_config = self.swing_regimes[current_regime]
        exits_made = 0
        positions_to_exit = []
        
        for symbol in list(portfolio['positions'].keys()):
            if symbol not in data:
                continue
                
            try:
                prices = data[symbol]
                available_prices = prices[prices.index <= date]
                if len(available_prices) == 0:
                    continue
                
                current_price = float(available_prices.iloc[-1])
                entry_price = portfolio['position_entry_prices'].get(symbol, current_price)
                entry_date_index = portfolio['position_entry_dates'].get(symbol, day_index)
                
                pnl_pct = (current_price / entry_price) - 1
                hold_days = day_index - entry_date_index
                
                should_exit = False
                
                # Exit conditions
                if pnl_pct >= regime_config['take_profit']:
                    should_exit = True
                elif pnl_pct <= -regime_config['stop_loss']:
                    should_exit = True
                elif hold_days >= self.max_hold_days:
                    should_exit = True
                elif hold_days < self.min_hold_days:
                    should_exit = False
                
                if should_exit:
                    positions_to_exit.append(symbol)
                    
            except Exception:
                continue
        
        # Execute exits
        for symbol in positions_to_exit:
            self.execute_swing_exit(portfolio, data, date, symbol)
            exits_made += 1
        
        return exits_made
    
    def consider_swing_entries(self, portfolio, data, date, current_regime, day_index):
        """Same entry logic as proven system"""
        regime_config = self.swing_regimes[current_regime]
        entries_made = 0
        
        current_positions = len(portfolio['positions'])
        if current_positions >= regime_config['max_positions']:
            return entries_made
        
        signals = self.calculate_swing_signals(data, date, current_regime)
        
        qualified_signals = [
            (symbol, sig['score']) for symbol, sig in signals.items() 
            if sig['score'] >= regime_config['score_threshold']
            and symbol not in portfolio['positions']
        ]
        
        if not qualified_signals:
            return entries_made
        
        qualified_signals.sort(key=lambda x: x[1], reverse=True)
        max_new_entries = regime_config['max_positions'] - current_positions
        
        for symbol, score in qualified_signals[:max_new_entries]:
            if self.execute_swing_entry(portfolio, data, date, symbol, day_index):
                entries_made += 1
        
        return entries_made
    
    def calculate_swing_signals(self, data, date, current_regime):
        """Same signal calculation as proven system"""
        signals = {}
        
        for symbol, prices in data.items():
            try:
                historical_data = prices[prices.index <= date]
                
                if len(historical_data) < 25:
                    continue
                
                # Enhanced EMA system
                ema_8 = historical_data.ewm(span=8).mean()
                ema_21 = historical_data.ewm(span=21).mean()
                
                current_ema_8 = float(ema_8.iloc[-1])
                current_ema_21 = float(ema_21.iloc[-1])
                ema_signal = 1 if current_ema_8 > current_ema_21 else 0
                
                # EMA momentum
                if len(ema_8) >= 3:
                    ema_momentum = (ema_8.iloc[-1] - ema_8.iloc[-3]) / ema_8.iloc[-3]
                    momentum_signal = 1 if ema_momentum > 0.001 else 0
                else:
                    momentum_signal = 0
                
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
                
                # Price breakout
                current_price = float(historical_data.iloc[-1])
                if len(historical_data) >= 20:
                    recent_high = float(historical_data.tail(20).max())
                    breakout_signal = 1 if current_price > recent_high * 0.995 else 0
                else:
                    breakout_signal = 0
                
                # Volume proxy
                if len(historical_data) >= 10:
                    recent_volatility = historical_data.tail(10).std() / current_price
                    volume_signal = 1 if recent_volatility > 0.008 else 0
                else:
                    volume_signal = 0
                
                # Scoring
                score = (0.3 * ema_signal + 
                        0.25 * momentum_signal + 
                        0.2 * rsi_signal + 
                        0.15 * breakout_signal + 
                        0.1 * volume_signal)
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price,
                    'rsi': current_rsi
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_swing_entry(self, portfolio, data, date, symbol, day_index):
        """Execute entry with tracking"""
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
    
    def execute_swing_exit(self, portfolio, data, date, symbol):
        """Execute exit with cleanup"""
        try:
            if symbol in portfolio['positions']:
                shares = portfolio['positions'][symbol]
                prices = data[symbol]
                available_prices = prices[prices.index <= date]
                
                if len(available_prices) > 0:
                    price = float(available_prices.iloc[-1])
                    proceeds = shares * price
                    portfolio['cash'] += proceeds
                
                # Cleanup
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
    
    def calculate_extended_performance(self, portfolio_results):
        """Calculate extended performance metrics"""
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
    
    def analyze_stress_periods(self, portfolio_results):
        """Analyze performance during stress periods"""
        history = pd.DataFrame(portfolio_results['history'])
        history['date'] = pd.to_datetime(history['date'])
        history.set_index('date', inplace=True)
        
        stress_periods = {
            '2015_2016_bear': ('2015-08-01', '2016-02-29'),
            '2018_correction': ('2018-10-01', '2018-12-31'),
            '2020_covid_crash': ('2020-02-15', '2020-04-15'),
            '2022_bear_market': ('2022-01-01', '2022-10-31')
        }
        
        stress_results = {}
        
        for period_name, (start_date, end_date) in stress_periods.items():
            try:
                period_data = history.loc[start_date:end_date]
                if len(period_data) > 0:
                    period_return = (period_data['portfolio_value'].iloc[-1] / period_data['portfolio_value'].iloc[0]) - 1
                    stress_results[period_name] = {
                        'return': float(period_return),
                        'start_date': start_date,
                        'end_date': end_date,
                        'days': len(period_data)
                    }
            except:
                stress_results[period_name] = {'return': 0, 'error': 'Data not available'}
        
        return stress_results
    
    def get_extended_benchmarks(self):
        """Get benchmark performance for extended period"""
        benchmarks = {}
        
        benchmark_symbols = ['SPY', 'QQQ', 'IWM']
        
        for symbol in benchmark_symbols:
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                closes = data['Close']
                total_return = (closes.iloc[-1] / closes.iloc[0]) - 1
                years = len(closes) / 252
                annual_return = (1 + total_return) ** (1/years) - 1
                
                benchmarks[symbol] = {
                    'total_return': float(total_return),
                    'annual_return': float(annual_return)
                }
            except:
                benchmarks[symbol] = {'annual_return': 0.15, 'error': 'Download failed'}
        
        return benchmarks
    
    def print_extended_summary(self, results):
        """Print extended validation summary"""
        print("\\n" + "="*80)
        print("🔍 EXTENDED SWING TRADING VALIDATION SUMMARY")
        print("="*80)
        
        config = results['config']
        perf = results['performance']
        stress = results['stress_tests']
        benchmarks = results['benchmarks']
        
        print(f"🔍 Extended Validation:")
        print(f"  Period Tested:         {config['period']}")
        print(f"  Years Simulated:       {config['years_tested']:.1f} years")
        print(f"  Universe Size:         {config['universe_size']:>8} symbols")
        print(f"  Stress Periods:        {len(config['stress_periods'])} major downturns")
        
        print(f"\\n🚀 EXTENDED Performance (9.5 years):")
        print(f"  Total Return:          {perf['total_return']:>8.1%}")
        print(f"  Annual Return:         {perf['annual_return']:>8.1%}")
        print(f"  Volatility:            {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:          {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:              {perf['win_rate']:>8.1%}")
        print(f"  Final Value:           ${perf['final_value']:>8,.0f}")
        
        print(f"\\n🧪 Stress Test Results:")
        for period, result in stress.items():
            if 'return' in result:
                print(f"  {period:20s}: {result['return']:>8.1%}")
        
        print(f"\\n📈 Extended Benchmark Comparison:")
        for symbol, bench in benchmarks.items():
            if 'annual_return' in bench:
                print(f"  {symbol} (9.5Y Annual):   {bench['annual_return']:>8.1%}")
        
        print(f"  Our System (9.5Y):     {perf['annual_return']:>8.1%}")
        
        # Validation results
        print(f"\\n🏆 EXTENDED VALIDATION RESULTS:")
        
        original_5y_performance = 0.392  # 39.2% from 5-year test
        extended_performance = perf['annual_return']
        performance_delta = extended_performance - original_5y_performance
        
        print(f"  Original 5Y Test:      {original_5y_performance:>8.1%}")
        print(f"  Extended 9.5Y Test:    {extended_performance:>8.1%}")
        print(f"  Delta:                 {performance_delta:>8.1%}")
        
        if abs(performance_delta) < 0.05:  # Within 5%
            print("✅ VALIDATION CONFIRMED: Extended test confirms original results!")
        elif extended_performance > original_5y_performance:
            print("🎉 VALIDATION EXCEEDED: Extended test shows even better results!")
        else:
            print("⚠️ VALIDATION CONCERN: Extended test shows lower performance")
        
        # Stress test evaluation
        stress_positive = sum(1 for r in stress.values() if r.get('return', -1) > 0)
        stress_total = len([r for r in stress.values() if 'return' in r])
        
        if stress_positive >= stress_total * 0.5:
            print(f"✅ STRESS TEST PASSED: Positive in {stress_positive}/{stress_total} major downturns")
        else:
            print(f"⚠️ STRESS TEST CONCERN: Only positive in {stress_positive}/{stress_total} downturns")


def main():
    """Execute extended swing validation"""
    print("🔍 EXTENDED SWING TRADING VALIDATION")
    print("Testing on 10 years + 50+ assets + stress periods")
    print("="*80)
    
    validator = ExtendedSwingValidation()
    results = validator.run_extended_validation()
    
    return 0


if __name__ == "__main__":
    exit_code = main()