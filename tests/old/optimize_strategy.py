#!/usr/bin/env python3
"""
Strategy Optimizer - Améliorer les 9.7% vers 12-15% annuel
Test de différentes combinaisons de signaux et paramètres
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class StrategyOptimizer:
    """
    Optimisation systématique de la stratégie AlphaBot
    """
    
    def __init__(self):
        # Universe simplifié pour tests rapides
        self.usa_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'JPM', 'BAC']
        self.europe_stocks = ['EWG', 'EWQ', 'EWU', 'EWI']
        
        # Périodes de test
        self.train_start = "2019-01-01"
        self.train_end = "2022-01-01"    # 3 ans pour optimisation
        self.test_start = "2022-01-01"
        self.test_end = "2024-01-01"     # 2 ans pour validation
        
        self.initial_capital = 100000
        
        print(f"🎯 STRATEGY OPTIMIZER")
        print(f"📊 Universe: {len(self.usa_stocks)} USA + {len(self.europe_stocks)} Europe")
        print(f"📅 Train: {self.train_start} to {self.train_end}")
        print(f"📅 Test: {self.test_start} to {self.test_end}")
    
    def run_optimization(self):
        """Lance l'optimisation complète"""
        print("\n" + "="*80)
        print("🚀 STRATEGY OPTIMIZATION - Sprint 35-36")
        print("="*80)
        
        # Step 1: Download data
        print("\n📊 Step 1: Downloading optimization data...")
        data = self.download_data()
        
        # Step 2: Define parameter space
        print("\n🎯 Step 2: Defining parameter space...")
        param_combinations = self.define_parameter_space()
        print(f"  Total combinations to test: {len(param_combinations)}")
        
        # Step 3: Optimize on training data
        print("\n🔧 Step 3: Optimizing on training data...")
        best_params = self.optimize_parameters(data, param_combinations)
        
        # Step 4: Validate on test data
        print("\n✅ Step 4: Validating on test data...")
        validation_results = self.validate_strategy(data, best_params)
        
        # Step 5: Generate optimization report
        print("\n📋 Step 5: Generating optimization report...")
        self.generate_optimization_report(best_params, validation_results)
        
        return best_params, validation_results
    
    def download_data(self):
        """Download data for optimization"""
        all_symbols = self.usa_stocks + self.europe_stocks
        data = {}
        
        for symbol in all_symbols:
            try:
                print(f"  📊 {symbol}...", end=" ")
                ticker_data = yf.download(
                    symbol, 
                    start=self.train_start, 
                    end=self.test_end,
                    progress=False
                )
                
                if len(ticker_data) > 500:
                    data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ Insufficient data")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:20]}...")
        
        print(f"  Downloaded: {len(data)} symbols")
        return data
    
    def define_parameter_space(self):
        """Définit l'espace des paramètres à optimiser"""
        
        # Paramètres à tester
        ema_short_periods = [10, 15, 20]
        ema_long_periods = [30, 45, 50, 60]
        rsi_periods = [10, 14, 18]
        rsi_thresholds = [70, 75, 80]
        score_thresholds = [0.3, 0.4, 0.5, 0.6]
        signal_weights = [(0.6, 0.4), (0.7, 0.3), (0.5, 0.5), (0.8, 0.2)]  # EMA, RSI
        
        combinations = []
        
        for ema_short, ema_long, rsi_period, rsi_thresh, score_thresh, weights in itertools.product(
            ema_short_periods, ema_long_periods, rsi_periods, 
            rsi_thresholds, score_thresholds, signal_weights
        ):
            if ema_short < ema_long:  # EMA court < EMA long
                combinations.append({
                    'ema_short': ema_short,
                    'ema_long': ema_long,
                    'rsi_period': rsi_period,
                    'rsi_threshold': rsi_thresh,
                    'score_threshold': score_thresh,
                    'ema_weight': weights[0],
                    'rsi_weight': weights[1]
                })
        
        print(f"  EMA periods: {ema_short_periods} x {ema_long_periods}")
        print(f"  RSI periods: {rsi_periods}, thresholds: {rsi_thresholds}")
        print(f"  Score thresholds: {score_thresholds}")
        print(f"  Signal weights: {signal_weights}")
        
        return combinations[:50]  # Limite pour test rapide
    
    def optimize_parameters(self, data, param_combinations):
        """Optimise les paramètres sur données d'entraînement"""
        
        # Split training data
        train_data = {}
        for symbol, prices in data.items():
            train_mask = (prices.index >= self.train_start) & (prices.index < self.train_end)
            train_data[symbol] = prices[train_mask]
        
        best_performance = -1
        best_params = None
        results = []
        
        print(f"  Testing {len(param_combinations)} combinations...")
        
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                print(f"    Progress: {i+1}/{len(param_combinations)}")
            
            # Test this parameter combination
            performance = self.backtest_parameters(train_data, params)
            results.append({
                'params': params,
                'annual_return': performance['annual_return'],
                'sharpe_ratio': performance['sharpe_ratio'],
                'max_drawdown': performance['max_drawdown'],
                'score': performance['score']
            })
            
            # Track best combination
            if performance['score'] > best_performance:
                best_performance = performance['score']
                best_params = params.copy()
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n  🏆 Best parameters found:")
        print(f"    EMA: {best_params['ema_short']}/{best_params['ema_long']}")
        print(f"    RSI: {best_params['rsi_period']} period, <{best_params['rsi_threshold']} threshold")
        print(f"    Score threshold: {best_params['score_threshold']}")
        print(f"    Weights: EMA {best_params['ema_weight']:.1f}, RSI {best_params['rsi_weight']:.1f}")
        print(f"    Performance score: {best_performance:.3f}")
        
        # Save detailed results
        self.save_optimization_results(results)
        
        return best_params
    
    def backtest_parameters(self, data, params):
        """Backtest avec paramètres spécifiques"""
        
        # Generate trading dates (monthly for speed)
        start_date = pd.to_datetime(self.train_start)
        end_date = pd.to_datetime(self.train_end)
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        
        portfolio_value = self.initial_capital
        values = [portfolio_value]
        
        for date in rebalance_dates:
            # Generate signals with these parameters
            signals = self.calculate_signals_with_params(data, date, params)
            
            # Simple equal-weight allocation of selected assets
            selected_assets = self.select_assets_with_params(signals, params)
            
            # Calculate month return (simplified)
            if len(selected_assets) > 0:
                month_returns = []
                for symbol in selected_assets:
                    if symbol in data:
                        symbol_data = data[symbol]
                        available_data = symbol_data[symbol_data.index <= date]
                        if len(available_data) >= 30:
                            month_return = available_data.pct_change(periods=21).iloc[-1]  # ~1 month
                            # Convert to float if pandas Series
                            if hasattr(month_return, 'item'):
                                month_return = month_return.item()
                            if not pd.isna(month_return):
                                month_returns.append(float(month_return))
                
                if month_returns:
                    avg_return = np.mean(month_returns)
                    portfolio_value *= (1 + avg_return)
            
            values.append(portfolio_value)
        
        # Calculate performance metrics
        returns = np.array(values)
        total_return = (returns[-1] / returns[0]) - 1
        periods = len(returns) - 1
        annual_return = (1 + total_return) ** (12 / periods) - 1 if periods > 0 else 0
        
        # Simplified metrics
        returns_series = pd.Series(returns).pct_change().dropna()
        volatility = returns_series.std() * np.sqrt(12)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = returns / returns[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Composite score (à maximiser)
        score = annual_return + sharpe_ratio - abs(max_drawdown)
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'score': score
        }
    
    def calculate_signals_with_params(self, data, date, params):
        """Calcule signaux avec paramètres spécifiques"""
        signals = {}
        
        for symbol, prices in data.items():
            try:
                historical_data = prices[prices.index <= date]
                
                if len(historical_data) >= max(params['ema_long'], params['rsi_period']) + 10:
                    # EMA with custom periods
                    ema_short = historical_data.ewm(span=params['ema_short']).mean()
                    ema_long = historical_data.ewm(span=params['ema_long']).mean()
                    
                    # RSI with custom period
                    delta = historical_data.diff()
                    gains = delta.where(delta > 0, 0.0)
                    losses = -delta.where(delta < 0, 0.0)
                    avg_gains = gains.ewm(alpha=1/params['rsi_period']).mean()
                    avg_losses = losses.ewm(alpha=1/params['rsi_period']).mean()
                    rs = avg_gains / avg_losses
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Latest values
                    current_ema_short = ema_short.iloc[-1]
                    current_ema_long = ema_long.iloc[-1]
                    current_rsi = rsi.iloc[-1]
                    
                    # Handle pandas Series
                    if hasattr(current_ema_short, 'item'):
                        current_ema_short = current_ema_short.item()
                    if hasattr(current_ema_long, 'item'):
                        current_ema_long = current_ema_long.item()
                    if hasattr(current_rsi, 'item'):
                        current_rsi = current_rsi.item()
                    
                    # Signals with custom weights
                    ema_signal = 1 if current_ema_short > current_ema_long else 0
                    rsi_signal = 1 if current_rsi < params['rsi_threshold'] else 0
                    
                    score = (params['ema_weight'] * ema_signal + 
                            params['rsi_weight'] * rsi_signal)
                    
                    signals[symbol] = {
                        'score': score,
                        'ema_signal': ema_signal,
                        'rsi_signal': rsi_signal
                    }
                    
            except Exception:
                continue
        
        return signals
    
    def select_assets_with_params(self, signals, params):
        """Sélectionne actifs avec paramètres"""
        
        # Filter by score threshold
        qualified = [(symbol, sig['score']) for symbol, sig in signals.items() 
                    if sig['score'] >= params['score_threshold']]
        
        # Sort by score
        qualified.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers by region
        usa_selected = [s for s, _ in qualified if s in self.usa_stocks][:5]
        europe_selected = [s for s, _ in qualified if s in self.europe_stocks][:2]
        
        return usa_selected + europe_selected
    
    def validate_strategy(self, data, best_params):
        """Valide la stratégie optimisée sur données de test"""
        
        # Split test data
        test_data = {}
        for symbol, prices in data.items():
            test_mask = (prices.index >= self.test_start) & (prices.index < self.test_end)
            test_data[symbol] = prices[test_mask]
        
        print(f"  Validating optimized strategy...")
        validation_performance = self.backtest_parameters(test_data, best_params)
        
        print(f"  📊 Validation Results:")
        print(f"    Annual Return: {validation_performance['annual_return']:.1%}")
        print(f"    Sharpe Ratio: {validation_performance['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: {validation_performance['max_drawdown']:.1%}")
        
        return validation_performance
    
    def save_optimization_results(self, results):
        """Sauvegarde résultats d'optimisation"""
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"  💾 Optimization results saved to {output_dir / filename}")
    
    def generate_optimization_report(self, best_params, validation_results):
        """Génère rapport d'optimisation"""
        
        report = f"""
# 🚀 OPTIMIZATION REPORT - AlphaBot Strategy Enhancement

## 🎯 OPTIMIZED PARAMETERS

**EMA Configuration:**
- Short Period: {best_params['ema_short']} days
- Long Period: {best_params['ema_long']} days

**RSI Configuration:**
- Period: {best_params['rsi_period']} days  
- Threshold: <{best_params['rsi_threshold']}

**Scoring:**
- Score Threshold: {best_params['score_threshold']}
- EMA Weight: {best_params['ema_weight']:.1%}
- RSI Weight: {best_params['rsi_weight']:.1%}

## 📊 VALIDATION PERFORMANCE

**Returns:**
- Annual Return: {validation_results['annual_return']:.1%}
- Total Return: {validation_results['total_return']:.1%}

**Risk Metrics:**
- Sharpe Ratio: {validation_results['sharpe_ratio']:.2f}
- Max Drawdown: {validation_results['max_drawdown']:.1%}

## 🎖️ IMPROVEMENT vs BASELINE

**Baseline (9.7% annual):**
- Target Improvement: +2-5% annual
- Status: {'✅ IMPROVED' if validation_results['annual_return'] > 0.10 else '⚠️ NEEDS MORE WORK'}

## 🔧 NEXT STEPS

1. Implement optimized parameters in main system
2. Test with additional signal combinations
3. Add momentum and volume indicators
4. Dynamic position sizing optimization

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        output_path = Path("OPTIMIZATION_REPORT.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  📋 Optimization report saved to {output_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("🎯 OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Best Annual Return: {validation_results['annual_return']:.1%}")
        print(f"Best Sharpe Ratio: {validation_results['sharpe_ratio']:.2f}")
        print(f"Improvement: {'+' if validation_results['annual_return'] > 0.097 else ''}{(validation_results['annual_return'] - 0.097)*100:.1f}%")
        
        if validation_results['annual_return'] > 0.12:
            print("🎉 EXCELLENT: >12% annual achieved!")
        elif validation_results['annual_return'] > 0.10:
            print("✅ GOOD: >10% annual achieved!")
        else:
            print("⚠️ MORE OPTIMIZATION NEEDED")


def main():
    """Lance l'optimisation"""
    print("🚀 ALPHABOT STRATEGY OPTIMIZATION")
    print("Sprint 35-36: Maximize returns from 9.7% baseline")
    print("="*80)
    
    optimizer = StrategyOptimizer()
    best_params, validation = optimizer.run_optimization()
    
    return 0


if __name__ == "__main__":
    exit_code = main()