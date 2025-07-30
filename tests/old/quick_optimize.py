#!/usr/bin/env python3
"""
Quick Strategy Optimizer - Version simplifiée et rapide
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class QuickOptimizer:
    """Optimiseur rapide et simplifié"""
    
    def __init__(self):
        # Symboles de test
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'EWG', 'EWQ']
        self.start_date = "2020-01-01"
        self.end_date = "2024-01-01"
        
        print("🚀 QUICK STRATEGY OPTIMIZER")
        print(f"📊 Testing {len(self.symbols)} symbols")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
    
    def run_quick_optimization(self):
        """Lance optimisation rapide"""
        
        print("\n" + "="*60)
        print("⚡ QUICK OPTIMIZATION")
        print("="*60)
        
        # Download data
        print("\n📊 Downloading data...")
        data = self.download_data()
        
        # Test parameter combinations
        print("\n🔧 Testing parameter combinations...")
        results = self.test_combinations(data)
        
        # Find best
        best_result = max(results, key=lambda x: x['annual_return'])
        
        print(f"\n🏆 BEST CONFIGURATION FOUND:")
        print(f"  EMA: {best_result['ema_short']}/{best_result['ema_long']}")
        print(f"  RSI: {best_result['rsi_period']} period, <{best_result['rsi_threshold']}")
        print(f"  Score threshold: {best_result['score_threshold']}")
        print(f"  Annual Return: {best_result['annual_return']:.1%}")
        print(f"  Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {best_result['max_drawdown']:.1%}")
        
        # Save best configuration
        self.save_best_config(best_result)
        
        return best_result
    
    def download_data(self):
        """Download data rapide"""
        data = {}
        
        for symbol in self.symbols:
            try:
                print(f"  📊 {symbol}...", end=" ")
                ticker_data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                
                if len(ticker_data) > 500:
                    data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ {len(ticker_data)} days")
                    
            except Exception as e:
                print(f"❌ Error")
        
        print(f"  Downloaded: {len(data)} symbols")
        return data
    
    def test_combinations(self, data):
        """Test combinaisons de paramètres"""
        
        # Paramètres à tester (réduits pour rapidité)
        combinations = [
            # EMA_short, EMA_long, RSI_period, RSI_threshold, score_threshold
            (10, 30, 10, 70, 0.3),
            (10, 30, 14, 75, 0.4),
            (15, 45, 14, 80, 0.3),
            (20, 50, 14, 75, 0.4),  # Baseline actuel
            (12, 40, 12, 78, 0.35),
            (8, 25, 16, 72, 0.45),
            (18, 55, 10, 85, 0.25),
        ]
        
        results = []
        
        for i, (ema_short, ema_long, rsi_period, rsi_threshold, score_threshold) in enumerate(combinations):
            print(f"  🔧 Testing combination {i+1}/{len(combinations)}: EMA({ema_short}/{ema_long}), RSI({rsi_period}, <{rsi_threshold})")
            
            performance = self.backtest_simple(data, {
                'ema_short': ema_short,
                'ema_long': ema_long,
                'rsi_period': rsi_period,
                'rsi_threshold': rsi_threshold,
                'score_threshold': score_threshold
            })
            
            results.append({
                'ema_short': ema_short,
                'ema_long': ema_long,
                'rsi_period': rsi_period,
                'rsi_threshold': rsi_threshold,
                'score_threshold': score_threshold,
                **performance
            })
            
            print(f"    Result: {performance['annual_return']:.1%} annual, Sharpe {performance['sharpe_ratio']:.2f}")
        
        return results
    
    def backtest_simple(self, data, params):
        """Backtest simplifié mais fonctionnel"""
        
        # Dates de rebalancing mensuelles
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        rebalance_dates = pd.date_range(start=start, end=end, freq='MS')  # Month start
        
        portfolio_value = 100000
        history = [portfolio_value]
        
        for date in rebalance_dates[1:]:  # Skip first date
            try:
                # Calculate signals for this date
                signals = {}
                
                for symbol, prices in data.items():
                    historical_data = prices[prices.index <= date]
                    
                    if len(historical_data) >= max(params['ema_long'], params['rsi_period']) + 5:
                        # EMA calculation
                        ema_short = historical_data.ewm(span=params['ema_short']).mean().iloc[-1]
                        ema_long = historical_data.ewm(span=params['ema_long']).mean().iloc[-1]
                        
                        # RSI calculation
                        delta = historical_data.diff()
                        gains = delta.where(delta > 0, 0.0)
                        losses = -delta.where(delta < 0, 0.0)
                        avg_gains = gains.ewm(alpha=1/params['rsi_period']).mean()
                        avg_losses = losses.ewm(alpha=1/params['rsi_period']).mean()
                        rs = avg_gains / avg_losses
                        rsi = (100 - (100 / (1 + rs))).iloc[-1]
                        
                        # Convert to float
                        if hasattr(ema_short, 'item'):
                            ema_short = ema_short.item()
                        if hasattr(ema_long, 'item'):
                            ema_long = ema_long.item()
                        if hasattr(rsi, 'item'):
                            rsi = rsi.item()
                        
                        # Signal generation
                        ema_signal = 1 if ema_short > ema_long else 0
                        rsi_signal = 1 if rsi < params['rsi_threshold'] else 0
                        score = 0.6 * ema_signal + 0.4 * rsi_signal
                        
                        if score >= params['score_threshold']:
                            signals[symbol] = score
                
                # Calculate portfolio return for this month
                if len(signals) > 0:
                    # Select top signals
                    top_symbols = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:4]
                    selected_symbols = [s for s, _ in top_symbols]
                    
                    # Calculate equal-weight return for selected assets
                    returns = []
                    for symbol in selected_symbols:
                        if symbol in data:
                            symbol_prices = data[symbol]
                            # Get prices around this date
                            before_date = date - pd.DateOffset(months=1)
                            price_before = symbol_prices[symbol_prices.index <= before_date]
                            price_current = symbol_prices[symbol_prices.index <= date]
                            
                            if len(price_before) > 0 and len(price_current) > 0:
                                return_1m = (price_current.iloc[-1] / price_before.iloc[-1]) - 1
                                if hasattr(return_1m, 'item'):
                                    return_1m = return_1m.item()
                                returns.append(return_1m)
                    
                    if returns:
                        avg_return = np.mean(returns)
                        portfolio_value *= (1 + avg_return)
                
                history.append(portfolio_value)
                
            except Exception as e:
                history.append(portfolio_value)  # Keep same value if error
        
        # Calculate performance metrics
        if len(history) > 1:
            values = np.array(history)
            total_return = (values[-1] / values[0]) - 1
            periods = len(values) - 1
            annual_return = (1 + total_return) ** (12 / periods) - 1 if periods > 0 else 0
            
            # Calculate returns and volatility
            returns_series = pd.Series(values).pct_change().dropna()
            if len(returns_series) > 0:
                volatility = returns_series.std() * np.sqrt(12)
                sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
                
                # Drawdown
                cumulative = values / values[0]
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown)
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            annual_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
            total_return = 0
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'final_value': portfolio_value
        }
    
    def save_best_config(self, best_result):
        """Sauvegarde la meilleure configuration"""
        
        config = {
            'optimized_parameters': {
                'ema_short': best_result['ema_short'],
                'ema_long': best_result['ema_long'],
                'rsi_period': best_result['rsi_period'],
                'rsi_threshold': best_result['rsi_threshold'],
                'score_threshold': best_result['score_threshold']
            },
            'performance': {
                'annual_return': best_result['annual_return'],
                'sharpe_ratio': best_result['sharpe_ratio'],
                'max_drawdown': best_result['max_drawdown']
            },
            'optimization_date': datetime.now().isoformat()
        }
        
        with open('OPTIMIZED_CONFIG.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Best configuration saved to OPTIMIZED_CONFIG.json")
        
        # Create implementation instructions
        instructions = f"""
# 🚀 OPTIMIZED PARAMETERS - IMPLEMENTATION

## Configuration Optimisée
- **EMA Short:** {best_result['ema_short']} jours
- **EMA Long:** {best_result['ema_long']} jours  
- **RSI Period:** {best_result['rsi_period']} jours
- **RSI Threshold:** <{best_result['rsi_threshold']}
- **Score Threshold:** {best_result['score_threshold']}

## Performance Améliorée
- **Return Annuel:** {best_result['annual_return']:.1%}
- **Sharpe Ratio:** {best_result['sharpe_ratio']:.2f}
- **Max Drawdown:** {best_result['max_drawdown']:.1%}

## Implémentation
1. Modifier les paramètres dans le script principal
2. Tester sur données complètes 28 actifs
3. Valider performance sur 5 ans complets

*Optimisé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('IMPLEMENTATION_INSTRUCTIONS.md', 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        print(f"💡 Implementation instructions saved to IMPLEMENTATION_INSTRUCTIONS.md")


def main():
    """Lance optimisation rapide"""
    optimizer = QuickOptimizer()
    best_config = optimizer.run_quick_optimization()
    
    improvement = best_config['annual_return'] - 0.097  # vs baseline 9.7%
    
    print(f"\n🎯 OPTIMIZATION COMPLETE")
    print(f"Improvement vs baseline: {improvement*100:+.1f}%")
    
    if best_config['annual_return'] > 0.12:
        print("🎉 EXCELLENT: >12% annual achieved!")
    elif best_config['annual_return'] > 0.10:
        print("✅ SUCCESS: >10% annual achieved!")
    else:
        print("⚠️ Need more optimization")
    
    return 0


if __name__ == "__main__":
    exit_code = main()