#!/usr/bin/env python3
"""
Simplified Historical Backtest - Sans problèmes timezone
Backtest réel simplifié pour validation performance
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SimplifiedBacktester:
    """
    Backtest simplifié sans problèmes timezone
    Focus sur validation performance système simplifié
    """
    
    def __init__(self):
        # Univers simplifié 70% USA / 30% Europe
        self.usa_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'JPM', 'BAC', 'JNJ', 'UNH', 'KO']
        self.europe_symbols = ['EWG', 'EWQ', 'EWU', 'ASML', 'SAP']  # ETFs + ADRs
        
        # Période de test
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        self.initial_capital = 100000
        
        print(f"🌍 Universe: {len(self.usa_symbols)} USA + {len(self.europe_symbols)} Europe")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"💰 Capital: ${self.initial_capital:,}")
    
    async def run_simplified_backtest(self) -> dict:
        """Execute backtest simplifié"""
        print("\n" + "="*70)
        print("📈 SIMPLIFIED HISTORICAL BACKTEST")
        print("="*70)
        
        # 1. Télécharger données
        print("\n📊 Step 1: Downloading data...")
        all_data = self.download_data()
        
        # 2. Calculer signaux mensuels (plus simple)
        print("\n🎯 Step 2: Calculating monthly signals...")
        monthly_signals = self.calculate_monthly_signals(all_data)
        
        # 3. Simulation avec rebalancing mensuel
        print("\n💼 Step 3: Running monthly simulation...")
        portfolio_performance = self.simulate_monthly_rebalancing(all_data, monthly_signals)
        
        # 4. Calculer métriques
        print("\n📊 Step 4: Calculating metrics...")
        performance_metrics = self.calculate_metrics(portfolio_performance)
        
        # 5. Comparer vs benchmarks
        print("\n🎯 Step 5: Benchmark comparison...")
        benchmark_comparison = self.compare_benchmarks(portfolio_performance)
        
        # 6. Résultats
        results = {
            'timestamp': datetime.now().isoformat(),
            'performance': performance_metrics,
            'benchmark_comparison': benchmark_comparison,
            'portfolio_history': portfolio_performance
        }
        
        self.save_results(results)
        self.print_summary(results)
        
        return results
    
    def download_data(self) -> dict:
        """Télécharge données pour tous les actifs"""
        all_symbols = self.usa_symbols + self.europe_symbols
        data = {}
        
        for i, symbol in enumerate(all_symbols, 1):
            try:
                print(f"  📊 ({i}/{len(all_symbols)}) {symbol}...", end=" ")
                ticker_data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                
                if len(ticker_data) > 100:
                    # Garder la Series pandas directement
                    data[symbol] = ticker_data['Close']
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ Insufficient data")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:20]}...")
        
        print(f"  ✅ Downloaded: {len(data)} symbols")
        return data
    
    def calculate_monthly_signals(self, data: dict) -> dict:
        """Calcule signaux mensuels pour chaque actif"""
        signals = {}
        
        for symbol, prices in data.items():
            try:
                # S'assurer que c'est une Series pandas
                if isinstance(prices, pd.Series):
                    price_series = prices
                else:
                    price_series = pd.Series(prices)
                    price_series.index = pd.to_datetime(price_series.index)
                
                price_series = price_series.sort_index().dropna()
                
                # Calculer EMA et RSI
                ema_20 = price_series.ewm(span=20).mean()
                ema_50 = price_series.ewm(span=50).mean()
                
                # RSI
                delta = price_series.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/14).mean()
                avg_losses = losses.ewm(alpha=1/14).mean()
                rs = avg_gains / avg_losses
                rsi = 100 - (100 / (1 + rs))
                
                # Signal mensuel (dernier jour de chaque mois)
                monthly_data = price_series.resample('M').last()
                monthly_signals = []
                
                for date in monthly_data.index:
                    if date in ema_20.index and date in rsi.index:
                        ema_signal = 1 if ema_20[date] > ema_50[date] else 0
                        rsi_signal = 1 if rsi[date] < 70 else 0  # Pas overbought
                        
                        # Score combiné
                        score = 0.6 * ema_signal + 0.4 * rsi_signal
                        
                        monthly_signals.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'score': score,
                            'ema_bullish': ema_signal,
                            'rsi_ok': rsi_signal,
                            'price': monthly_data[date]
                        })
                
                signals[symbol] = monthly_signals
                
            except Exception as e:
                print(f"    ❌ {symbol} signal calculation failed")
                signals[symbol] = []
        
        print(f"  ✅ Calculated signals for {len(signals)} symbols")
        return signals
    
    def simulate_monthly_rebalancing(self, data: dict, signals: dict) -> list:
        """Simulation avec rebalancing mensuel"""
        
        # Dates de rebalancing (fin de chaque mois)
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        rebalance_dates = pd.date_range(start=start, end=end, freq='ME')  # Month End
        
        portfolio_history = []
        current_value = self.initial_capital
        
        for i, rebalance_date in enumerate(rebalance_dates):
            date_str = rebalance_date.strftime('%Y-%m-%d')
            
            # Sélectionner top actifs pour ce mois
            month_scores = {}
            
            for symbol, symbol_signals in signals.items():
                # Trouver signal le plus proche de cette date
                for signal in symbol_signals:
                    signal_date = pd.to_datetime(signal['date'])
                    if abs((signal_date - rebalance_date).days) <= 15:  # Tolérance 15 jours
                        month_scores[symbol] = signal['score']
                        break
            
            # Sélectionner top actifs par région
            usa_scores = {s: score for s, score in month_scores.items() if s in self.usa_symbols}
            europe_scores = {s: score for s, score in month_scores.items() if s in self.europe_symbols}
            
            # Top 5 USA, Top 2 Europe
            usa_top = sorted(usa_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            europe_top = sorted(europe_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            selected_symbols = [s for s, _ in usa_top] + [s for s, _ in europe_top]
            
            # Allocation égale entre sélectionnés
            allocation_per_asset = 1.0 / len(selected_symbols) if selected_symbols else 0
            
            # Calculer performance depuis dernier rebalancing
            if i > 0:
                # Performance des positions précédentes
                prev_positions = portfolio_history[-1]['positions']
                month_return = 0.0
                
                for symbol, weight in prev_positions.items():
                    if symbol in data:
                        prices = pd.Series(data[symbol])
                        prices.index = pd.to_datetime(prices.index)
                        
                        # Prix début et fin de période
                        prev_date = rebalance_dates[i-1]
                        
                        try:
                            start_price = prices.asof(prev_date)
                            end_price = prices.asof(rebalance_date)
                            
                            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                                asset_return = (end_price / start_price - 1)
                                month_return += weight * asset_return
                        except:
                            pass  # Skip si erreur de prix
                
                current_value *= (1 + month_return)
            
            # Enregistrer état du portfolio
            portfolio_state = {
                'date': date_str,
                'value': current_value,
                'positions': {symbol: allocation_per_asset for symbol in selected_symbols},
                'selected_usa': [s for s, _ in usa_top],
                'selected_europe': [s for s, _ in europe_top],
                'monthly_return': month_return if i > 0 else 0
            }
            
            portfolio_history.append(portfolio_state)
            
            if i % 6 == 0:  # Progress every 6 months
                print(f"    Progress: {i+1}/{len(rebalance_dates)} months, Value: ${current_value:,.0f}")
        
        return portfolio_history
    
    def calculate_metrics(self, portfolio_history: list) -> dict:
        """Calcule métriques de performance"""
        
        values = [p['value'] for p in portfolio_history]
        monthly_returns = [p['monthly_return'] for p in portfolio_history[1:]]
        
        # Performance totale
        total_return = (values[-1] / values[0]) - 1
        annual_return = (1 + total_return) ** (12 / len(monthly_returns)) - 1
        
        # Volatilité
        volatility = np.std(monthly_returns) * np.sqrt(12)
        
        # Sharpe
        risk_free = 0.02
        sharpe = (annual_return - risk_free) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative_values = np.array(values)
        running_max = np.maximum.accumulate(cumulative_values)
        drawdowns = (cumulative_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = np.mean([r > 0 for r in monthly_returns]) if monthly_returns else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': values[-1],
            'months_simulated': len(portfolio_history)
        }
    
    def compare_benchmarks(self, portfolio_history: list) -> dict:
        """Compare vs benchmarks"""
        
        try:
            # Download benchmark data
            spy_data = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)['Close']
            ezu_data = yf.download('EZU', start=self.start_date, end=self.end_date, progress=False)['Close']
            
            # Portfolio return
            portfolio_return = (portfolio_history[-1]['value'] / portfolio_history[0]['value']) - 1
            
            # Benchmark returns
            spy_return = (spy_data.iloc[-1] / spy_data.iloc[0]) - 1
            ezu_return = (ezu_data.iloc[-1] / ezu_data.iloc[0]) - 1
            
            # Mixed benchmark (70% SPY + 30% EZU)
            mixed_return = 0.7 * spy_return + 0.3 * ezu_return
            
            return {
                'alphabot_return': portfolio_return,
                'spy_return': spy_return,
                'europe_return': ezu_return,
                'mixed_benchmark': mixed_return,
                'alpha_vs_spy': portfolio_return - spy_return,
                'alpha_vs_mixed': portfolio_return - mixed_return,
                'outperformance': portfolio_return > mixed_return
            }
            
        except Exception as e:
            print(f"    ⚠️ Benchmark comparison failed: {e}")
            return {'error': 'Benchmark data unavailable'}
    
    def save_results(self, results: dict):
        """Sauvegarde résultats"""
        output_dir = Path("backtests/simplified_historical")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simplified_backtest_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {output_dir / filename}")
    
    def print_summary(self, results: dict):
        """Affiche résumé"""
        print("\n" + "="*70)
        print("📋 SIMPLIFIED BACKTEST SUMMARY")
        print("="*70)
        
        perf = results['performance']
        bench = results['benchmark_comparison']
        
        print(f"📊 AlphaBot Simplified Performance:")
        print(f"  Total Return:      {perf['total_return']:>8.1%}")
        print(f"  Annual Return:     {perf['annual_return']:>8.1%}")
        print(f"  Volatility:        {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:      {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:      {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:      {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:          {perf['win_rate']:>8.1%}")
        print(f"  Final Value:       ${perf['final_value']:>8,.0f}")
        
        if 'error' not in bench:
            print(f"\n🎯 vs Benchmarks:")
            print(f"  AlphaBot:          {float(bench['alphabot_return']):>8.1%}")
            print(f"  S&P 500:           {float(bench['spy_return']):>8.1%}")
            print(f"  Europe:            {float(bench['europe_return']):>8.1%}")
            print(f"  Mixed (70/30):     {float(bench['mixed_benchmark']):>8.1%}")
            print(f"  Alpha vs Mixed:    {float(bench['alpha_vs_mixed']):>8.1%}")
            print(f"  Outperformance:    {'✅ YES' if bench['outperformance'] else '❌ NO'}")
        else:
            print(f"\n🎯 Benchmark comparison: {bench.get('error', 'Failed')}")
        
        print(f"\n💡 System Validation:")
        if perf['annual_return'] > 0.08:  # >8%
            print("✅ Return target exceeded")
        if perf['sharpe_ratio'] > 1.0:
            print("✅ Sharpe ratio target met")
        if abs(perf['max_drawdown']) < 0.20:
            print("✅ Drawdown control successful")
        
        # Verdict final
        targets_met = (perf['annual_return'] > 0.08 and 
                      perf['sharpe_ratio'] > 1.0 and 
                      abs(perf['max_drawdown']) < 0.20)
        
        if targets_met:
            print("\n🎉 SYSTÈME VALIDÉ - Prêt pour Sprint 35-36!")
        else:
            print("\n⚠️ Optimisations nécessaires avant production")


async def main():
    """Test principal"""
    print("📈 SIMPLIFIED HISTORICAL BACKTEST")
    print("Validation system performance without timezone issues")
    print("="*70)
    
    backtester = SimplifiedBacktester()
    results = await backtester.run_simplified_backtest()
    
    return 0 if 'error' not in results else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())