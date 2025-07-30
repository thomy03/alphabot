#!/usr/bin/env python3
"""
Real Historical Backtest - Simulation jour par jour sur 5 ans
Backtest réel avec répartition 70% USA / 30% Europe
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yfinance as yf
import json
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphabot.agents.technical.simplified_technical_agent import SimplifiedTechnicalAgent
from alphabot.agents.risk.enhanced_risk_agent import EnhancedRiskAgent


class RealHistoricalBacktester:
    """
    Backtest historique réel jour par jour
    Répartition géographique : 70% USA, 30% Europe
    """
    
    def __init__(self):
        self.technical_agent = SimplifiedTechnicalAgent()
        self.risk_agent = EnhancedRiskAgent()
        
        # Univers d'investissement 70% USA / 30% Europe
        self.universe = {
            'USA': {
                'weight': 0.70,
                'symbols': [
                    # Tech (25%)
                    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
                    # Finance (15%)
                    'JPM', 'BAC', 'WFC',
                    # Healthcare (10%)
                    'JNJ', 'UNH',
                    # Consumer (10%)
                    'KO', 'PG',
                    # Energy (5%)
                    'XOM', 'CVX',
                    # Industrials (5%)
                    'CAT', 'GE'
                ]
            },
            'Europe': {
                'weight': 0.30,
                'symbols': [
                    # ETFs Europe (plus liquides que actions individuelles)
                    'EWG',   # Germany ETF
                    'EWQ',   # France ETF 
                    'EWI',   # Italy ETF
                    'EWP',   # Spain ETF
                    'EWU',   # UK ETF
                    'EWN',   # Netherlands ETF
                    'EWO',   # Austria ETF
                    'EWK',   # Belgium ETF
                    # Actions européennes majeures (ADR US)
                    'ASML',  # ASML (Netherlands)
                    'SAP',   # SAP (Germany)
                    'NVO',   # Novo Nordisk (Denmark)
                    'NESN.SW' # Nestlé (peut nécessiter ajustement)
                ]
            }
        }
        
        # Configuration backtest
        self.start_date = datetime(2019, 1, 1)  # 5 ans
        self.end_date = datetime(2024, 1, 1)
        self.initial_capital = 100000  # $100k
        self.rebalance_frequency = 7  # Weekly (tous les 7 jours)
        
        # Limites de risque
        self.max_position_size = 0.05  # 5% max par actif
        self.max_sector_weight = 0.30  # 30% max par secteur
        
        print(f"🌍 Universe: {len(self.universe['USA']['symbols'])} USA + {len(self.universe['Europe']['symbols'])} Europe")
        print(f"📅 Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"💰 Initial capital: ${self.initial_capital:,}")
        print(f"🔄 Rebalancing: Every {self.rebalance_frequency} days")
    
    async def run_historical_backtest(self) -> dict:
        """Execute le backtest historique complet"""
        print("\n" + "="*80)
        print("📈 REAL HISTORICAL BACKTEST - Day by Day Simulation")
        print("="*80)
        
        # 1. Télécharger toutes les données historiques
        print("\n📊 Step 1: Downloading historical data...")
        all_data = await self.download_all_historical_data()
        
        if not all_data:
            return {'error': 'Failed to download historical data'}
        
        # 2. Initialiser portfolio
        print("\n💼 Step 2: Initializing portfolio...")
        portfolio = self.initialize_portfolio()
        
        # 3. Simulation jour par jour
        print("\n🕐 Step 3: Running day-by-day simulation...")
        simulation_results = await self.run_daily_simulation(all_data, portfolio)
        
        # 4. Calculer métriques de performance
        print("\n📊 Step 4: Calculating performance metrics...")
        performance_metrics = self.calculate_performance_metrics(simulation_results)
        
        # 5. Comparer vs benchmarks
        print("\n🎯 Step 5: Benchmark comparison...")
        benchmark_comparison = await self.compare_vs_benchmarks(simulation_results)
        
        # 6. Sauvegarder résultats
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'initial_capital': self.initial_capital,
                'rebalance_frequency': self.rebalance_frequency,
                'universe': self.universe
            },
            'simulation_results': simulation_results,
            'performance_metrics': performance_metrics,
            'benchmark_comparison': benchmark_comparison
        }
        
        await self.save_backtest_results(results)
        self.print_backtest_summary(results)
        
        return results
    
    async def download_all_historical_data(self) -> dict:
        """Télécharge données historiques pour tous les actifs"""
        all_symbols = self.universe['USA']['symbols'] + self.universe['Europe']['symbols']
        all_data = {}
        failed_symbols = []
        
        print(f"  📥 Downloading data for {len(all_symbols)} symbols...")
        
        for i, symbol in enumerate(all_symbols, 1):
            try:
                print(f"  📊 ({i}/{len(all_symbols)}) {symbol}...", end=" ")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=self.start_date - timedelta(days=100),  # Extra data for indicators
                    end=self.end_date,
                    interval="1d"
                )
                
                if len(data) > 200:  # Sufficient data
                    all_data[symbol] = data
                    print(f"✅ {len(data)} days")
                else:
                    failed_symbols.append(symbol)
                    print(f"❌ Insufficient data ({len(data)} days)")
                
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"❌ Error: {str(e)[:30]}...")
        
        print(f"\n  ✅ Successfully downloaded: {len(all_data)} symbols")
        print(f"  ❌ Failed: {len(failed_symbols)} symbols {failed_symbols if failed_symbols else ''}")
        
        return all_data
    
    def initialize_portfolio(self) -> dict:
        """Initialise le portfolio"""
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},  # {symbol: {'shares': float, 'value': float}}
            'total_value': self.initial_capital,
            'history': [],
            'trades': [],
            'last_rebalance': pd.Timestamp(self.start_date).tz_localize('UTC')
        }
        
        print(f"  💰 Starting cash: ${portfolio['cash']:,.2f}")
        return portfolio
    
    async def run_daily_simulation(self, all_data: dict, portfolio: dict) -> dict:
        """Simulation jour par jour"""
        
        # Générer dates de trading (jours ouvrables seulement)
        trading_dates = self.generate_trading_dates()
        total_days = len(trading_dates)
        
        print(f"  📅 Simulating {total_days} trading days...")
        
        rebalance_counter = 0
        
        for i, current_date in enumerate(trading_dates):
            # Progress indicator
            if i % 50 == 0 or i == total_days - 1:
                progress = (i + 1) / total_days * 100
                print(f"    Progress: {progress:.1f}% ({current_date.strftime('%Y-%m-%d')})")
            
            # 1. Update portfolio value avec prix actuels
            portfolio_value = self.update_portfolio_value(portfolio, all_data, current_date)
            
            # 2. Check si rebalancing nécessaire
            days_since_rebalance = (current_date - portfolio['last_rebalance']).days
            should_rebalance = days_since_rebalance >= self.rebalance_frequency
            
            # 3. Rebalancing si nécessaire
            if should_rebalance:
                rebalance_counter += 1
                await self.execute_rebalancing(portfolio, all_data, current_date)
                portfolio['last_rebalance'] = current_date
            
            # 4. Log daily performance
            portfolio['history'].append({
                'date': current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date),
                'total_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions_value': portfolio_value - portfolio['cash'],
                'rebalanced': should_rebalance
            })
        
        simulation_summary = {
            'total_trading_days': total_days,
            'rebalancing_events': rebalance_counter,
            'final_value': portfolio['total_value'],
            'total_return': (portfolio['total_value'] / self.initial_capital - 1),
            'portfolio_history': portfolio['history'],
            'trades_executed': portfolio['trades']
        }
        
        print(f"  ✅ Simulation completed:")
        print(f"    📊 Trading days: {total_days}")
        print(f"    🔄 Rebalances: {rebalance_counter}")
        print(f"    💰 Final value: ${portfolio['total_value']:,.2f}")
        print(f"    📈 Total return: {simulation_summary['total_return']:.1%}")
        
        return simulation_summary
    
    def generate_trading_dates(self) -> list:
        """Génère liste des dates de trading (jours ouvrables)"""
        dates = []
        current = self.start_date
        
        while current < self.end_date:
            # Jours ouvrables seulement (lundi=0, dimanche=6)
            if current.weekday() < 5:  # Lundi à Vendredi
                # Convertir en timestamp pandas avec timezone UTC
                dates.append(pd.Timestamp(current).tz_localize('UTC'))
            current += timedelta(days=1)
        
        return dates
    
    def update_portfolio_value(self, portfolio: dict, all_data: dict, current_date) -> float:
        """Met à jour la valeur du portfolio"""
        total_value = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            if symbol in all_data:
                # Trouver prix du jour (ou dernier disponible)
                symbol_data = all_data[symbol]
                
                # Assurer compatibilité timezone
                if hasattr(current_date, 'tz_localize'):
                    search_date = current_date
                else:
                    search_date = pd.Timestamp(current_date).tz_localize('UTC')
                
                # Chercher prix exact ou plus proche
                try:
                    current_prices = symbol_data[symbol_data.index <= search_date]
                    
                    if len(current_prices) > 0:
                        latest_price = current_prices['Close'].iloc[-1]
                        position_value = position['shares'] * latest_price
                        position['value'] = position_value
                        total_value += position_value
                except Exception as e:
                    # Fallback: utiliser dernier prix disponible
                    if len(symbol_data) > 0:
                        latest_price = symbol_data['Close'].iloc[-1]
                        position_value = position['shares'] * latest_price
                        position['value'] = position_value
                        total_value += position_value
        
        portfolio['total_value'] = total_value
        return total_value
    
    async def execute_rebalancing(self, portfolio: dict, all_data: dict, current_date):
        """Execute le rebalancing du portfolio"""
        
        # 1. Générer signaux pour tous les actifs
        signals = {}
        available_symbols = []
        
        # Assurer compatibilité timezone
        if hasattr(current_date, 'tz_localize'):
            search_date = current_date
        else:
            search_date = pd.Timestamp(current_date).tz_localize('UTC')
        
        for region, config in self.universe.items():
            for symbol in config['symbols']:
                if symbol in all_data:
                    # Vérifier données disponibles à cette date
                    symbol_data = all_data[symbol]
                    
                    try:
                        data_until_date = symbol_data[symbol_data.index <= search_date]
                        
                        if len(data_until_date) >= 50:  # Minimum pour calculs techniques
                            # Utiliser données jusqu'à current_date seulement
                            signal = await self.get_signal_for_date(symbol, data_until_date)
                            signals[symbol] = signal
                            available_symbols.append(symbol)
                    except Exception as e:
                        # Fallback: prendre toutes les données disponibles
                        if len(symbol_data) >= 50:
                            try:
                                signal = await self.get_signal_for_date(symbol, symbol_data)
                                signals[symbol] = signal
                                available_symbols.append(symbol)
                            except:
                                pass  # Skip si erreur
        
        # 2. Sélectionner top actifs par région
        usa_symbols = [s for s in available_symbols if s in self.universe['USA']['symbols']]
        eur_symbols = [s for s in available_symbols if s in self.universe['Europe']['symbols']]
        
        # Top 10 USA, Top 5 Europe
        usa_top = self.select_top_signals(signals, usa_symbols, 10)
        eur_top = self.select_top_signals(signals, eur_symbols, 5)
        
        # 3. Calculer allocations target
        target_allocations = {}
        
        # 70% USA
        if usa_top:
            usa_weight_per_stock = 0.70 / len(usa_top)
            for symbol in usa_top:
                target_allocations[symbol] = min(usa_weight_per_stock, self.max_position_size)
        
        # 30% Europe  
        if eur_top:
            eur_weight_per_stock = 0.30 / len(eur_top)
            for symbol in eur_top:
                target_allocations[symbol] = min(eur_weight_per_stock, self.max_position_size)
        
        # 4. Execute trades
        total_trades = self.execute_trades(portfolio, all_data, current_date, target_allocations)
        
        if total_trades > 0:
            portfolio['trades'].append({
                'date': current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date),
                'trades_count': total_trades,
                'selected_usa': usa_top,
                'selected_europe': eur_top,
                'allocations': target_allocations
            })
    
    async def get_signal_for_date(self, symbol: str, historical_data: pd.DataFrame) -> dict:
        """Génère signal pour un symbole à une date donnée"""
        
        # Utiliser les dernières données disponibles
        if len(historical_data) < 50:
            return {'score': 0.5, 'confidence': 0.5}
        
        # Calcul EMA crossover
        closes = historical_data['Close']
        ema_20 = closes.ewm(span=20).mean()
        ema_50 = closes.ewm(span=50).mean()
        
        current_diff = ema_20.iloc[-1] - ema_50.iloc[-1]
        signal_strength = np.tanh(current_diff / ema_50.iloc[-1] * 100)
        
        # Calcul RSI  
        delta = closes.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        avg_gains = gains.ewm(alpha=1/14).mean()
        avg_losses = losses.ewm(alpha=1/14).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]
        
        # Score combiné
        score = 0.5  # Base neutre
        if signal_strength > 0.1:
            score += 0.3 * min(signal_strength, 1.0)
        if rsi_value < 30:
            score += 0.2
        elif rsi_value > 70:
            score -= 0.2
        
        score = max(0, min(1, score))
        
        return {
            'score': score,
            'confidence': abs(score - 0.5) * 2,  # Distance de neutralité
            'ema_signal': signal_strength,
            'rsi': rsi_value
        }
    
    def select_top_signals(self, signals: dict, symbols: list, top_n: int) -> list:
        """Sélectionne les top N signaux"""
        
        symbol_scores = [(symbol, signals[symbol]['score']) for symbol in symbols if symbol in signals]
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [symbol for symbol, score in symbol_scores[:top_n] if score > 0.6]
    
    def execute_trades(self, portfolio: dict, all_data: dict, current_date, target_allocations: dict) -> int:
        """Execute les trades pour atteindre allocations target"""
        
        trades_count = 0
        total_portfolio_value = portfolio['total_value']
        
        # 1. Sell positions not in target
        positions_to_close = []
        for symbol in portfolio['positions'].keys():
            if symbol not in target_allocations:
                positions_to_close.append(symbol)
        
        for symbol in positions_to_close:
            if symbol in all_data:
                trades_count += self.sell_position(portfolio, symbol, all_data, current_date)
        
        # 2. Adjust existing positions + buy new ones
        for symbol, target_weight in target_allocations.items():
            if symbol in all_data:
                target_value = total_portfolio_value * target_weight
                current_value = portfolio['positions'].get(symbol, {}).get('value', 0)
                
                if abs(target_value - current_value) > total_portfolio_value * 0.01:  # 1% threshold
                    trades_count += self.adjust_position(portfolio, symbol, target_value, all_data, current_date)
        
        return trades_count
    
    def sell_position(self, portfolio: dict, symbol: str, all_data: dict, current_date) -> int:
        """Vend une position complètement"""
        if symbol not in portfolio['positions']:
            return 0
        
        # Prix actuel avec gestion timezone
        symbol_data = all_data[symbol]
        
        try:
            if hasattr(current_date, 'tz_localize'):
                search_date = current_date
            else:
                search_date = pd.Timestamp(current_date).tz_localize('UTC')
            
            current_prices = symbol_data[symbol_data.index <= search_date]
            if len(current_prices) == 0:
                return 0
            
            price = current_prices['Close'].iloc[-1]
        except:
            # Fallback: dernier prix disponible
            if len(symbol_data) == 0:
                return 0
            price = symbol_data['Close'].iloc[-1]
        
        position = portfolio['positions'][symbol]
        
        # Execute vente
        proceeds = position['shares'] * price
        portfolio['cash'] += proceeds
        del portfolio['positions'][symbol]
        
        return 1
    
    def adjust_position(self, portfolio: dict, symbol: str, target_value: float, all_data: dict, current_date) -> int:
        """Ajuste une position vers valeur target"""
        
        # Prix actuel avec gestion timezone
        symbol_data = all_data[symbol]
        
        try:
            if hasattr(current_date, 'tz_localize'):
                search_date = current_date
            else:
                search_date = pd.Timestamp(current_date).tz_localize('UTC')
            
            current_prices = symbol_data[symbol_data.index <= search_date]
            if len(current_prices) == 0:
                return 0
            
            price = current_prices['Close'].iloc[-1]
        except:
            # Fallback: dernier prix disponible
            if len(symbol_data) == 0:
                return 0
            price = symbol_data['Close'].iloc[-1]
        
        # Position actuelle
        current_position = portfolio['positions'].get(symbol, {'shares': 0, 'value': 0})
        current_shares = current_position['shares']
        target_shares = target_value / price
        
        shares_diff = target_shares - current_shares
        trade_value = abs(shares_diff * price)
        
        # Execute trade si assez de cash ou de valeur à vendre
        if shares_diff > 0:  # Buy
            if portfolio['cash'] >= trade_value:
                portfolio['cash'] -= trade_value
                portfolio['positions'][symbol] = {
                    'shares': target_shares,
                    'value': target_value
                }
                return 1
        else:  # Sell
            portfolio['cash'] += trade_value
            if target_shares > 0:
                portfolio['positions'][symbol] = {
                    'shares': target_shares,
                    'value': target_value
                }
            else:
                if symbol in portfolio['positions']:
                    del portfolio['positions'][symbol]
            return 1
        
        return 0
    
    def calculate_performance_metrics(self, simulation_results: dict) -> dict:
        """Calcule métriques de performance"""
        
        history = pd.DataFrame(simulation_results['portfolio_history'])
        history['date'] = pd.to_datetime(history['date'])
        history.set_index('date', inplace=True)
        
        # Returns
        values = history['total_value']
        daily_returns = values.pct_change().dropna()
        
        # Métriques de base
        total_return = simulation_results['total_return']
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = values / values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio (recommandation expert)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (daily_returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trading_days': len(daily_returns),
            'rebalancing_frequency': simulation_results['rebalancing_events']
        }
    
    async def compare_vs_benchmarks(self, simulation_results: dict) -> dict:
        """Compare vs benchmarks SPY et Europe"""
        
        try:
            # Download benchmark data
            spy = yf.download('SPY', start=self.start_date, end=self.end_date)['Close']
            ezu = yf.download('EZU', start=self.start_date, end=self.end_date)['Close']  # Europe ETF
            
            # Portfolio performance
            history = pd.DataFrame(simulation_results['portfolio_history'])
            history['date'] = pd.to_datetime(history['date'])
            portfolio_values = history['total_value'].values
            
            # Align dates
            portfolio_returns = (portfolio_values[-1] / portfolio_values[0]) - 1
            spy_returns = (spy.iloc[-1] / spy.iloc[0]) - 1
            ezu_returns = (ezu.iloc[-1] / ezu.iloc[0]) - 1
            
            # Mixed benchmark (70% SPY + 30% EZU)
            mixed_benchmark_return = 0.7 * spy_returns + 0.3 * ezu_returns
            
            return {
                'alphabot_return': portfolio_returns,
                'spy_return': spy_returns,
                'europe_return': ezu_returns,
                'mixed_benchmark_return': mixed_benchmark_return,
                'alpha_vs_spy': portfolio_returns - spy_returns,
                'alpha_vs_mixed': portfolio_returns - mixed_benchmark_return,
                'outperformance': portfolio_returns > mixed_benchmark_return
            }
            
        except Exception as e:
            return {'error': f'Benchmark comparison failed: {e}'}
    
    async def save_backtest_results(self, results: dict):
        """Sauvegarde résultats"""
        output_dir = Path("backtests/real_historical")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"real_backtest_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {output_dir / filename}")
    
    def print_backtest_summary(self, results: dict):
        """Affiche résumé du backtest"""
        print("\n" + "="*80)
        print("📋 REAL HISTORICAL BACKTEST SUMMARY")
        print("="*80)
        
        perf = results['performance_metrics']
        bench = results['benchmark_comparison']
        
        print(f"📊 Performance AlphaBot Simplifié:")
        print(f"  Total Return:      {perf['total_return']:>8.1%}")
        print(f"  Annual Return:     {perf['annual_return']:>8.1%}")
        print(f"  Volatility:        {perf['volatility']:>8.1%}")
        print(f"  Sharpe Ratio:      {perf['sharpe_ratio']:>8.2f}")
        print(f"  Calmar Ratio:      {perf['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:      {perf['max_drawdown']:>8.1%}")
        print(f"  Win Rate:          {perf['win_rate']:>8.1%}")
        
        if 'error' not in bench:
            print(f"\n🎯 Benchmark Comparison:")
            print(f"  AlphaBot:          {bench['alphabot_return']:>8.1%}")
            print(f"  S&P 500:           {bench['spy_return']:>8.1%}")
            print(f"  Europe:            {bench['europe_return']:>8.1%}")
            print(f"  Mixed (70/30):     {bench['mixed_benchmark_return']:>8.1%}")
            print(f"  Alpha vs Mixed:    {bench['alpha_vs_mixed']:>8.1%}")
            print(f"  Outperformance:    {'✅ YES' if bench['outperformance'] else '❌ NO'}")
        
        print(f"\n📈 Trading Activity:")
        print(f"  Trading Days:      {perf['trading_days']:>8,}")
        print(f"  Rebalances:        {perf['rebalancing_frequency']:>8,}")
        print(f"  Avg Days/Rebal:    {perf['trading_days']/max(perf['rebalancing_frequency'],1):>8.1f}")


async def main():
    """Backtest principal"""
    print("📈 REAL HISTORICAL BACKTEST")
    print("Day-by-day simulation with 70% USA / 30% Europe allocation")
    print("="*80)
    
    backtester = RealHistoricalBacktester()
    results = await backtester.run_historical_backtest()
    
    if 'error' not in results:
        print("\n🎉 REAL BACKTEST COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n❌ BACKTEST FAILED: {results['error']}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())