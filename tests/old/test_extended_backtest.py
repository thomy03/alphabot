#!/usr/bin/env python3
"""
Extended Backtest - Test sur plus d'actifs et plus d'historique
Test étendu pour validation robuste système simplifié
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphabot.agents.technical.simplified_technical_agent import SimplifiedTechnicalAgent
from alphabot.agents.risk.enhanced_risk_agent import EnhancedRiskAgent


class ExtendedBacktester:
    """
    Test étendu sur plus d'actifs et plus d'historique
    """
    
    def __init__(self):
        self.technical_agent = SimplifiedTechnicalAgent()
        self.risk_agent = EnhancedRiskAgent()
        
        # Univers étendu par secteur
        self.test_universe = {
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA'],
            'Finance': ['JPM', 'BAC', 'GS', 'WFC', 'C'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABT', 'MRK'],
            'Consumer': ['KO', 'PG', 'WMT', 'HD', 'MCD'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
            'International': ['^STOXX50E', '^N225', 'EWJ', 'EFA']  # Indices pour diversification
        }
        
        # Périodes de test
        self.test_periods = {
            'short_term': 90,    # 3 mois
            'medium_term': 252,  # 1 an  
            'long_term': 1260,   # 5 ans
            'full_history': 2520 # 10 ans
        }
        
        print(f"🌍 Extended universe: {sum(len(stocks) for stocks in self.test_universe.values())} assets")
        print(f"📅 Test periods: {list(self.test_periods.keys())}")
    
    async def run_extended_analysis(self) -> dict:
        """Analyse étendue par secteur et période"""
        print("\n" + "="*80)
        print("🌍 EXTENDED BACKTEST - Multi-Asset, Multi-Period")
        print("="*80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'universe_tested': self.test_universe,
            'periods_tested': self.test_periods,
            'sector_analysis': {},
            'period_analysis': {},
            'summary_stats': {}
        }
        
        # Test par secteur
        print("\n📊 Sector Analysis:")
        for sector, symbols in self.test_universe.items():
            print(f"\n🏢 Testing {sector} sector ({len(symbols)} assets)...")
            sector_results = await self.test_sector(sector, symbols)
            results['sector_analysis'][sector] = sector_results
        
        # Test par période (sur tech stocks seulement pour vitesse)
        print("\n📅 Period Analysis:")
        tech_symbols = self.test_universe['Tech'][:3]  # AAPL, MSFT, GOOGL
        for period_name, days in self.test_periods.items():
            print(f"\n⏰ Testing {period_name} ({days} days)...")
            period_results = await self.test_period(tech_symbols, days)
            results['period_analysis'][period_name] = period_results
        
        # Statistiques globales
        results['summary_stats'] = self.calculate_summary_stats(results)
        
        # Sauvegarde
        await self.save_extended_results(results)
        
        # Affichage résumé
        self.print_extended_summary(results)
        
        return results
    
    async def test_sector(self, sector_name: str, symbols: list) -> dict:
        """Test d'un secteur complet"""
        sector_results = {
            'symbols_tested': symbols,
            'signals_generated': {},
            'risk_assessments': {},
            'sector_metrics': {}
        }
        
        successful_signals = 0
        total_confidence = 0
        risk_scores = []
        
        for symbol in symbols:
            try:
                # Signaux techniques
                signals = await self.technical_agent.get_simplified_signals(symbol)
                sector_results['signals_generated'][symbol] = signals
                
                if 'error' not in signals:
                    successful_signals += 1
                    total_confidence += signals.get('confidence', 0)
                
                # Risk assessment
                risk_assessment = await self.risk_agent.assess_single_asset_risk(symbol)
                sector_results['risk_assessments'][symbol] = risk_assessment
                
                if 'error' not in risk_assessment:
                    risk_scores.append(risk_assessment.get('risk_score', 0.5))
                
                print(f"  ✅ {symbol}: {signals.get('action', 'ERROR')} (conf: {signals.get('confidence', 0):.2f})")
                
            except Exception as e:
                print(f"  ❌ {symbol}: {str(e)[:50]}...")
                sector_results['signals_generated'][symbol] = {'error': str(e)}
        
        # Métriques secteur
        sector_results['sector_metrics'] = {
            'success_rate': successful_signals / len(symbols),
            'avg_confidence': total_confidence / max(successful_signals, 1),
            'avg_risk_score': np.mean(risk_scores) if risk_scores else 0.5,
            'risk_score_std': np.std(risk_scores) if len(risk_scores) > 1 else 0,
            'buy_signals': sum(1 for s in sector_results['signals_generated'].values() if s.get('action') == 'BUY'),
            'sell_signals': sum(1 for s in sector_results['signals_generated'].values() if s.get('action') == 'SELL'),
            'hold_signals': sum(1 for s in sector_results['signals_generated'].values() if s.get('action') == 'HOLD')
        }
        
        metrics = sector_results['sector_metrics']
        print(f"    📊 Success: {metrics['success_rate']:.1%} | Confidence: {metrics['avg_confidence']:.2f} | Risk: {metrics['avg_risk_score']:.2f}")
        
        return sector_results
    
    async def test_period(self, symbols: list, days: int) -> dict:
        """Test sur une période donnée"""
        period_results = {
            'period_days': days,
            'symbols': symbols,
            'historical_data_available': {},
            'metrics_evolution': {}
        }
        
        # Test disponibilité données
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=f"{days}d", interval="1d")
                
                period_results['historical_data_available'][symbol] = {
                    'days_available': len(data),
                    'data_completeness': len(data) / days,
                    'oldest_date': data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else None,
                    'newest_date': data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else None
                }
                
                print(f"  📊 {symbol}: {len(data)}/{days} days ({len(data)/days:.1%})")
                
            except Exception as e:
                print(f"  ❌ {symbol}: Data unavailable")
                period_results['historical_data_available'][symbol] = {'error': str(e)}
        
        # Calcul métriques moyennes
        available_data = [d for d in period_results['historical_data_available'].values() if 'error' not in d]
        if available_data:
            period_results['metrics_evolution'] = {
                'avg_data_completeness': np.mean([d['data_completeness'] for d in available_data]),
                'min_data_completeness': np.min([d['data_completeness'] for d in available_data]),
                'max_data_completeness': np.max([d['data_completeness'] for d in available_data])
            }
        
        return period_results
    
    def calculate_summary_stats(self, results: dict) -> dict:
        """Calcule statistiques globales"""
        # Compter actifs par secteur
        total_assets = sum(len(symbols) for symbols in self.test_universe.values())
        
        # Success rates par secteur
        sector_success_rates = {}
        for sector, data in results['sector_analysis'].items():
            sector_success_rates[sector] = data['sector_metrics']['success_rate']
        
        # Données historiques
        period_completeness = {}
        for period, data in results['period_analysis'].items():
            if 'metrics_evolution' in data and data['metrics_evolution']:
                period_completeness[period] = data['metrics_evolution']['avg_data_completeness']
        
        return {
            'total_assets_tested': total_assets,
            'sectors_tested': len(self.test_universe),
            'periods_tested': len(self.test_periods),
            'overall_success_rate': np.mean(list(sector_success_rates.values())),
            'best_sector': max(sector_success_rates.items(), key=lambda x: x[1]) if sector_success_rates else None,
            'worst_sector': min(sector_success_rates.items(), key=lambda x: x[1]) if sector_success_rates else None,
            'data_availability': period_completeness
        }
    
    async def save_extended_results(self, results: dict):
        """Sauvegarde résultats étendus"""
        from pathlib import Path
        
        output_dir = Path("backtests/extended_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"extended_backtest_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Extended results saved to {output_dir / filename}")
    
    def print_extended_summary(self, results: dict):
        """Affiche résumé étendu"""
        print("\n" + "="*80)
        print("📋 EXTENDED BACKTEST SUMMARY")
        print("="*80)
        
        summary = results['summary_stats']
        
        print(f"🌍 Assets tested: {summary['total_assets_tested']}")
        print(f"🏢 Sectors tested: {summary['sectors_tested']}")
        print(f"📅 Periods tested: {summary['periods_tested']}")
        print(f"✅ Overall success rate: {summary['overall_success_rate']:.1%}")
        
        if summary['best_sector']:
            best_sector, best_rate = summary['best_sector']
            print(f"🏆 Best sector: {best_sector} ({best_rate:.1%})")
        
        if summary['worst_sector']:
            worst_sector, worst_rate = summary['worst_sector']
            print(f"⚠️ Worst sector: {worst_sector} ({worst_rate:.1%})")
        
        print(f"\n📊 Data Availability by Period:")
        for period, completeness in summary['data_availability'].items():
            print(f"  {period}: {completeness:.1%}")
        
        # Recommandations
        print(f"\n💡 RECOMMENDATIONS:")
        if summary['overall_success_rate'] > 0.8:
            print("✅ System robust across sectors")
        else:
            print("⚠️ Sector-specific optimizations needed")
        
        if all(c > 0.9 for c in summary['data_availability'].values()):
            print("✅ Data coverage excellent for all periods")
        else:
            print("⚠️ Some periods have limited data")


async def main():
    """Test principal étendu"""
    print("🌍 EXTENDED BACKTEST - Multi-Asset, Multi-Period Analysis")
    print("Testing AlphaBot simplified system on broader universe")
    print("="*80)
    
    backtester = ExtendedBacktester()
    results = await backtester.run_extended_analysis()
    
    if 'error' not in results:
        print("\n🎉 EXTENDED ANALYSIS COMPLETED!")
        return 0
    else:
        print(f"\n❌ EXTENDED ANALYSIS FAILED: {results['error']}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())