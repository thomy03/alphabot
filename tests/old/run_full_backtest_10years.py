#!/usr/bin/env python3
"""
Backtest AlphaBot complet - 10 ans de données historiques
Phase 5 : Validation performance sur données réelles
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def run_decade_backtest():
    """Backtest principal sur 10 ans (2014-2024)"""
    try:
        from alphabot.core.backtesting_engine import BacktestingEngine, BacktestConfig
        
        print("🚀 BACKTEST ALPHABOT - DÉCENNIE 2014-2024")
        print("=" * 70)
        
        # Configuration complète
        config = BacktestConfig(
            start_date="2014-01-01",
            end_date="2024-01-01", 
            initial_capital=100000.0,
            
            # Univers S&P 500 top 50
            universe=[
                # Technology
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "CRM", "ORCL", "ADBE",
                # Finance  
                "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI", "COF",
                # Healthcare
                "UNH", "JNJ", "PFE", "ABBV", "TMO", "ABT", "LLY", "MRK", "BMY", "AMGN",
                # Consumer
                "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
                # Industrial/Energy
                "BA", "CAT", "GE", "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO"
            ],
            
            # Paramètres optimisés
            commission=0.001,  # 0.1%
            slippage_bps=5,    # 5 bps
            max_position_size=0.04,  # 4% max par titre
            max_sector_exposure=0.25, # 25% max par secteur
            max_leverage=0.95,        # 95% investi max
            
            benchmark="SPY"
        )
        
        print(f"📊 Configuration:")
        print(f"   Période: {config.start_date} → {config.end_date}")
        print(f"   Capital: ${config.initial_capital:,}")
        print(f"   Univers: {len(config.universe)} actifs")
        print(f"   Benchmark: {config.benchmark}")
        
        # Créer engine et exécuter
        engine = BacktestingEngine(config)
        
        print(f"\n🔄 Chargement données historiques...")
        await engine.load_historical_data()
        
        print(f"\n⚡ Exécution backtest...")
        start_time = datetime.now()
        
        result = await engine.run_backtest()
        
        execution_time = datetime.now() - start_time
        print(f"\n⏱️ Temps d'exécution: {execution_time}")
        
        # Sauvegarder résultats détaillés
        print(f"\n💾 Sauvegarde résultats...")
        engine.save_results(result, "backtests/decade_2014_2024")
        
        return result, config
        
    except Exception as e:
        print(f"❌ Erreur backtest: {e}")
        import traceback
        traceback.print_exc()
        return None, None

async def run_crisis_analysis():
    """Analyse performance durant les crises"""
    try:
        from alphabot.core.backtesting_engine import BacktestingEngine, BacktestConfig
        
        print(f"\n🚨 ANALYSE CRISES FINANCIÈRES")
        print("=" * 50)
        
        crisis_periods = [
            {
                'name': 'Oil Crash 2014-2016',
                'start': '2014-06-01',
                'end': '2016-02-29',
                'description': 'Chute du pétrole -75%'
            },
            {
                'name': 'China Crash 2015',
                'start': '2015-06-01', 
                'end': '2016-01-31',
                'description': 'Krach boursier chinois'
            },
            {
                'name': 'COVID-19 2020',
                'start': '2020-01-01',
                'end': '2020-12-31', 
                'description': 'Pandémie mondiale'
            },
            {
                'name': 'Inflation 2022',
                'start': '2022-01-01',
                'end': '2022-12-31',
                'description': 'Hausse taux Fed'
            },
            {
                'name': 'Bank Crisis 2023',
                'start': '2023-01-01',
                'end': '2023-06-30',
                'description': 'SVB, Credit Suisse'
            }
        ]
        
        crisis_results = []
        
        for crisis in crisis_periods:
            print(f"\n📉 {crisis['name']} ({crisis['description']})")
            print(f"   Période: {crisis['start']} → {crisis['end']}")
            
            config = BacktestConfig(
                start_date=crisis['start'],
                end_date=crisis['end'],
                initial_capital=100000.0,
                universe=[
                    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                    "JPM", "BAC", "XOM", "PG", "JNJ"
                ],  # Core holdings
                max_position_size=0.08,
                benchmark="SPY"
            )
            
            engine = BacktestingEngine(config)
            result = await engine.run_backtest()
            
            # Calculer survie vs benchmark
            outperformance = result.total_return - result.benchmark_return
            
            print(f"   📊 AlphaBot: {result.total_return:+.1%}")
            print(f"   📊 SPY:      {result.benchmark_return:+.1%}")
            print(f"   📊 Alpha:    {outperformance:+.1%}")
            print(f"   📊 Max DD:   {result.max_drawdown:.1%}")
            print(f"   📊 Sharpe:   {result.sharpe_ratio:.2f}")
            
            crisis_results.append({
                'name': crisis['name'],
                'period': f"{crisis['start']}/{crisis['end']}", 
                'alphabot_return': result.total_return,
                'spy_return': result.benchmark_return,
                'outperformance': outperformance,
                'max_drawdown': result.max_drawdown,
                'sharpe': result.sharpe_ratio
            })
        
        # Synthèse crises
        avg_outperf = np.mean([r['outperformance'] for r in crisis_results])
        worst_dd = min([r['max_drawdown'] for r in crisis_results])
        crisis_wins = sum([1 for r in crisis_results if r['outperformance'] > 0])
        
        print(f"\n📊 SYNTHÈSE ANALYSE CRISES:")
        print(f"   Outperformance moyenne: {avg_outperf:+.1%}")
        print(f"   Pire drawdown:          {worst_dd:.1%}")
        print(f"   Crises gagnantes:       {crisis_wins}/{len(crisis_results)}")
        print(f"   Résilience:             {'✅' if avg_outperf > 0 and worst_dd > -0.30 else '❌'}")
        
        return crisis_results
        
    except Exception as e:
        print(f"❌ Erreur analyse crises: {e}")
        return []

async def run_sector_analysis():
    """Analyse performance par secteur"""
    try:
        from alphabot.core.backtesting_engine import BacktestingEngine, BacktestConfig
        
        print(f"\n🏭 ANALYSE SECTORIELLE")
        print("=" * 30)
        
        sectors = {
            'Technology': ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "ORCL", "CRM", "ADBE"],
            'Finance': ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK"],
            'Healthcare': ["UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK", "ABT", "TMO"], 
            'Consumer': ["PG", "KO", "WMT", "HD", "MCD", "NKE", "PEP", "SBUX"],
            'Energy': ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO"]
        }
        
        sector_results = []
        
        for sector_name, symbols in sectors.items():
            print(f"\n📊 Secteur {sector_name}")
            
            config = BacktestConfig(
                start_date="2019-01-01",  # 5 ans pour rapidité
                end_date="2024-01-01",
                initial_capital=100000.0,
                universe=symbols,
                max_position_size=0.15,  # Plus concentré par secteur
                benchmark="SPY"
            )
            
            engine = BacktestingEngine(config)
            result = await engine.run_backtest()
            
            print(f"   Rendement: {result.annualized_return:+.1%}")
            print(f"   Sharpe:    {result.sharpe_ratio:.2f}")
            print(f"   Max DD:    {result.max_drawdown:.1%}")
            
            sector_results.append({
                'sector': sector_name,
                'return': result.annualized_return,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown,
                'trades': result.total_trades
            })
        
        # Classement des secteurs
        sector_results.sort(key=lambda x: x['sharpe'], reverse=True)
        
        print(f"\n🏆 CLASSEMENT SECTEURS (par Sharpe):")
        for i, sector in enumerate(sector_results, 1):
            print(f"   {i}. {sector['sector']:<12} Sharpe: {sector['sharpe']:.2f}, Rendement: {sector['return']:+.1%}")
        
        return sector_results
        
    except Exception as e:
        print(f"❌ Erreur analyse sectorielle: {e}")
        return []

def create_performance_report(result, config, crisis_results, sector_results):
    """Génère rapport de performance complet"""
    
    print(f"\n📋 RAPPORT DE PERFORMANCE COMPLET")
    print("=" * 70)
    
    # Performance globale
    print(f"\n💰 PERFORMANCE 10 ANS (2014-2024):")
    print(f"   Capital initial:       ${config.initial_capital:,}")
    print(f"   Capital final:         ${result.portfolio_value.iloc[-1]:,.0f}")
    print(f"   Rendement total:       {result.total_return:+.1%}")
    print(f"   Rendement annualisé:   {result.annualized_return:+.1%}")
    print(f"   CAGR vs SPY:           {result.annualized_return - (result.benchmark_return * (10/10)):+.1%}")
    
    # Métriques de risque
    print(f"\n📊 MÉTRIQUES DE RISQUE:")
    print(f"   Volatilité annuelle:   {result.volatility:.1%}")
    print(f"   Sharpe ratio:          {result.sharpe_ratio:.2f}")
    print(f"   Maximum drawdown:      {result.max_drawdown:.1%}")
    print(f"   Calmar ratio:          {result.calmar_ratio:.2f}")
    print(f"   Alpha vs SPY:          {result.alpha:+.1%}")
    print(f"   Beta vs SPY:           {result.beta:.2f}")
    print(f"   Information ratio:     {result.information_ratio:.2f}")
    
    # Métriques de trading
    print(f"\n⚡ MÉTRIQUES DE TRADING:")
    print(f"   Nombre total trades:   {result.total_trades:,}")
    print(f"   Taux de réussite:      {result.win_rate:.1%}")
    print(f"   Rendement moyen/trade: {result.avg_trade_return:.2%}")
    print(f"   Période détention moy: {result.avg_holding_period:.1f} jours")
    
    # Validation objectifs
    print(f"\n🎯 VALIDATION OBJECTIFS ALPHABOT:")
    sharpe_target = 1.5
    dd_target = -0.15
    return_target = 0.12
    
    sharpe_ok = result.sharpe_ratio >= sharpe_target
    dd_ok = result.max_drawdown >= dd_target
    return_ok = result.annualized_return >= return_target
    
    print(f"   Sharpe ≥ 1.5:          {'✅' if sharpe_ok else '❌'} ({result.sharpe_ratio:.2f})")
    print(f"   Drawdown ≤ 15%:        {'✅' if dd_ok else '❌'} ({result.max_drawdown:.1%})")
    print(f"   Rendement ≥ 12%:       {'✅' if return_ok else '❌'} ({result.annualized_return:.1%})")
    
    all_targets_met = sharpe_ok and dd_ok and return_ok
    
    # Résumé crises
    if crisis_results:
        avg_crisis_alpha = np.mean([r['outperformance'] for r in crisis_results])
        crisis_resilience = avg_crisis_alpha > 0
        
        print(f"\n🚨 RÉSILIENCE AUX CRISES:")
        print(f"   Alpha moyen en crise:  {avg_crisis_alpha:+.1%}")
        print(f"   Résilience:            {'✅' if crisis_resilience else '❌'}")
    
    # Meilleur secteur
    if sector_results:
        best_sector = sector_results[0]
        print(f"\n🏆 MEILLEUR SECTEUR:")
        print(f"   {best_sector['sector']} (Sharpe: {best_sector['sharpe']:.2f})")
    
    # Conclusion
    print(f"\n🏁 CONCLUSION:")
    if all_targets_met:
        print("   🎉 TOUS LES OBJECTIFS ATTEINTS!")
        print("   ✅ AlphaBot prêt pour production")
        print("   💰 Recommandation: GO-LIVE avec capital limité")
    else:
        print("   ⚠️ Objectifs partiellement atteints")
        print("   🔧 Optimisations nécessaires avant production")
        
        if not sharpe_ok:
            print("   📊 Améliorer Sharpe ratio (diversification, signaux)")
        if not dd_ok:
            print("   🛡️ Réduire drawdown (risk management)")
        if not return_ok:
            print("   📈 Augmenter rendement (signaux, allocation)")
    
    return all_targets_met

async def main():
    """Exécution principale"""
    print("🔬 BACKTEST COMPLET ALPHABOT - PHASE 5")
    print("Validation performance sur 10 ans de données réelles")
    print("=" * 70)
    
    # 1. Backtest principal 10 ans
    print("\n1️⃣ Backtest principal 10 ans...")
    result, config = await run_decade_backtest()
    
    if result is None:
        print("❌ Échec backtest principal")
        return 1
    
    # 2. Analyse des crises
    print("\n2️⃣ Analyse performance durant crises...")
    crisis_results = await run_crisis_analysis()
    
    # 3. Analyse sectorielle  
    print("\n3️⃣ Analyse performance par secteur...")
    sector_results = await run_sector_analysis()
    
    # 4. Rapport final
    print("\n4️⃣ Génération rapport final...")
    targets_met = create_performance_report(result, config, crisis_results, sector_results)
    
    # Créer répertoire de sortie
    output_dir = Path("backtests/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder rapport JSON
    import json
    full_report = {
        'timestamp': timestamp,
        'backtest_period': f"{config.start_date} to {config.end_date}",
        'main_results': {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate
        },
        'objectives_met': targets_met,
        'crisis_analysis': crisis_results,
        'sector_analysis': sector_results
    }
    
    with open(output_dir / f"full_report_{timestamp}.json", 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"\n💾 Rapport complet sauvegardé: backtests/reports/full_report_{timestamp}.json")
    
    if targets_met:
        print("\n🎉 PHASE 5 RÉUSSIE - VALIDATION COMPLÈTE!")
        print("🚀 AlphaBot prêt pour Phase 6 (Production)")
        return 0
    else:
        print("\n⚠️ Optimisations nécessaires")
        print("🔧 Retour Phase 4 pour ajustements")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())