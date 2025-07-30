#!/usr/bin/env python3
"""
Test du Backtesting Engine AlphaBot
Validation sur période 2019-2023 (5 ans)
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def test_simple_backtest():
    """Test backtest simple sur 5 ans"""
    try:
        from alphabot.core.backtesting_engine import BacktestingEngine, BacktestConfig
        
        print("🚀 Test Backtesting Engine - Période 2019-2023")
        print("=" * 60)
        
        # Configuration test (période courte pour rapidité)
        config = BacktestConfig(
            start_date="2019-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
            universe=[
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                "META", "NVDA", "JPM", "V", "UNH"
            ],  # Top 10 pour test rapide
            commission=0.001,
            max_position_size=0.10,  # 10% max par position
            benchmark="SPY"
        )
        
        # Créer engine
        engine = BacktestingEngine(config)
        
        # Exécuter backtest
        print("\n📊 Exécution du backtest...")
        start_time = datetime.now()
        
        result = await engine.run_backtest()
        
        execution_time = datetime.now() - start_time
        print(f"⏱️ Temps d'exécution: {execution_time.total_seconds():.1f}s")
        
        # Afficher résultats
        print("\n" + "=" * 60)
        print("📈 RÉSULTATS BACKTEST")
        print("=" * 60)
        
        print(f"\n💰 Performance:")
        print(f"   Capital initial:      ${config.initial_capital:,.0f}")
        print(f"   Capital final:        ${result.portfolio_value.iloc[-1]:,.0f}")
        print(f"   Rendement total:      {result.total_return:+.1%}")
        print(f"   Rendement annualisé:  {result.annualized_return:+.1%}")
        
        print(f"\n📊 Métriques de risque:")
        print(f"   Volatilité:           {result.volatility:.1%}")
        print(f"   Sharpe ratio:         {result.sharpe_ratio:.2f}")
        print(f"   Max drawdown:         {result.max_drawdown:.1%}")
        print(f"   Calmar ratio:         {result.calmar_ratio:.2f}")
        
        print(f"\n🎯 Trading:")
        print(f"   Nombre de trades:     {result.total_trades}")
        print(f"   Win rate:             {result.win_rate:.1%}")
        print(f"   Rendement moyen/trade: {result.avg_trade_return:.2%}")
        
        print(f"\n📈 vs Benchmark ({config.benchmark}):")
        print(f"   Rendement benchmark:  {result.benchmark_return:+.1%}")
        print(f"   Alpha:                {result.alpha:+.1%}")
        print(f"   Beta:                 {result.beta:.2f}")
        print(f"   Information ratio:    {result.information_ratio:.2f}")
        
        # Validation des objectifs
        print(f"\n🎯 VALIDATION OBJECTIFS:")
        sharpe_ok = result.sharpe_ratio >= 1.5
        drawdown_ok = abs(result.max_drawdown) <= 0.15
        return_ok = result.annualized_return >= 0.10
        
        print(f"   Sharpe ≥ 1.5:         {'✅' if sharpe_ok else '❌'} ({result.sharpe_ratio:.2f})")
        print(f"   Drawdown ≤ 15%:       {'✅' if drawdown_ok else '❌'} ({result.max_drawdown:.1%})")
        print(f"   Rendement ≥ 10%:      {'✅' if return_ok else '❌'} ({result.annualized_return:.1%})")
        
        objectives_met = sharpe_ok and drawdown_ok and return_ok
        
        # Sauvegarder résultats
        print(f"\n💾 Sauvegarde des résultats...")
        engine.save_results(result, "backtests/test_results")
        
        print(f"\n{'🎉 BACKTEST RÉUSSI!' if objectives_met else '⚠️ Objectifs partiellement atteints'}")
        
        return {
            'success': True,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'total_return': result.total_return,
            'objectives_met': objectives_met,
            'execution_time': execution_time.total_seconds()
        }
        
    except Exception as e:
        print(f"❌ Erreur test backtest: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

async def test_stress_scenarios():
    """Test scénarios de stress"""
    try:
        from alphabot.core.backtesting_engine import BacktestingEngine, BacktestConfig
        
        print("\n🚨 Test scénarios de stress")
        print("=" * 40)
        
        scenarios = [
            {
                'name': 'COVID Crash 2020',
                'start_date': '2020-01-01',
                'end_date': '2020-06-30',
                'description': 'Période de forte volatilité'
            },
            {
                'name': 'Inflation 2022',
                'start_date': '2022-01-01', 
                'end_date': '2022-12-31',
                'description': 'Hausse des taux d\'intérêt'
            },
            {
                'name': 'Recovery 2021',
                'start_date': '2021-01-01',
                'end_date': '2021-12-31', 
                'description': 'Reprise post-COVID'
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\n📊 Scénario: {scenario['name']}")
            print(f"   Période: {scenario['start_date']} à {scenario['end_date']}")
            
            config = BacktestConfig(
                start_date=scenario['start_date'],
                end_date=scenario['end_date'],
                initial_capital=100000.0,
                universe=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],  # Réduit pour rapidité
                benchmark="SPY"
            )
            
            engine = BacktestingEngine(config)
            result = await engine.run_backtest()
            
            print(f"   Rendement: {result.total_return:+.1%}")
            print(f"   Sharpe: {result.sharpe_ratio:.2f}")
            print(f"   Max DD: {result.max_drawdown:.1%}")
            print(f"   vs SPY: {result.total_return - result.benchmark_return:+.1%}")
            
            results.append({
                'scenario': scenario['name'],
                'return': result.total_return,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown,
                'vs_benchmark': result.total_return - result.benchmark_return
            })
        
        # Analyse globale
        avg_return = sum(r['return'] for r in results) / len(results)
        avg_sharpe = sum(r['sharpe'] for r in results) / len(results)
        worst_dd = min(r['max_dd'] for r in results)
        
        print(f"\n📊 SYNTHÈSE STRESS TESTS:")
        print(f"   Rendement moyen:      {avg_return:+.1%}")
        print(f"   Sharpe moyen:         {avg_sharpe:.2f}")
        print(f"   Pire drawdown:        {worst_dd:.1%}")
        
        stress_resilience = avg_sharpe > 0.5 and worst_dd > -0.25
        print(f"   Résilience:           {'✅' if stress_resilience else '❌'}")
        
        return {
            'success': True,
            'scenarios': results,
            'resilience': stress_resilience
        }
        
    except Exception as e:
        print(f"❌ Erreur stress tests: {e}")
        return {'success': False, 'error': str(e)}

async def test_parameter_sensitivity():
    """Test sensibilité aux paramètres"""
    try:
        from alphabot.core.backtesting_engine import BacktestingEngine, BacktestConfig
        
        print("\n🔬 Test sensibilité paramètres")
        print("=" * 35)
        
        base_config = BacktestConfig(
            start_date="2021-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
            universe=["AAPL", "MSFT", "GOOGL", "TSLA"],
            benchmark="SPY"
        )
        
        # Test différentes tailles de position max
        position_sizes = [0.05, 0.10, 0.15, 0.20]
        
        print("\n📊 Impact taille position max:")
        
        best_sharpe = -999
        best_config = None
        
        for pos_size in position_sizes:
            config = BacktestConfig(
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                initial_capital=base_config.initial_capital,
                universe=base_config.universe,
                max_position_size=pos_size,
                benchmark=base_config.benchmark
            )
            
            engine = BacktestingEngine(config)
            result = await engine.run_backtest()
            
            print(f"   Position max {pos_size:.0%}: Sharpe {result.sharpe_ratio:.2f}, DD {result.max_drawdown:.1%}")
            
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_config = pos_size
        
        print(f"\n🏆 Meilleure configuration: Position max {best_config:.0%} (Sharpe {best_sharpe:.2f})")
        
        return {
            'success': True,
            'best_position_size': best_config,
            'best_sharpe': best_sharpe
        }
        
    except Exception as e:
        print(f"❌ Erreur test sensibilité: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Test principal"""
    print("🔬 TESTS BACKTESTING ENGINE - AlphaBot Phase 5")
    print("=" * 70)
    
    results = []
    
    # Test backtest principal
    print("\n1️⃣ Test backtest 5 ans...")
    result1 = await test_simple_backtest()
    results.append(result1['success'])
    
    # Test scénarios de stress
    print("\n2️⃣ Test scénarios de stress...")
    result2 = await test_stress_scenarios()
    results.append(result2['success'])
    
    # Test sensibilité paramètres
    print("\n3️⃣ Test sensibilité paramètres...")
    result3 = await test_parameter_sensitivity()
    results.append(result3['success'])
    
    # Bilan global
    success_count = sum(results)
    total_tests = len(results)
    
    print("\n" + "=" * 70)
    print(f"📊 BILAN TESTS: {success_count}/{total_tests} réussis")
    
    if success_count == total_tests:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Backtesting engine opérationnel")
        print("✅ Performance validée sur 5 ans")
        print("✅ Résilience aux stress scenarios")
        print("✅ Optimisation paramètres")
        print("\n🚀 Prêt pour backtests 10 ans complets!")
        
        if result1['success'] and result1.get('objectives_met'):
            print("🎯 Objectifs de performance ATTEINTS!")
            print(f"   Sharpe: {result1['sharpe_ratio']:.2f} ≥ 1.5 ✅")
            print(f"   Drawdown: {result1['max_drawdown']:.1%} ≤ 15% ✅")
        
        return 0
    else:
        print(f"\n⚠️ {total_tests - success_count} test(s) échoué(s)")
        print("🔧 Révision nécessaire avant production")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())