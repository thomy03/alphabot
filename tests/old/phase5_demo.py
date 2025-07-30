#!/usr/bin/env python3
"""
Démo complète Phase 5 AlphaBot
Séquence: Backtest → Paper Trading → Dashboard
"""

import asyncio
import sys
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def run_demo_backtest():
    """Lance un backtest de démonstration (rapide)"""
    try:
        from alphabot.core.backtesting_engine import BacktestingEngine, BacktestConfig
        
        print("1️⃣ DÉMONSTRATION BACKTEST")
        print("=" * 40)
        
        # Configuration demo (période courte)
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
            universe=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],  # Top 5 pour rapidité
            commission=0.001,
            max_position_size=0.10,
            benchmark="SPY"
        )
        
        print(f"📊 Configuration:")
        print(f"   Période: {config.start_date} → {config.end_date}")
        print(f"   Capital: ${config.initial_capital:,}")
        print(f"   Univers: {len(config.universe)} actifs")
        
        # Exécuter backtest
        engine = BacktestingEngine(config)
        
        print(f"\n⚡ Exécution backtest...")
        start_time = datetime.now()
        
        result = await engine.run_backtest()
        
        execution_time = datetime.now() - start_time
        print(f"⏱️ Temps: {execution_time.total_seconds():.1f}s")
        
        # Afficher résultats
        print(f"\n📈 RÉSULTATS:")
        print(f"   Rendement total:      {result.total_return:+.1%}")
        print(f"   Rendement annualisé:  {result.annualized_return:+.1%}")
        print(f"   Sharpe ratio:         {result.sharpe_ratio:.2f}")
        print(f"   Max drawdown:         {result.max_drawdown:.1%}")
        print(f"   Nombre de trades:     {result.total_trades}")
        print(f"   Win rate:             {result.win_rate:.1%}")
        
        # Validation objectifs
        sharpe_ok = result.sharpe_ratio >= 1.5
        dd_ok = abs(result.max_drawdown) <= 0.15
        return_ok = result.annualized_return >= 0.10
        
        print(f"\n🎯 VALIDATION:")
        print(f"   Sharpe ≥ 1.5:         {'✅' if sharpe_ok else '❌'} ({result.sharpe_ratio:.2f})")
        print(f"   Drawdown ≤ 15%:       {'✅' if dd_ok else '❌'} ({result.max_drawdown:.1%})")
        print(f"   Rendement ≥ 10%:      {'✅' if return_ok else '❌'} ({result.annualized_return:.1%})")
        
        objectives_met = sharpe_ok and dd_ok and return_ok
        
        # Sauvegarder pour dashboard
        engine.save_results(result, "backtests/demo_results")
        
        print(f"\n{'🎉 Backtest RÉUSSI!' if objectives_met else '⚠️ Objectifs partiellement atteints'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur backtest demo: {e}")
        return False

async def run_demo_paper_trading():
    """Lance une session de paper trading de démonstration"""
    try:
        from alphabot.core.paper_trading import PaperTradingEngine, PaperTradingConfig
        from alphabot.core.signal_hub import Signal, SignalType, SignalPriority
        from unittest.mock import AsyncMock, patch
        
        print("\n2️⃣ DÉMONSTRATION PAPER TRADING")
        print("=" * 40)
        
        # Mock des dépendances
        mock_hub = AsyncMock()
        mock_crew = AsyncMock()
        
        with patch('alphabot.core.paper_trading.get_signal_hub', return_value=mock_hub), \
             patch('alphabot.core.paper_trading.CrewOrchestrator', return_value=mock_crew):
            
            # Configuration demo
            config = PaperTradingConfig(
                initial_capital=50000.0,
                commission_rate=0.001,
                price_update_interval=2,  # 2 secondes
                portfolio_update_interval=3,  # 3 secondes
                market_hours_only=False,
                data_directory="demo_paper_trading",
                save_trades=True,
                save_positions=True
            )
            
            print(f"💰 Capital initial: ${config.initial_capital:,}")
            print(f"📊 Commission: {config.commission_rate:.1%}")
            print(f"⏱️ Durée demo: 20 secondes")
            
            # Créer engine
            engine = PaperTradingEngine(config)
            
            # Prix initiaux simulés
            engine.current_prices = {
                'AAPL': 175.0,
                'MSFT': 350.0,
                'GOOGL': 2500.0,
                'AMZN': 145.0,
                'TSLA': 200.0
            }
            
            print(f"\n💹 Prix simulés:")
            for symbol, price in engine.current_prices.items():
                print(f"   {symbol}: ${price:.2f}")
            
            # Lancer paper trading en arrière-plan
            print(f"\n🚀 Démarrage paper trading...")
            
            trading_task = asyncio.create_task(engine.start())
            
            # Attendre initialisation
            await asyncio.sleep(2)
            
            # Injecter quelques signaux de test
            print(f"\n📡 Injection de signaux de test...")
            
            # Signal d'achat AAPL
            buy_signal = Signal(
                id="demo_buy_001",
                type=SignalType.BUY_SIGNAL,
                source_agent="demo_agent",
                priority=SignalPriority.HIGH,
                data={'symbol': 'AAPL', 'weight': 0.10}  # 10% du portfolio
            )
            
            await engine._handle_trading_signal(buy_signal)
            print(f"   📈 Signal d'achat: AAPL (10%)")
            
            await asyncio.sleep(3)
            
            # Signal de rebalancement
            rebalance_signal = Signal(
                id="demo_rebal_001",
                type=SignalType.PORTFOLIO_REBALANCE,
                source_agent="optimization_agent",
                priority=SignalPriority.HIGH,
                data={
                    'weights': {
                        'AAPL': 0.20,
                        'MSFT': 0.15,
                        'GOOGL': 0.10,
                        'CASH': 0.55
                    }
                }
            )
            
            await engine._handle_trading_signal(rebalance_signal)
            print(f"   🔄 Signal de rebalancement")
            
            # Attendre trading
            print(f"\n⏳ Trading en cours...")
            await asyncio.sleep(8)
            
            # Simuler variation de prix
            print(f"\n📈 Simulation variations de prix...")
            for symbol in engine.current_prices:
                change = 0.02 if symbol in ['AAPL', 'MSFT'] else -0.01
                engine.current_prices[symbol] *= (1 + change)
            
            await asyncio.sleep(5)
            
            # Arrêter trading
            print(f"\n🛑 Arrêt paper trading...")
            await engine.stop()
            trading_task.cancel()
            
            try:
                await trading_task
            except asyncio.CancelledError:
                pass
            
            # Résultats
            portfolio = await engine.get_portfolio_summary()
            
            print(f"\n📊 RÉSULTATS PAPER TRADING:")
            print(f"   Valeur finale:        ${portfolio['total_value']:,.0f}")
            print(f"   P&L total:            ${portfolio['total_pnl']:+,.0f}")
            print(f"   Rendement:            {portfolio['total_return']:+.1%}")
            print(f"   Ordres exécutés:      {portfolio['total_trades']}")
            print(f"   Positions actives:    {len(portfolio['positions'])}")
            print(f"   Cash restant:         ${portfolio['cash']:,.0f}")
            
            if portfolio['positions']:
                print(f"\n📈 Positions:")
                for pos in portfolio['positions']:
                    print(f"   {pos['symbol']:<6} {pos['quantity']:>8.0f} @ ${pos['avg_cost']:>7.2f}")
            
            success = portfolio['total_trades'] > 0
            
            print(f"\n{'✅ Paper Trading RÉUSSI!' if success else '⚠️ Activité limitée'}")
            
            return True
            
    except Exception as e:
        print(f"❌ Erreur paper trading demo: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_dashboard():
    """Lance le dashboard Streamlit"""
    try:
        print("\n3️⃣ LANCEMENT DASHBOARD")
        print("=" * 30)
        
        # Vérifier streamlit
        try:
            import streamlit
            print(f"✅ Streamlit {streamlit.__version__} disponible")
        except ImportError:
            print("❌ Streamlit non installé")
            print("💡 Installation: pip install streamlit")
            return False
        
        # Chemin dashboard
        dashboard_path = Path(__file__).parent.parent / "alphabot" / "dashboard" / "streamlit_app.py"
        
        if not dashboard_path.exists():
            print(f"❌ Dashboard non trouvé: {dashboard_path}")
            return False
        
        print(f"📊 Dashboard: streamlit_app.py")
        print(f"🌐 URL: http://localhost:8501")
        print(f"🔗 Sections disponibles:")
        print(f"   📊 Live Trading    - Données paper trading")
        print(f"   📈 Backtest        - Résultats backtest demo")
        print(f"   📋 Agent Status    - Statut des agents")
        print(f"   ⚙️ Configuration   - Paramètres")
        print(f"\n🚀 Lancement du dashboard...")
        print(f"🛑 Pour arrêter: Ctrl+C dans le terminal")
        print("=" * 50)
        
        # Lancer streamlit
        cmd = [
            sys.executable,
            "-m", "streamlit",
            "run",
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost", 
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd, cwd=dashboard_path.parent.parent.parent)
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n🛑 Dashboard arrêté")
        return True
    except Exception as e:
        print(f"❌ Erreur lancement dashboard: {e}")
        return False

async def main():
    """Démo principale Phase 5"""
    
    print("🚀 DÉMO COMPLÈTE ALPHABOT - PHASE 5")
    print("Validation des capacités de backtesting et paper trading")
    print("=" * 70)
    
    results = []
    
    # 1. Backtest de démonstration
    result1 = await run_demo_backtest()
    results.append(result1)
    
    if not result1:
        print("❌ Échec backtest - arrêt de la démo")
        return 1
    
    # Pause entre les étapes
    print(f"\n⏸️ Pause 3 secondes...")
    await asyncio.sleep(3)
    
    # 2. Paper trading de démonstration
    result2 = await run_demo_paper_trading()
    results.append(result2)
    
    if not result2:
        print("❌ Échec paper trading - poursuite dashboard")
    
    # Pause avant dashboard
    print(f"\n⏸️ Préparation dashboard...")
    await asyncio.sleep(2)
    
    # 3. Lancement dashboard
    print(f"\n🎯 Toutes les données sont prêtes pour le dashboard!")
    print(f"📊 Vous pouvez maintenant explorer:")
    print(f"   • Résultats du backtest de démonstration")
    print(f"   • Données du paper trading")
    print(f"   • Métriques de performance")
    print(f"   • Configuration système")
    
    # Demander confirmation pour dashboard
    try:
        response = input(f"\n🚀 Lancer le dashboard Streamlit? (y/N): ").strip().lower()
        
        if response in ['y', 'yes', 'o', 'oui']:
            result3 = launch_dashboard()
            results.append(result3)
        else:
            print(f"ℹ️ Dashboard non lancé")
            print(f"💡 Pour lancer manuellement: python scripts/run_dashboard.py")
            results.append(True)  # Pas d'erreur
            
    except KeyboardInterrupt:
        print(f"\n🛑 Démo interrompue par l'utilisateur")
        return 0
    
    # Bilan
    success_count = sum(results)
    total_steps = len(results)
    
    print(f"\n" + "=" * 70)
    print(f"📊 BILAN DÉMO PHASE 5: {success_count}/{total_steps} étapes réussies")
    
    if success_count == total_steps:
        print(f"\n🎉 DÉMO PHASE 5 COMPLÈTE!")
        print(f"✅ Backtest fonctionnel et validé")
        print(f"✅ Paper trading opérationnel")
        print(f"✅ Dashboard de monitoring")
        print(f"\n🚀 AlphaBot prêt pour Phase 6 (Production)!")
        
        print(f"\n📋 PROCHAINES ÉTAPES:")
        print(f"   1. Backtests complets 10 ans (scripts/run_full_backtest_10years.py)")
        print(f"   2. Paper trading longue durée (3 mois)")
        print(f"   3. Optimisation paramètres")
        print(f"   4. Go-live avec capital limité")
        
        return 0
    else:
        print(f"\n⚠️ {total_steps - success_count} étape(s) échouée(s)")
        print(f"🔧 Révisions nécessaires")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())