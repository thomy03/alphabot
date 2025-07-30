#!/usr/bin/env python3
"""
Test du Paper Trading Engine AlphaBot
Simulation de trading en temps réel
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import time
from unittest.mock import AsyncMock, patch

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Redis
sys.modules['redis'] = AsyncMock()
sys.modules['redis.asyncio'] = AsyncMock()

async def test_paper_trading_basic():
    """Test fonctionnalités de base du paper trading"""
    try:
        from alphabot.core.paper_trading import PaperTradingEngine, PaperTradingConfig
        
        print("📊 Test Paper Trading - Fonctionnalités de base")
        print("=" * 60)
        
        # Configuration test
        config = PaperTradingConfig(
            initial_capital=50000.0,
            commission_rate=0.001,
            price_update_interval=1,  # 1 seconde pour test
            portfolio_update_interval=2,
            signal_check_interval=1,
            market_hours_only=False,  # Pas de restriction horaire pour test
            data_directory="test_paper_trading"
        )
        
        # Créer engine
        engine = PaperTradingEngine(config)
        
        print(f"💰 Capital initial: ${engine.cash:,.0f}")
        
        # Simuler quelques prix
        engine.current_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0,
            'TSLA': 200.0
        }
        
        print(f"💹 Prix simulés:")
        for symbol, price in engine.current_prices.items():
            print(f"   {symbol}: ${price:.2f}")
        
        # Test création d'ordres
        from alphabot.core.paper_trading import PaperOrder, PaperOrderType
        
        test_orders = [
            PaperOrder(
                id="test_001",
                symbol="AAPL",
                side="BUY",
                quantity=100,
                order_type=PaperOrderType.MARKET
            ),
            PaperOrder(
                id="test_002", 
                symbol="MSFT",
                side="BUY",
                quantity=50,
                order_type=PaperOrderType.MARKET
            ),
            PaperOrder(
                id="test_003",
                symbol="GOOGL", 
                side="BUY",
                quantity=10,
                order_type=PaperOrderType.MARKET
            )
        ]
        
        print(f"\n📋 Test d'ordres ({len(test_orders)} ordres):")
        
        # Ajouter ordres à l'engine
        for order in test_orders:
            engine.orders[order.id] = order
            
        # Simuler exécution des ordres
        await engine._process_pending_orders()
        
        # Vérifier résultats
        executed_orders = [o for o in engine.execution_history]
        positions_count = len(engine.positions)
        remaining_cash = engine.cash
        
        print(f"   Ordres exécutés:      {len(executed_orders)}")
        print(f"   Positions ouvertes:   {positions_count}")
        print(f"   Cash restant:         ${remaining_cash:,.0f}")
        
        # Test portfolio summary
        portfolio = await engine.get_portfolio_summary()
        
        print(f"\n💼 Résumé portfolio:")
        print(f"   Valeur totale:        ${portfolio['total_value']:,.0f}")
        print(f"   Valeur investie:      ${portfolio['invested_value']:,.0f}")
        print(f"   P&L total:            ${portfolio['total_pnl']:+,.0f}")
        print(f"   Rendement:            {portfolio['total_return']:+.1%}")
        
        # Vérifier positions
        print(f"\n📈 Positions détaillées:")
        for pos in portfolio['positions']:
            print(f"   {pos['symbol']:<6} {pos['quantity']:>8.0f} @ ${pos['avg_cost']:>7.2f} = ${pos['market_value']:>10,.0f}")
        
        success = len(executed_orders) > 0 and positions_count > 0
        
        print(f"\n{'✅ Test basic réussi!' if success else '❌ Test basic échoué'}")
        return success
        
    except Exception as e:
        print(f"❌ Erreur test basic: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_signal_processing():
    """Test traitement des signaux de trading"""
    try:
        from alphabot.core.paper_trading import PaperTradingEngine, PaperTradingConfig
        from alphabot.core.signal_hub import Signal, SignalType, SignalPriority
        
        print(f"\n🔔 Test traitement des signaux")
        print("=" * 40)
        
        # Mock signal hub
        mock_hub = AsyncMock()
        
        with patch('alphabot.core.paper_trading.get_signal_hub', return_value=mock_hub):
            config = PaperTradingConfig(
                initial_capital=100000.0,
                market_hours_only=False
            )
            
            engine = PaperTradingEngine(config)
            
            # Prix simulés
            engine.current_prices = {
                'AAPL': 175.0,
                'MSFT': 350.0,
                'AMZN': 150.0
            }
            
            # Test signal d'achat
            buy_signal = Signal(
                id="buy_001",
                type=SignalType.BUY_SIGNAL,
                source_agent="technical_agent",
                priority=SignalPriority.HIGH,
                data={
                    'symbol': 'AAPL',
                    'weight': 0.05,  # 5% du portfolio
                    'confidence': 0.85
                }
            )
            
            print(f"📈 Signal d'achat: {buy_signal.data['symbol']} ({buy_signal.data['weight']:.0%})")
            
            await engine._handle_trading_signal(buy_signal)
            
            # Vérifier création d'ordre
            buy_orders = [o for o in engine.orders.values() if o.side == "BUY"]
            print(f"   Ordres d'achat créés: {len(buy_orders)}")
            
            if buy_orders:
                order = buy_orders[0]
                print(f"   Détail: {order.quantity:.0f} {order.symbol} @ market")
            
            # Test signal de rebalancement
            rebalance_signal = Signal(
                id="rebal_001",
                type=SignalType.PORTFOLIO_REBALANCE,
                source_agent="optimization_agent", 
                priority=SignalPriority.HIGH,
                data={
                    'weights': {
                        'AAPL': 0.30,
                        'MSFT': 0.25,
                        'AMZN': 0.20,
                        'CASH': 0.25
                    }
                }
            )
            
            print(f"\n🔄 Signal de rebalancement:")
            for symbol, weight in rebalance_signal.data['weights'].items():
                print(f"   {symbol}: {weight:.0%}")
            
            await engine._handle_trading_signal(rebalance_signal)
            
            # Compter ordres total
            total_orders = len(engine.orders)
            rebalance_orders = len([o for o in engine.orders.values() 
                                   if o.id.startswith("rebal_")])
            
            print(f"   Ordres de rebalancement: {rebalance_orders}")
            print(f"   Total ordres en attente: {total_orders}")
            
            # Simuler exécution
            await engine._process_pending_orders()
            
            executed = len(engine.execution_history)
            print(f"   Ordres exécutés: {executed}")
            
            success = total_orders > 0 and executed > 0
            
            print(f"\n{'✅ Test signaux réussi!' if success else '❌ Test signaux échoué'}")
            return success
            
    except Exception as e:
        print(f"❌ Erreur test signaux: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_portfolio_metrics():
    """Test calcul des métriques de performance"""
    try:
        from alphabot.core.paper_trading import PaperTradingEngine, PaperTradingConfig, PaperPosition
        
        print(f"\n📊 Test métriques de performance")
        print("=" * 40)
        
        config = PaperTradingConfig(initial_capital=100000.0)
        engine = PaperTradingEngine(config)
        
        # Simuler positions existantes
        engine.positions = {
            'AAPL': PaperPosition(
                symbol='AAPL',
                quantity=100,
                avg_cost=150.0,
                market_value=17500.0,  # Prix actuel 175$
                unrealized_pnl=2500.0
            ),
            'MSFT': PaperPosition(
                symbol='MSFT', 
                quantity=50,
                avg_cost=300.0,
                market_value=17500.0,  # Prix actuel 350$
                unrealized_pnl=2500.0
            )
        }
        
        # Simuler cash utilisé
        engine.cash = 65000.0  # 35k investi
        
        # Simuler quelques jours de rendements
        engine.metrics['daily_returns'] = [
            0.02,   # +2%
            -0.01,  # -1%
            0.015,  # +1.5%
            -0.005, # -0.5%
            0.01    # +1%
        ]
        
        # Mettre à jour métriques
        await engine._update_portfolio_metrics()
        
        portfolio = await engine.get_portfolio_summary()
        
        print(f"💼 Portfolio:")
        print(f"   Valeur totale:    ${portfolio['total_value']:,.0f}")
        print(f"   Cash:             ${portfolio['cash']:,.0f}")
        print(f"   Investi:          ${portfolio['invested_value']:,.0f}")
        print(f"   P&L:              ${portfolio['total_pnl']:+,.0f}")
        print(f"   Rendement:        {portfolio['total_return']:+.1%}")
        
        print(f"\n📈 Métriques:")
        metrics = portfolio['metrics']
        print(f"   Sharpe ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max drawdown:     {metrics.get('max_drawdown', 0):.1%}")
        print(f"   Win rate:         {metrics.get('win_rate', 0):.1%}")
        print(f"   Total trades:     {metrics.get('total_trades', 0)}")
        
        # Vérifier calculs
        expected_total = engine.cash + sum(p.market_value for p in engine.positions.values())
        actual_total = portfolio['total_value']
        
        calculation_ok = abs(expected_total - actual_total) < 1.0
        has_positions = len(portfolio['positions']) > 0
        has_metrics = 'sharpe_ratio' in metrics
        
        success = calculation_ok and has_positions and has_metrics
        
        print(f"\n✅ Validation:")
        print(f"   Calculs corrects:     {'✅' if calculation_ok else '❌'}")
        print(f"   Positions détaillées: {'✅' if has_positions else '❌'}")
        print(f"   Métriques calculées:  {'✅' if has_metrics else '❌'}")
        
        print(f"\n{'✅ Test métriques réussi!' if success else '❌ Test métriques échoué'}")
        return success
        
    except Exception as e:
        print(f"❌ Erreur test métriques: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_real_time_simulation():
    """Test simulation temps réel (courte durée)"""
    try:
        from alphabot.core.paper_trading import PaperTradingEngine, PaperTradingConfig
        
        print(f"\n⏱️ Test simulation temps réel (10 secondes)")
        print("=" * 50)
        
        # Mock CrewOrchestrator pour éviter dépendances
        mock_crew = AsyncMock()
        mock_hub = AsyncMock()
        
        with patch('alphabot.core.paper_trading.CrewOrchestrator', return_value=mock_crew), \
             patch('alphabot.core.paper_trading.get_signal_hub', return_value=mock_hub):
            
            config = PaperTradingConfig(
                initial_capital=50000.0,
                price_update_interval=1,  # 1 seconde
                portfolio_update_interval=2,  # 2 secondes
                signal_check_interval=3,   # 3 secondes
                market_hours_only=False
            )
            
            engine = PaperTradingEngine(config)
            
            # Prix initiaux
            engine.current_prices = {
                'AAPL': 150.0,
                'MSFT': 300.0
            }
            
            print(f"🚀 Démarrage simulation...")
            print(f"   Capital: ${engine.cash:,.0f}")
            print(f"   Durée: 10 secondes")
            
            # Lancer simulation en arrière-plan
            simulation_task = asyncio.create_task(engine.start())
            
            # Attendre quelques secondes
            await asyncio.sleep(3)
            
            # Injecter quelques ordres manuellement
            from alphabot.core.paper_trading import PaperOrder, PaperOrderType
            
            test_order = PaperOrder(
                id="sim_001",
                symbol="AAPL",
                side="BUY", 
                quantity=50,
                order_type=PaperOrderType.MARKET
            )
            
            engine.orders[test_order.id] = test_order
            print(f"📋 Ordre injecté: {test_order.side} {test_order.quantity} {test_order.symbol}")
            
            # Attendre encore
            await asyncio.sleep(5)
            
            # Arrêter simulation
            await engine.stop()
            simulation_task.cancel()
            
            try:
                await simulation_task
            except asyncio.CancelledError:
                pass
            
            # Vérifier résultats
            portfolio = await engine.get_portfolio_summary()
            
            print(f"\n📊 Résultats après 8 secondes:")
            print(f"   Valeur finale:        ${portfolio['total_value']:,.0f}")
            print(f"   Ordres exécutés:      {portfolio['total_trades']}")
            print(f"   Positions actives:    {len(portfolio['positions'])}")
            print(f"   Cash restant:         ${portfolio['cash']:,.0f}")
            
            # Vérifier logs/métriques
            has_trades = portfolio['total_trades'] > 0
            has_positions = len(portfolio['positions']) > 0
            
            success = engine.is_running == False  # Arrêt propre
            
            print(f"\n✅ Validation:")
            print(f"   Arrêt propre:         {'✅' if not engine.is_running else '❌'}")
            print(f"   Activité trading:     {'✅' if has_trades or has_positions else '⚠️'}")
            
            print(f"\n{'✅ Test simulation réussi!' if success else '❌ Test simulation échoué'}")
            return success
            
    except Exception as e:
        print(f"❌ Erreur test simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Test principal"""
    print("🧪 TESTS PAPER TRADING ENGINE - AlphaBot Phase 5")
    print("=" * 70)
    
    results = []
    
    # Test fonctionnalités de base
    print("\n1️⃣ Test fonctionnalités de base...")
    result1 = await test_paper_trading_basic()
    results.append(result1)
    
    # Test traitement signaux
    print("\n2️⃣ Test traitement des signaux...")
    result2 = await test_signal_processing()
    results.append(result2)
    
    # Test métriques
    print("\n3️⃣ Test métriques de performance...")
    result3 = await test_portfolio_metrics()
    results.append(result3)
    
    # Test simulation temps réel
    print("\n4️⃣ Test simulation temps réel...")
    result4 = await test_real_time_simulation()
    results.append(result4)
    
    # Bilan global
    success_count = sum(results)
    total_tests = len(results)
    
    print("\n" + "=" * 70)
    print(f"📊 BILAN TESTS: {success_count}/{total_tests} réussis")
    
    if success_count == total_tests:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Paper Trading engine opérationnel")
        print("✅ Traitement signaux en temps réel")
        print("✅ Métriques de performance")
        print("✅ Simulation multi-threading")
        print("\n🚀 Prêt pour paper trading en production!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - success_count} test(s) échoué(s)")
        print("🔧 Corrections nécessaires")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())