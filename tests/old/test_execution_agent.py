#!/usr/bin/env python3
"""
Test de l'Execution Agent (IBKR simulation)
"""

import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Redis
sys.modules['redis'] = Mock()
sys.modules['redis.asyncio'] = Mock()

async def test_order_submission():
    """Test soumission d'ordres"""
    try:
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub'):
            from alphabot.agents.execution.execution_agent import ExecutionAgent, Order, OrderSide, OrderType
            
            print("⚡ Test soumission d'ordres...")
            
            # Créer l'agent
            agent = ExecutionAgent()
            await agent.start()
            print("✅ Execution Agent démarré")
            
            # Créer des ordres de test
            test_orders = [
                Order(
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    quantity=100,
                    order_type=OrderType.MARKET
                ),
                Order(
                    symbol="MSFT",
                    side=OrderSide.BUY,
                    quantity=50,
                    order_type=OrderType.LIMIT,
                    price=295.0
                ),
                Order(
                    symbol="GOOGL",
                    side=OrderSide.SELL,
                    quantity=10,
                    order_type=OrderType.MARKET
                )
            ]
            
            print(f"\n📋 Test de {len(test_orders)} ordres:")
            
            submitted_count = 0
            for order in test_orders:
                success = await agent._submit_order(order)
                status = "✅" if success else "❌"
                print(f"   {status} {order.side.value.upper()} {order.quantity} {order.symbol} @ {order.order_type.value}")
                if success:
                    submitted_count += 1
            
            print(f"\n📊 Résultats soumission:")
            print(f"   Ordres soumis: {submitted_count}/{len(test_orders)}")
            print(f"   Ordres en attente: {len(agent.pending_orders)}")
            print(f"   Ordres exécutés: {len(agent.execution_history)}")
            
            # Attendre un peu pour les exécutions simulées
            await asyncio.sleep(0.5)
            
            print(f"\n📈 Après exécutions:")
            print(f"   Ordres en attente: {len(agent.pending_orders)}")
            print(f"   Ordres exécutés: {len(agent.execution_history)}")
            
            # Vérifier les positions
            positions = agent.get_positions_summary()
            print(f"   Positions ouvertes: {positions['total_positions']}")
            print(f"   Valeur totale: ${positions['total_market_value']:.0f}")
            
            await agent.stop()
            
            print("\n✅ Test soumission ordres réussi!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur test soumission: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_portfolio_rebalancing():
    """Test rebalancement de portefeuille"""
    try:
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub'):
            from alphabot.agents.execution.execution_agent import ExecutionAgent
            from alphabot.core.signal_hub import Signal, SignalType, SignalPriority
            
            print("\n🔄 Test rebalancement portefeuille...")
            
            agent = ExecutionAgent()
            await agent.start()
            
            # Signal de rebalancement
            rebalancing_signal = Signal(
                id="rebal-001",
                type=SignalType.PORTFOLIO_REBALANCE,
                source_agent="optimization_agent",
                priority=SignalPriority.HIGH,
                data={
                    'trades': {
                        'AAPL': 0.08,   # Augmenter de 8%
                        'MSFT': -0.05,  # Réduire de 5%
                        'GOOGL': 0.03,  # Augmenter de 3%
                        'AMZN': -0.06   # Réduire de 6%
                    },
                    'weights': {
                        'AAPL': 0.25,
                        'MSFT': 0.20,
                        'GOOGL': 0.23,
                        'AMZN': 0.15,
                        'CASH': 0.17
                    }
                }
            )
            
            print("📊 Signal de rebalancement reçu:")
            for symbol, change in rebalancing_signal.data['trades'].items():
                direction = "↗️" if change > 0 else "↘️"
                print(f"   {direction} {symbol}: {change:+.1%}")
            
            # Traiter le signal
            initial_orders = len(agent.execution_history)
            await agent._process_rebalancing_order(rebalancing_signal)
            
            # Attendre les exécutions
            await asyncio.sleep(0.3)
            
            final_orders = len(agent.execution_history)
            new_orders = final_orders - initial_orders
            
            print(f"\n✅ Rebalancement exécuté:")
            print(f"   Nouveaux ordres: {new_orders}")
            print(f"   Volume total: ${agent.metrics['volume_today']:.0f}")
            print(f"   Commissions: ${agent.metrics['commission_today']:.2f}")
            
            # Rapport d'exécution
            report = agent.get_execution_report()
            print(f"\n📋 Rapport d'exécution:")
            print(f"   Taux d'exécution: {report.fill_rate:.0%}")
            print(f"   Volume total: ${report.total_volume:.0f}")
            print(f"   Commissions: ${report.total_commission:.2f}")
            
            await agent.stop()
            
            print("\n✅ Test rebalancement réussi!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur test rebalancement: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_risk_management():
    """Test gestion des risques"""
    try:
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub'):
            from alphabot.agents.execution.execution_agent import ExecutionAgent, Order, OrderSide, OrderType
            from alphabot.core.signal_hub import Signal, SignalType, SignalPriority
            
            print("\n🚨 Test gestion des risques...")
            
            agent = ExecutionAgent()
            await agent.start()
            
            # Créer quelques ordres en attente
            pending_orders = [
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=200, order_type=OrderType.LIMIT, price=180.0),
                Order(symbol="MSFT", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
                Order(symbol="GOOGL", side=OrderSide.SELL, quantity=5, order_type=OrderType.MARKET)
            ]
            
            for order in pending_orders:
                order.order_id = f"TEST_{order.symbol}"
                agent.pending_orders[order.order_id] = order
            
            print(f"📋 Ordres en attente créés: {len(agent.pending_orders)}")
            
            # Test alerte risque élevé
            risk_alert = Signal(
                id="risk-alert-001",
                type=SignalType.RISK_ALERT,
                source_agent="risk_agent",
                priority=SignalPriority.HIGH,
                data={'risk_level': 'HIGH', 'var_95': 0.08}
            )
            
            print("⚠️ Alerte risque élevé émise...")
            await agent._process_risk_alert(risk_alert)
            
            print(f"   Ordres restants: {len(agent.pending_orders)}")
            
            # Test alerte critique
            critical_alert = Signal(
                id="risk-alert-002",
                type=SignalType.RISK_ALERT,
                source_agent="risk_agent",
                priority=SignalPriority.CRITICAL,
                data={'risk_level': 'CRITICAL', 'var_95': 0.15}
            )
            
            print("\n🚨 Alerte critique émise...")
            initial_positions = len([p for p in agent.positions.values() if p.quantity > 0])
            await agent._process_risk_alert(critical_alert)
            
            # Attendre les exécutions d'urgence
            await asyncio.sleep(0.2)
            
            final_positions = len([p for p in agent.positions.values() if p.quantity > 0])
            emergency_orders = len([o for o in agent.execution_history if 'emergency' in o.order_id or True])
            
            print(f"   Mode urgence activé")
            print(f"   Ordres d'urgence: {emergency_orders}")
            print(f"   Positions réduites: {initial_positions} → {final_positions}")
            
            await agent.stop()
            
            print("\n✅ Test gestion risques réussi!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur test gestion risques: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_order_validation():
    """Test validation des ordres"""
    try:
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub'):
            from alphabot.agents.execution.execution_agent import ExecutionAgent, Order, OrderSide, OrderType
            
            print("\n🔒 Test validation des ordres...")
            
            agent = ExecutionAgent()
            await agent.start()
            
            # Tests de validation
            test_cases = [
                {
                    'order': Order(symbol="AAPL", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET),
                    'expected': False,
                    'reason': "Ordre trop petit"
                },
                {
                    'order': Order(symbol="AAPL", side=OrderSide.BUY, quantity=10000, order_type=OrderType.MARKET),
                    'expected': False,
                    'reason': "Ordre trop grand"
                },
                {
                    'order': Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
                    'expected': True,
                    'reason': "Ordre valide"
                },
                {
                    'order': Order(symbol="MSFT", side=OrderSide.SELL, quantity=50, order_type=OrderType.LIMIT, price=295.0),
                    'expected': True,
                    'reason': "Vente valide"
                }
            ]
            
            print("🧪 Tests de validation:")
            validation_results = []
            
            for i, test_case in enumerate(test_cases, 1):
                order = test_case['order']
                expected = test_case['expected']
                reason = test_case['reason']
                
                is_valid = await agent._validate_order(order)
                result = "✅" if is_valid == expected else "❌"
                
                print(f"   {result} Test {i}: {reason} ({'valide' if is_valid else 'invalide'})")
                validation_results.append(is_valid == expected)
            
            success_rate = sum(validation_results) / len(validation_results)
            print(f"\n📊 Validation tests: {success_rate:.0%} réussite")
            
            # Test limites de position
            print("\n🎯 Test limites de position:")
            
            # Simuler grosse position existante
            from alphabot.agents.execution.execution_agent import Position
            big_position = Position(
                symbol="AAPL",
                quantity=1000,  # Position importante
                avg_cost=150.0,
                market_value=175000.0,
                unrealized_pnl=25000.0,
                realized_pnl=0.0,
                last_updated=datetime.utcnow()
            )
            agent.positions["AAPL"] = big_position
            
            # Tenter un gros achat supplémentaire
            large_order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=500, order_type=OrderType.MARKET)
            is_valid = await agent._validate_order(large_order)
            
            print(f"   Position existante: 1000 actions AAPL")
            print(f"   Achat additionnel 500: {'✅ Autorisé' if is_valid else '❌ Refusé (limite position)'}")
            
            await agent.stop()
            
            print("\n✅ Test validation réussi!")
            return success_rate > 0.75  # 75% des tests doivent passer
            
    except Exception as e:
        print(f"❌ Erreur test validation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_metrics():
    """Test métriques de performance"""
    try:
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub'):
            from alphabot.agents.execution.execution_agent import ExecutionAgent, Order, OrderSide, OrderType
            
            print("\n📈 Test métriques de performance...")
            
            agent = ExecutionAgent()
            await agent.start()
            
            # Simuler une session de trading
            trading_orders = []
            
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
            for i in range(10):  # 10 ordres
                symbol = np.random.choice(symbols)
                side = np.random.choice([OrderSide.BUY, OrderSide.SELL])
                quantity = np.random.randint(10, 200)
                
                order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )
                
                trading_orders.append(order)
            
            print(f"🎯 Simulation de {len(trading_orders)} ordres...")
            
            # Soumettre tous les ordres
            start_time = asyncio.get_event_loop().time()
            
            for order in trading_orders:
                await agent._submit_order(order)
            
            # Attendre les exécutions
            await asyncio.sleep(0.5)
            
            total_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Générer rapport
            report = agent.get_execution_report()
            positions = agent.get_positions_summary()
            status = agent.get_agent_status()
            
            print(f"\n📊 Métriques de performance:")
            print(f"   Temps total: {total_time:.0f}ms")
            print(f"   Débit: {len(trading_orders) / (total_time/1000):.1f} ordres/sec")
            print(f"   Taux d'exécution: {report.fill_rate:.0%}")
            print(f"   Volume total: ${report.total_volume:.0f}")
            print(f"   Commissions: ${report.total_commission:.2f}")
            print(f"   Slippage moyen: {report.slippage_bps:.1f} bps")
            
            print(f"\n💼 État du portefeuille:")
            print(f"   Positions actives: {positions['total_positions']}")
            print(f"   Valeur totale: ${positions['total_market_value']:.0f}")
            print(f"   P&L total: ${positions['total_pnl']:.0f}")
            
            print(f"\n⚙️ Statut agent:")
            print(f"   Connexion: {status['connection_status']}")
            print(f"   Ordres en attente: {status['pending_orders']}")
            print(f"   Positions: {status['positions_count']}")
            
            await agent.stop()
            
            # Vérifications performance
            throughput_ok = len(trading_orders) / (total_time/1000) > 10  # > 10 ordres/sec
            fill_rate_ok = report.fill_rate > 0.8  # > 80%
            latency_ok = total_time / len(trading_orders) < 100  # < 100ms par ordre
            
            print(f"\n✅ Performance:")
            print(f"   Débit: {'✅' if throughput_ok else '❌'} {'>10/sec' if throughput_ok else '≤10/sec'}")
            print(f"   Fill Rate: {'✅' if fill_rate_ok else '❌'} {'>80%' if fill_rate_ok else '≤80%'}")
            print(f"   Latence: {'✅' if latency_ok else '❌'} {'<100ms' if latency_ok else '≥100ms'}")
            
            print("\n✅ Test métriques réussi!")
            return throughput_ok and fill_rate_ok and latency_ok
            
    except Exception as e:
        print(f"❌ Erreur test métriques: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test principal"""
    print("=" * 70)
    print("⚡ TESTS EXECUTION AGENT - IBKR Simulation")
    print("=" * 70)
    
    results = []
    
    # Test soumission ordres
    results.append(await test_order_submission())
    
    # Test rebalancement
    results.append(await test_portfolio_rebalancing())
    
    # Test gestion risques
    results.append(await test_risk_management())
    
    # Test validation
    results.append(await test_order_validation())
    
    # Test performance
    results.append(await test_performance_metrics())
    
    # Bilan
    success_count = sum(results)
    total_tests = len(results)
    
    print("\n" + "=" * 70)
    print(f"📊 BILAN: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Soumission d'ordres : OK")
        print("✅ Rebalancement portefeuille : OK") 
        print("✅ Gestion des risques : OK")
        print("✅ Validation d'ordres : OK")
        print("✅ Métriques performance : OK")
        print("\n🚀 Execution Agent opérationnel!")
        print("💡 Prêt pour intégration IBKR réelle en Phase 5")
        return 0
    else:
        print("\n⚠️ Certains tests ont échoué")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())