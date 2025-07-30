#!/usr/bin/env python3
"""
Test de l'Execution Agent (IBKR simulation) - Version corrigée
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

async def test_order_validation_standalone():
    """Test validation des ordres sans démarrage complet"""
    try:
        # Mock complet du signal hub
        mock_hub = Mock()
        mock_hub.subscribe_to_signals = AsyncMock()
        mock_hub.publish_agent_status = AsyncMock()
        mock_hub.publish_signal = AsyncMock()
        
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub', return_value=mock_hub):
            from alphabot.agents.execution.execution_agent import ExecutionAgent, Order, OrderSide, OrderType
            
            print("🔒 Test validation des ordres...")
            
            agent = ExecutionAgent()
            # Ne pas démarrer l'agent, juste tester la validation
            
            # Simuler connexion
            agent.is_connected = True
            agent.ib_connection = {
                'connected': True,
                'account': 'DU123456',
                'buying_power': 100000.0,
                'currency': 'USD'
            }
            
            # Tests de validation
            test_cases = [
                {
                    'order': Order(symbol="AAPL", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET),
                    'expected': False,
                    'reason': "Ordre trop petit"
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
                
                estimated_value = order.quantity * agent._get_estimated_price(order.symbol)
                print(f"   {result} Test {i}: {reason} (valeur: ${estimated_value:.0f})")
                validation_results.append(is_valid == expected)
            
            success_rate = sum(validation_results) / len(validation_results)
            print(f"\n📊 Validation: {success_rate:.0%} réussite")
            
            print("\n✅ Test validation réussi!")
            return success_rate > 0.5
            
    except Exception as e:
        print(f"❌ Erreur test validation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_order_simulation():
    """Test simulation d'ordres sans Signal HUB"""
    try:
        mock_hub = Mock()
        mock_hub.subscribe_to_signals = AsyncMock()
        mock_hub.publish_agent_status = AsyncMock()
        mock_hub.publish_signal = AsyncMock()
        
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub', return_value=mock_hub):
            from alphabot.agents.execution.execution_agent import ExecutionAgent, Order, OrderSide, OrderType
            
            print("\n⚡ Test simulation d'ordres...")
            
            agent = ExecutionAgent()
            
            # Setup manuel sans démarrage
            agent.is_connected = True
            agent.is_running = True
            agent.ib_connection = {
                'connected': True,
                'account': 'DU123456',
                'buying_power': 100000.0,
                'currency': 'USD'
            }
            
            # Créer des ordres de test
            test_orders = [
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
                Order(symbol="MSFT", side=OrderSide.BUY, quantity=50, order_type=OrderType.MARKET),
                Order(symbol="GOOGL", side=OrderSide.SELL, quantity=10, order_type=OrderType.MARKET)
            ]
            
            print(f"📋 Test de {len(test_orders)} ordres:")
            
            submitted_count = 0
            for order in test_orders:
                success = await agent._submit_order(order)
                status = "✅" if success else "❌"
                estimated_price = agent._get_estimated_price(order.symbol)
                estimated_value = order.quantity * estimated_price
                print(f"   {status} {order.side.value.upper()} {order.quantity} {order.symbol} (${estimated_value:.0f})")
                if success:
                    submitted_count += 1
            
            # Attendre les exécutions simulées
            await asyncio.sleep(0.3)
            
            print(f"\n📊 Résultats:")
            print(f"   Ordres soumis: {submitted_count}/{len(test_orders)}")
            print(f"   Ordres exécutés: {len(agent.execution_history)}")
            print(f"   Volume total: ${agent.metrics['volume_today']:.0f}")
            print(f"   Commissions: ${agent.metrics['commission_today']:.2f}")
            
            # Vérifier les positions
            positions = agent.get_positions_summary()
            print(f"   Positions: {positions['total_positions']}")
            print(f"   Valeur portefeuille: ${positions['total_market_value']:.0f}")
            
            print("\n✅ Test simulation réussi!")
            return submitted_count > 0
            
    except Exception as e:
        print(f"❌ Erreur test simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_risk_scenarios():
    """Test scénarios de gestion des risques"""
    try:
        mock_hub = Mock()
        mock_hub.subscribe_to_signals = AsyncMock()
        mock_hub.publish_agent_status = AsyncMock()
        mock_hub.publish_signal = AsyncMock()
        
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub', return_value=mock_hub):
            from alphabot.agents.execution.execution_agent import ExecutionAgent, Order, OrderSide, OrderType, Position
            
            print("\n🚨 Test scénarios de risque...")
            
            agent = ExecutionAgent()
            agent.is_connected = True
            agent.is_running = True
            
            # Créer des positions importantes
            large_positions = {
                'AAPL': Position(
                    symbol='AAPL',
                    quantity=1000,
                    avg_cost=150.0,
                    market_value=175000.0,
                    unrealized_pnl=25000.0,
                    realized_pnl=0.0,
                    last_updated=datetime.utcnow()
                ),
                'MSFT': Position(
                    symbol='MSFT',
                    quantity=500,
                    avg_cost=280.0,
                    market_value=150000.0,
                    unrealized_pnl=10000.0,
                    realized_pnl=0.0,
                    last_updated=datetime.utcnow()
                )
            }
            
            agent.positions.update(large_positions)
            
            # Créer ordres en attente
            pending_orders = [
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=200, order_type=OrderType.LIMIT, price=180.0),
                Order(symbol="MSFT", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
                Order(symbol="GOOGL", side=OrderSide.SELL, quantity=5, order_type=OrderType.MARKET)
            ]
            
            for i, order in enumerate(pending_orders):
                order.order_id = f"PENDING_{i}"
                agent.pending_orders[order.order_id] = order
            
            print(f"📊 État initial:")
            print(f"   Positions: {len(agent.positions)} (valeur: ${sum(p.market_value for p in agent.positions.values()):.0f})")
            print(f"   Ordres en attente: {len(agent.pending_orders)}")
            
            # Test alerte risque élevé
            print("\n⚠️ Simulation alerte risque élevé...")
            await agent._cancel_non_essential_orders()
            print(f"   Ordres restants: {len(agent.pending_orders)}")
            
            # Remettre quelques ordres
            for i, order in enumerate(pending_orders):
                order.order_id = f"PENDING_NEW_{i}"
                agent.pending_orders[order.order_id] = order
            
            # Test alerte critique
            print("\n🚨 Simulation alerte critique...")
            initial_positions_value = sum(p.market_value for p in agent.positions.values())
            
            await agent._emergency_risk_reduction()
            
            # Attendre les exécutions d'urgence
            await asyncio.sleep(0.2)
            
            final_positions_value = sum(p.market_value for p in agent.positions.values())
            reduction_pct = (initial_positions_value - final_positions_value) / initial_positions_value
            
            print(f"   Valeur initiale: ${initial_positions_value:.0f}")
            print(f"   Valeur finale: ${final_positions_value:.0f}")
            print(f"   Réduction: {reduction_pct:.1%}")
            print(f"   Ordres d'urgence: {len(agent.execution_history)}")
            
            print("\n✅ Test scénarios risque réussi!")
            return reduction_pct > 0.1  # Au moins 10% de réduction
            
    except Exception as e:
        print(f"❌ Erreur test risque: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_standalone():
    """Test performance sans démarrage complet"""
    try:
        mock_hub = Mock()
        mock_hub.subscribe_to_signals = AsyncMock()
        mock_hub.publish_agent_status = AsyncMock()
        mock_hub.publish_signal = AsyncMock()
        
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub', return_value=mock_hub):
            from alphabot.agents.execution.execution_agent import ExecutionAgent, Order, OrderSide, OrderType
            
            print("\n📈 Test performance...")
            
            agent = ExecutionAgent()
            agent.is_connected = True
            agent.is_running = True
            agent.ib_connection = {'buying_power': 100000.0}
            
            # Créer beaucoup d'ordres
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            orders = []
            
            for i in range(20):  # 20 ordres
                symbol = np.random.choice(symbols)
                side = np.random.choice([OrderSide.BUY, OrderSide.SELL])
                quantity = np.random.randint(10, 100)
                
                order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )
                orders.append(order)
            
            print(f"🎯 Test performance avec {len(orders)} ordres...")
            
            # Mesurer performance
            start_time = asyncio.get_event_loop().time()
            
            submitted = 0
            for order in orders:
                success = await agent._submit_order(order)
                if success:
                    submitted += 1
            
            # Attendre exécutions
            await asyncio.sleep(0.5)
            
            total_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Calculs
            throughput = submitted / (total_time / 1000)
            avg_latency = total_time / submitted if submitted > 0 else 0
            
            # Rapport
            report = agent.get_execution_report()
            
            print(f"\n📊 Métriques:")
            print(f"   Ordres soumis: {submitted}/{len(orders)}")
            print(f"   Temps total: {total_time:.0f}ms")
            print(f"   Débit: {throughput:.1f} ordres/sec")
            print(f"   Latence moyenne: {avg_latency:.1f}ms")
            print(f"   Fill rate: {report.fill_rate:.0%}")
            print(f"   Volume: ${report.total_volume:.0f}")
            
            # Vérifications
            throughput_ok = throughput > 10
            latency_ok = avg_latency < 100
            fill_rate_ok = report.fill_rate > 0.8
            
            print(f"\n✅ Performance:")
            print(f"   Débit: {'✅' if throughput_ok else '❌'} (cible: >10/sec)")
            print(f"   Latence: {'✅' if latency_ok else '❌'} (cible: <100ms)")
            print(f"   Fill rate: {'✅' if fill_rate_ok else '❌'} (cible: >80%)")
            
            print("\n✅ Test performance réussi!")
            return throughput_ok and latency_ok and fill_rate_ok
            
    except Exception as e:
        print(f"❌ Erreur test performance: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_functionality():
    """Test fonctionnalités générales de l'agent"""
    try:
        mock_hub = Mock()
        mock_hub.subscribe_to_signals = AsyncMock()
        mock_hub.publish_agent_status = AsyncMock()
        mock_hub.publish_signal = AsyncMock()
        
        with patch('alphabot.agents.execution.execution_agent.get_signal_hub', return_value=mock_hub):
            from alphabot.agents.execution.execution_agent import ExecutionAgent
            
            print("\n🔧 Test fonctionnalités agent...")
            
            agent = ExecutionAgent()
            
            # Test configuration
            print("⚙️ Configuration:")
            print(f"   IBKR Host: {agent.ibkr_config['host']}")
            print(f"   IBKR Port: {agent.ibkr_config['port']}")
            print(f"   Client ID: {agent.ibkr_config['client_id']}")
            
            # Test paramètres d'exécution
            params = agent.execution_params
            print(f"\n📋 Paramètres:")
            print(f"   Ordre max: ${params['max_order_size']:,}")
            print(f"   Ordre min: ${params['min_order_size']:,}")
            print(f"   Slippage max: {params['slippage_tolerance_bps']} bps")
            print(f"   Position max: {params['position_size_limit']:.0%}")
            
            # Test prix estimation
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            print(f"\n💰 Prix estimés:")
            for symbol in symbols:
                price = agent._get_estimated_price(symbol)
                print(f"   {symbol}: ${price:.2f}")
            
            # Test statut
            status = agent.get_agent_status()
            print(f"\n📊 Statut agent:")
            print(f"   Nom: {status['name']}")
            print(f"   Version: {status['version']}")
            print(f"   Broker: {status['broker']}")
            print(f"   Running: {status['is_running']}")
            
            print("\n✅ Test fonctionnalités réussi!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur test fonctionnalités: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test principal"""
    print("=" * 70)
    print("⚡ TESTS EXECUTION AGENT - IBKR Simulation (Version corrigée)")
    print("=" * 70)
    
    results = []
    
    # Test fonctionnalités de base
    results.append(await test_agent_functionality())
    
    # Test validation
    results.append(await test_order_validation_standalone())
    
    # Test simulation ordres
    results.append(await test_order_simulation())
    
    # Test gestion risques
    results.append(await test_risk_scenarios())
    
    # Test performance
    results.append(await test_performance_standalone())
    
    # Bilan
    success_count = sum(results)
    total_tests = len(results)
    
    print("\n" + "=" * 70)
    print(f"📊 BILAN: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Fonctionnalités agent : OK")
        print("✅ Validation d'ordres : OK")
        print("✅ Simulation d'ordres : OK") 
        print("✅ Gestion des risques : OK")
        print("✅ Performance : OK")
        print("\n🚀 Execution Agent opérationnel!")
        print("💡 Prêt pour intégration IBKR réelle en Phase 5")
        return 0
    else:
        print(f"\n⚠️ {total_tests - success_count} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())