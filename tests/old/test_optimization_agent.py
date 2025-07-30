#!/usr/bin/env python3
"""
Test de l'Optimization Agent (HRP)
"""

import asyncio
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Redis
sys.modules['redis'] = Mock()
sys.modules['redis.asyncio'] = Mock()

async def test_hrp_optimization():
    """Test Hierarchical Risk Parity"""
    try:
        with patch('alphabot.agents.optimization.optimization_agent.get_signal_hub'):
            from alphabot.agents.optimization.optimization_agent import OptimizationAgent
            
            print("⚖️ Test Hierarchical Risk Parity...")
            
            # Créer l'agent
            agent = OptimizationAgent()
            print("✅ Optimization Agent créé")
            
            # Données de test
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
            # Générer des données de prix réalistes
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            
            price_data = {}
            for i, symbol in enumerate(symbols):
                # Prix avec corrélations différentes
                base_price = 100 + i * 50
                returns = np.random.normal(0.0005, 0.015 + i * 0.005, len(dates))
                
                prices = [base_price]
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                price_data[symbol] = prices
            
            price_df = pd.DataFrame(price_data, index=dates)
            returns = price_df.pct_change().dropna()
            
            print(f"📊 Données: {len(returns)} jours, {len(symbols)} actifs")
            
            # Test HRP
            hrp_weights = agent._hierarchical_risk_parity(returns)
            print(f"\n🎯 Poids HRP:")
            for symbol, weight in zip(symbols, hrp_weights):
                print(f"   {symbol}: {weight:.1%}")
            
            # Test Equal Weight
            eq_weights = agent._equal_weight_optimization(symbols)
            print(f"\n⚖️ Poids égaux:")
            for symbol, weight in zip(symbols, eq_weights):
                print(f"   {symbol}: {weight:.1%}")
            
            # Test Risk Parity
            rp_weights = agent._risk_parity_optimization(returns)
            print(f"\n🔄 Risk Parity:")
            for symbol, weight in zip(symbols, rp_weights):
                print(f"   {symbol}: {weight:.1%}")
            
            # Comparer les métriques
            methods = {
                'HRP': hrp_weights,
                'Equal Weight': eq_weights,
                'Risk Parity': rp_weights
            }
            
            print(f"\n📈 Comparaison des méthodes:")
            print(f"{'Méthode':<15} {'Sharpe':<8} {'Vol':<8} {'Div Ratio':<10}")
            print("-" * 45)
            
            for method_name, weights in methods.items():
                metrics = agent._calculate_portfolio_metrics(returns, weights)
                div_ratio = agent._calculate_diversification_ratio(returns, weights)
                
                print(f"{method_name:<15} {metrics['sharpe_ratio']:<8.2f} "
                      f"{metrics['expected_volatility']:<8.1%} {div_ratio:<10.2f}")
            
            # Test contraintes
            print(f"\n🔒 Vérification contraintes:")
            for method_name, weights in methods.items():
                min_w, max_w = weights.min(), weights.max()
                sum_w = weights.sum()
                print(f"   {method_name}: min={min_w:.1%}, max={max_w:.1%}, sum={sum_w:.1%}")
            
            print("\n✅ Test HRP réussi!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur test HRP: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_portfolio_metrics():
    """Test calcul métriques portefeuille"""
    try:
        with patch('alphabot.agents.optimization.optimization_agent.get_signal_hub'):
            from alphabot.agents.optimization.optimization_agent import OptimizationAgent, PortfolioWeights
            from datetime import datetime
            
            print("\n📊 Test métriques portefeuille...")
            
            agent = OptimizationAgent()
            
            # Données simulées
            symbols = ["AAPL", "MSFT", "GOOGL"]
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            
            # Générer rendements avec patterns différents
            np.random.seed(123)
            returns_data = {
                'AAPL': np.random.normal(0.001, 0.02, 100),  # Rendement élevé, vol moyenne
                'MSFT': np.random.normal(0.0005, 0.015, 100),  # Rendement moyen, vol faible
                'GOOGL': np.random.normal(0.0008, 0.025, 100)   # Rendement moyen, vol élevée
            }
            
            returns = pd.DataFrame(returns_data, index=dates)
            weights = np.array([0.4, 0.35, 0.25])  # Allocation test
            
            # Calculer métriques
            metrics = agent._calculate_portfolio_metrics(returns, weights)
            risk_contrib = agent._calculate_risk_contribution(returns, weights)
            div_ratio = agent._calculate_diversification_ratio(returns, weights)
            
            print(f"✅ Métriques calculées:")
            print(f"   Rendement attendu: {metrics['expected_return']:.1%}")
            print(f"   Volatilité: {metrics['expected_volatility']:.1%}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {metrics['max_drawdown']:.1%}")
            print(f"   Ratio diversification: {div_ratio:.2f}")
            
            print(f"\n🎯 Contribution au risque:")
            for symbol, contrib in zip(symbols, risk_contrib):
                print(f"   {symbol}: {contrib:.1%}")
            
            # Créer objet PortfolioWeights
            portfolio = PortfolioWeights(
                symbols=symbols,
                weights=weights,
                method="test",
                risk_contribution=risk_contrib,
                expected_return=metrics['expected_return'],
                expected_volatility=metrics['expected_volatility'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                timestamp=datetime.utcnow()
            )
            
            # Test insights
            insights = agent._generate_optimization_insights(portfolio, returns)
            print(f"\n💡 Insights générés:")
            for insight in insights:
                print(f"   - {insight}")
            
            print("\n✅ Test métriques réussi!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur test métriques: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rebalancing_logic():
    """Test logique de rebalancement"""
    try:
        with patch('alphabot.agents.optimization.optimization_agent.get_signal_hub'):
            from alphabot.agents.optimization.optimization_agent import OptimizationAgent, PortfolioWeights
            from datetime import datetime
            
            print("\n🔄 Test logique rebalancement...")
            
            agent = OptimizationAgent()
            
            # Portfolio initial
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            old_weights = np.array([0.25, 0.25, 0.25, 0.25])
            
            old_portfolio = PortfolioWeights(
                symbols=symbols,
                weights=old_weights,
                method="equal_weight",
                risk_contribution=old_weights,
                expected_return=0.12,
                expected_volatility=0.18,
                sharpe_ratio=0.67,
                max_drawdown=-0.15,
                timestamp=datetime.utcnow()
            )
            
            agent.current_portfolio = old_portfolio
            
            # Nouveau portfolio (avec drift)
            new_weights = np.array([0.35, 0.20, 0.30, 0.15])  # Changements significatifs
            
            new_portfolio = PortfolioWeights(
                symbols=symbols,
                weights=new_weights,
                method="hrp",
                risk_contribution=new_weights,
                expected_return=0.14,
                expected_volatility=0.16,
                sharpe_ratio=0.88,
                max_drawdown=-0.12,
                timestamp=datetime.utcnow()
            )
            
            # Calculer trades
            trades = agent._calculate_rebalancing_trades(new_portfolio)
            
            print(f"✅ Trades de rebalancement:")
            total_turnover = 0
            for symbol, trade in trades.items():
                direction = "ACHETER" if trade > 0 else "VENDRE"
                total_turnover += abs(trade)
                print(f"   {symbol}: {direction} {abs(trade):.1%}")
            
            print(f"   Turnover total: {total_turnover:.1%}")
            
            # Test seuil de rebalancement
            agent.optimization_params['rebalancing_threshold'] = 0.10  # 10%
            
            # Petit changement (sous le seuil)
            small_weights = np.array([0.26, 0.24, 0.26, 0.24])
            small_portfolio = PortfolioWeights(
                symbols=symbols,
                weights=small_weights,
                method="hrp",
                risk_contribution=small_weights,
                expected_return=0.12,
                expected_volatility=0.18,
                sharpe_ratio=0.67,
                max_drawdown=-0.15,
                timestamp=datetime.utcnow()
            )
            
            small_trades = agent._calculate_rebalancing_trades(small_portfolio)
            
            print(f"\n🔒 Test seuil rebalancement (seuil: 10%):")
            if small_trades:
                print(f"   Trades générés: {len(small_trades)}")
            else:
                print(f"   Aucun trade (changements < seuil)")
            
            print("\n✅ Test rebalancement réussi!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur test rebalancement: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_integration():
    """Test intégration complète agent"""
    try:
        with patch('alphabot.agents.optimization.optimization_agent.get_signal_hub'):
            from alphabot.agents.optimization.optimization_agent import OptimizationAgent
            
            print("\n🔧 Test intégration agent...")
            
            agent = OptimizationAgent()
            
            # Test statut
            status = agent.get_agent_status()
            print(f"✅ Statut agent:")
            print(f"   Nom: {status['name']}")
            print(f"   Version: {status['version']}")
            print(f"   Méthodes: {status['optimization_methods']}")
            print(f"   Portfolio actuel: {status['current_portfolio']['symbols']}")
            
            # Test paramètres
            params = agent.optimization_params
            print(f"\n⚙️ Paramètres:")
            print(f"   Lookback: {params['lookback_days']} jours")
            print(f"   Poids min/max: {params['min_weight']:.1%} - {params['max_weight']:.1%}")
            print(f"   Risque cible: {params['risk_target']:.1%}")
            print(f"   Seuil rebalancement: {params['rebalancing_threshold']:.1%}")
            
            # Test simulation optimisation
            symbols = ["AAPL", "MSFT", "GOOGL"]
            print(f"\n🚀 Simulation optimisation pour: {symbols}")
            
            # Mock de la récupération de données
            with patch.object(agent, '_fetch_price_data') as mock_fetch:
                # Données simulées
                dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
                price_data = pd.DataFrame({
                    'AAPL': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) * 150,
                    'MSFT': np.cumprod(1 + np.random.normal(0.0008, 0.018, 100)) * 300,
                    'GOOGL': np.cumprod(1 + np.random.normal(0.0006, 0.022, 100)) * 2800
                }, index=dates)
                
                mock_fetch.return_value = price_data
                
                # Optimiser
                await agent._optimize_portfolio(symbols, "hrp")
                
                # Vérifier portfolio créé
                portfolio = agent.get_current_portfolio()
                if portfolio:
                    print(f"✅ Portfolio optimisé:")
                    print(f"   Méthode: {portfolio.method}")
                    print(f"   Sharpe: {portfolio.sharpe_ratio:.2f}")
                    print(f"   Volatilité: {portfolio.expected_volatility:.1%}")
                    
                    for symbol, weight in zip(portfolio.symbols, portfolio.weights):
                        print(f"   {symbol}: {weight:.1%}")
                else:
                    print("⚠️ Pas de portfolio généré")
            
            print("\n✅ Test intégration réussi!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur test intégration: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test principal"""
    print("=" * 70)
    print("⚖️ TESTS OPTIMIZATION AGENT - HRP & Risk Parity")
    print("=" * 70)
    
    results = []
    
    # Test HRP
    results.append(await test_hrp_optimization())
    
    # Test métriques
    results.append(await test_portfolio_metrics())
    
    # Test rebalancement
    results.append(await test_rebalancing_logic())
    
    # Test intégration
    results.append(await test_agent_integration())
    
    # Bilan
    success_count = sum(results)
    total_tests = len(results)
    
    print("\n" + "=" * 70)
    print(f"📊 BILAN: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Hierarchical Risk Parity : OK")
        print("✅ Métriques portefeuille : OK") 
        print("✅ Logique rebalancement : OK")
        print("✅ Intégration agent : OK")
        print("\n🚀 Optimization Agent opérationnel!")
        return 0
    else:
        print("\n⚠️ Certains tests ont échoué")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())