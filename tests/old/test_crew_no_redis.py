#!/usr/bin/env python3
"""
Test CrewAI Orchestrator SANS Redis
Mode simulation pour test local
"""

import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Redis avant import
sys.modules['redis'] = Mock()
sys.modules['redis.asyncio'] = Mock()

async def test_crew_orchestrator():
    """Test basique de l'orchestrateur sans Redis"""
    try:
        from alphabot.core.crew_orchestrator import CrewOrchestrator, WorkflowType
        
        print("🚀 Test CrewAI Orchestrator (mode simulation)...")
        
        # Créer l'orchestrateur
        orchestrator = CrewOrchestrator()
        print("✅ Orchestrateur créé")
        
        # Tester création des agents
        orchestrator._create_crewai_agents()
        print(f"✅ {len(orchestrator.agents_crewai)} agents CrewAI créés:")
        for name, agent in orchestrator.agents_crewai.items():
            print(f"   - {name}: {agent.role}")
        
        # Tester création des tâches
        orchestrator._create_tasks(["AAPL", "MSFT"], WorkflowType.SIGNAL_ANALYSIS)
        print(f"\n✅ {len(orchestrator.tasks)} tâches créées:")
        for name, task in orchestrator.tasks.items():
            print(f"   - {name}")
        
        # Tester métriques
        metrics = orchestrator.get_metrics()
        print(f"\n✅ Métriques orchestrateur:")
        for key, value in metrics.items():
            print(f"   - {key}: {value}")
        
        print("\n✅ CrewAI Orchestrator OK!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fundamental_agent():
    """Test Fundamental Agent sans Redis"""
    try:
        # Mock Signal Hub
        with patch('alphabot.agents.fundamental.fundamental_agent.get_signal_hub'):
            from alphabot.agents.fundamental.fundamental_agent import FundamentalAgent, FundamentalMetrics
            
            print("\n📊 Test Fundamental Agent (mode simulation)...")
            
            # Créer l'agent
            agent = FundamentalAgent()
            print("✅ Fundamental Agent créé")
            
            # Tester métriques simulées pour 3 symboles
            symbols = ["AAPL", "MSFT", "GOOGL"]
            for symbol in symbols:
                metrics = await agent._fetch_fundamental_data(symbol)
                if metrics:
                    print(f"\n📈 {symbol}:")
                    print(f"   - P/E Ratio: {metrics.pe_ratio:.1f}")
                    print(f"   - ROE: {metrics.roe:.2%}")
                    print(f"   - Piotroski Score: {metrics.piotroski_score}/9")
                    print(f"   - Altman Z-Score: {metrics.altman_z_score}")
                    
                    # Calculer le score
                    signal = await agent._calculate_fundamental_score(metrics)
                    print(f"   - Recommandation: {signal.recommendation}")
                    print(f"   - Score: {signal.score:.1f}/100")
                    print(f"   - Confiance: {signal.confidence:.2%}")
                    print(f"   - Facteurs clés: {', '.join(signal.key_factors)}")
            
            # Statut agent
            status = agent.get_agent_status()
            print(f"\n✅ Statut agent: {status['name']}")
            print(f"   - Capacités: {len(status['capabilities'])}")
            print(f"   - Cache: {status['cache_size']} symboles")
            
            print("\n✅ Fundamental Agent OK!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur Fundamental Agent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_signal_types():
    """Test types de signaux"""
    try:
        from alphabot.core.signal_hub import Signal, SignalType, SignalPriority
        from datetime import datetime
        
        print("\n📡 Test types de signaux...")
        
        # Créer différents types de signaux
        signals = [
            Signal(
                id="tech-001",
                type=SignalType.TECHNICAL_SIGNAL,
                source_agent="technical_agent",
                symbol="AAPL",
                priority=SignalPriority.HIGH,
                data={'ema_cross': 'bullish', 'rsi': 45}
            ),
            Signal(
                id="fund-001", 
                type=SignalType.FUNDAMENTAL_SIGNAL,
                source_agent="fundamental_agent",
                symbol="MSFT",
                priority=SignalPriority.MEDIUM,
                data={'pe_ratio': 25, 'recommendation': 'BUY'}
            ),
            Signal(
                id="risk-001",
                type=SignalType.RISK_ALERT,
                source_agent="risk_agent",
                priority=SignalPriority.CRITICAL,
                data={'var_breach': True, 'exposure': 0.95}
            )
        ]
        
        print(f"✅ {len(signals)} signaux créés:")
        for signal in signals:
            print(f"   - {signal.type.value}: priorité {signal.priority.value}")
        
        # Tester sérialisation
        for signal in signals:
            dict_form = signal.to_dict()
            restored = Signal.from_dict(dict_form)
            assert restored.id == signal.id
            assert restored.type == signal.type
        
        print("✅ Sérialisation OK!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur types signaux: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trading_decision():
    """Test structure décision trading"""
    try:
        from alphabot.core.crew_orchestrator import TradingDecision
        from datetime import datetime
        
        print("\n💰 Test décision de trading...")
        
        # Créer une décision
        decision = TradingDecision(
            symbol="AAPL",
            action="BUY",
            confidence=0.85,
            target_weight=0.05,
            reasoning=[
                "Technical: EMA crossover bullish",
                "Fundamental: P/E below sector average", 
                "Sentiment: Positive news coverage"
            ],
            risk_score=60.0,
            technical_score=85.0,
            fundamental_score=90.0,
            sentiment_score=75.0,
            timestamp=datetime.utcnow()
        )
        
        print(f"✅ Décision créée:")
        print(f"   - Action: {decision.action} {decision.symbol}")
        print(f"   - Confiance: {decision.confidence:.0%}")
        print(f"   - Poids cible: {decision.target_weight:.1%}")
        print(f"   - Score moyen: {(decision.technical_score + decision.fundamental_score + decision.sentiment_score) / 3:.1f}")
        print(f"   - Raisons: {len(decision.reasoning)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur décision trading: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test principal sans Redis"""
    print("=" * 70)
    print("🧪 TESTS CRÉWAI + AGENTS - Phase 4 (Mode Simulation)")
    print("=" * 70)
    print("⚠️  Mode simulation : Redis mocké, pas de connexion requise")
    
    results = []
    
    # Test types de signaux
    results.append(await test_signal_types())
    
    # Test Fundamental Agent
    results.append(await test_fundamental_agent())
    
    # Test Trading Decision
    results.append(await test_trading_decision())
    
    # Test CrewAI Orchestrator
    results.append(await test_crew_orchestrator())
    
    # Bilan
    success_count = sum(results)
    total_tests = len(results)
    
    print("\n" + "=" * 70)
    print(f"📊 BILAN: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Types de signaux : OK")
        print("✅ Fundamental Agent : OK") 
        print("✅ Trading Decision : OK")
        print("✅ CrewAI Orchestrator : OK")
        print("\n🚀 Phase 4 - Partie 1 validée (sans Redis)")
        return 0
    else:
        print("\n⚠️ Certains tests ont échoué")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())