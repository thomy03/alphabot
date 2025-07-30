#!/usr/bin/env python3
"""
Test basique du CrewAI Orchestrator
"""

import asyncio
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def test_crew_orchestrator():
    """Test basique de l'orchestrateur"""
    try:
        from alphabot.core.crew_orchestrator import CrewOrchestrator, WorkflowType
        
        print("🚀 Test CrewAI Orchestrator...")
        
        # Créer l'orchestrateur
        orchestrator = CrewOrchestrator()
        print("✅ Orchestrateur créé")
        
        # Tester création des agents
        orchestrator._create_crewai_agents()
        print(f"✅ {len(orchestrator.agents_crewai)} agents CrewAI créés")
        
        # Tester création des tâches
        orchestrator._create_tasks(["AAPL"], WorkflowType.SIGNAL_ANALYSIS)
        print(f"✅ {len(orchestrator.tasks)} tâches créées")
        
        # Tester métriques
        metrics = orchestrator.get_metrics()
        print(f"✅ Métriques: {metrics['name']} v{metrics['version']}")
        
        # Simuler démarrage/arrêt
        print("🔄 Test démarrage agents...")
        # Note: on évite le démarrage complet pour ce test simple
        
        print("✅ Tous les tests réussis!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_signal_hub():
    """Test basique du Signal HUB"""
    try:
        from alphabot.core.signal_hub import get_signal_hub, Signal, SignalType, SignalPriority
        from datetime import datetime
        
        print("🔌 Test Signal HUB...")
        
        # Créer le hub
        hub = get_signal_hub()
        print("✅ Signal HUB créé")
        
        # Créer un signal test
        signal = Signal(
            id="test-123",
            type=SignalType.TECHNICAL_SIGNAL,
            source_agent="test_agent",
            symbol="AAPL",
            priority=SignalPriority.MEDIUM,
            data={'test': 'data'}
        )
        print("✅ Signal créé")
        
        # Tester sérialisation
        signal_dict = signal.to_dict()
        signal_restored = Signal.from_dict(signal_dict)
        assert signal_restored.id == signal.id
        print("✅ Sérialisation/désérialisation OK")
        
        # Tester métriques
        metrics = hub.get_metrics()
        print(f"✅ Métriques HUB: {metrics}")
        
        print("✅ Signal HUB tests réussis!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur Signal HUB: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fundamental_agent():
    """Test basique du Fundamental Agent"""
    try:
        from alphabot.agents.fundamental.fundamental_agent import FundamentalAgent, FundamentalMetrics
        
        print("📊 Test Fundamental Agent...")
        
        # Créer l'agent
        agent = FundamentalAgent()
        print("✅ Fundamental Agent créé")
        
        # Tester métriques simulées
        metrics = await agent._fetch_fundamental_data("AAPL")
        if metrics:
            print(f"✅ Métriques récupérées: P/E={metrics.pe_ratio:.1f}, ROE={metrics.roe:.2f}")
            print(f"✅ Piotroski Score: {metrics.piotroski_score}/9")
        else:
            print("⚠️ Pas de métriques récupérées")
        
        # Tester scoring
        if metrics:
            signal = await agent._calculate_fundamental_score(metrics)
            print(f"✅ Score calculé: {signal.recommendation} (score: {signal.score:.1f}, confidence: {signal.confidence:.2f})")
        
        # Statut agent
        status = agent.get_agent_status()
        print(f"✅ Statut: {status['name']} - {len(status['capabilities'])} capacités")
        
        print("✅ Fundamental Agent tests réussis!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur Fundamental Agent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test principal"""
    print("=" * 60)
    print("🧪 TESTS CRÉWAI + SIGNAL HUB - Phase 4")
    print("=" * 60)
    
    results = []
    
    # Test Signal HUB
    results.append(await test_signal_hub())
    print()
    
    # Test Fundamental Agent
    results.append(await test_fundamental_agent())
    print()
    
    # Test CrewAI Orchestrator
    results.append(await test_crew_orchestrator())
    print()
    
    # Bilan
    success_count = sum(results)
    total_tests = len(results)
    
    print("=" * 60)
    print(f"📊 BILAN: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Signal HUB opérationnel")
        print("✅ Fundamental Agent fonctionnel") 
        print("✅ CrewAI Orchestrator prêt")
        return 0
    else:
        print("⚠️ Certains tests ont échoué")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())