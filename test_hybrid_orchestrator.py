#!/usr/bin/env python3
"""
Test du Hybrid Orchestrator - AlphaBot
Script de test pour valider l'intégration complète du système hybride
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphabot.core.hybrid_orchestrator import (
    HybridOrchestrator, 
    HybridWorkflowType,
    get_hybrid_orchestrator
)
from alphabot.ml import get_ml_info

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hybrid_orchestrator_test.log')
    ]
)

logger = logging.getLogger(__name__)


async def test_ml_components():
    """Tester les composants ML disponibles"""
    logger.info("🧠 Test des composants ML...")
    
    try:
        ml_info = get_ml_info()
        logger.info(f"✅ ML Info: {ml_info}")
        
        # Tester l'import des composants
        from alphabot.ml.pattern_detector import MLPatternDetector
        from alphabot.ml.sentiment_analyzer import SentimentDLAnalyzer
        from alphabot.ml.rag_integrator import RAGIntegrator
        
        logger.info("✅ Tous les composants ML importés avec succès")
        
        # Initialiser chaque composant
        pattern_detector = MLPatternDetector()
        sentiment_analyzer = SentimentDLAnalyzer()
        rag_integrator = RAGIntegrator()
        
        logger.info("✅ Tous les composants ML initialisés avec succès")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Échec du test des composants ML: {e}")
        return False


async def test_hybrid_orchestrator_basic():
    """Tester les fonctionnalités de base de l'orchestrateur hybride"""
    logger.info("🚀 Test de base de l'orchestrateur hybride...")
    
    try:
        # Initialiser l'orchestrateur
        orchestrator = get_hybrid_orchestrator(enable_ml=True)
        logger.info("✅ Orchestrateur hybride initialisé")
        
        # Vérifier les métriques de performance
        metrics = orchestrator.get_performance_metrics()
        logger.info(f"📊 Métriques initiales: {metrics}")
        
        # Vérifier les composants disponibles
        ml_components = metrics['ml_components_available']
        logger.info(f"🔧 Composants ML disponibles: {ml_components}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Échec du test de base: {e}")
        return False


async def test_core_analysis():
    """Tester l'analyse Core System (sans ML)"""
    logger.info("🎯 Test de l'analyse Core System...")
    
    try:
        orchestrator = get_hybrid_orchestrator(enable_ml=False)  # Désactiver ML pour ce test
        
        # Symboles de test
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Exécuter l'analyse Core
        decisions = await orchestrator.analyze_portfolio_hybrid(
            test_symbols, 
            HybridWorkflowType.CORE_ANALYSIS
        )
        
        logger.info(f"✅ Analyse Core terminée pour {len(decisions)} symboles")
        
        # Afficher les résultats
        for symbol, decision in decisions.items():
            logger.info(f"  {symbol}: {decision.action} (confiance: {decision.confidence:.2f})")
        
        return len(decisions) > 0
        
    except Exception as e:
        logger.error(f"❌ Échec du test Core Analysis: {e}")
        return False


async def test_ml_enhanced_analysis():
    """Tester l'analyse ML Enhanced"""
    logger.info("🧠 Test de l'analyse ML Enhanced...")
    
    try:
        orchestrator = get_hybrid_orchestrator(enable_ml=True)
        
        # Symboles de test
        test_symbols = ['AAPL', 'MSFT']
        
        # Exécuter l'analyse ML Enhanced
        decisions = await orchestrator.analyze_portfolio_hybrid(
            test_symbols, 
            HybridWorkflowType.ML_ENHANCED
        )
        
        logger.info(f"✅ Analyse ML Enhanced terminée pour {len(decisions)} symboles")
        
        # Afficher les résultats détaillés
        for symbol, decision in decisions.items():
            logger.info(f"  {symbol}:")
            logger.info(f"    Action: {decision.action}")
            logger.info(f"    Confiance: {decision.confidence:.2f}")
            logger.info(f"    Composants ML utilisés: {decision.ml_components_used}")
            if decision.ml_pattern_score:
                logger.info(f"    Score Pattern: {decision.ml_pattern_score:.2f}")
            if decision.ml_sentiment_score:
                logger.info(f"    Score Sentiment: {decision.ml_sentiment_score:.2f}")
            if decision.rag_confidence:
                logger.info(f"    Confiance RAG: {decision.rag_confidence:.2f}")
        
        return len(decisions) > 0
        
    except Exception as e:
        logger.error(f"❌ Échec du test ML Enhanced Analysis: {e}")
        return False


async def test_performance_metrics():
    """Tester le suivi des métriques de performance"""
    logger.info("📊 Test des métriques de performance...")
    
    try:
        orchestrator = get_hybrid_orchestrator(enable_ml=True)
        
        # Exécuter quelques analyses pour générer des métriques
        test_symbols = ['AAPL', 'MSFT']
        
        # Analyse Core
        await orchestrator.analyze_portfolio_hybrid(
            test_symbols, 
            HybridWorkflowType.CORE_ANALYSIS
        )
        
        # Analyse ML Enhanced
        await orchestrator.analyze_portfolio_hybrid(
            test_symbols, 
            HybridWorkflowType.ML_ENHANCED
        )
        
        # Récupérer les métriques
        metrics = orchestrator.get_performance_metrics()
        
        logger.info("📈 Métriques de performance:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Vérifier que les métriques ont été mises à jour
        assert metrics['core_decisions'] > 0, "Les décisions Core devraient être comptabilisées"
        assert metrics['ml_enabled'] == True, "Le ML devrait être activé"
        
        logger.info("✅ Métriques de fonctionnement correctement")
        return True
        
    except Exception as e:
        logger.error(f"❌ Échec du test des métriques: {e}")
        return False


async def test_weekly_rebalance():
    """Tester le rebalancement hebdomadaire"""
    logger.info("🔄 Test du rebalancement hebdomadaire...")
    
    try:
        orchestrator = get_hybrid_orchestrator(enable_ml=True)
        
        # Symboles de test
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Exécuter le rebalancement
        result = await orchestrator.execute_weekly_rebalance(test_symbols)
        
        logger.info(f"✅ Rebalancement terminé: {result['status']}")
        logger.info(f"   Décisions: {result.get('decisions_count', 0)}")
        logger.info(f"   Trades exécutés: {result.get('trades_executed', 0)}")
        logger.info(f"   Décisions ML: {result.get('ml_enhanced', 0)}")
        
        return result['status'] == 'completed'
        
    except Exception as e:
        logger.error(f"❌ Échec du test de rebalancement: {e}")
        return False


async def run_all_tests():
    """Exécuter tous les tests"""
    logger.info("🚀 Démarrage des tests complets du Hybrid Orchestrator")
    logger.info("=" * 60)
    
    tests = [
        ("Composants ML", test_ml_components),
        ("Orchestrateur Base", test_hybrid_orchestrator_basic),
        ("Analyse Core", test_core_analysis),
        ("Analyse ML Enhanced", test_ml_enhanced_analysis),
        ("Métriques Performance", test_performance_metrics),
        ("Rebalancement Hebdomadaire", test_weekly_rebalance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Test: {test_name}")
        logger.info("-" * 40)
        
        try:
            success = await test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name}: SUCCÈS")
            else:
                logger.error(f"❌ {test_name}: ÉCHEC")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERREUR - {e}")
            results[test_name] = False
    
    # Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        logger.info(f"{test_name:<30} {status}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS ONT RÉUSSI!")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests ont échoué")
        return False


if __name__ == "__main__":
    # Exécuter les tests
    success = asyncio.run(run_all_tests())
    
    # Code de sortie
    sys.exit(0 if success else 1)
