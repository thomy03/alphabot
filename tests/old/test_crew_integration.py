#!/usr/bin/env python3
"""
Tests d'intégration pour CrewAI Orchestrator
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from alphabot.core.crew_orchestrator import CrewOrchestrator, WorkflowType, TradingDecision
from alphabot.core.signal_hub import Signal, SignalType, SignalPriority


class TestCrewIntegration:
    """Tests d'intégration Crew + Signal HUB"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Fixture orchestrateur"""
        orchestrator = CrewOrchestrator()
        await orchestrator.start()
        yield orchestrator
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_orchestrator_startup(self, orchestrator):
        """Test démarrage orchestrateur"""
        assert orchestrator.is_running
        assert len(orchestrator.agents_crewai) == 5
        assert orchestrator.risk_agent.is_running
        assert orchestrator.technical_agent.is_running
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, orchestrator):
        """Test exécution workflow complet"""
        symbols = ["AAPL", "MSFT"]
        
        # Mock du Crew pour éviter l'exécution réelle
        with patch.object(orchestrator, '_create_tasks'), \
             patch('alphabot.core.crew_orchestrator.Crew') as mock_crew:
            
            mock_crew_instance = Mock()
            mock_crew_instance.kickoff.return_value = "BUY AAPL: score=85, confidence=0.8"
            mock_crew.return_value = mock_crew_instance
            
            decision = await orchestrator.execute_workflow(symbols, WorkflowType.SIGNAL_ANALYSIS)
            
            assert decision is not None
            assert decision.symbol == "AAPL"
            assert decision.action in ["BUY", "SELL", "HOLD"]
            assert 0 <= decision.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_signal_handling(self, orchestrator):
        """Test traitement des signaux"""
        # Signal de rebalancement
        signal = Signal(
            id="test-123",
            type=SignalType.PORTFOLIO_REBALANCE,
            source_agent="test",
            data={'symbols': ['AAPL', 'GOOGL']}
        )
        
        with patch.object(orchestrator, 'execute_workflow') as mock_execute:
            await orchestrator._handle_signal(signal)
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trading_decision_publishing(self, orchestrator):
        """Test publication décision de trading"""
        decision = TradingDecision(
            symbol="AAPL",
            action="BUY",
            confidence=0.85,
            target_weight=0.05,
            reasoning=["Strong fundamentals"],
            risk_score=60.0,
            technical_score=85.0,
            fundamental_score=90.0,
            sentiment_score=75.0,
            timestamp=datetime.utcnow()
        )
        
        with patch.object(orchestrator.signal_hub, 'publish_signal') as mock_publish:
            await orchestrator._publish_trading_decision(decision)
            mock_publish.assert_called_once()
            
            # Vérifier le signal publié
            call_args = mock_publish.call_args[0][0]
            assert call_args.type == SignalType.EXECUTION_ORDER
            assert call_args.symbol == "AAPL"
            assert call_args.data['action'] == "BUY"
    
    @pytest.mark.asyncio 
    async def test_multiple_symbols_analysis(self, orchestrator):
        """Test analyse de plusieurs symboles"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        with patch.object(orchestrator, 'execute_workflow') as mock_execute:
            mock_execute.return_value = TradingDecision(
                symbol="AAPL", action="HOLD", confidence=0.7,
                target_weight=0.05, reasoning=[], risk_score=50,
                technical_score=60, fundamental_score=70, 
                sentiment_score=65, timestamp=datetime.utcnow()
            )
            
            decisions = await orchestrator.analyze_symbols(symbols)
            
            assert len(decisions) == len(symbols)
            assert mock_execute.call_count == len(symbols)
    
    def test_metrics_tracking(self, orchestrator):
        """Test suivi des métriques"""
        metrics = orchestrator.get_metrics()
        
        assert 'workflows_executed' in metrics
        assert 'decisions_made' in metrics
        assert 'avg_decision_time_ms' in metrics
        assert 'is_running' in metrics
        assert metrics['name'] == 'crew_orchestrator'


class TestCrewWorkflowTypes:
    """Tests des différents types de workflows"""
    
    @pytest.fixture
    def orchestrator(self):
        return CrewOrchestrator()
    
    def test_signal_analysis_tasks(self, orchestrator):
        """Test création tâches analyse signaux"""
        orchestrator._create_crewai_agents()
        orchestrator._create_tasks(["AAPL"], WorkflowType.SIGNAL_ANALYSIS)
        
        expected_tasks = [
            'technical_analysis',
            'fundamental_analysis', 
            'sentiment_analysis',
            'risk_assessment',
            'trading_decision'
        ]
        
        for task_name in expected_tasks:
            assert task_name in orchestrator.tasks
            assert orchestrator.tasks[task_name].agent is not None
    
    def test_agents_creation(self, orchestrator):
        """Test création agents CrewAI"""
        orchestrator._create_crewai_agents()
        
        expected_agents = [
            'coordinator',
            'technical_analyst',
            'fundamental_analyst',
            'sentiment_analyst',
            'risk_manager'
        ]
        
        for agent_name in expected_agents:
            assert agent_name in orchestrator.agents_crewai
            agent = orchestrator.agents_crewai[agent_name]
            assert agent.role is not None
            assert agent.goal is not None


@pytest.mark.integration
class TestFullPipeline:
    """Tests d'intégration complète"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test pipeline complet end-to-end"""
        # Démarrer orchestrateur
        orchestrator = CrewOrchestrator()
        await orchestrator.start()
        
        try:
            # Simuler signal d'entrée
            signal = Signal(
                id="e2e-test",
                type=SignalType.PRICE_UPDATE,
                source_agent="market_data",
                symbol="AAPL",
                data={'price': 150.0, 'volume': 1000000}
            )
            
            # Mock du workflow pour éviter l'exécution CrewAI réelle
            with patch.object(orchestrator, 'execute_workflow') as mock_workflow:
                decision = TradingDecision(
                    symbol="AAPL", action="BUY", confidence=0.8,
                    target_weight=0.1, reasoning=["Strong signals"],
                    risk_score=55, technical_score=80, 
                    fundamental_score=85, sentiment_score=75,
                    timestamp=datetime.utcnow()
                )
                mock_workflow.return_value = decision
                
                # Traiter le signal
                await orchestrator._handle_signal(signal)
                
                # Vérifier métriques
                metrics = orchestrator.get_metrics()
                assert metrics['is_running']
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_stress_test_multiple_signals(self):
        """Test de charge avec multiples signaux"""
        orchestrator = CrewOrchestrator()
        await orchestrator.start()
        
        try:
            symbols = [f"STOCK{i}" for i in range(10)]
            
            with patch.object(orchestrator, 'execute_workflow') as mock_workflow:
                mock_workflow.return_value = TradingDecision(
                    symbol="STOCK1", action="HOLD", confidence=0.6,
                    target_weight=0.05, reasoning=[], risk_score=60,
                    technical_score=65, fundamental_score=70,
                    sentiment_score=60, timestamp=datetime.utcnow()
                )
                
                # Analyser tous les symboles
                start_time = asyncio.get_event_loop().time()
                decisions = await orchestrator.analyze_symbols(symbols)
                elapsed = asyncio.get_event_loop().time() - start_time
                
                # Vérifications performance
                assert len(decisions) == len(symbols)
                assert elapsed < 5.0  # Moins de 5 secondes
                assert mock_workflow.call_count == len(symbols)
                
        finally:
            await orchestrator.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])