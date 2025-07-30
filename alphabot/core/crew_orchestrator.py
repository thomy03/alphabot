#!/usr/bin/env python3
"""
CrewAI Orchestrator - AlphaBot Multi-Agent Trading System
Orchestrateur principal pour coordonner les agents via CrewAI
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from alphabot.core.signal_hub import (
    Signal, SignalType, SignalPriority, get_signal_hub
)
from alphabot.core.config import get_settings
from alphabot.agents.risk.risk_agent import RiskAgent
from alphabot.agents.technical.technical_agent import TechnicalAgent
from alphabot.agents.sentiment.sentiment_agent import SentimentAgent
from alphabot.agents.fundamental.fundamental_agent import FundamentalAgent

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types de workflows d'analyse"""
    SIGNAL_ANALYSIS = "signal_analysis"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_SCAN = "market_scan"


@dataclass
class TradingDecision:
    """Décision de trading consolidée"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    target_weight: float
    reasoning: List[str]
    risk_score: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    timestamp: datetime


class SignalHubTool(BaseTool):
    """Outil CrewAI pour interaction avec Signal HUB"""
    name: str = "signal_hub"
    description: str = "Publier et récupérer des signaux via le Signal HUB"
    
    def _run(self, action: str, **kwargs) -> str:
        """Exécuter une action sur le Signal HUB"""
        # Récupérer le hub à chaque utilisation
        signal_hub = get_signal_hub()
        
        if action == "publish":
            return "Signal publié via HUB"
        elif action == "get_history":
            return "Historique récupéré du HUB"
        else:
            return f"Action inconnue: {action}"


class DataFetchTool(BaseTool):
    """Outil pour récupérer des données de marché"""
    name: str = "data_fetch"
    description: str = "Récupérer des données de prix, volume et news"
    
    def _run(self, symbol: str, data_type: str = "price") -> str:
        """Récupérer des données"""
        # Simulation - en production, connecter aux APIs
        return f"Données {data_type} pour {symbol}: prix=100.50, volume=1000000"


class CrewOrchestrator:
    """Orchestrateur principal CrewAI"""
    
    def __init__(self):
        self.settings = get_settings()
        self.signal_hub = get_signal_hub()
        self.is_running = False
        
        # Agents individuels
        self.risk_agent = RiskAgent()
        self.technical_agent = TechnicalAgent()
        self.sentiment_agent = SentimentAgent()
        self.fundamental_agent = FundamentalAgent()
        
        # Outils CrewAI
        self.tools = [
            SignalHubTool(),
            DataFetchTool()
        ]
        
        # Crew CrewAI
        self.crew = None
        self.agents_crewai = {}
        self.tasks = {}
        
        # Métriques
        self.metrics = {
            'workflows_executed': 0,
            'decisions_made': 0,
            'avg_decision_time_ms': 0,
            'success_rate': 0.0
        }
    
    def _create_crewai_agents(self):
        """Créer les agents CrewAI"""
        
        # Agent Coordinateur
        self.agents_crewai['coordinator'] = Agent(
            role='Trading Coordinator',
            goal='Coordonner l\'analyse multi-agents pour prendre des décisions de trading optimales',
            backstory="""Tu es un coordinateur expert en trading algorithmique. 
            Tu orchestre l'analyse de 4 agents spécialisés (Risk, Technical, Sentiment, Fundamental) 
            pour prendre des décisions de trading éclairées.""",
            verbose=True,
            allow_delegation=True,
            tools=self.tools,
            max_iter=3
        )
        
        # Agent Analyste Technique
        self.agents_crewai['technical_analyst'] = Agent(
            role='Technical Analyst',
            goal='Analyser les indicateurs techniques et identifier les signaux de trading',
            backstory="""Tu es un analyste technique expert utilisant EMA, RSI, ATR et autres indicateurs.
            Tu identifies les points d'entrée et sortie optimaux basés sur l'analyse technique.""",
            verbose=True,
            tools=self.tools,
            max_iter=2
        )
        
        # Agent Analyste Fondamental
        self.agents_crewai['fundamental_analyst'] = Agent(
            role='Fundamental Analyst',
            goal='Évaluer la valeur intrinsèque des actifs via l\'analyse fondamentale',
            backstory="""Tu es un analyste fondamental expert en ratios financiers, Piotroski F-Score,
            et valorisation d'entreprises. Tu identifies les opportunités sous-évaluées.""",
            verbose=True,
            tools=self.tools,
            max_iter=2
        )
        
        # Agent Analyste Sentiment
        self.agents_crewai['sentiment_analyst'] = Agent(
            role='Sentiment Analyst',
            goal='Analyser le sentiment de marché via NLP et médias sociaux',
            backstory="""Tu es un expert en analyse de sentiment utilisant FinBERT et NLP.
            Tu captures l'humeur du marché à partir des news et réseaux sociaux.""",
            verbose=True,
            tools=self.tools,
            max_iter=2
        )
        
        # Agent Gestionnaire de Risque
        self.agents_crewai['risk_manager'] = Agent(
            role='Risk Manager',
            goal='Évaluer et contrôler les risques de portefeuille',
            backstory="""Tu es un gestionnaire de risque expert en VaR, Expected Shortfall,
            et gestion de portefeuille. Tu assures que tous les trades respectent les limites de risque.""",
            verbose=True,
            tools=self.tools,
            max_iter=2
        )
    
    def _create_tasks(self, symbols: List[str], workflow_type: WorkflowType):
        """Créer les tâches CrewAI"""
        
        symbol_list = ', '.join(symbols)
        
        if workflow_type == WorkflowType.SIGNAL_ANALYSIS:
            
            # Tâche d'analyse technique
            self.tasks['technical_analysis'] = Task(
                description=f"""Analyser les indicateurs techniques pour: {symbol_list}
                
                Pour chaque symbole, calculer:
                - EMA 20/50 et signal de croisement
                - RSI (14) et niveaux de surachat/survente
                - ATR pour la volatilité
                - Support/résistance
                
                Retourner un score technique 0-100 et recommandation BUY/HOLD/SELL pour chaque symbole.""",
                agent=self.agents_crewai['technical_analyst'],
                expected_output="Score technique et recommandation pour chaque symbole"
            )
            
            # Tâche d'analyse fondamentale
            self.tasks['fundamental_analysis'] = Task(
                description=f"""Analyser les fondamentaux pour: {symbol_list}
                
                Pour chaque symbole, évaluer:
                - Ratios de valorisation (P/E, P/B)
                - Ratios de profitabilité (ROE, ROA)
                - Piotroski F-Score
                - Santé financière (debt/equity, current ratio)
                
                Retourner un score fondamental 0-100 et recommandation pour chaque symbole.""",
                agent=self.agents_crewai['fundamental_analyst'],
                expected_output="Score fondamental et recommandation pour chaque symbole"
            )
            
            # Tâche d'analyse sentiment
            self.tasks['sentiment_analysis'] = Task(
                description=f"""Analyser le sentiment de marché pour: {symbol_list}
                
                Pour chaque symbole, analyser:
                - Sentiment des news récentes via FinBERT
                - Volume et fréquence des mentions
                - Évolution du sentiment sur 7 jours
                
                Retourner un score de sentiment 0-100 et impact sur le prix.""",
                agent=self.agents_crewai['sentiment_analyst'],
                expected_output="Score de sentiment et impact estimé pour chaque symbole"
            )
            
            # Tâche d'évaluation des risques
            self.tasks['risk_assessment'] = Task(
                description=f"""Évaluer les risques pour: {symbol_list}
                
                Pour chaque symbole et le portefeuille global:
                - VaR 95% et Expected Shortfall
                - Corrélations avec le portefeuille existant
                - Stress tests (scénarios extrêmes)
                - Limites de position et secteur
                
                Retourner le niveau de risque et contraintes de position.""",
                agent=self.agents_crewai['risk_manager'],
                expected_output="Évaluation des risques et limites de position"
            )
            
            # Tâche de coordination finale
            self.tasks['trading_decision'] = Task(
                description=f"""Synthétiser les analyses pour décision finale sur: {symbol_list}
                
                Basé sur les analyses technique, fondamentale, sentiment et risque:
                - Pondérer chaque signal selon sa fiabilité
                - Appliquer les contraintes de risque
                - Optimiser l'allocation de portefeuille
                - Générer les ordres de trading
                
                Retourner une décision de trading consolidée avec justification.""",
                agent=self.agents_crewai['coordinator'],
                expected_output="Décision de trading finale avec allocation et justification",
                context=[
                    self.tasks['technical_analysis'],
                    self.tasks['fundamental_analysis'], 
                    self.tasks['sentiment_analysis'],
                    self.tasks['risk_assessment']
                ]
            )
    
    async def start(self):
        """Démarrer l'orchestrateur"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🎭 Crew Orchestrator démarré")
        
        # Créer les agents CrewAI
        self._create_crewai_agents()
        
        # Démarrer les agents individuels
        await self.risk_agent.start()
        await self.technical_agent.start()
        await self.sentiment_agent.start()
        await self.fundamental_agent.start()
        
        # S'abonner aux signaux système
        await self.signal_hub.subscribe_to_signals(
            agent_name="crew_orchestrator",
            callback=self._handle_signal,
            signal_types=[SignalType.SYSTEM_STATUS, SignalType.PORTFOLIO_REBALANCE]
        )
        
        # Publier le statut
        await self.signal_hub.publish_agent_status(
            "crew_orchestrator",
            "started",
            {
                "version": "1.0",
                "agents_count": len(self.agents_crewai),
                "workflows": [w.value for w in WorkflowType]
            }
        )
    
    async def stop(self):
        """Arrêter l'orchestrateur"""
        self.is_running = False
        
        # Arrêter les agents individuels
        await self.risk_agent.stop()
        await self.technical_agent.stop()
        await self.sentiment_agent.stop()
        await self.fundamental_agent.stop()
        
        await self.signal_hub.publish_agent_status("crew_orchestrator", "stopped")
        logger.info("🎭 Crew Orchestrator arrêté")
    
    async def _handle_signal(self, signal: Signal):
        """Traiter un signal reçu"""
        try:
            if signal.type == SignalType.PORTFOLIO_REBALANCE:
                symbols = signal.data.get('symbols', [])
                if symbols:
                    await self.execute_workflow(symbols, WorkflowType.SIGNAL_ANALYSIS)
                    
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal orchestrateur: {e}")
    
    async def execute_workflow(self, 
                             symbols: List[str], 
                             workflow_type: WorkflowType) -> Optional[TradingDecision]:
        """Exécuter un workflow d'analyse"""
        
        if not self.is_running:
            logger.error("Orchestrateur non démarré")
            return None
        
        try:
            start_time = time.time()
            logger.info(f"🚀 Exécution workflow {workflow_type.value} pour {len(symbols)} symboles")
            
            # Créer les tâches pour ce workflow
            self._create_tasks(symbols, workflow_type)
            
            # Créer et exécuter le Crew
            crew = Crew(
                agents=list(self.agents_crewai.values()),
                tasks=list(self.tasks.values()),
                process=Process.sequential,
                verbose=True
            )
            
            # Exécuter le workflow (attention: bloquant)
            result = crew.kickoff()
            
            # Traiter le résultat
            decision = await self._process_crew_result(result, symbols)
            
            # Métriques
            execution_time = (time.time() - start_time) * 1000
            self.metrics['workflows_executed'] += 1
            self.metrics['avg_decision_time_ms'] = (
                (self.metrics['avg_decision_time_ms'] * (self.metrics['workflows_executed'] - 1) + execution_time)
                / self.metrics['workflows_executed']
            )
            
            logger.info(f"✅ Workflow terminé en {execution_time:.1f}ms")
            
            # Publier la décision
            if decision:
                await self._publish_trading_decision(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Erreur exécution workflow: {e}")
            return None
    
    async def _process_crew_result(self, result: str, symbols: List[str]) -> Optional[TradingDecision]:
        """Traiter le résultat du Crew pour extraire la décision"""
        try:
            # Parser le résultat textuel du Crew
            # En production, utiliser un format structuré (JSON)
            
            # Pour la démo, créer une décision simulée
            primary_symbol = symbols[0] if symbols else "AAPL"
            
            decision = TradingDecision(
                symbol=primary_symbol,
                action="HOLD",  # Par défaut
                confidence=0.75,
                target_weight=0.05,
                reasoning=[
                    "Analyse technique neutre",
                    "Fondamentaux solides", 
                    "Sentiment positif"
                ],
                risk_score=65.0,
                technical_score=70.0,
                fundamental_score=80.0,
                sentiment_score=75.0,
                timestamp=datetime.utcnow()
            )
            
            # Logique de décision simplifiée
            avg_score = (decision.technical_score + decision.fundamental_score + decision.sentiment_score) / 3
            
            if avg_score >= 80 and decision.risk_score <= 70:
                decision.action = "BUY"
                decision.confidence = min(0.95, avg_score / 100 * 1.1)
            elif avg_score <= 40 or decision.risk_score >= 85:
                decision.action = "SELL"
                decision.confidence = min(0.90, (100 - avg_score) / 100 * 1.1)
            else:
                decision.action = "HOLD"
                decision.confidence = 0.6 + abs(avg_score - 60) / 40 * 0.2
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement résultat Crew: {e}")
            return None
    
    async def _publish_trading_decision(self, decision: TradingDecision):
        """Publier une décision de trading"""
        
        signal = Signal(
            id=None,
            type=SignalType.EXECUTION_ORDER,
            source_agent="crew_orchestrator",
            symbol=decision.symbol,
            priority=SignalPriority.HIGH if decision.action in ["BUY", "SELL"] else SignalPriority.MEDIUM,
            data={
                'action': decision.action,
                'confidence': decision.confidence,
                'target_weight': decision.target_weight,
                'reasoning': decision.reasoning,
                'scores': {
                    'risk': decision.risk_score,
                    'technical': decision.technical_score,
                    'fundamental': decision.fundamental_score,
                    'sentiment': decision.sentiment_score
                }
            },
            metadata={
                'workflow_type': 'crew_analysis',
                'agents_involved': len(self.agents_crewai),
                'decision_timestamp': decision.timestamp.isoformat()
            }
        )
        
        await self.signal_hub.publish_signal(signal)
        self.metrics['decisions_made'] += 1
        
        logger.info(f"📊 Décision publiée: {decision.action} {decision.symbol} (confidence: {decision.confidence:.2f})")
    
    async def analyze_symbols(self, symbols: List[str]) -> List[TradingDecision]:
        """Analyser une liste de symboles"""
        decisions = []
        
        for symbol in symbols:
            decision = await self.execute_workflow([symbol], WorkflowType.SIGNAL_ANALYSIS)
            if decision:
                decisions.append(decision)
        
        return decisions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtenir les métriques de l'orchestrateur"""
        return {
            'name': 'crew_orchestrator',
            'version': '1.0',
            'is_running': self.is_running,
            'workflows_executed': self.metrics['workflows_executed'],
            'decisions_made': self.metrics['decisions_made'],
            'avg_decision_time_ms': round(self.metrics['avg_decision_time_ms'], 2),
            'agents_count': len(self.agents_crewai),
            'active_agents': sum(1 for agent in [
                self.risk_agent, self.technical_agent, 
                self.sentiment_agent, self.fundamental_agent
            ] if hasattr(agent, 'is_running') and agent.is_running)
        }


# Instance globale
_orchestrator_instance: Optional[CrewOrchestrator] = None


def get_orchestrator() -> CrewOrchestrator:
    """Obtenir l'instance globale de l'orchestrateur"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = CrewOrchestrator()
    return _orchestrator_instance