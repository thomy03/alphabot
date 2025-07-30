#!/usr/bin/env python3
"""
Stress Test Script - AlphaBot Multi-Agent Trading System
Test de charge : génère 600 signaux en 10 minutes et mesure les performances

Objectifs:
- Latence moyenne < 200ms par signal
- Throughput ≥ 1 signal/seconde
- Uptime agents ≥ 99.5%
- Aucune perte de signal
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
import json
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphabot.agents.technical.technical_agent import TechnicalAgent
from alphabot.agents.sentiment.sentiment_agent import SentimentAgent
from alphabot.agents.risk.risk_agent import RiskAgent


@dataclass
class StressTestConfig:
    """Configuration du test de stress"""
    total_signals: int = 600
    duration_minutes: int = 10
    max_concurrent: int = 50
    target_latency_ms: float = 200.0
    target_throughput: float = 1.0  # signals/sec
    test_agents: List[str] = None
    
    def __post_init__(self):
        if self.test_agents is None:
            self.test_agents = ['technical', 'sentiment', 'risk']


@dataclass
class SignalResult:
    """Résultat d'un signal individual"""
    signal_id: int
    agent_type: str
    ticker: str
    start_time: float
    end_time: float
    latency_ms: float
    success: bool
    error_message: str = None
    result_data: Dict[str, Any] = None


@dataclass
class StressTestResults:
    """Résultats consolidés du stress test"""
    total_signals: int
    successful_signals: int
    failed_signals: int
    total_duration_seconds: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_signals_per_sec: float
    success_rate_percent: float
    agent_stats: Dict[str, Dict[str, float]]
    errors: List[str]
    timestamp: str


class StressTestRunner:
    """Runner principal pour les tests de stress"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.agents = {}
        self.results: List[SignalResult] = []
        self.start_time = None
        self.end_time = None
        
        # Données de test
        self.test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'NFLX']
        self.test_data = self._generate_test_data()
        self.test_texts = self._generate_test_texts()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure le logger pour le stress test"""
        logger = logging.getLogger('stress_test')
        logger.setLevel(logging.INFO)
        
        # Handler console
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_agents(self):
        """Initialise les agents pour le test"""
        self.logger.info("Initialisation des agents...")
        
        try:
            if 'technical' in self.config.test_agents:
                self.agents['technical'] = TechnicalAgent()
                self.logger.info("✅ Technical Agent initialisé")
            
            if 'sentiment' in self.config.test_agents:
                self.agents['sentiment'] = SentimentAgent()
                self.logger.info("✅ Sentiment Agent initialisé")
            
            if 'risk' in self.config.test_agents:
                self.agents['risk'] = RiskAgent()
                self.logger.info("✅ Risk Agent initialisé")
            
            # Health check
            for agent_type, agent in self.agents.items():
                if not agent.health_check():
                    raise Exception(f"Health check failed for {agent_type} agent")
            
            self.logger.info(f"✅ Tous les agents ({len(self.agents)}) sont opérationnels")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation agents: {e}")
            raise
    
    def _generate_test_data(self) -> Dict[str, pd.DataFrame]:
        """Génère des données de test pour les calculs techniques"""
        np.random.seed(42)
        test_data = {}
        
        for ticker in self.test_tickers:
            # Générer 1000 jours de données OHLCV
            dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
            
            # Prix de base avec marche aléatoire
            base_price = 100.0
            returns = np.random.normal(0.0005, 0.02, 1000)  # 0.05% moyenne, 2% vol
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # OHLC basé sur le close
            df = pd.DataFrame({
                'date': dates,
                'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 1000)
            })
            
            test_data[ticker] = df
        
        return test_data
    
    def _generate_test_texts(self) -> List[str]:
        """Génère des textes de test pour l'analyse de sentiment"""
        positive_texts = [
            "Company reports strong quarterly earnings beating analyst estimates",
            "Revenue growth accelerates as demand increases across all segments",
            "Stock price surges on bullish guidance and strong fundamentals",
            "Analyst upgrades rating citing excellent execution and market position"
        ]
        
        negative_texts = [
            "Company misses earnings expectations amid weak demand",
            "Revenue decline continues as market conditions deteriorate",
            "Stock falls on bearish outlook and regulatory concerns",
            "Analyst downgrades citing increased competition and margin pressure"
        ]
        
        neutral_texts = [
            "Company maintains steady performance in line with expectations",
            "Market conditions remain stable with unchanged outlook",
            "Stock trades sideways as investors await next earnings report",
            "Analyst maintains neutral rating pending further developments"
        ]
        
        # Créer une liste de 100 textes variés
        all_texts = positive_texts * 25 + negative_texts * 25 + neutral_texts * 25
        np.random.shuffle(all_texts)
        
        return all_texts[:100]  # Retourner 100 textes
    
    def _create_signal_payload(self, signal_id: int, agent_type: str) -> Tuple[str, Dict[str, Any]]:
        """Crée le payload pour un signal de test"""
        ticker = np.random.choice(self.test_tickers)
        
        if agent_type == 'technical':
            return ticker, {
                'type': 'analyze_technical',
                'ticker': ticker,
                'price_data': self.test_data[ticker].to_dict('records')
            }
        
        elif agent_type == 'sentiment':
            text = np.random.choice(self.test_texts)
            return ticker, {
                'type': 'analyze_sentiment',
                'text': f"{text} for {ticker}",
                'ticker': ticker
            }
        
        elif agent_type == 'risk':
            # Utiliser des données du technical pour les calculs de risque
            returns_data = np.random.normal(0.001, 0.02, (252, 3))  # 1 an, 3 actifs
            weights = np.array([0.4, 0.4, 0.2])
            
            return ticker, {
                'type': 'calculate_risk',
                'returns_data': returns_data,
                'weights': weights,
                'positions': {ticker: 1000, 'SPY': 2000, 'QQQ': 1000}
            }
        
        else:
            raise ValueError(f"Agent type non supporté: {agent_type}")
    
    def _execute_signal(self, signal_id: int, agent_type: str) -> SignalResult:
        """Exécute un signal individual et mesure la performance"""
        start_time = time.time()
        
        try:
            ticker, payload = self._create_signal_payload(signal_id, agent_type)
            agent = self.agents[agent_type]
            
            # Exécution du signal
            result = agent._process_message(payload)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            success = result.get('status') == 'success'
            error_message = result.get('message') if not success else None
            
            return SignalResult(
                signal_id=signal_id,
                agent_type=agent_type,
                ticker=ticker,
                start_time=start_time,
                end_time=end_time,
                latency_ms=latency_ms,
                success=success,
                error_message=error_message,
                result_data=result if success else None
            )
            
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            return SignalResult(
                signal_id=signal_id,
                agent_type=agent_type,
                ticker="ERROR",
                start_time=start_time,
                end_time=end_time,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e)
            )
    
    async def run_stress_test(self) -> StressTestResults:
        """Exécute le test de stress principal"""
        self.logger.info(f"🚀 Démarrage du stress test:")
        self.logger.info(f"   📊 {self.config.total_signals} signaux en {self.config.duration_minutes} minutes")
        self.logger.info(f"   🎯 Cible: {self.config.target_latency_ms}ms latence, {self.config.target_throughput} signals/sec")
        self.logger.info(f"   🤖 Agents: {', '.join(self.config.test_agents)}")
        
        # Initialiser les agents
        self._initialize_agents()
        
        # Calculer timing
        signals_per_second = self.config.total_signals / (self.config.duration_minutes * 60)
        interval_seconds = 1.0 / signals_per_second if signals_per_second > 0 else 0.1
        
        self.logger.info(f"   ⏱️  Intervalle entre signaux: {interval_seconds:.3f}s")
        
        self.start_time = time.time()
        
        # Utiliser ThreadPoolExecutor pour la concurrence
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            futures = []
            signal_count = 0
            
            # Générer et soumettre les signaux
            for i in range(self.config.total_signals):
                # Choisir un agent aléatoirement
                agent_type = np.random.choice(self.config.test_agents)
                
                # Soumettre le signal
                future = executor.submit(self._execute_signal, i, agent_type)
                futures.append(future)
                signal_count += 1
                
                # Affichage du progrès
                if signal_count % 50 == 0:
                    elapsed = time.time() - self.start_time
                    rate = signal_count / elapsed if elapsed > 0 else 0
                    self.logger.info(f"   📈 {signal_count}/{self.config.total_signals} signaux soumis ({rate:.1f}/sec)")
                
                # Respect de l'intervalle (optionnel, pour éviter la surcharge)
                if interval_seconds > 0.01:  # Seulement si intervalle > 10ms
                    await asyncio.sleep(interval_seconds)
            
            # Collecter tous les résultats
            self.logger.info("📥 Collecte des résultats...")
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # Timeout de 30s par signal
                    self.results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Erreur dans un signal: {e}")
        
        self.end_time = time.time()
        
        # Analyser les résultats
        return self._analyze_results()
    
    def _analyze_results(self) -> StressTestResults:
        """Analyse les résultats du stress test"""
        if not self.results:
            raise Exception("Aucun résultat à analyser")
        
        # Métriques globales
        total_duration = self.end_time - self.start_time
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        # Latences
        latencies = [r.latency_ms for r in successful_results]
        avg_latency = np.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        
        # Throughput
        throughput = len(self.results) / total_duration if total_duration > 0 else 0
        success_rate = len(successful_results) / len(self.results) * 100 if self.results else 0
        
        # Statistiques par agent
        agent_stats = {}
        for agent_type in self.config.test_agents:
            agent_results = [r for r in self.results if r.agent_type == agent_type]
            agent_successful = [r for r in agent_results if r.success]
            agent_latencies = [r.latency_ms for r in agent_successful]
            
            agent_stats[agent_type] = {
                'total_signals': len(agent_results),
                'successful_signals': len(agent_successful),
                'success_rate_percent': len(agent_successful) / len(agent_results) * 100 if agent_results else 0,
                'avg_latency_ms': np.mean(agent_latencies) if agent_latencies else 0,
                'p95_latency_ms': np.percentile(agent_latencies, 95) if agent_latencies else 0
            }
        
        # Erreurs
        errors = [r.error_message for r in failed_results if r.error_message]
        
        return StressTestResults(
            total_signals=len(self.results),
            successful_signals=len(successful_results),
            failed_signals=len(failed_results),
            total_duration_seconds=total_duration,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_signals_per_sec=throughput,
            success_rate_percent=success_rate,
            agent_stats=agent_stats,
            errors=errors[:10],  # Limiter à 10 erreurs
            timestamp=datetime.now().isoformat()
        )
    
    def _print_results(self, results: StressTestResults):
        """Affiche les résultats du stress test"""
        print("\n" + "="*70)
        print("🏁 RÉSULTATS DU STRESS TEST - AlphaBot")
        print("="*70)
        
        # Métriques globales
        print(f"\n📊 MÉTRIQUES GLOBALES:")
        print(f"   Total signaux      : {results.total_signals}")
        print(f"   Signaux réussis    : {results.successful_signals}")
        print(f"   Signaux échoués    : {results.failed_signals}")
        print(f"   Taux de succès     : {results.success_rate_percent:.1f}%")
        print(f"   Durée totale       : {results.total_duration_seconds:.1f}s")
        print(f"   Throughput         : {results.throughput_signals_per_sec:.2f} signals/sec")
        
        # Latences
        print(f"\n⏱️  LATENCES:")
        print(f"   Latence moyenne    : {results.avg_latency_ms:.1f}ms")
        print(f"   Latence P95        : {results.p95_latency_ms:.1f}ms")
        print(f"   Latence P99        : {results.p99_latency_ms:.1f}ms")
        
        # Statuts par rapport aux objectifs
        print(f"\n🎯 CONFORMITÉ AUX OBJECTIFS:")
        latency_ok = results.avg_latency_ms <= self.config.target_latency_ms
        throughput_ok = results.throughput_signals_per_sec >= self.config.target_throughput
        success_ok = results.success_rate_percent >= 99.5
        
        print(f"   Latence < {self.config.target_latency_ms}ms     : {'✅' if latency_ok else '❌'} ({results.avg_latency_ms:.1f}ms)")
        print(f"   Throughput ≥ {self.config.target_throughput}/sec   : {'✅' if throughput_ok else '❌'} ({results.throughput_signals_per_sec:.2f}/sec)")
        print(f"   Taux succès ≥ 99.5% : {'✅' if success_ok else '❌'} ({results.success_rate_percent:.1f}%)")
        
        overall_pass = latency_ok and throughput_ok and success_ok
        print(f"\n🏆 RÉSULTAT GLOBAL     : {'✅ PASS' if overall_pass else '❌ FAIL'}")
        
        # Détails par agent
        print(f"\n🤖 DÉTAILS PAR AGENT:")
        for agent_type, stats in results.agent_stats.items():
            print(f"   {agent_type.upper():<12} : {stats['successful_signals']}/{stats['total_signals']} "
                  f"({stats['success_rate_percent']:.1f}%) - {stats['avg_latency_ms']:.1f}ms avg")
        
        # Erreurs
        if results.errors:
            print(f"\n❌ ERREURS ({len(results.errors)} échantillon):")
            for i, error in enumerate(results.errors[:5], 1):
                print(f"   {i}. {error}")
        
        print("\n" + "="*70)
    
    def save_results(self, results: StressTestResults, output_file: str = None):
        """Sauvegarde les résultats dans un fichier JSON"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"stress_test_results_{timestamp}.json"
        
        # Convertir en dict pour JSON
        results_dict = asdict(results)
        
        # Ajouter les résultats détaillés
        results_dict['detailed_results'] = [asdict(r) for r in self.results]
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📁 Résultats sauvegardés dans: {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde: {e}")
            return None


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Stress Test AlphaBot - 600 signaux en 10 minutes")
    parser.add_argument("--signals", type=int, default=600, help="Nombre total de signaux (défaut: 600)")
    parser.add_argument("--duration", type=int, default=10, help="Durée en minutes (défaut: 10)")
    parser.add_argument("--concurrent", type=int, default=50, help="Max signaux concurrents (défaut: 50)")
    parser.add_argument("--agents", nargs='+', default=['technical', 'sentiment', 'risk'], 
                      choices=['technical', 'sentiment', 'risk'], help="Agents à tester")
    parser.add_argument("--output", type=str, help="Fichier de sortie pour les résultats")
    parser.add_argument("--quick", action='store_true', help="Test rapide (60 signaux en 1 minute)")
    
    args = parser.parse_args()
    
    # Configuration rapide si demandée
    if args.quick:
        config = StressTestConfig(
            total_signals=60,
            duration_minutes=1,
            max_concurrent=20,
            test_agents=args.agents
        )
        print("🚀 Mode test rapide: 60 signaux en 1 minute")
    else:
        config = StressTestConfig(
            total_signals=args.signals,
            duration_minutes=args.duration,
            max_concurrent=args.concurrent,
            test_agents=args.agents
        )
    
    # Exécuter le test
    runner = StressTestRunner(config)
    
    try:
        results = await runner.run_stress_test()
        runner._print_results(results)
        
        # Sauvegarder si demandé
        if args.output:
            runner.save_results(results, args.output)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur durant le test: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())