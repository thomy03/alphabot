"""
Tests unitaires pour Risk Agent
Tests des fonctionnalités de gestion des risques et calculs VaR/ES
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import yaml
import os

from alphabot.agents.risk.risk_agent import (
    RiskAgent, VaRCalculator, EVTCalculator, RiskMetrics
)


class TestVaRCalculator:
    """Tests pour le calculateur VaR"""
    
    def test_historical_var_normal_returns(self):
        """Test VaR historique avec rendements normaux"""
        # Génération de rendements test
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)  # Rendements journaliers
        
        var_95 = VaRCalculator.historical_var(returns, 0.95)
        var_99 = VaRCalculator.historical_var(returns, 0.99)
        
        # VaR doit être négatif (perte)
        assert var_95 < 0
        assert var_99 < 0
        # VaR 99% doit être plus extrême que VaR 95%
        assert var_99 < var_95
    
    def test_historical_var_empty_returns(self):
        """Test VaR avec tableau vide"""
        var = VaRCalculator.historical_var(np.array([]), 0.95)
        assert var == 0.0
    
    def test_parametric_var(self):
        """Test VaR paramétrique"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        var_param = VaRCalculator.parametric_var(returns, 0.95)
        var_hist = VaRCalculator.historical_var(returns, 0.95)
        
        # Les deux méthodes doivent donner des résultats proches
        assert abs(var_param - var_hist) < 0.01
    
    def test_expected_shortfall(self):
        """Test Expected Shortfall"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        es = VaRCalculator.expected_shortfall(returns, 0.975)
        var_95 = VaRCalculator.historical_var(returns, 0.95)
        
        # ES doit être plus extrême que VaR
        assert es < var_95
        assert es < 0


class TestEVTCalculator:
    """Tests pour l'EVT Calculator"""
    
    def test_fit_gpd_normal_data(self):
        """Test ajustement GPD sur données normales"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        threshold = np.percentile(data, 90)
        
        shape, scale = EVTCalculator.fit_gpd(data, threshold)
        
        # Paramètres doivent être positifs et raisonnables
        assert scale > 0
        assert -0.5 < shape < 0.5  # Shape typique pour données financières
    
    def test_evt_var_calculation(self):
        """Test calcul VaR via EVT"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        evt_var = EVTCalculator.evt_var(returns, 0.99)
        hist_var = VaRCalculator.historical_var(returns, 0.99)
        
        # Les deux méthodes doivent donner des résultats dans le même ordre
        assert abs(evt_var / hist_var - 1) < 0.5  # Max 50% de différence
    
    def test_evt_var_insufficient_data(self):
        """Test EVT avec données insuffisantes"""
        returns = np.random.normal(0, 0.02, 10)  # Peu de données
        
        evt_var = EVTCalculator.evt_var(returns, 0.99)
        hist_var = VaRCalculator.historical_var(returns, 0.99)
        
        # Doit revenir à VaR historique
        assert evt_var == hist_var


class TestRiskMetrics:
    """Tests pour les métriques de risque portfolio"""
    
    def test_portfolio_volatility(self):
        """Test calcul volatilité portfolio"""
        # Portfolio 2 actifs égaux
        weights = np.array([0.5, 0.5])
        cov_matrix = np.array([[0.04, 0.02], [0.02, 0.09]])  # Corrélation positive
        
        vol = RiskMetrics.portfolio_volatility(weights, cov_matrix)
        
        # Volatilité doit être entre les deux actifs individuels
        vol_asset1 = np.sqrt(0.04)  # 20%
        vol_asset2 = np.sqrt(0.09)  # 30%
        assert vol_asset1 < vol < vol_asset2
    
    def test_portfolio_var(self):
        """Test VaR portfolio"""
        np.random.seed(42)
        # Générer rendements pour 2 actifs corrélés
        returns1 = np.random.normal(0.001, 0.02, 1000)
        returns2 = 0.8 * returns1 + 0.6 * np.random.normal(0.001, 0.015, 1000)
        returns_matrix = np.column_stack([returns1, returns2])
        
        weights = np.array([0.6, 0.4])
        
        port_var = RiskMetrics.portfolio_var(weights, returns_matrix, 0.95)
        
        assert port_var < 0  # VaR est une perte
    
    def test_correlation_matrix(self):
        """Test calcul matrice corrélation"""
        np.random.seed(42)
        returns1 = np.random.normal(0, 0.02, 1000)
        returns2 = 0.7 * returns1 + 0.71 * np.random.normal(0, 0.02, 1000)  # Corrélation ~0.7
        returns_matrix = np.column_stack([returns1, returns2])
        
        corr_matrix = RiskMetrics.correlation_matrix(returns_matrix)
        
        assert corr_matrix.shape == (2, 2)
        assert corr_matrix[0, 0] == 1.0  # Auto-corrélation
        assert corr_matrix[1, 1] == 1.0
        assert 0.6 < corr_matrix[0, 1] < 0.8  # Corrélation attendue
    
    def test_max_drawdown(self):
        """Test calcul maximum drawdown"""
        # Série avec drawdown connu
        returns = np.array([0.05, 0.03, -0.10, -0.05, 0.02, 0.08, -0.15, 0.10])
        
        max_dd = RiskMetrics.max_drawdown(returns)
        
        assert max_dd < 0  # Drawdown est négatif
        assert max_dd > -1  # Pas plus de 100%


class TestRiskAgent:
    """Tests pour l'agent de risque principal"""
    
    @pytest.fixture
    def temp_config(self):
        """Crée un fichier de config temporaire"""
        config = {
            'risk_global': {
                'max_drawdown_percent': 15.0,
                'total_capital_eur': 10000,
                'max_daily_var_percent': 3.0,
                'max_expected_shortfall_percent': 5.0
            },
            'position_limits': {
                'max_single_stock_percent': 5.0,
                'max_portfolio_exposure_percent': 95.0
            },
            'volatility_management': {
                'max_portfolio_correlation': 0.7
            },
            'stress_scenarios': {
                'covid_scenario': {
                    'volatility_multiplier': 1.5,
                    'correlation_increase': 0.2,
                    'max_drawdown_tolerance_percent': 25.0
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_risk_agent_initialization(self, temp_config):
        """Test initialisation Risk Agent"""
        agent = RiskAgent(config_path=temp_config)
        
        assert agent.agent_name == "RiskAgent"
        assert agent.config is not None
        assert 'risk_global' in agent.config
    
    def test_calculate_portfolio_risk(self, temp_config):
        """Test calcul des métriques de risque portfolio"""
        agent = RiskAgent(config_path=temp_config)
        
        # Données test
        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, (1000, 3))  # 1000 jours, 3 actifs
        weights = np.array([0.4, 0.4, 0.2])
        
        message = {
            'type': 'calculate_risk',
            'returns_data': returns_data,
            'weights': weights,
            'positions': {'AAPL': 2000, 'MSFT': 2000, 'GOOGL': 1000}
        }
        
        response = agent._process_message(message)
        
        assert response['status'] == 'success'
        assert 'risk_metrics' in response
        assert 'var_95_percent' in response['risk_metrics']
        assert 'expected_shortfall_975' in response['risk_metrics']
        assert 'portfolio_volatility' in response['risk_metrics']
    
    def test_validate_position(self, temp_config):
        """Test validation de position"""
        agent = RiskAgent(config_path=temp_config)
        
        message = {
            'type': 'validate_position',
            'ticker': 'AAPL',
            'position_size': 300,  # 3% du capital (acceptable)
            'current_portfolio': {'MSFT': 400, 'GOOGL': 300}
        }
        
        response = agent._process_message(message)
        
        assert response['status'] == 'success'
        assert response['position_valid'] is True
        
        # Test avec position trop grande
        message['position_size'] = 800  # 8% du capital (trop grand)
        response = agent._process_message(message)
        
        assert response['position_valid'] is False
        assert len(response['violations']) > 0
    
    def test_stress_test(self, temp_config):
        """Test exécution stress test"""
        agent = RiskAgent(config_path=temp_config)
        
        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, (1000, 2))
        weights = np.array([0.6, 0.4])
        
        message = {
            'type': 'stress_test',
            'scenario': 'covid_scenario',
            'returns_data': returns_data,
            'weights': weights
        }
        
        response = agent._process_message(message)
        
        assert response['status'] == 'success'
        assert response['scenario'] == 'covid_scenario'
        assert 'stressed_metrics' in response
        assert 'pass_stress_test' in response
    
    def test_risk_limits_validation(self, temp_config):
        """Test validation des limites de risque"""
        agent = RiskAgent(config_path=temp_config)
        
        # Métriques qui violent les limites
        metrics = {
            'var_95': -0.05,  # -5% > limite 3%
            'es_975': -0.07,  # -7% > limite 5%
            'portfolio_vol': 0.25,
            'max_correlation': 0.85,  # > limite 0.7
            'max_drawdown': -0.20  # -20% > limite 15%
        }
        
        violations = agent._check_risk_limits(metrics)
        
        # Doit détecter plusieurs violations
        assert len(violations) >= 3
        violation_types = [v['type'] for v in violations]
        assert 'var_breach' in violation_types
        assert 'es_breach' in violation_types
        assert 'correlation_high' in violation_types
    
    def test_caching_mechanism(self, temp_config):
        """Test mécanisme de cache"""
        agent = RiskAgent(config_path=temp_config)
        
        # Premier calcul
        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, (100, 2))
        weights = np.array([0.5, 0.5])
        
        message = {
            'type': 'calculate_risk',
            'returns_data': returns_data,
            'weights': weights
        }
        
        response1 = agent._process_message(message)
        
        # Vérifier cache
        cached = agent.get_cached_risk_metrics()
        assert cached is not None
        assert cached['status'] == 'success'
    
    def test_health_check(self, temp_config):
        """Test health check de l'agent"""
        agent = RiskAgent(config_path=temp_config)
        
        assert agent.health_check() is True
        
        status = agent.get_status()
        assert status['agent_name'] == 'RiskAgent'
        assert status['config_loaded'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])