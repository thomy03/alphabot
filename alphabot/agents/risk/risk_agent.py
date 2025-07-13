"""
Risk Agent pour AlphaBot - Gestion des risques et contrôles
Calcule VaR, Expected Shortfall, EVT, et applique les politiques de risque
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings

from ..TEMPLATE_agent import AlphaBotAgentTemplate
from crewai.tools import BaseTool


class VaRCalculator:
    """Calculateur de Value at Risk et Expected Shortfall"""
    
    @staticmethod
    def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calcule VaR historique"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def parametric_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calcule VaR paramétrique (normale)"""
        if len(returns) == 0:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence)
        return mean + z_score * std
    
    @staticmethod
    def expected_shortfall(returns: np.ndarray, confidence: float = 0.975) -> float:
        """Calcule Expected Shortfall (CVaR)"""
        if len(returns) == 0:
            return 0.0
        var = VaRCalculator.historical_var(returns, confidence)
        tail_returns = returns[returns <= var]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var


class EVTCalculator:
    """Calculateur Extreme Value Theory pour tail risk"""
    
    @staticmethod
    def fit_gpd(data: np.ndarray, threshold: float) -> Tuple[float, float]:
        """
        Ajuste distribution Generalized Pareto pour queues
        
        Returns:
            Tuple (shape_param, scale_param)
        """
        exceedances = data[data > threshold] - threshold
        if len(exceedances) < 10:
            return 0.1, np.std(data)  # Valeurs par défaut
        
        try:
            # Estimation par maximum de vraisemblance
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
            return shape, scale
        except:
            return 0.1, np.std(data)
    
    @staticmethod
    def evt_var(returns: np.ndarray, confidence: float = 0.99, 
                threshold_percentile: float = 0.9) -> float:
        """Calcule VaR via EVT"""
        if len(returns) < 50:
            return VaRCalculator.historical_var(returns, confidence)
        
        threshold = np.percentile(np.abs(returns), threshold_percentile * 100)
        losses = -returns  # Convertir en pertes
        
        shape, scale = EVTCalculator.fit_gpd(losses, threshold)
        n_exceedances = np.sum(losses > threshold)
        n_total = len(losses)
        
        if n_exceedances == 0:
            return VaRCalculator.historical_var(returns, confidence)
        
        # Calcul VaR EVT
        prob_exceed = n_exceedances / n_total
        q = (1 - confidence) / prob_exceed
        
        if shape != 0:
            evt_var = threshold + (scale / shape) * (q**(-shape) - 1)
        else:
            evt_var = threshold + scale * np.log(q)
        
        return -evt_var  # Retourner comme perte négative


class RiskMetrics:
    """Calcul des métriques de risque portfolio"""
    
    @staticmethod
    def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calcule volatilité du portefeuille"""
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    @staticmethod
    def portfolio_var(weights: np.ndarray, returns_matrix: np.ndarray, 
                     confidence: float = 0.95) -> float:
        """Calcule VaR du portefeuille"""
        portfolio_returns = returns_matrix @ weights
        return VaRCalculator.historical_var(portfolio_returns, confidence)
    
    @staticmethod
    def correlation_matrix(returns_matrix: np.ndarray) -> np.ndarray:
        """Calcule matrice de corrélation"""
        return np.corrcoef(returns_matrix.T)
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Calcule maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)


class RiskAgent(AlphaBotAgentTemplate):
    """
    Agent de gestion des risques pour AlphaBot
    Responsable du calcul et monitoring des métriques de risque
    """
    
    def __init__(self, config_path: str = "risk_policy.yaml"):
        super().__init__(
            agent_name="RiskAgent",
            description="Calculate and monitor portfolio risk metrics including VaR, ES, and stress scenarios",
            config_path=config_path
        )
        
        self.var_calculator = VaRCalculator()
        self.evt_calculator = EVTCalculator()
        self.risk_metrics = RiskMetrics()
        
        # Cache pour optimiser les calculs
        self._cache = {}
        self._last_calculation = None
    
    def _validate_config(self) -> None:
        """Valide la configuration spécifique au Risk Agent"""
        super()._validate_config()
        
        required_sections = [
            'risk_global', 'position_limits', 'volatility_management'
        ]
        
        for section in required_sections:
            if section not in self.config:
                self.logger.warning(f"Missing config section: {section}")
    
    def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les demandes de calcul de risque"""
        message_type = message.get('type', 'calculate_risk')
        
        if message_type == 'calculate_risk':
            return self._calculate_portfolio_risk(message)
        elif message_type == 'validate_position':
            return self._validate_position(message)
        elif message_type == 'stress_test':
            return self._run_stress_test(message)
        else:
            return {
                "status": "error",
                "message": f"Unknown message type: {message_type}"
            }
    
    def _calculate_portfolio_risk(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les métriques de risque du portefeuille"""
        try:
            # Extraction des données
            returns_data = message.get('returns_data')
            weights = np.array(message.get('weights', []))
            positions = message.get('positions', {})
            
            if returns_data is None:
                return {"status": "error", "message": "Missing returns_data"}
            
            # Conversion en numpy si nécessaire
            if isinstance(returns_data, pd.DataFrame):
                returns_matrix = returns_data.values
            else:
                returns_matrix = np.array(returns_data)
            
            # Calculs de base
            portfolio_returns = returns_matrix @ weights if len(weights) > 0 else np.array([])
            
            # VaR et ES
            var_95 = self.var_calculator.historical_var(portfolio_returns, 0.95)
            var_99 = self.var_calculator.historical_var(portfolio_returns, 0.99)
            es_975 = self.var_calculator.expected_shortfall(portfolio_returns, 0.975)
            
            # EVT pour tail risk
            evt_var_99 = self.evt_calculator.evt_var(portfolio_returns, 0.99)
            
            # Métriques portfolio
            if len(weights) > 0 and returns_matrix.shape[1] == len(weights):
                cov_matrix = np.cov(returns_matrix.T)
                portfolio_vol = self.risk_metrics.portfolio_volatility(weights, cov_matrix)
                correlation_matrix = self.risk_metrics.correlation_matrix(returns_matrix)
                max_correlation = np.max(correlation_matrix[correlation_matrix < 1])
            else:
                portfolio_vol = 0.0
                max_correlation = 0.0
            
            max_dd = self.risk_metrics.max_drawdown(portfolio_returns)
            
            # Validation des limites
            risk_violations = self._check_risk_limits({
                'var_95': var_95,
                'es_975': es_975,
                'portfolio_vol': portfolio_vol,
                'max_correlation': max_correlation,
                'max_drawdown': max_dd
            })
            
            result = {
                "status": "success",
                "risk_metrics": {
                    "var_95_percent": var_95 * 100,
                    "var_99_percent": var_99 * 100,
                    "expected_shortfall_975": es_975 * 100,
                    "evt_var_99": evt_var_99 * 100,
                    "portfolio_volatility": portfolio_vol,
                    "max_correlation": max_correlation,
                    "max_drawdown": max_dd * 100,
                    "portfolio_returns_sharpe": np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
                },
                "risk_violations": risk_violations,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache le résultat
            self._cache['last_risk_calculation'] = result
            self._last_calculation = datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_position(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Valide une position proposée contre les limites de risque"""
        try:
            ticker = message.get('ticker')
            position_size = message.get('position_size', 0)
            current_portfolio = message.get('current_portfolio', {})
            
            # Récupération des limites
            risk_global = self.config.get('risk_global', {})
            position_limits = self.config.get('position_limits', {})
            
            total_capital = risk_global.get('total_capital_eur', 10000)
            max_single_stock = position_limits.get('max_single_stock_percent', 5.0) / 100
            max_position_eur = max_single_stock * total_capital
            
            # Validations
            violations = []
            
            if position_size > max_position_eur:
                violations.append({
                    "type": "position_size",
                    "message": f"Position size {position_size}€ exceeds limit {max_position_eur}€",
                    "severity": "error"
                })
            
            # Validation exposition totale
            total_exposure = sum(current_portfolio.values()) + position_size
            max_exposure = risk_global.get('max_portfolio_exposure_percent', 95.0) / 100 * total_capital
            
            if total_exposure > max_exposure:
                violations.append({
                    "type": "total_exposure",
                    "message": f"Total exposure would exceed limit",
                    "severity": "warning"
                })
            
            return {
                "status": "success",
                "position_valid": len([v for v in violations if v["severity"] == "error"]) == 0,
                "violations": violations,
                "recommended_size": min(position_size, max_position_eur),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _run_stress_test(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute des stress tests sur le portefeuille"""
        try:
            scenario = message.get('scenario', 'covid_scenario')
            returns_data = message.get('returns_data')
            weights = np.array(message.get('weights', []))
            
            # Récupération du scénario
            stress_scenarios = self.config.get('stress_scenarios', {})
            scenario_params = stress_scenarios.get(scenario, {})
            
            if not scenario_params:
                return {
                    "status": "error",
                    "message": f"Unknown stress scenario: {scenario}"
                }
            
            # Application du stress
            vol_multiplier = scenario_params.get('volatility_multiplier', 1.0)
            corr_increase = scenario_params.get('correlation_increase', 0.0)
            
            if isinstance(returns_data, pd.DataFrame):
                stressed_returns = returns_data.values * vol_multiplier
            else:
                stressed_returns = np.array(returns_data) * vol_multiplier
            
            # Calcul métriques sous stress
            portfolio_returns = stressed_returns @ weights if len(weights) > 0 else np.array([])
            
            stressed_var = self.var_calculator.historical_var(portfolio_returns, 0.95)
            stressed_es = self.var_calculator.expected_shortfall(portfolio_returns, 0.975)
            stressed_dd = self.risk_metrics.max_drawdown(portfolio_returns)
            
            return {
                "status": "success",
                "scenario": scenario,
                "stressed_metrics": {
                    "var_95": stressed_var * 100,
                    "expected_shortfall": stressed_es * 100,
                    "max_drawdown": stressed_dd * 100,
                    "volatility_multiplier": vol_multiplier,
                    "correlation_increase": corr_increase
                },
                "pass_stress_test": abs(stressed_dd) <= scenario_params.get('max_drawdown_tolerance_percent', 25.0) / 100,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _check_risk_limits(self, metrics: Dict[str, float]) -> List[Dict[str, str]]:
        """Vérifie les violations des limites de risque"""
        violations = []
        
        # Récupération des limites
        risk_global = self.config.get('risk_global', {})
        volatility_mgmt = self.config.get('volatility_management', {})
        
        # VaR
        if abs(metrics['var_95']) > risk_global.get('max_daily_var_percent', 3.0) / 100:
            violations.append({
                "type": "var_breach",
                "message": f"VaR 95% exceeds limit: {metrics['var_95']*100:.2f}%",
                "severity": "error"
            })
        
        # Expected Shortfall
        if abs(metrics['es_975']) > risk_global.get('max_expected_shortfall_percent', 5.0) / 100:
            violations.append({
                "type": "es_breach", 
                "message": f"Expected Shortfall exceeds limit: {metrics['es_975']*100:.2f}%",
                "severity": "error"
            })
        
        # Corrélation
        if metrics['max_correlation'] > volatility_mgmt.get('max_portfolio_correlation', 0.7):
            violations.append({
                "type": "correlation_high",
                "message": f"Portfolio correlation too high: {metrics['max_correlation']:.3f}",
                "severity": "warning"
            })
        
        # Drawdown
        if abs(metrics['max_drawdown']) > risk_global.get('max_drawdown_percent', 15.0):
            violations.append({
                "type": "drawdown_breach",
                "message": f"Max drawdown exceeds limit: {metrics['max_drawdown']:.2f}%",
                "severity": "error"
            })
        
        return violations
    
    def get_cached_risk_metrics(self) -> Optional[Dict[str, Any]]:
        """Retourne les dernières métriques calculées si récentes"""
        if (self._last_calculation and 
            datetime.now() - self._last_calculation < timedelta(minutes=5)):
            return self._cache.get('last_risk_calculation')
        return None