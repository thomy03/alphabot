"""
Enhanced Risk Agent - CVaR + Ulcer Index + Advanced Metrics
Agent de risque amélioré selon recommandations expert
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import yfinance as yf
from scipy import stats

logger = logging.getLogger(__name__)


class EnhancedRiskAgent:
    """
    Agent de risque amélioré avec :
    - CVaR (Conditional VaR) vs VaR traditionnel
    - Ulcer Index pour downside volatility
    - Calmar Ratio pour recovery analysis
    - TVaR (Tail VaR) pour severity analysis
    """
    
    def __init__(self):
        # Paramètres de risque
        self.confidence_levels = [0.95, 0.975, 0.99]
        self.lookback_days = 252  # 1 year
        
        # Limites de risque (selon risk_policy.yaml)
        self.max_position_size = 0.05  # 5%
        self.max_sector_exposure = 0.30  # 30%
        self.max_portfolio_var = 0.03  # 3%
        self.max_drawdown = 0.15  # 15%
        
        # Cache optimisé
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info("✅ Enhanced Risk Agent initialized - CVaR + Ulcer + Calmar")
    
    async def assess_portfolio_risk_cvar(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """
        Évaluation complète du risque portfolio avec CVaR
        
        Args:
            portfolio: {symbol: weight} dict
            
        Returns:
            Métriques de risque avancées
        """
        try:
            # Récupérer returns pour tous les actifs
            returns_data = {}
            for symbol, weight in portfolio.items():
                if weight > 0:
                    returns = await self._get_cached_returns(symbol)
                    if returns is not None:
                        returns_data[symbol] = returns
            
            if not returns_data:
                return self._default_risk_assessment()
            
            # Calculs directs (pas parallèles pour éviter erreurs)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, portfolio)
            
            cvar_metrics = {
                'cvar_95': self._calculate_cvar(portfolio_returns, 0.05),
                'cvar_975': self._calculate_cvar(portfolio_returns, 0.025),
                'cvar_99': self._calculate_cvar(portfolio_returns, 0.01)
            }
            
            # Portfolio prices pour ulcer/calmar
            portfolio_prices = self._calculate_portfolio_prices(returns_data, portfolio)
            
            ulcer_metrics = {
                'ulcer_index': self._calculate_ulcer_index(portfolio_prices)
            }
            
            calmar_metrics = {
                'calmar_ratio': self._calculate_calmar_ratio(portfolio_returns, portfolio_prices)
            }
            
            tvar_metrics = {
                'tvar_95': self._calculate_tvar(portfolio_returns, 0.05),
                'tvar_975': self._calculate_tvar(portfolio_returns, 0.025)
            }
            
            # Consolidation
            risk_assessment = {
                **cvar_metrics,
                **ulcer_metrics,
                **calmar_metrics,
                **tvar_metrics,
                'portfolio_size': len(returns_data),
                'assessment_timestamp': datetime.now().isoformat(),
                'risk_level': self._determine_risk_level(cvar_metrics, ulcer_metrics)
            }
            
            logger.debug(f"Portfolio risk assessment: CVaR {cvar_metrics['cvar_95']:.2%}")
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return self._default_risk_assessment()
    
    async def assess_single_asset_risk(self, symbol: str) -> Dict[str, Any]:
        """
        Évaluation risque d'un actif individuel
        """
        try:
            returns = await self._get_cached_returns(symbol)
            if returns is None:
                return self._default_asset_risk(symbol)
            
            prices = await self._get_cached_prices(symbol)
            
            # Calculs de risque
            risk_metrics = {
                'symbol': symbol,
                'cvar_95': self._calculate_cvar(returns, 0.05),
                'cvar_975': self._calculate_cvar(returns, 0.025),
                'cvar_99': self._calculate_cvar(returns, 0.01),
                'tvar_95': self._calculate_tvar(returns, 0.05),
                'ulcer_index': self._calculate_ulcer_index(prices),
                'volatility': float(returns.std() * np.sqrt(252)),
                'max_drawdown': self._calculate_max_drawdown(prices),
                'calmar_ratio': self._calculate_calmar_ratio(returns, prices),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'var_95': float(np.percentile(returns, 5))
            }
            
            # Score de risque composite
            risk_metrics['risk_score'] = self._calculate_composite_risk_score(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Asset risk assessment failed for {symbol}: {e}")
            return self._default_asset_risk(symbol)
    
    async def check_position_limits(self, symbol: str, target_weight: float, 
                                   current_portfolio: Dict[str, float]) -> Dict[str, Any]:
        """
        Vérification des limites de position selon politique de risque
        """
        violations = []
        warnings = []
        
        # Limite position individuelle
        if target_weight > self.max_position_size:
            violations.append(f"Position size {target_weight:.1%} > limit {self.max_position_size:.1%}")
        
        # Limite exposition sectorielle
        sector_exposure = await self._calculate_sector_exposure(symbol, target_weight, current_portfolio)
        if sector_exposure > self.max_sector_exposure:
            violations.append(f"Sector exposure {sector_exposure:.1%} > limit {self.max_sector_exposure:.1%}")
        
        # Avertissements
        if target_weight > self.max_position_size * 0.8:
            warnings.append(f"Position approaching limit: {target_weight:.1%}")
        
        return {
            'symbol': symbol,
            'target_weight': target_weight,
            'violations': violations,
            'warnings': warnings,
            'can_trade': len(violations) == 0,
            'sector_exposure': sector_exposure,
            'position_utilization': target_weight / self.max_position_size
        }
    
    async def stress_test_cvar(self, portfolio: Dict[str, float], 
                              scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stress test avec CVaR selon scénarios
        
        Args:
            scenario: {'vol_multiplier': float, 'correlation_boost': float, 'name': str}
        """
        try:
            # Récupérer données historiques
            returns_data = {}
            for symbol, weight in portfolio.items():
                if weight > 0:
                    returns = await self._get_cached_returns(symbol)
                    if returns is not None:
                        returns_data[symbol] = returns
            
            # Appliquer stress scenario
            stressed_returns = self._apply_stress_scenario(returns_data, scenario)
            
            # Calculs CVaR sous stress
            portfolio_returns = self._calculate_portfolio_returns(stressed_returns, portfolio)
            
            stress_results = {
                'scenario_name': scenario.get('name', 'Custom'),
                'base_cvar_95': self._calculate_cvar(portfolio_returns, 0.05),
                'stress_cvar_95': self._calculate_cvar(stressed_returns, 0.05),
                'cvar_deterioration': 0.0,
                'max_loss_scenario': float(np.min(stressed_returns)),
                'breach_probability': float(np.mean(stressed_returns < -0.15))  # P(loss > 15%)
            }
            
            # Calcul détérioration
            if stress_results['base_cvar_95'] != 0:
                stress_results['cvar_deterioration'] = (
                    stress_results['stress_cvar_95'] / stress_results['base_cvar_95'] - 1
                )
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return {'error': str(e)}
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        Calcul CVaR (Conditional Value at Risk)
        Recommandation expert : Meilleur que VaR pour tail risks
        """
        if len(returns) == 0:
            return -0.05
        
        var = np.percentile(returns, confidence_level * 100)
        tail_returns = returns[returns <= var]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var
        return float(cvar)
    
    def _calculate_tvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        Calcul TVaR (Tail Value at Risk) = Expected value given breach
        Capture severity beyond threshold
        """
        var = np.percentile(returns, confidence_level * 100)
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return var
        return float(np.mean(tail_returns))
    
    def _calculate_ulcer_index(self, prices: pd.Series) -> float:
        """
        Calcul Ulcer Index - Recommandation expert
        Focus sur downside volatility uniquement
        """
        if len(prices) < 2:
            return 5.0
        
        # Calcul drawdowns depuis peaks
        cummax = prices.expanding().max()
        drawdowns = (prices / cummax - 1) * 100
        
        # Ulcer Index = RMS des drawdowns
        ulcer = np.sqrt(np.mean(drawdowns ** 2))
        return float(ulcer)
    
    def _calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """
        Calcul Calmar Ratio - Recommandation expert
        Rendement annualisé / Max Drawdown
        """
        if len(returns) < 2:
            return 0.0
        
        # Rendement annualisé
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        periods = len(returns)
        annual_return = (1 + total_return) ** (252 / periods) - 1
        
        # Max drawdown
        max_dd = abs(self._calculate_max_drawdown(prices))
        
        if max_dd == 0:
            return 0.0
        
        calmar = annual_return / max_dd
        return float(calmar)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calcul maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        cummax = prices.expanding().max()
        drawdowns = (prices / cummax - 1)
        return float(drawdowns.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcul Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        if volatility == 0:
            return 0.0
        
        return float(excess_returns / volatility)
    
    def _calculate_composite_risk_score(self, metrics: Dict[str, float]) -> float:
        """
        Score de risque composite basé sur CVaR, Ulcer, etc.
        Score [0,1] où 0 = très risqué, 1 = peu risqué
        """
        # Normalisation CVaR (plus négatif = plus risqué)
        cvar_score = max(0, min(1, 1 + metrics['cvar_95'] / 0.1))
        
        # Normalisation Ulcer Index (plus élevé = plus risqué)
        ulcer_score = max(0, min(1, 1 - metrics['ulcer_index'] / 15))
        
        # Normalisation volatilité
        vol_score = max(0, min(1, 1 - metrics['volatility'] / 0.5))
        
        # Score composite pondéré
        composite_score = (
            0.4 * cvar_score +      # CVaR poids principal
            0.3 * ulcer_score +     # Ulcer index
            0.2 * vol_score +       # Volatilité
            0.1 * max(0, min(1, metrics['calmar_ratio'] / 3))  # Calmar bonus
        )
        
        return float(composite_score)
    
    def _determine_risk_level(self, cvar_metrics: Dict, ulcer_metrics: Dict) -> str:
        """Déterminer niveau de risque global"""
        cvar_95 = abs(cvar_metrics.get('cvar_95', 0))
        ulcer = ulcer_metrics.get('ulcer_index', 5)
        
        if cvar_95 > 0.10 or ulcer > 12:
            return "HIGH"
        elif cvar_95 > 0.05 or ulcer > 8:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def _get_cached_returns(self, symbol: str, days: int = 252) -> Optional[pd.Series]:
        """Récupère returns avec cache"""
        cache_key = f"{symbol}_returns"
        current_time = datetime.now()
        
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if (current_time - timestamp).total_seconds() < self._cache_ttl:
                return cached_data
        
        try:
            prices = await self._get_cached_prices(symbol, days + 1)
            if prices is None or len(prices) < 2:
                return None
            
            returns = prices.pct_change().dropna()
            self._cache[cache_key] = (returns, current_time)
            return returns
            
        except Exception as e:
            logger.error(f"Failed to get returns for {symbol}: {e}")
            return None
    
    async def _get_cached_prices(self, symbol: str, days: int = 252) -> Optional[pd.Series]:
        """Récupère prix avec cache"""
        cache_key = f"{symbol}_prices"
        current_time = datetime.now()
        
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if (current_time - timestamp).total_seconds() < self._cache_ttl:
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if data.empty:
                return None
            
            prices = data['Close']
            self._cache[cache_key] = (prices, current_time)
            return prices
            
        except Exception as e:
            logger.error(f"Failed to get prices for {symbol}: {e}")
            return None
    
    async def _calculate_sector_exposure(self, symbol: str, target_weight: float, 
                                       portfolio: Dict[str, float]) -> float:
        """Calcul exposition sectorielle (simplifié)"""
        # Simplified - assume tech sector for major stocks
        tech_symbols = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'META', 'AMZN'}
        
        if symbol in tech_symbols:
            # Ajouter exposition existante du secteur tech
            tech_exposure = sum(
                weight for sym, weight in portfolio.items() 
                if sym in tech_symbols
            )
            return tech_exposure + target_weight
        
        return target_weight  # Assume autre secteur
    
    def _default_risk_assessment(self) -> Dict[str, Any]:
        """Assessment par défaut en cas d'erreur"""
        return {
            'cvar_95': -0.05,
            'ulcer_index': 5.0,
            'calmar_ratio': 1.0,
            'risk_level': 'MEDIUM',
            'error': 'Insufficient data'
        }
    
    def _default_asset_risk(self, symbol: str) -> Dict[str, Any]:
        """Risk assessment par défaut pour un actif"""
        return {
            'symbol': symbol,
            'cvar_95': -0.05,
            'ulcer_index': 5.0,
            'risk_score': 0.5,
            'error': 'Insufficient data'
        }
    
    def _calculate_portfolio_returns(self, returns_data: Dict[str, pd.Series], 
                                   portfolio: Dict[str, float]) -> pd.Series:
        """Calcul des returns du portfolio pondéré"""
        # Aligner toutes les séries sur les mêmes dates
        all_returns = pd.DataFrame(returns_data)
        all_returns = all_returns.dropna()
        
        # Calcul returns pondérés
        portfolio_returns = pd.Series(0.0, index=all_returns.index)
        for symbol, weight in portfolio.items():
            if symbol in all_returns.columns and weight > 0:
                portfolio_returns += all_returns[symbol] * weight
        
        return portfolio_returns
    
    def _calculate_portfolio_prices(self, returns_data: Dict[str, pd.Series], 
                                  portfolio: Dict[str, float]) -> pd.Series:
        """Calcul des prix du portfolio (pour Ulcer/Calmar)"""
        portfolio_returns = self._calculate_portfolio_returns(returns_data, portfolio)
        # Créer série de prix cumulés (base 100)
        portfolio_prices = (1 + portfolio_returns).cumprod() * 100
        return portfolio_prices
    
    def _apply_stress_scenario(self, returns_data: Dict[str, pd.Series], 
                             scenario: Dict[str, Any]) -> pd.Series:
        """Applique scénario de stress aux returns"""
        portfolio_returns = self._calculate_portfolio_returns(returns_data, {symbol: 1/len(returns_data) for symbol in returns_data.keys()})
        
        # Appliquer multiplicateurs de stress
        vol_mult = scenario.get('vol_multiplier', 1.0)
        stressed_returns = portfolio_returns * vol_mult
        
        return stressed_returns

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Métriques de performance de l'agent de risque"""
        return {
            'agent_name': 'EnhancedRisk',
            'metrics': ['CVaR', 'TVaR', 'Ulcer_Index', 'Calmar_Ratio'],
            'cache_size': len(self._cache),
            'confidence_levels': self.confidence_levels,
            'architecture': 'enhanced_risk_metrics'
        }


# Test rapide
async def test_enhanced_risk():
    """Test de l'agent de risque amélioré"""
    agent = EnhancedRiskAgent()
    
    # Test portfolio simple
    portfolio = {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.5}
    
    print("🛡️ Testing Enhanced Risk Agent:")
    
    # Test assessment portfolio
    risk_assessment = await agent.assess_portfolio_risk_cvar(portfolio)
    print(f"Portfolio CVaR 95%: {risk_assessment.get('cvar_95', 0):.2%}")
    print(f"Risk Level: {risk_assessment.get('risk_level', 'Unknown')}")
    
    # Test asset individuel
    asset_risk = await agent.assess_single_asset_risk('AAPL')
    print(f"AAPL Risk Score: {asset_risk.get('risk_score', 0):.2f}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_risk())