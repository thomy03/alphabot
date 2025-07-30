"""
AlphaBot Backtesting Engine avec vectorbt
Système de backtest pour validation des stratégies multi-agents
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from alphabot.core.config import get_settings
from alphabot.core.crew_orchestrator import CrewOrchestrator
from alphabot.core.signal_hub import Signal, SignalType, SignalPriority

@dataclass
class BacktestConfig:
    """Configuration du backtest"""
    start_date: str = "2014-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 100000.0
    rebalance_frequency: str = "1D"  # Daily rebalancing
    commission: float = 0.001  # 0.1% commission
    slippage_bps: int = 5  # 5 bps slippage
    
    # Univers d'investissement
    universe: List[str] = None
    benchmark: str = "SPY"
    
    # Contraintes de risque
    max_position_size: float = 0.05
    max_sector_exposure: float = 0.30
    max_leverage: float = 1.0
    
    def __post_init__(self):
        if self.universe is None:
            # Top 20 S&P 500 par capitalisation
            self.universe = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                "TSLA", "META", "UNH", "XOM", "JNJ",
                "JPM", "V", "PG", "MA", "HD",
                "CVX", "ABBV", "LLY", "BAC", "AVGO"
            ]

@dataclass
class BacktestResult:
    """Résultats du backtest"""
    portfolio_value: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    
    # Métriques de performance
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Métriques de trading
    total_trades: int
    win_rate: float
    avg_trade_return: float
    avg_holding_period: float
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float

class BacktestingEngine:
    """Engine de backtesting pour AlphaBot"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.settings = get_settings()
        self.data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.Series] = None
        
        # Cache pour les données
        self.data_cache = {}
        
    async def load_historical_data(self) -> pd.DataFrame:
        """Charge les données historiques pour l'univers"""
        print(f"📊 Chargement données historiques ({self.config.start_date} à {self.config.end_date})...")
        
        all_symbols = self.config.universe + [self.config.benchmark]
        
        try:
            # Télécharger toutes les données en une fois
            data = yf.download(
                all_symbols,
                start=self.config.start_date,
                end=self.config.end_date,
                auto_adjust=True,
                progress=False
            )
            
            if len(all_symbols) == 1:
                # Un seul symbole
                prices = data['Close'].to_frame()
                prices.columns = all_symbols
            else:
                # Plusieurs symboles
                prices = data['Close']
            
            # Nettoyer les données
            prices = prices.dropna(how='all')
            
            # Séparer benchmark
            self.benchmark_data = prices[self.config.benchmark]
            self.data = prices[self.config.universe]
            
            print(f"✅ Données chargées: {len(self.data)} jours, {len(self.data.columns)} actifs")
            print(f"   Période: {self.data.index[0].date()} à {self.data.index[-1].date()}")
            print(f"   Données manquantes: {self.data.isnull().sum().sum()}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Erreur chargement données: {e}")
            raise
    
    def _create_signals_from_agents(self, date: pd.Timestamp, prices: pd.Series) -> Dict[str, float]:
        """Simule les signaux des agents pour une date donnée"""
        
        # Simuler les signaux (en Phase 5, on intégrera les vrais agents)
        signals = {}
        
        # Signal technique simple : momentum 20 jours
        if len(self.data.loc[:date]) >= 20:
            returns_20d = self.data.loc[:date].tail(20).pct_change().mean()
            
            for symbol in self.config.universe:
                if symbol in prices.index and not pd.isna(prices[symbol]):
                    momentum = returns_20d[symbol] if symbol in returns_20d else 0
                    
                    # Signal strength basé sur momentum
                    if momentum > 0.02:  # +2%
                        signals[symbol] = 0.05  # Position 5%
                    elif momentum > 0.01:  # +1%
                        signals[symbol] = 0.03  # Position 3%
                    elif momentum < -0.02:  # -2%
                        signals[symbol] = -0.02  # Short 2%
                    else:
                        signals[symbol] = 0.01  # Position neutre
        
        return signals
    
    def _apply_risk_constraints(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """Applique les contraintes de risque"""
        
        # Limiter taille position individuelle
        for symbol in target_weights:
            target_weights[symbol] = np.clip(
                target_weights[symbol], 
                -self.config.max_position_size, 
                self.config.max_position_size
            )
        
        # Normaliser pour éviter sur-leverage
        total_exposure = sum(abs(w) for w in target_weights.values())
        if total_exposure > self.config.max_leverage:
            scale_factor = self.config.max_leverage / total_exposure
            target_weights = {k: v * scale_factor for k, v in target_weights.items()}
        
        return target_weights
    
    async def run_backtest(self) -> BacktestResult:
        """Exécute le backtest complet"""
        print("🚀 Démarrage backtest AlphaBot...")
        
        if self.data is None:
            await self.load_historical_data()
        
        # Initialisation
        dates = self.data.index
        portfolio_values = []
        positions_history = []
        trades_history = []
        
        current_positions = {symbol: 0.0 for symbol in self.config.universe}
        portfolio_value = self.config.initial_capital
        cash = self.config.initial_capital
        
        print(f"📈 Simulation sur {len(dates)} jours...")
        
        for i, date in enumerate(dates):
            if i % 252 == 0:  # Progress chaque année
                print(f"   Année {date.year}: Portfolio ${portfolio_value:,.0f}")
            
            current_prices = self.data.loc[date]
            
            # Générer signaux
            target_weights = self._create_signals_from_agents(date, current_prices)
            target_weights = self._apply_risk_constraints(target_weights)
            
            # Calculer positions cibles
            target_positions = {}
            for symbol, weight in target_weights.items():
                if symbol in current_prices and not pd.isna(current_prices[symbol]):
                    target_value = portfolio_value * weight
                    target_qty = target_value / current_prices[symbol]
                    target_positions[symbol] = target_qty
                else:
                    target_positions[symbol] = 0.0
            
            # Exécuter trades
            for symbol in self.config.universe:
                if symbol in target_positions and symbol in current_prices:
                    current_qty = current_positions.get(symbol, 0.0)
                    target_qty = target_positions[symbol]
                    trade_qty = target_qty - current_qty
                    
                    if abs(trade_qty) > 0.001:  # Seuil minimal
                        price = current_prices[symbol]
                        if not pd.isna(price) and price > 0:
                            # Appliquer slippage
                            execution_price = price * (1 + self.config.slippage_bps / 10000 * np.sign(trade_qty))
                            
                            trade_value = trade_qty * execution_price
                            commission = abs(trade_value) * self.config.commission
                            
                            # Vérifier liquidité
                            if cash >= trade_value + commission:
                                current_positions[symbol] = target_qty
                                cash -= (trade_value + commission)
                                
                                # Enregistrer trade
                                trades_history.append({
                                    'date': date,
                                    'symbol': symbol,
                                    'quantity': trade_qty,
                                    'price': execution_price,
                                    'value': trade_value,
                                    'commission': commission
                                })
            
            # Calculer valeur portfolio
            positions_value = sum(
                qty * current_prices.get(symbol, 0) 
                for symbol, qty in current_positions.items()
                if not pd.isna(current_prices.get(symbol, np.nan))
            )
            portfolio_value = cash + positions_value
            
            portfolio_values.append(portfolio_value)
            
            # Sauvegarder positions
            position_record = {'date': date, 'cash': cash, 'total_value': portfolio_value}
            position_record.update(current_positions)
            positions_history.append(position_record)
        
        # Créer DataFrames résultats
        portfolio_series = pd.Series(portfolio_values, index=dates)
        returns_series = portfolio_series.pct_change().dropna()
        
        positions_df = pd.DataFrame(positions_history).set_index('date')
        trades_df = pd.DataFrame(trades_history)
        if not trades_df.empty:
            trades_df = trades_df.set_index('date')
        
        # Calculer métriques
        result = self._calculate_metrics(portfolio_series, returns_series, positions_df, trades_df)
        
        print("✅ Backtest terminé!")
        print(f"📊 Performance: {result.total_return:.1%} total, {result.annualized_return:.1%} annualisé")
        print(f"📈 Sharpe: {result.sharpe_ratio:.2f}, Max DD: {result.max_drawdown:.1%}")
        
        return result
    
    def _calculate_metrics(
        self, 
        portfolio_series: pd.Series,
        returns_series: pd.Series, 
        positions_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> BacktestResult:
        """Calcule les métriques de performance"""
        
        # Performance globale
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        days = (portfolio_series.index[-1] - portfolio_series.index[0]).days
        annualized_return = (1 + total_return) ** (365.25 / days) - 1
        
        # Volatilité et Sharpe
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Métriques de trading
        total_trades = len(trades_df) if not trades_df.empty else 0
        
        if not trades_df.empty and 'value' in trades_df.columns:
            # Grouper par position (buy/sell pairs)
            trade_returns = []
            # Simplified: calculate per-trade return
            avg_trade_return = trades_df['value'].mean() / self.config.initial_capital
            win_rate = (trades_df['value'] > 0).mean() if len(trades_df) > 0 else 0
            avg_holding_period = 5.0  # Estimation
        else:
            avg_trade_return = 0.0
            win_rate = 0.0
            avg_holding_period = 0.0
        
        # Benchmark comparison
        benchmark_returns = self.benchmark_data.pct_change().dropna()
        benchmark_total_return = (self.benchmark_data.iloc[-1] / self.benchmark_data.iloc[0]) - 1
        
        # Alpha/Beta via regression
        common_dates = returns_series.index.intersection(benchmark_returns.index)
        if len(common_dates) > 10:
            portfolio_aligned = returns_series.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]
            
            covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
            benchmark_var = np.var(benchmark_aligned)
            beta = covariance / benchmark_var if benchmark_var > 0 else 0
            
            benchmark_ann_return = benchmark_total_return * (365.25 / days) 
            alpha = annualized_return - beta * benchmark_ann_return
            
            # Information ratio
            active_returns = portfolio_aligned - benchmark_aligned
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
        else:
            alpha = beta = information_ratio = 0.0
        
        return BacktestResult(
            portfolio_value=portfolio_series,
            returns=returns_series,
            positions=positions_df,
            trades=trades_df,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            avg_holding_period=avg_holding_period,
            benchmark_return=benchmark_total_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio
        )
    
    def save_results(self, result: BacktestResult, output_dir: str = "backtests/results"):
        """Sauvegarde les résultats"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder données
        result.portfolio_value.to_csv(output_path / f"portfolio_value_{timestamp}.csv")
        result.positions.to_csv(output_path / f"positions_{timestamp}.csv")
        
        if not result.trades.empty:
            result.trades.to_csv(output_path / f"trades_{timestamp}.csv")
        
        # Rapport de synthèse
        report = {
            'config': self.config.__dict__,
            'metrics': {
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'benchmark_return': result.benchmark_return,
                'alpha': result.alpha,
                'beta': result.beta,
                'information_ratio': result.information_ratio
            }
        }
        
        import json
        with open(output_path / f"summary_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Résultats sauvegardés dans {output_path}")

def create_backtest_engine(
    start_date: str = "2014-01-01",
    end_date: str = "2024-01-01",
    initial_capital: float = 100000.0
) -> BacktestingEngine:
    """Factory pour créer un engine de backtest"""
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    return BacktestingEngine(config)