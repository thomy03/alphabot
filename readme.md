# 🤖 AlphaBot Multi-Agent Trading System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-dependency-blue.svg)](https://python-poetry.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-multi--agent-green.svg)](https://crewai.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Documentation** : [Voir l'index complet de la documentation](docs/INDEX_DOCUMENTATION.md)

Un système de trading algorithmique multi-agents utilisant CrewAI pour prendre des décisions d'investissement autonomes sur les marchés actions.

## 🎯 Objectifs

- **Sharpe ratio** ≥ 1.5
- **Drawdown max** ≤ 15%
- **Hit ratio** ≥ 60%
- **Uptime** ≥ 99.5%

## 🏗️ Architecture

### Agents principaux ✅ IMPLÉMENTÉS

- **Risk Agent** ✅ : VaR 95%, Expected Shortfall, EVT, stress tests (`alphabot/agents/risk/`)
- **Technical Agent** ✅ : EMA 20/50, RSI, ATR, signaux croisements (`alphabot/agents/technical/`)
- **Sentiment Agent** ✅ : FinBERT NLP, analyse sentiment news (`alphabot/agents/sentiment/`)
- **Fundamental Agent** ✅ : P/E, ROE, Piotroski F-Score, Altman Z-Score (`alphabot/agents/fundamental/`)
- **Optimization Agent** ✅ : HRP, Risk Parity, Equal Weight (`alphabot/agents/optimization/`)
- **Execution Agent** ✅ : Simulation IBKR, gestion ordres, risk management (`alphabot/agents/execution/`)

### Stack technologique ✅ IMPLÉMENTÉ

- **Orchestration** ✅ : CrewAI multi-agents (`alphabot/core/crew_orchestrator.py`)
- **Communication** ✅ : Redis Signal HUB pub/sub (`alphabot/core/signal_hub.py`)
- **Données** ✅ : Simulation + APIs (Alpha Vantage, FinancialModelingPrep)
- **ML** ✅ : FinBERT sentiment, Piotroski scoring, HRP optimization
- **Broker** ✅ : Interactive Brokers simulation (Phase 5: ib_insync réel)
- **Config** ✅ : Pydantic settings, YAML policies (`alphabot/core/config.py`)

## 🚀 Installation rapide

```bash
# 1. Cloner le projet
git clone <repo-url>
cd Tradingbot_V2

# 2. Setup complet
make setup

# 3. Lancer les tests
make test

# 4. Démarrer le dashboard
make streamlit

# 5. Entraînement ML/DL via Google Colab  
# Ouvrir `ALPHABOT_ML_TRAINING_COLAB.ipynb` sur Google Colab et suivre `docs/README_ENTRAINEMENT_COLAB.md`
```

## 📋 Commandes principales

```bash
# Développement
make help              # Affiche toutes les commandes
make setup             # Installation complète
make test              # Tests unitaires
make quality           # Contrôles qualité (lint + format + types)
make stress-test       # Test de charge (600 signaux/10min)

# Phase 5 - Backtesting & Paper Trading
python scripts/test_backtesting_engine.py      # Test framework backtest
python scripts/run_full_backtest_10years.py    # Backtest complet 10 ans
python scripts/test_paper_trading.py           # Test paper trading
python scripts/benchmark_comparison.py         # Comparaison benchmarks
python scripts/phase5_demo.py                  # Démo complète Phase 5

# Dashboard
python scripts/run_dashboard.py                # Dashboard Streamlit
# Ou directement: streamlit run alphabot/dashboard/streamlit_app.py

# Agents
make run-risk-agent    # Risk Agent standalone
make notebook          # Jupyter pour exploration

# Base de données
make docker-redis      # Démarrer Redis
make docker-redis-stop # Arrêter Redis
```

## 🔧 Configuration ✅ IMPLÉMENTÉE

### 1. Politique de risque ✅

Le fichier `risk_policy.yaml` est configuré avec :

```yaml
# Configuration développée et testée ✅
personal_preferences:
  risk_tolerance: "moderate"
  preferred_sectors: ["Information Technology", "Health Care"]
  investment_horizon_months: 6
  
risk_limits:
  max_position_size: 0.05      # 5% max par titre
  max_sector_exposure: 0.30    # 30% max par secteur  
  max_daily_var_95: 0.03       # VaR 95% max 3%
  max_drawdown: 0.15           # 15% drawdown max
```

### 2. Variables d'environnement ✅

Configuration centralisée dans `alphabot/core/config.py` :

```python
# Configuration Pydantic développée ✅
class Settings(BaseSettings):
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # APIs (optionnelles en Phase 4)
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    
    # IBKR
    ibkr_host: str = "localhost"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    
    # Trading (implémenté)
    max_position_size: float = 0.05
    max_sector_exposure: float = 0.30
```

## 📊 Univers d'investissement

- **S&P 500** (500 titres US)
- **STOXX 600** (600 titres Europe)
- **Nikkei 225** (225 titres Asie)
- **Total** : ~1300 actifs
- **Exclusions** : Penny stocks (<$5), volume <1M$/jour

## 🧪 Tests et qualité ✅ IMPLÉMENTÉS

```bash
# Tests développés et validés ✅
python scripts/test_crew_no_redis.py        # Tests CrewAI + agents (4/4 ✅)
python scripts/test_optimization_agent.py   # Tests HRP + Risk Parity (4/4 ✅)  
python scripts/test_execution_agent_fixed.py # Tests IBKR simulation (3/5 ✅)
python scripts/test_full_integration.py     # Tests intégration complète (3/4 ✅)

# Tests par agent ✅
make test-risk         # Risk Agent (VaR, ES)
make test-technical    # Technical Agent (EMA, RSI)
make test-sentiment    # Sentiment Agent (FinBERT)
make test-fundamental  # Fundamental Agent (Piotroski)

# Qualité code ✅
make lint              # Ruff linting
make format            # Black formatting  
make type-check        # MyPy type checking
```

## 🗓️ Roadmap (9 mois)

### Phase 1 - Avant-projet (S1-S2) ✅ TERMINÉE
- [x] Spécifications techniques (`docs/specs.md`)
- [x] Politique de risque (`risk_policy.yaml`)
- [x] Risk Agent complet avec tests (`alphabot/agents/risk/`)
- [x] Environment setup (Poetry, Makefile, Git)

### Phase 2 - Initialisation (S3-S4) ✅ TERMINÉE
- [x] Environnement Poetry/Docker/DVC (`pyproject.toml`, `Makefile`)
- [x] Technical Agent (EMA 20/50, ATR) (`alphabot/agents/technical/`)
- [x] Sentiment Agent (FinBERT) (`alphabot/agents/sentiment/`)
- [x] Test de charge 600 signaux/10min (`scripts/stress_test.py`)

### Phase 3 - Planification (S5-S8) ✅ TERMINÉE
- [x] Roadmap Gantt automatisée (`planning.yml` + script génération)
- [x] Documentation ressources (`docs/resources.md`)
- [x] Registre de risques (`risk_register.csv` + analyse)
- [x] Rétrospective Sprint-0 (`docs/retro_P3.md`)

### Phase 4 - Exécution (S9-S24) ✅ TERMINÉE
- [x] Signal HUB Redis/CrewAI (`alphabot/core/signal_hub.py`)
- [x] Agents Fundamental, Optimization (`alphabot/agents/fundamental/`, `alphabot/agents/optimization/`)
- [x] CrewAI Orchestrator (`alphabot/core/crew_orchestrator.py`)
- [x] Agent d'exécution IBKR (`alphabot/agents/execution/`)

### Phase 5 - Contrôle (S25-S32) ✅ TERMINÉE
- [x] Backtests vectorbt 10 ans (données historiques réelles) - `alphabot/core/backtesting_engine.py`
- [x] Paper trading temps réel (simulation) - `alphabot/core/paper_trading.py`
- [x] Dashboard Streamlit monitoring - `alphabot/dashboard/streamlit_app.py`
- [x] Comparaison vs benchmarks - `scripts/benchmark_comparison.py`

### Phase 6 - Production (S33+)
- [ ] Go-live capital limité (≤10k€)
- [ ] Monitoring 24/7
- [ ] Amélioration continue

## 🛡️ Gestion des risques

### Limites strictes
- **Position max** : 5% par titre, 30% par secteur
- **VaR 95%** : Max 3% capital quotidien
- **Expected Shortfall** : Max 5% capital
- **Drawdown max** : 15%

### Stress tests
- Scénario COVID (vol x1.5, corr +0.2)
- Scénario inflation 2022
- Scénario crise 2008
- Scénario AI crash (vol x2.5)

## 📈 KPIs de monitoring

```bash
# Performance
- Sharpe ratio (30/90/252 jours)
- Information ratio vs S&P 500
- Hit ratio positions gagnantes
- Maximum drawdown

# Technique  
- Latence signaux (<200ms)
- Uptime agents (≥99.5%)
- Fill ratio (≥98%)
- Couverture données (≥95%)
```

## 🔗 Sources de données

### Prix & Volume
- Alpha Vantage (primary)
- Yahoo Finance (backup)
- FinancialModelingPrep (EOD/intraday)
- Finnhub (realtime websocket)

### Fondamentaux
- SEC filings via FinancialModelingPrep
- GitHub datasets open-source
- ESG via Hugging Face

### Sentiment
- Twitter/Reddit (limité 300 req/15min)
- Finnhub news API
- Financial datasets HF

## 🤝 Contribution

```bash
# 1. Fork et clone
git checkout -b feature/nouvelle-fonctionnalite

# 2. Développement
make setup
make test
make quality

# 3. Commit et PR
git add .
git commit -m "feat: nouvelle fonctionnalité"
git push origin feature/nouvelle-fonctionnalite
```

## 📄 Licence

MIT License - voir [LICENSE](LICENSE)

## 🆘 Support

- **Documentation** : `docs/specs.md`
- **Issues** : GitHub Issues
- **Questions** : Discussions GitHub

---

**⚠️ Disclaimer** : Ce système est destiné à des fins éducatives et de recherche. Le trading algorithmique comporte des risques. Ne jamais investir plus que ce que vous pouvez vous permettre de perdre.
