# 🤖 AlphaBot Multi-Agent Trading System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-dependency-blue.svg)](https://python-poetry.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-multi--agent-green.svg)](https://crewai.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Un système de trading algorithmique multi-agents utilisant CrewAI pour prendre des décisions d'investissement autonomes sur les marchés actions.

## 🎯 Objectifs

- **Sharpe ratio** ≥ 1.5
- **Drawdown max** ≤ 15%
- **Hit ratio** ≥ 60%
- **Uptime** ≥ 99.5%

## 🏗️ Architecture

### Agents principaux

- **Data Agent** : Ingestion et validation des données
- **Technical Agent** : Analyse technique (EMA, RSI, ATR)
- **Fundamental Agent** : Analyse fondamentale (P/E, Piotroski-F)
- **Sentiment Agent** : NLP avec FinBERT fine-tuné
- **Risk Agent** : VaR, Expected Shortfall, EVT
- **Optimization Agent** : Hierarchical Risk Parity (HRP)
- **Execution Agent** : Ordres via Interactive Brokers

### Stack technologique

- **Orchestration** : CrewAI
- **Données** : DuckDB, Redis, Polars
- **ML** : Riskfolio-Lib, FinBERT, FinRL
- **Broker** : Interactive Brokers (ib_insync)
- **Monitoring** : Streamlit, MLflow

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
```

## 📋 Commandes principales

```bash
# Développement
make help              # Affiche toutes les commandes
make setup             # Installation complète
make test              # Tests unitaires
make quality           # Contrôles qualité (lint + format + types)
make stress-test       # Test de charge (600 signaux/10min)

# Agents
make run-risk-agent    # Risk Agent standalone
make notebook          # Jupyter pour exploration
make streamlit         # Dashboard de monitoring

# Base de données
make docker-redis      # Démarrer Redis
make docker-redis-stop # Arrêter Redis
```

## 🔧 Configuration

### 1. Politique de risque

Éditer `risk_policy.yaml` pour vos préférences :

```yaml
personal_preferences:
  risk_tolerance: "moderate"          # conservative/moderate/aggressive
  preferred_sectors: 
    - "Information Technology"
    - "Health Care"
  investment_horizon_months: 6
  additional_constraints:
    - "No crypto exposure"
    - "ESG screening preferred"
```

### 2. Variables d'environnement

Créer `.env` :

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
FINANCIAL_MODELING_PREP_KEY=your_key_here

# IBKR
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

## 📊 Univers d'investissement

- **S&P 500** (500 titres US)
- **STOXX 600** (600 titres Europe)
- **Nikkei 225** (225 titres Asie)
- **Total** : ~1300 actifs
- **Exclusions** : Penny stocks (<$5), volume <1M$/jour

## 🧪 Tests et qualité

```bash
# Tests complets avec couverture
make test-cov

# Tests spécifiques Risk Agent
make test-risk

# Contrôles qualité
make lint              # Ruff linting
make format            # Black formatting  
make type-check        # MyPy type checking
make check             # Tout en une fois
```

## 🗓️ Roadmap (9 mois)

### Phase 1 - Avant-projet (S1-S2) ✅
- [x] Spécifications techniques
- [x] Politique de risque
- [x] Architecture agents

### Phase 2 - Initialisation (S3-S4)
- [ ] Environnement Poetry/Docker/DVC
- [ ] Technical Agent (EMA 20/50, ATR)
- [ ] Sentiment Agent (FinBERT)
- [ ] Test de charge 600 signaux/10min

### Phase 3 - Planification (S5-S8)
- [ ] Roadmap Gantt automatisée
- [ ] Documentation ressources
- [ ] Registre de risques

### Phase 4 - Exécution (S9-S24)
- [ ] Signal HUB Redis/CrewAI
- [ ] Agents Fundamental, Optimization
- [ ] FinRL-DeepSeek intégration
- [ ] Agent d'exécution IBKR

### Phase 5 - Contrôle (S25-S32)
- [ ] Backtests vectorbt 10 ans
- [ ] Paper trading 3 mois
- [ ] Dashboard Streamlit

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