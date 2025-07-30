# AlphaBot Makefile - Automatisation des tâches de développement

.PHONY: help setup install test lint format type-check clean run-tests coverage dev-install docker-redis

# Variables
PYTHON := python
POETRY := poetry
PYTEST := pytest
BLACK := black
RUFF := ruff
MYPY := mypy

# Couleurs pour l'affichage
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Affiche l'aide
	@echo "$(BLUE)AlphaBot Multi-Agent Trading System$(RESET)"
	@echo "$(YELLOW)Commandes disponibles:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

setup: ## Configuration initiale complète du projet
	@echo "$(BLUE)🚀 Configuration d'AlphaBot...$(RESET)"
	@$(MAKE) install
	@$(MAKE) dev-install
	@$(MAKE) docker-redis
	@echo "$(GREEN)✅ Setup terminé !$(RESET)"

install: ## Installe les dépendances avec Poetry
	@echo "$(YELLOW)📦 Installation des dépendances...$(RESET)"
	$(POETRY) install

dev-install: ## Installe les dépendances de développement
	@echo "$(YELLOW)🔧 Installation des outils de développement...$(RESET)"
	$(POETRY) install --with dev

test: ## Lance tous les tests
	@echo "$(YELLOW)🧪 Exécution des tests...$(RESET)"
	$(POETRY) run $(PYTEST)

test-verbose: ## Lance les tests en mode verbose
	@echo "$(YELLOW)🧪 Tests en mode verbose...$(RESET)"
	$(POETRY) run $(PYTEST) -v -s

test-cov: ## Lance les tests avec couverture
	@echo "$(YELLOW)📊 Tests avec couverture de code...$(RESET)"
	$(POETRY) run $(PYTEST) --cov=alphabot --cov-report=html --cov-report=term

test-risk: ## Lance uniquement les tests du Risk Agent
	@echo "$(YELLOW)⚠️  Tests Risk Agent...$(RESET)"
	$(POETRY) run $(PYTEST) tests/test_risk_agent.py -v

lint: ## Vérifie le code avec ruff
	@echo "$(YELLOW)🔍 Vérification du code avec ruff...$(RESET)"
	$(POETRY) run $(RUFF) check alphabot tests

lint-fix: ## Corrige automatiquement les erreurs de lint
	@echo "$(YELLOW)🔧 Correction automatique avec ruff...$(RESET)"
	$(POETRY) run $(RUFF) check --fix alphabot tests

format: ## Formate le code avec black
	@echo "$(YELLOW)✨ Formatage du code avec black...$(RESET)"
	$(POETRY) run $(BLACK) alphabot tests

format-check: ## Vérifie le formatage sans modifier
	@echo "$(YELLOW)📋 Vérification du formatage...$(RESET)"
	$(POETRY) run $(BLACK) --check alphabot tests

type-check: ## Vérifie les types avec mypy
	@echo "$(YELLOW)🔍 Vérification des types avec mypy...$(RESET)"
	$(POETRY) run $(MYPY) alphabot

quality: ## Lance tous les contrôles qualité (lint + format + types)
	@echo "$(BLUE)🎯 Contrôles qualité complets...$(RESET)"
	@$(MAKE) lint
	@$(MAKE) format-check
	@$(MAKE) type-check
	@echo "$(GREEN)✅ Contrôles qualité terminés !$(RESET)"

docker-redis: ## Lance Redis avec Docker
	@echo "$(YELLOW)🐳 Démarrage de Redis...$(RESET)"
	docker run -d --name alphabot-redis -p 6379:6379 redis:7-alpine redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru

docker-redis-stop: ## Arrête Redis
	@echo "$(YELLOW)🛑 Arrêt de Redis...$(RESET)"
	docker stop alphabot-redis || true
	docker rm alphabot-redis || true

stress-test: ## Lance le test de stress (600 signaux/10min)
	@echo "$(YELLOW)⚡ Test de stress en cours...$(RESET)"
	$(POETRY) run python scripts/stress_test.py

run-risk-agent: ## Lance le Risk Agent en mode standalone
	@echo "$(YELLOW)🤖 Démarrage Risk Agent...$(RESET)"
	$(POETRY) run python -c "from alphabot.agents.risk.risk_agent import RiskAgent; agent = RiskAgent(); print('Risk Agent ready')"

notebook: ## Lance Jupyter Notebook
	@echo "$(YELLOW)📊 Démarrage Jupyter...$(RESET)"
	$(POETRY) run jupyter notebook notebooks/

streamlit: ## Lance le dashboard Streamlit
	@echo "$(YELLOW)📈 Démarrage Streamlit Dashboard...$(RESET)"
	$(POETRY) run streamlit run app.py

clean: ## Nettoie les fichiers temporaires
	@echo "$(YELLOW)🧹 Nettoyage...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

clean-all: clean docker-redis-stop ## Nettoyage complet incluant Docker
	@echo "$(GREEN)✅ Nettoyage complet terminé !$(RESET)"

check: quality test ## Lance tous les contrôles (qualité + tests)
	@echo "$(GREEN)✅ Tous les contrôles sont passés !$(RESET)"

ci: ## Pipeline CI (utilisé par GitHub Actions)
	@echo "$(BLUE)🚀 Pipeline CI...$(RESET)"
	@$(MAKE) install
	@$(MAKE) quality
	@$(MAKE) test-cov

# Phase 1 - Issue P1-S1-1
init-phase1: ## Initialise Phase 1 selon plan_détaillé.md
	@echo "$(BLUE)📋 Phase 1 - Avant-projet (S1-S2)$(RESET)"
	@echo "$(GREEN)✅ Specs créées: docs/specs.md$(RESET)"
	@echo "$(GREEN)✅ Risk policy: risk_policy.yaml$(RESET)"
	@echo "$(GREEN)✅ Risk Agent: alphabot/agents/risk/$(RESET)"
	@echo "$(YELLOW)📝 TODO: Compléter personal_preferences dans risk_policy.yaml$(RESET)"

# Phase 2 - Environnement (S3-S4)  
init-phase2: setup docker-redis ## Initialise Phase 2 - Environnement
	@echo "$(BLUE)⚙️  Phase 2 - Initialisation (S3-S4)$(RESET)"
	@echo "$(GREEN)✅ Poetry configuré$(RESET)"
	@echo "$(GREEN)✅ Redis démarré$(RESET)"
	@echo "$(YELLOW)📝 Prêt pour développement agents Technical & Sentiment$(RESET)"

status: ## Affiche le statut du projet
	@echo "$(BLUE)📊 Statut AlphaBot$(RESET)"
	@echo "$(YELLOW)Python:$(RESET) $(shell python --version)"
	@echo "$(YELLOW)Poetry:$(RESET) $(shell poetry --version)"
	@echo "$(YELLOW)Redis:$(RESET) $(shell docker ps --filter name=alphabot-redis --format "table {{.Status}}" | tail -n +2 || echo "Non démarré")"
	@echo "$(YELLOW)Tests:$(RESET) $(shell find tests -name "*.py" | wc -l) fichiers"
	@echo "$(YELLOW)Agents:$(RESET) $(shell find alphabot/agents -name "*.py" | grep -v __pycache__ | wc -l) fichiers"

help-dev: ## Aide pour développeurs
	@echo "$(BLUE)🔧 Guide développeur AlphaBot$(RESET)"
	@echo ""
	@echo "$(YELLOW)Workflow recommandé:$(RESET)"
	@echo "  1. $(GREEN)make setup$(RESET)           - Configuration initiale"
	@echo "  2. $(GREEN)make test$(RESET)            - Vérifier que tout fonctionne"
	@echo "  3. $(GREEN)make quality$(RESET)         - Contrôles code avant commit"
	@echo "  4. $(GREEN)git add . && git commit$(RESET) - Commit des changements"
	@echo ""
	@echo "$(YELLOW)Pendant le développement:$(RESET)"
	@echo "  - $(GREEN)make test-risk$(RESET)        - Tests rapides Risk Agent"
	@echo "  - $(GREEN)make lint-fix$(RESET)         - Correction automatique code"
	@echo "  - $(GREEN)make notebook$(RESET)         - Exploration données"
	@echo ""
	@echo "$(YELLOW)Avant production:$(RESET)"
	@echo "  - $(GREEN)make stress-test$(RESET)      - Test de charge"
	@echo "  - $(GREEN)make check$(RESET)           - Validation complète"