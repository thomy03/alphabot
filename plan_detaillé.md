Voici un **plan d’exécution “prêt-à-coder”** : il transforme votre feuille de route en **tâches unitaires, artefacts livrables, critères de réussite (“Definition of Done”) et check-lists hebdo**.
Les jalons restent ceux du planning à 9 mois, mais chaque phase est maintenant découpée en **sprints de deux semaines** avec la granularité suffisante pour ouvrir directement des issues GitHub.

---

## 0. Convention générale

| Élément                | Convention                                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Repo**               | `alphabot/` racine. Sous-dossiers : `data/`, `notebooks/`, `alphabot/agents/`, `scripts/`, `tests/`, `docs/`         |
| **Branches Git**       | `main`, `dev`, puis `feature/<phase>-<num>` (ex. `feature/2.2-data-ingestion`)                                       |
| **Issues**             | Numérotation “P <phase>-S <sprint>-<n>” (ex. `P2-S1-3`) + labels: `agent`, `bug`, `doc`, `infra`, `research`, `risk` |
| **CI**                 | GitHub Actions ― lint (`ruff`), tests (`pytest`), build wheels                                                       |
| **Definition of Done** | ➊ code mergé sur `dev`, ➋ tests ≥ 80 % coverage, ➌ artefact versionné (DVC ou docs), ➍ issue fermée                  |

---

## 1. Phase 1 – Avant-projet (S1-S2)

| Semaine | Livrable                         | Tâches unitaires                                                                                                                                 | DoD                       |
| ------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| **S1**  | **`docs/specs.md`**              | - Ouvrir issue `P1-S1-1` : remplir template (objectif, KPIs).<br>- Copier OKRs dans tableau.<br>- Valider univers : tickers S\&P 500 + STOXX600. | Doc push + revue PR       |
| **S1**  | **`risk_policy.yaml`**           | - Renseigner DD max, sizing, couverture.<br>- Ajouter PREMIER test YAML-schema dans `tests/`.                                                    | Fichier parse sans erreur |
| **S2**  | **`project_plan.xlsx`** (ou CSV) | - Dresser WBS (Work-Breakdown Structure) niveau tâche.<br>- Estimer charges (h/dev).                                                             | Fichier dans DVC          |

---

## 2. Phase 2 – Initialisation (S3-S4)

| Sprint | Livrables & artefacts                                                | Back-log détaillé                                                                                                                                                                                                                                      |
| ------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **S3** | *Environnement* :<br>`poetry.lock`, `docker-compose.yml`, `Makefile` | - Issue `P2-S3-1` : script `make setup` ➞ crée venv + installe CrewAI, Redis, Polars.<br>- `P2-S3-2` : Docker file Redis (ttl 24 h).<br>- `P2-S3-3` : config DVC remote (local).                                                                       |
| **S4** | *2 agents* + *pipeline de test charge*                               | - `alphabot/agents/technical.py` (EMA 20/50, ATR stop).<br>- `alphabot/agents/sentiment.py` (FinBERT HF).<br>- Script `scripts/stress_test.py` → génère 600 signaux/10 min et mesure latence.<br>- Benchmark stocké dans `docs/benchmarks/latency.md`. |

---

## 3. Phase 3 – Planification (S5-S8)

| Semaine | Objectif               | Tâches pratiques                                                                                                                                         |
| ------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **S5**  | Roadmap Gantt          | - Utiliser `gantt-lab` GitHub-action pour générer PNG à partir de fichier YAML (`planning.yml`).                                                         |
| **S6**  | Ressources             | - Créer `docs/resources.md` : CPU, RAM, temps dispo.<br>- Ajouter check list matériel (GPU ? SSD ?).                                                     |
| **S7**  | Gestion risques        | - Ouvrir `risk_register.csv` : colonnes ID, description, impact, prob, owner, mitigation.<br>- Première revue : API X, biais de survie, fail Redis, etc. |
| **S8**  | Sprint-0 rétrospective | - Document `docs/retro_P3.md` : ce qui a marché, à améliorer.                                                                                            |

---

## 4. Phase 4 – Exécution (S9-S24)

### Organisation en **sous-modules** CrewAI

```
alphabot/
└── agents/
    ├── data/
    ├── signals/
    ├── risk/
    └── execution/
```

### Tableau des Sprints (extrait)

| Sprint     | Épic                              | Tâches clés                                                                             | KPIs sprint             |
| ---------- | --------------------------------- | --------------------------------------------------------------------------------------- | ----------------------- |
| **S9**     | **Signal HUB v1**                 | - Créer broker d’événements Redis->CrewAI.<br>- Implémenter fusion WMA (poids manuels). | latence <150 ms         |
| **S10**    | **Fundamental Agent**             | - Scraper SEC / FinancialModelingPrep.<br>- Score Piotroski-F (unit-test sur AAPL).     | couverture 90 % S\&P500 |
| **S11**    | **Risk Agent**                    | - Calcul VaR 95%, ES 97.5% (Polars).<br>- YAML de limites auto checké par pytest.       | VaR runtime <50 ms      |
| **S12**    | **Optimizer v1**                  | - HRP via Riskfolio-Lib.<br>- Enregistrer poids dans DuckDB table `port_weights`.       | turnover scripté        |
| **S13-14** | **NLP Upgrade**                   | - Fine-tune FinBERT-ESG → HF Trainer.<br>- `sentiment_risk_overlay()` dans risk agent.  | F1 score ≥0.82          |
| **S15-16** | **FinRL-DeepSeek** (feature flag) | - Cloner branch, wrapper CrewAI.<br>- Bench Sharpe ∆ vs baseline.                       | +0.15 Sharpe            |
| **S17-18** | **R\&D-Agent-Quant**              | - Intégrer Ray backend.<br>- Auto-feature search 500 facteurs.                          | Search runtime ≤4 h     |
| **S19-20** | **EVT + CVaR**                    | - Batch mensuel : fit GPD, store params.                                                | ES erro < 5 %           |
| **S21-22** | **Execution Agent**               | - ib\_insync wrapper + fail-safe retry.<br>- Dry-run orders.                            | fill ratio ≥98 %        |
| **S23-24** | **Hardening & Docs**              | - Docstring 100 %, mkdocs-material site.<br>- Code freeze pour Phase 5.                 | test cov ≥85 %          |

---

## 5. Phase 5 – Contrôle & suivi (S25-S32)

| Sprint     | Action                 | Détail                                                                                                            |
| ---------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **S25**    | **Backtests vectorbt** | - Générer 10 ans, 5 000 scénarios param-grid (EMA windows, stop ATR).<br>- Résultats stockés `backtests.parquet`. |
| **S26**    | **Stress 2020 & 2022** | - Scénario COVID + inflation : simuler volatilité *x 1.5*, corrélations *+0.2*.                                   |
| **S27-29** | **Paper Trading**      | - Branch `paper-live` ➞ IBKR paper.<br>- Toute exécution loggée dans DuckDB `executions`.                         |
| **S30**    | **Dashboard v1**       | - Streamlit app `app.py` (drawdown, PnL, VaR).<br>- Deploy local `localhost:8501`.                                |
| **S31-32** | **Rétro & tuning**     | - Analyser hit-ratio, ES violations.<br>- Ajuster poids HUB et risk overlay.                                      |

---

## 6. Phase 6 – Clôture & production (≥ S33)

| Étape                     | Checklist Go-Live                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| **Cut-over**              | ▢ Clés API rotées / Vault ▢ Capital limité (≤ 10 k €) ▢ Exposition maxi 30 % par secteur          |
| **Run-book**              | ▢ Procédure restart Redis ▢ Procédures fail order ▢ Contact broker                                |
| **Post-mortem**           | template `docs/postmortem.md` auto-pré-rempli par GitHub Action après chaque drawdown >5 %.       |
| **Amélioration continue** | - Cron `scripts/retrain_weekly.sh` (feature flag RL).<br>- Étude migration LangGraph (>8 agents). |

---

## 7. Rituels “vibe-coding”

| Moment           | Durée  | Agenda                                                 | Support                    |
| ---------------- | ------ | ------------------------------------------------------ | -------------------------- |
| **Lundi 20h**    | 90 min | *Kick-off sprint* : revue backlog, assignation issues. | Google Meet + screen-share |
| **Mercredi 21h** | 60 min | *Pair coding* (vous codez, j’explique tests).          | Jupyter Live Share         |
| **Vendredi 20h** | 30 min | *Daily short* : check KPIs, blocages.                  | Slack huddle               |
| **Fin sprint**   | 45 min | Demo + rétro (retro doc automatique).                  | Streamlit report           |

---

## 8. Tableau de bord KPI (extrait)

| KPI                 | Source           | Seuil alerte | Widget             |
| ------------------- | ---------------- | ------------ | ------------------ |
| Latence signal (ms) | Redis metrics    | >200         | Gauge              |
| Coverage data (%)   | Agent QC         | <95          | Line               |
| Sharpe 30d          | backtest vs live | <1.2         | Number + sparkline |
| ES 97.5 %           | Risk agent       | >10          | Bar                |

---

## 9. Gabarits & ressources fournis

* **`TEMPLATE_spec.md`** – structure des specs.
* **`TEMPLATE_agent.py`** – squelette d’un agent CrewAI avec hooks `on_start`, `on_message`, `on_stop`.
* **`TEMPLATE_test_agent.py`** – exemple pytest avec fixtures Redis fake.
* **`TEMPLATE_notebook.ipynb`** – notebook ingestion + QC.
* **`mkdocs.yml`** – site de doc prêt à déployer via GitHub Pages.

---

### Prochaine action

1. **Créez le dépôt Git vide** et invitez-moi comme collaborateur.
2. Lancez l’issue `P1-S1-1` (spécifications) et collez vos exigences risque personnelles.
3. Lors de notre prochaine session “lundi 20h”, nous remplirons ensemble `docs/specs.md` et pousserons le premier commit.

Avec cette granularité, chaque pas est exécutable en < 2 heures, ce qui garantit un flux continu et mesurable jusqu’au go-live.
