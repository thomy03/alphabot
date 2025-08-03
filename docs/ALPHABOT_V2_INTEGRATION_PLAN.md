# ALPHABOT V2 - Plan d’Intégration des Modèles ML/DL au Modèle Élite

Dernière mise à jour: 2025-08-02 (15h16 - Corrections Colab finalisées)

## 1) Contexte et Objectifs

La V2 vise à:
- Stabiliser le pipeline d’entraînement sur Colab (ALPHABOT_ML_TRAINING_COLAB_v2.ipynb, cellules 1–9)
- Résoudre les fragilités HF/accelerate (FinBERT import-safe, RAG simple sans dépendances bloquantes)
- Produire des artefacts robustes (pattern_model, finbert_sentiment, rag_index)
- Mettre en place une Évaluation Quantitative standardisée (Cellule 9)
- Intégrer ces briques au modèle élite existant (elite_superintelligence_enhanced_v2.py) via des interfaces d’inférence propres

Objectifs KPI (Go/No-Go):
- Pattern: F1-macro ≥ 0.55 (puis ≥ 0.65)
- FinBERT: F1-macro ≥ 0.80 (≥ 50 phrases annotées)
- RAG: recall@5 ≥ 0.60 (≥ 20 requêtes difficiles)
- Backtest élite: amélioration du hit-rate ou PnL simulé vs baseline sur 3–6 mois

## 2) État Actuel (résumé)

- Pipeline Colab v2 opérationnel, suivi/reprise et artefacts générés (cellules 1–8)
- Cellule 9 ajoutée (évaluation)
- eval_metrics.json (exemple récent):
  - Pattern: collapse de classe (toutes prédictions “up”), accuracy 0.48 → non satisfaisant
  - FinBERT: F1=0 sur 3 phrases → benchmark non représentatif
  - RAG: recall@k=1.0 sur 3 requêtes faciles → test fonctionnel mais trop simple

Conclusion: robuste sur l’industrialisation, mais nécessité d’une évaluation scientifique plus crédible + corrections ciblées sur Pattern.

## 3) Briques V2 et Critères de Succès

### 3.1 Pattern Detector (LSTM non-cuDNN)
Problème observé: collapse de classe majoritaire.
Actions:
- Rééquilibrage: class_weight = inverse des fréquences de classes
- Seuils de labellisation: passer de ±0.2% à ±0.5–1.0% (horizon 2 jours)
- Split temporel: éviter fuite (train 2015–2023 / test 2024–2025)
- Régularisation/architecture:
  - label_smoothing = 0.05
  - dropout = 0.4
  - ajouter Conv1D (kernel 3–5) avant LSTM
- Features: momentum (pct-change cumulée), volatilité locale (rolling std)
- Évaluation: F1/precision/recall par classe, CM normalisée, distribution des prédictions
Succès: F1-macro ≥ 0.55 (puis ≥ 0.65) en holdout temporel

### 3.2 FinBERT (Sentiment)
Problème: benchmark 3 phrases → métrique non représentative.
Actions:
- Dataset d’éval: ≥ 50 phrases financières annotées (neg/neutral/pos) → /eval/finbert_benchmark.json
- Paramètres d’inférence: max_length=256, gestion d’un seuil neutre si max prob < 0.55
- Évaluation: F1 macro/micro/accuracy, matrice de confusion, 10 erreurs typiques avec probas
Succès: F1-macro ≥ 0.80

### 3.3 RAG (Contexte)
Problème: queries trop simples (recall@k=1.0)
Actions:
- Dataset d’éval: ≥ 20 requêtes difficiles → /eval/rag_queries.json
- Comparaison:
  - Mean pooling (AutoModel) vs sentence-transformers (all-MiniLM-L6-v2)
  - Index: simple vs FAISS (faiss-cpu)
  - Normalisation L2 systématique
- Évaluation: recall@3/5/10 + exemples top-3 sur 5 requêtes
Succès: recall@5 ≥ 0.60

## 4) Intégration au Modèle Élite (elite_superintelligence_enhanced_v2.py)

Objectif: enrichir le pipeline décisionnel existant avec les signaux ML V2.

### 4.1 Interfaces d’inférence standard (alphabot/ml/)
- pattern_detector.py:
  - load_pattern_model(base_path) → model, scaler
  - predict_pattern(X_batch_15x3) → classes, probas
- sentiment_analyzer.py:
  - load_finbert(base_path) → tokenizer, model
  - predict_sentiment(texts) → logits, probas
- rag_integrator.py:
  - load_rag(base_path) → index, vectors, docs
  - rag_search(query, k) → top_idx, scores

### 4.2 IntegrationAdapter_v2
Créer alphabot/core/integration_adapter_v2.py:
- Responsabilité: charger/mettre en cache les artefacts V2 et exposer des méthodes simples à l’orchestrateur élite
- Garde-fous: si artefact manquant, fallback (signaux techniques only)

### 4.3 Orchestration élite
- Ingestion:
  - Pattern: moduler l’exposition directionnelle (poids position/scaling)
  - FinBERT: score agrégé par symbole/secteur sur les news récentes
  - RAG: contexte explicatif, features sémantiques pour arbitrer signaux ambigus
- Logging/audit: tracer la contribution des signaux ML et décisions finales

## 5) Backtest de Validation

- Période: 3–6 mois récents (ou 12 mois selon univers)
- Métriques: hit-rate directionnel, PnL simulé, max drawdown, Sharpe-like
- Comparaison: baseline élite vs élite + signaux ML V2
- Conditions Go/No-Go: amélioration statistiquement significative (au moins 52–55% hit-rate ou PnL > baseline)

## 6) Roadmap & Livrables

### Sprint A — Pattern
- Implémenter class_weight, seuils ±0.5–1.0%, split temporel, label_smoothing, Conv1D
- Relancer entraînement (Cellule 5) et évaluation (Cellule 9)
- Livrables: eval_metrics.json, pattern_confusion_matrix.png

### Sprint B — FinBERT
- Créer /eval/finbert_benchmark.json (≥ 50 phrases)
- Mettre à jour Cellule 9 (chargement benchmark, F1/CM, erreurs)
- Livrables: eval_metrics.json (finbert), confusion PNG, exemples erreurs

### Sprint C — RAG
- Créer /eval/rag_queries.json (≥ 20 requêtes)
- Comparer mean-pooling vs sentence-transformers + FAISS
- Livrables: eval_metrics.json (rag), liste top-3 d'exemples

### Sprint D — Intégration & Backtest
- Ajouter IntegrationAdapter_v2 + interfaces d’inférence
- Câbler dans élite (elite_superintelligence_enhanced_v2.py)
- Lancer backtest (script minimal)
- Livrables: rapport backtest (JSON/CSV), graphiques, logs

### Sprint E — Reporting & Doc
- Générer /exports/model_report.md (compilation auto des métriques et images)
- Mettre à jour docs/ (INDEX + synthèses) avec chiffres clés et décisions

## 6.1) Sélection par Régime et Routing des Modèles (NOUVEAU)
Objectif: exploiter les résultats de la cellule 9 et d’eval_metrics.json pour sélectionner des "modèles champions" par régime de marché et router dynamiquement les signaux dans l’orchestrateur hybride.

A. Standardisation eval_metrics v2
- Schéma cible:
  {
    "schema_version": "2.0",
    "model_name": "lstm_conv",
    "horizon": 2,
    "window": "2019-01-01:2024-12-31",
    "regime": "bull|bear|sideways|all",
    "metrics": {
      "accuracy": float, "precision_macro": float, "recall_macro": float, "f1_macro": float,
      "hit_rate": float, "sharpe": float, "sortino": float, "calmar": float, "mdd_adj_return": float
    },
    "cv": {"folds": int, "seed": int},
    "data": {"train_range": "YYYY-MM-DD:YYYY-MM-DD", "val_range": "YYYY-MM-DD:YYYY-MM-DD", "test_range": "YYYY-MM-DD:YYYY-MM-DD"},
    "hyperparams": {...}
  }
- Actions: mettre à jour train_ml_models.py pour produire ce schéma, ajouter "schema_version": "2.0".

B. Sélection des champions par régime
- Script: scripts/select_models_by_regime.py
- Entrée: eval_metrics.json (liste de runs ou agrégations par modèle)
- Règle de sélection: pour chaque régime, score de ranking primaire sur f1_macro ou hit_rate, tie-breaker sur calmar/sortino.
- Sortie: champions.json
  {
    "bull": {"model_name": "...", "path": "...", "thresholds": {"min_confidence_prob": 0.55, "min_expected_edge": 0.0}},
    "bear": {"model_name": "...", "path": "...", "thresholds": {...}},
    "sideways": {"model_name": "...", "path": "...", "thresholds": {...}},
    "baseline": {"model_name": "baseline_robuste", "path": "..."}
  }

C. Intégration Orchestrateur (ModelSelector)
- Fichier: alphabot/core/hybrid_orchestrator.py
- Ajout d’un composant ModelSelector chargé de:
  - Charger champions.json
  - Détecter le régime courant (via Risk/Technical agents)
  - Retourner le modèle champion et ses seuils
  - Gérer le fallback baseline si confiance insuffisante ou métriques live sous seuils

D. Table de routing (référence)
Regime -> Modèle champion -> Seuils -> Fallback:
- bull -> champion_bull -> min_confidence_prob=0.55, min_expected_edge=0.00 -> baseline
- bear -> champion_bear -> min_confidence_prob=0.60, min_expected_edge=0.05 -> baseline
- sideways -> champion_sideways -> min_confidence_prob=0.58, min_expected_edge=0.02 -> baseline

E. Monitoring live et kill-switch
- Rolling window (ex: 50 trades) du hit-rate et PnL simulé.
- Si sous-seuils: passer automatiquement en mode baseline ou désactiver ML.

## 6.2) Coût/Complexité: Pareto-pruning (NOUVEAU)
- Flag --pareto-prune dans train_ml_models.py pour restreindre la grille aux architectures/hparams qui dominent sur ≥2 régimes.
- Règle: conserver ≤2 modèles séquentiels + 1 baseline robuste.

## 6.3) Optimisation d’Inférence (NOUVEAU)
- Export ONNX si bénéfice stable, batching multi-actifs, cache des features avec invalidation journalière.
- Intégration: alphabot/ml/ utilitaires + config dédiée.

## 6.4) Feature Gating par Régime (NOUVEAU)
- Pondération des signaux ML conditionnée par:
  - sentiment_score >= w_sentiment_regime
  - pattern_score >= w_pattern_regime
- Paramétrage via docs/OPTIMIZED_CONFIG.json (section model_selection.gating_weights) et application dans signal_hub / orchestrateur.

## 6.5) Critères Go/No-Go Production (NOUVEAU)
- Écart val/test contrôlé (< seuil à définir) sur 3 régimes.
- Hit-rate out-of-sample supérieur au baseline d’au moins x% (ex: +3–5 pts).
- Ratios de risque (Sortino/Calmar) ≥ baseline.
- Stabilité en paper trading ≥ 4 semaines sans déclenchement kill-switch.


## 7) Schémas JSON d’Évaluation

### 7.1 /eval/finbert_benchmark.json
```json
[
  {"text": "Company beats expectations with strong guidance", "label": 2},
  {"text": "Layoffs announced amid declining revenue", "label": 0},
  {"text": "Mixed signals keep investors uncertain", "label": 1}
]
```
Labels: 0=neg, 1=neutral, 2=pos

### 7.2 /eval/rag_queries.json
```json
[
  {"query": "electric vehicles manufacturer", "target_idx": 4},
  {"query": "cloud enterprise software provider", "target_idx": 2},
  {"query": "online retail and subscriptions", "target_idx": 3}
]
```
Note: target_idx correspond à l’indice du document de référence dans documents.pkl

## 8) Risques & Mitigations

- Données marché hétérogènes → limiter aux tickers liquides, gérer les trous
- Déséquilibre classes persistant → ajuster seuils, focal loss, features additionnelles
- Dépendances HF/FAISS → pinner versions compatibles, fallback simple robuste
- Surajustement → split temporel strict + backtest séparé

## 9) Références

- Notebook: ALPHABOT_ML_TRAINING_COLAB_v2.ipynb (Cellules 1–9)
- Code ML: alphabot/ml/pattern_detector.py, sentiment_analyzer.py, rag_integrator.py
- Élites: old_scripts/elite_superintelligence_enhanced_v2.py
- Orchestrateur: alphabot/core/hybrid_orchestrator.py
- Docs: docs/README_ENTRAINEMENT_COLAB.md, docs/README_ENTRAINEMENT_MODELES.md
