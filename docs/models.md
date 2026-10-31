# Modèles et évaluation

## Phase 5 — modèles de référence et modèles de buts

| Modèle | Fichier | Sortie | Principe |
|---|---|---|---|
| `frequency` | `models/baselines` | 1N2 | fréquences historiques par compétition (lissage de Laplace) — la référence minimale |
| `elo` | `models/baselines` | 1N2 | régression logistique multinomiale sur l'écart Elo avant match |
| `poisson` | `models/poisson` | matrice de scores | attaque/défense par équipe + avantage du terrain, log-vraisemblance pondérée (décroissance temporelle ξ = 0,0019/jour ≈ demi-vie d'un an), pénalité L2, gradient analytique |
| `dixon_coles` | `models/dixon_coles` | matrice de scores | Poisson + correction ρ des scores 0-0, 1-0, 0-1, 1-1 (estimation en deux temps) |

Tous les modèles partagent l'interface `OutcomeModel` (`fit(train)`, `predict_proba(rows)` →
`[domicile, nul, extérieur]`) ; les modèles de buts exposent aussi `score_matrix` et
`expected_goals`. Une équipe inconnue à l'entraînement reçoit la force moyenne
(`known_teams()` permet de signaler la prédiction comme moins fiable).

### Marchés dérivés de la matrice de scores (`prediction/`)

1N2, double chance, buts attendus, Over/Under 1.5/2.5/3.5, BTTS, probabilités qu'une équipe
marque ≥ 1/2/3 buts, top 5 des scores exacts, score le plus probable.

## Phase 6 — modèles tabulaires (gradient boosting)

| Modèle | Sortie | Principe |
|---|---|---|
| `xgboost` | 1N2 | `XGBClassifier` multi:softprob sur les ~170 variables (NaN gérés nativement) |
| `lightgbm` | 1N2 | `LGBMClassifier` multiclass |
| `lightgbm_goals` | matrice de scores | deux régressions LightGBM à objectif Poisson (buts domicile / extérieur) → matrice de Poisson → tous les marchés |
| `xgboost_market`, `lightgbm_market` | 1N2 | variantes **avec** les variables de marché (cotes avant-match) |

Choix méthodologiques :

- **Variables de marché exclues par défaut** : un modèle entraîné sur les cotes reproduit le
  marché et ne peut plus servir à détecter un écart avec lui (phase 10). Les variantes
  `*_market` mesurent l'apport des cotes.
- **Early stopping sur une réserve chronologique interne** (les 15 % les plus récents de
  l'entraînement), jamais sur la validation ni sur le test.
- **Optimisation d'hyperparamètres** (`--tune N`) : recherche aléatoire, chaque essai entraîné
  sur le train et noté par la log loss de **validation** ; le test n'est jamais lu (vérifié par test).
- **Importance des variables** (gain) exportée avec chaque modèle enregistré
  (`feature_importance.csv`).
- Hypothèse à vérifier sur données réelles : les modèles de boosting ne sont pas supposés
  meilleurs que Dixon-Coles. Sur des données synthétiques générées par un processus de Poisson,
  Dixon-Coles (correctement spécifié) reste meilleur ; seule la validation sur données réelles
  tranchera.

### Suivi des expériences (MLflow)

Chaque évaluation et chaque optimisation est enregistrée dans MLflow (base SQLite locale
`var/experiments/mlflow.db` par défaut) et en JSON (`var/experiments/runs/`). Si MLflow
n'est pas disponible, le suivi bascule automatiquement en JSON seul.

```bash
mlflow ui --backend-store-uri sqlite:///var/experiments/mlflow.db   # nécessite le paquet mlflow complet
```

## Protocole d'évaluation

- **Découpage strictement chronologique** (`training/datasets.py`) : entraînement = saisons
  antérieures, validation = saisons suivantes, test = dernières saisons. Un découpage qui
  chevauche dans le temps lève une erreur.
- **Walk-forward** (`training/backtesting.py`) : la période évaluée est découpée en fenêtres
  (30 jours par défaut) ; avant chaque fenêtre le modèle est ré-entraîné sur les seuls matchs
  joués avant le début de la fenêtre. Un test vérifie qu'aucun entraînement ne voit le futur.
- **Le jeu de test est verrouillé** : il n'est évalué qu'avec `--evaluate-test`, pour le
  rapport final ; les choix (modèles, hyperparamètres, poids d'ensemble) se font sur la validation.
- Seuls les matchs `valid_for_training` sont utilisés (voir data_quality.md).

### Métriques (`training/evaluate.py`)

| Métrique | Usage |
|---|---|
| **Log loss** | critère principal (règle de score propre) |
| **Brier** (multi-classes) | qualité probabiliste |
| **RPS** | standard du football (issues ordonnées : domicile > nul > extérieur) |
| Accuracy, précision, rappel, F1 (macro) | indicatifs, ignorent la qualité des probabilités |
| ECE + table de calibration | fiabilité des probabilités |
| MAE des buts (domicile, extérieur, total) | modèles de buts |

## Commandes

```bash
football-ai build-features
football-ai train-baselines --validation-seasons 2023 --test-seasons 2024
football-ai train-baselines --validation-seasons 2023 --test-seasons 2024 --evaluate-test   # rapport final
football-ai train-baselines --validation-seasons 2023 --register   # ré-entraîne sur train+validation et enregistre
football-ai train-models --models frequency,elo,dixon_coles,xgboost,lightgbm,lightgbm_goals \
    --validation-seasons 2023 --test-seasons 2024 --tune 20 --register
```

Rapports : `var/reports/baselines_*.json`. Modèles : `var/models/<nom>/<version>/` (`model.joblib`,
`metadata.json`) + ligne `model_versions` (paramètres, métriques, période d'entraînement).

## Résultats

Aucun résultat sur données réelles n'est encore disponible : l'environnement de développement
n'a pas accès aux sources de données. Sur des ligues **synthétiques** (vérité connue), les tests
vérifient que Poisson, Dixon-Coles et Elo battent la référence de fréquence et que le modèle de
Poisson retrouve les forces latentes (corrélation > 0,9) et l'avantage du terrain.
