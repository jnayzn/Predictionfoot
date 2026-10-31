# Football AI Predictor

Système d'analyse probabiliste des matchs de football (Premier League, La Liga,
Bundesliga, Serie A, Ligue 1, UEFA Champions League) basé sur des données réelles,
des modèles statistiques et du machine learning.

> **État actuel : PHASE 12 terminée** — données (API-Football, football-data.co.uk),
> PostgreSQL, ETL, rapprochement multi-sources, contrôles qualité, feature engineering sans
> fuite, modèles de référence (fréquences, Elo), modèles de buts (Poisson, Dixon-Coles),
> gradient boosting (XGBoost, LightGBM), réseau de neurones (PyTorch), ensemble, calibration,
> backtesting scientifique, suivi des prédictions réelles, moteur d'analyse et de
> recommandation, API FastAPI sécurisée, bot Telegram, automatisation planifiée. Aucune prédiction n'est encore produite. Voir la [feuille de route](#feuille-de-route).

## Sommaire

- [Fonctionnalités de la phase 1](#fonctionnalités-de-la-phase-1)
- [Fonctionnalités de la phase 2](#fonctionnalités-de-la-phase-2)
- [Fonctionnalités de la phase 3](#fonctionnalités-de-la-phase-3)
- [Fonctionnalités de la phase 4](#fonctionnalités-de-la-phase-4)
- [Fonctionnalités de la phase 5](#fonctionnalités-de-la-phase-5)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation (CLI)](#utilisation-cli)
- [Tests et qualité](#tests-et-qualité)
- [Structure du projet](#structure-du-projet)
- [Documentation](#documentation)
- [Feuille de route](#feuille-de-route)

## Fonctionnalités de la phase 1

| Fonction | Détail |
|---|---|
| Client API-Football v3 asynchrone | `httpx`, en-tête `x-apisports-key`, clé jamais journalisée ni écrite sur disque |
| Gestion des quotas | quota journalier persistant (reset 00:00 UTC), réserve non consommable, limiteur glissant par minute, resynchronisation via les en-têtes `x-ratelimit-requests-*` |
| Cache | cache disque avec TTL par type de requête ; un hit ne consomme aucun quota |
| Retry | backoff exponentiel + jitter sur erreurs réseau, 429 (respect de `Retry-After`) et 5xx |
| Erreurs | mapping des erreurs HTTP **et** des erreurs renvoyées en HTTP 200 dans le champ `errors` (token, quota journalier, rate limit, restriction de plan, paramètres) |
| Déduplication | requêtes identiques concurrentes fusionnées en un seul appel ; matchs dédupliqués par identifiant |
| Pagination | `get_all_pages()` suit `paging.current/total` |
| Archivage | chaque réponse JSON brute est conservée (append-only) avec ses métadonnées |
| Découverte des compétitions | les 6 compétitions + toutes leurs saisons et indicateurs de couverture en **1 seule requête** |
| Matchs par date | **1 seule requête** par jour, filtrage local des compétitions couvertes |
| Qualité des données | parsing strict (aucune valeur inventée), validations (équipes identiques, buts manquants sur un match terminé, dates incohérentes...) |
| Abstraction fournisseur | interface `FootballDataProvider` : un autre fournisseur peut être ajouté sans toucher au reste |

## Fonctionnalités de la phase 2

| Fonction | Détail |
|---|---|
| Schéma PostgreSQL | 18 tables (voir [docs/database.md](docs/database.md)), migrations Alembic |
| Intégrité | prédictions immuables (trigger), prédiction interdite après le coup d'envoi, probabilités contrôlées, classements en instantanés datés |
| ETL | Extract → Transform → Validate → Load, chargements idempotents |
| Corrections | changement de score sur un match terminé détecté, journalisé (`data_change_log`), `actual_results.revision` incrémenté |
| Import historique | saison par saison, priorisé, **budget de requêtes** par exécution, **reprise** automatique (`ingestion_jobs`), respect de la couverture déclarée par l'API |
| Détails de matchs | statistiques (xG si disponible), compositions, statistiques joueurs : 20 matchs par requête |
| Blessures, classements, cotes | collecte et normalisation (1x2, over/under, BTTS, double chance) |
| Rejeu | reconstruction complète de la base depuis l'archive JSON brute, sans aucun appel API |
| Seconde source | CSV football-data.co.uk (gratuit) : résultats, statistiques, cotes d'ouverture et de clôture depuis les années 1990 |

## Fonctionnalités de la phase 3

| Fonction | Détail |
|---|---|
| Rapprochement multi-sources | équipes reliées entre API-Football et football-data.co.uk par les matchs communs (date + score), avec confiance et preuves ; jamais de fusion sans preuve |
| Couche canonique | `canonical_teams`, `matches` : un match réel = une ligne, score à 90 min, sources liées, conflits détectés |
| Exclusions traçables | `valid_for_training` + raison (forfait, abandon, conflit de sources, score manquant...) |
| Contrôles qualité | doublons, double programmation, scores aberrants, résultats non mis à jour, saisons incomplètes, cohérence des stats et des cotes |
| Couverture | % de matchs avec stats, xG, compositions, cotes, par compétition et saison |
| Historique qualité | rapports JSON + table `data_quality_reports` |

Détails : [docs/data_quality.md](docs/data_quality.md).

## Fonctionnalités de la phase 4

| Fonction | Détail |
|---|---|
| ~170 variables | forme, attaque (buts, xG, tirs, efficacité), défense, domicile/extérieur, saison en cours, calendrier (repos, densité), confrontations directes, Elo dynamique, marché (cotes avant-match), joueurs absents |
| Zéro fuite | état « après match » + `merge_asof` strictement antérieur ; cotes de clôture, classements stockés et compositions exclus |
| Contrôle automatique | recalcul sur données tronquées au coup d'envoi (`--check-leakage`) |
| Reproductibilité | jeu de variables versionné (Parquet + métadonnées, empreinte config + données) |

Détails : [docs/features.md](docs/features.md).

## Fonctionnalités de la phase 5

| Fonction | Détail |
|---|---|
| Références | fréquences historiques, modèle Elo (régression logistique) |
| Modèles de buts | Poisson (attaque/défense, décroissance temporelle, L2, gradient analytique) et Dixon-Coles |
| Marchés | 1N2, double chance, buts attendus, Over/Under 1.5/2.5/3.5, BTTS, buts par équipe, top 5 des scores exacts |
| Évaluation | log loss, Brier, RPS, accuracy/précision/rappel/F1, calibration, MAE des buts ; walk-forward ; test verrouillé |
| Registre | artefacts versionnés + table `model_versions` |

## Fonctionnalités de la phase 6

| Fonction | Détail |
|---|---|
| XGBoost, LightGBM | classifieurs 1N2 ; variantes avec/sans variables de marché |
| LightGBM buts | régressions de Poisson → matrice de scores → tous les marchés |
| Optimisation | recherche aléatoire sur la validation, early stopping sur réserve chronologique interne |
| Suivi | MLflow (SQLite local) + JSON ; importance des variables exportée |

## Fonctionnalités des phases 7 et 8

| Fonction | Détail |
|---|---|
| Réseau de neurones | encodeur d'équipe siamois + embeddings d'équipes, multi-tâche (1N2 + buts de Poisson), early stopping, reproductible ; dépendance optionnelle (`pip install -e ".[deep]"`) |
| Ensemble | pool linéaire ou log-linéaire, poids appris sur la validation, élagage des modèles inutiles |
| Calibration | aucune / température / Platt / isotonique, choisie par validation croisée chronologique |
| Cohérence | matrice de scores réconciliée avec le 1N2 calibré → tous les marchés cohérents |
| Test verrouillé | poids et calibration gelés avant l'évaluation finale |

Détails : [docs/models.md](docs/models.md).

## Fonctionnalités de la phase 9

| Fonction | Détail |
|---|---|
| Backtest | walk-forward sur saisons de test, tous modèles + ensemble, par compétition et saison |
| Référence marché | cotes de clôture sans marge (évaluation uniquement), même sous-ensemble pour tous |
| Significativité | bootstrap par blocs de journées, IC 95 %, p-values |
| Rapports | JSON + Markdown, tables de calibration |
| Track record | prédictions immuables enregistrées avant match, évaluées automatiquement |

Détails : [docs/evaluation.md](docs/evaluation.md).

## Fonctionnalités de la phase 10

| Fonction | Détail |
|---|---|
| Moteur de prédiction | modèle de production du registre, 1N2, buts attendus, tous les marchés, enregistrement immuable avant match |
| Confiance | score et niveau distincts de la probabilité, incertitudes explicites (historique, xG, blessures, désaccord des modèles) |
| Analyse | favori statistique ou match équilibré, forme, forces/faiblesses, facteurs déterminants |
| Recommandation | uniquement avec cotes fiables, probabilité rétrécie vers le marché, seuils de prudence, « aucune recommandation fiable » explicite |
| API | FastAPI, clé API, limitation de débit, validation, request id, erreurs normalisées, OpenAPI |

Détails : [docs/api.md](docs/api.md).

## Fonctionnalités de la phase 11

| Fonction | Détail |
|---|---|
| Bot Telegram (aiogram 3) | /start /today /leagues /analyze /predictions /team /compare /results /performance /help + /subscribe /unsubscribe |
| Navigation | boutons inline : compétition → match → analyse → marchés détaillés |
| Architecture | interface `BotBackend` : services internes (même processus) ou API HTTP (conteneurs séparés) |
| Robustesse | anti-spam par utilisateur, liste blanche optionnelle, messages d'erreur clairs, HTML échappé |
| Notifications | rapport quotidien aux abonnés, désabonnement automatique des utilisateurs ayant bloqué le bot |

Détails : [docs/bot.md](docs/bot.md).

## Fonctionnalités de la phase 12

| Fonction | Détail |
|---|---|
| Planificateur | APScheduler, horaires cron configurables (`configs/schedule.toml`) |
| Tâches | programmes, résultats et corrections, détails, blessures, cotes, couche canonique, prédictions, rapport Telegram, évaluation, contrôle qualité, import historique |
| Quotas | budget vérifié avant chaque appel, arrêt propre et reprise |
| Recalcul | nouvelle version de prédiction seulement si l'information ou le modèle change |
| Supervision | historique `scheduled_job_runs`, alertes logs + Telegram administrateur |

Détails : [docs/automation.md](docs/automation.md).

## Installation

Prérequis : Python **3.12+**.

```bash
git clone <repo> && cd Predictionfoot
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # ou : pip install -r requirements-dev.txt && pip install -e .
cp .env.example .env
```

## Configuration

1. Créer un compte gratuit sur <https://dashboard.api-football.com/register>.
2. Copier la clé dans `.env` :

   ```dotenv
   API_FOOTBALL_KEY=votre_cle
   ```

3. Ajuster si besoin les limites à votre plan (`API_FOOTBALL_DAILY_LIMIT`,
   `API_FOOTBALL_PER_MINUTE_LIMIT`). Plan gratuit : 100 requêtes/jour.
   Les limites renvoyées par le serveur sont prioritaires si elles sont plus basses.

Toutes les variables sont documentées dans [`.env.example`](.env.example).
Le fichier `.env` est ignoré par Git : **ne jamais committer de secret**.

Les compétitions couvertes sont définies dans [`configs/competitions.toml`](configs/competitions.toml) ;
ajouter un championnat = ajouter un bloc puis lancer `football-ai discover` pour vérifier le mapping.

## Base de données

```bash
docker compose up -d postgres redis   # ou un PostgreSQL 16 existant
# DATABASE_URL dans .env (voir .env.example)
football-ai db-upgrade
```

## Utilisation (CLI)

```bash
football-ai quota                                  # état local du quota (aucun appel réseau)
football-ai status                                 # statut du compte API (ne consomme pas de quota)
football-ai discover                               # 6 compétitions + saisons (1 requête, cache 7 jours)
football-ai discover --json
football-ai fixtures --date 2026-10-31             # matchs du jour (1 requête, cache 15 min / 7 jours)
football-ai fixtures --date 2026-10-31 --league premier_league --league ligue_1 --json
```

Base de données et ETL (phase 2) :

```bash
football-ai db-upgrade                                    # migrations
football-ai sync-competitions                             # ligues + saisons + couverture
football-ai sync-fixtures --date 2026-10-31               # matchs du jour -> base
football-ai import-history --seasons 2021-2025 --max-requests 80     # reprenable
football-ai import-history --competition ligue_1 --seasons 2024 --steps fixtures,details
football-ai import-csv --seasons 2005-2024                # football-data.co.uk (sans clé)
football-ai import-csv --competition serie_a --seasons 2023 --file I1.csv
football-ai replay                                        # reconstruit la base depuis var/raw
football-ai jobs                                          # avancement de l'import
football-ai db-stats                                      # volumes par table
football-ai build-matches                                 # couche canonique multi-sources
football-ai quality-report [--competition ligue_1] [--json] [--fail-on-error]
football-ai build-features --check-leakage 50             # jeu de variables versionné
football-ai train-baselines --validation-seasons 2023 --test-seasons 2024 [--register]
football-ai train-models --validation-seasons 2023 --test-seasons 2024 --tune 20 --register
football-ai train-ensemble --validation-seasons 2022,2023 --test-seasons 2024 --evaluate-test --register
football-ai backtest --validation-seasons 2021,2022 --test-seasons 2023,2024
football-ai evaluate-predictions                          # track record des prédictions publiées
football-ai serve --host 0.0.0.0 --port 8000              # API (docs : /docs)
football-ai bot                                           # bot Telegram (TELEGRAM_BOT_TOKEN)
football-ai scheduler                                     # tâches planifiées
football-ai run-job sync_fixtures                         # une tâche immédiatement
scripts/import_history.sh 2015-2025 80                    # import quotidien (cron)
```

Équivalent sans installation du script : `python -m football_ai.cli <commande>`.

Codes de sortie : `0` OK, `1` erreur fournisseur, `2` configuration (ex. clé absente),
`3` quota épuisé, `4` authentification refusée.

Données produites (dossier `DATA_DIR`, par défaut `var/`, ignoré par Git) :

```
var/raw/api_football/<endpoint>/<YYYY-MM-DD>/<HHMMSS.micro>_<hash>.json   # archive brute, jamais écrasée
var/cache/api_football/<endpoint>/<hash>.json                             # cache TTL
var/state/api_football_quota.json                                         # compteur de quota du jour
```

## Tests et qualité

```bash
pytest                 # HTTP entièrement simulé (respx), aucun appel réseau ; base SQLite
# Mêmes tests + tests spécifiques PostgreSQL (migrations réelles, trigger, contraintes) :
TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/football_ai_test pytest
ruff check src tests
ruff format --check src tests
mypy                   # mode strict
```

Les payloads utilisés dans les tests sont **synthétiques** : ils reproduisent la
structure documentée de l'API mais leurs valeurs sont inventées (voir `tests/conftest.py`).

## Structure du projet

```
configs/competitions.toml          catalogue des compétitions (clé interne -> id par fournisseur)
configs/team_aliases.toml          overrides manuels du rapprochement d'équipes
configs/schedule.toml              horaires des tâches planifiées
migrations/                        migrations Alembic
scripts/                           scripts d'exploitation (import historique quotidien)
src/football_ai/
  config/                          settings (pydantic-settings) et chargement des compétitions
  core/                            exceptions, logging structuré (JSON/console, masquage des secrets), I/O atomique
  domain/models.py                 modèles indépendants du fournisseur (Competition, Season, Fixture...)
  data/collectors/base.py          interface FootballDataProvider
  data/collectors/api_football/    client HTTP, quotas, parsers, provider
  data/collectors/football_data_uk/  CSV historiques football-data.co.uk
  data/pipeline/                   ETL, import historique, rejeu de l'archive
  data/cleaning/                   rapprochement multi-sources, couche canonique
  data/quality/                    contrôles qualité du jeu de données, rapports
  features/                        variables (forme, attaque, défense, Elo, contexte, h2h,
                                   marché, joueurs), contrôle anti-fuite, stockage versionné
  models/                          baselines, poisson, dixon_coles, xgboost, lightgbm,
                                   deep_learning, ensemble, calibration
  training/                        découpages temporels, métriques, walk-forward, registre,
                                   optimisation, suivi d'expériences (MLflow), backtest
  prediction/                      moteur, confiance, cotes, matrices de scores, marchés
  analysis/                        analyse de match/équipe, moteur de recommandation
  backend/                         API FastAPI (routes, schémas, services, sécurité)
  bot/                             bot Telegram (handlers, keyboards, middlewares, notifications)
  automation/                      tâches planifiées, planificateur, alertes
  data/storage/                    cache de réponses, archive JSON brute
  data/validators/                 contrôles de qualité des données
  database/                        modèles SQLAlchemy, connexion, repositories
  bootstrap.py                     assemblage des dépendances
  cli/                             interface en ligne de commande
tests/unit, tests/integration      tests (pytest + respx ; SQLite et PostgreSQL)
docs/                              architecture, base de données, sources de données
```

Les dossiers des phases suivantes (`features/`, `models/`, `prediction/`, `backend/`, `bot/`...)
seront créés dans `src/football_ai/` au fil des phases ; voir [docs/architecture.md](docs/architecture.md).

## Documentation

- [docs/architecture.md](docs/architecture.md) — architecture et décisions techniques
- [docs/database.md](docs/database.md) — schéma et garanties d'intégrité
- [docs/data_quality.md](docs/data_quality.md) — nettoyage, rapprochement, contrôles qualité
- [docs/features.md](docs/features.md) — variables et garanties anti-fuite
- [docs/models.md](docs/models.md) — modèles, protocole d'évaluation, métriques
- [docs/evaluation.md](docs/evaluation.md) — backtesting, significativité, track record
- [docs/api.md](docs/api.md) — API, sécurité, moteur d'analyse et de recommandation
- [docs/bot.md](docs/bot.md) — bot Telegram
- [docs/automation.md](docs/automation.md) — tâches planifiées, quotas, alertes
- [docs/data_sources.md](docs/data_sources.md) — sources, couverture, quotas, licences

## Feuille de route

| Phase | Contenu | État |
|---|---|---|
| 1 | Initialisation + connecteur API-Football | ✅ terminé (validé par mocks, à valider avec une vraie clé) |
| 2 | Collecte historique + PostgreSQL | ✅ terminé (validé sur PostgreSQL 16 avec API simulée) |
| 3 | Nettoyage et validation | ✅ terminé |
| 4 | Feature engineering | ✅ terminé |
| 5 | Poisson / modèle de référence | ✅ terminé |
| 6 | XGBoost / LightGBM | ✅ terminé |
| 7 | Deep learning (PyTorch) | ✅ terminé |
| 8 | Ensemble + calibration | ✅ terminé |
| 9 | Backtesting et évaluation | ✅ terminé |
| 10 | API FastAPI | ✅ terminé |
| 11 | Bot Telegram | ✅ terminé |
| 12 | Automatisation | ✅ terminé |
| 13 | Docker, déploiement, monitoring | à venir |
| 14 | Optimisation, commercialisation | à venir |

## Avertissement

Ce projet produit des **probabilités**, jamais des certitudes. Il ne constitue pas un conseil
de pari. Respectez la législation de votre pays et les conditions d'utilisation des fournisseurs de données.
