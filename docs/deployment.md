# Déploiement, exploitation et monitoring (Phase 13)

## Image Docker

Une seule image pour l'API, le bot, le planificateur et les migrations
(multi-stage, utilisateur non-root `uid 10001`, logs JSON, `APP_ENV=production`).

```bash
docker build -t football-ai .
docker build --build-arg EXTRAS="[deep]" -t football-ai:deep .   # + modèle PyTorch (image plus lourde)
```

- `PYTHON_IMAGE` (build arg) permet d'utiliser un registre miroir (ex. `mirror.gcr.io/library/python:3.13-slim`).
- LightGBM nécessite `libgomp` : installé par `apt` ; si les dépôts Debian sont inaccessibles
  au build, la copie fournie par la roue scikit-learn est utilisée (vérifié au build :
  l'image échoue si `lightgbm`, `xgboost` ou `sklearn` ne s'importent pas).
- Derrière un proxy TLS d'entreprise : `--secret id=pip_cert,src=<ca-bundle.crt>`.

## Pile complète (docker compose)

```bash
cp .env.example .env        # renseigner : API_KEYS, POSTGRES_PASSWORD, API_FOOTBALL_KEY, TELEGRAM_*...
docker compose up -d --build
docker compose --profile bot up -d   # + bot Telegram (mode HTTP vers l'API)
```

| Service | Rôle |
|---|---|
| `postgres` | base de données (volume `pgdata`, healthcheck) |
| `redis` | limitation de débit partagée entre workers de l'API |
| `migrate` | `football-ai db-upgrade` puis s'arrête ; les autres services l'attendent |
| `api` | FastAPI (`API_WORKERS` workers), healthcheck `/health`, port lié à `127.0.0.1` |
| `scheduler` | tâches planifiées |
| `bot` (profil `bot`) | bot Telegram, appelle l'API avec `BOT_API_KEY` |
| `backup` | `pg_dump` quotidien vérifié (`pg_restore --list`), rétention `BACKUP_KEEP_DAYS` |

Variables d'images : `POSTGRES_IMAGE`, `REDIS_IMAGE`, `PYTHON_IMAGE` ; ports : `API_PORT`,
`POSTGRES_PORT` ; fichier d'environnement : `APP_ENV_FILE` (défaut `.env`).

**Sécurité** : l'API refuse de démarrer en production sans `API_KEYS` (arrêt immédiat avec
code d'erreur). Changer `POSTGRES_PASSWORD`. Les ports sont liés à `127.0.0.1` : exposer l'API
derrière un reverse proxy TLS (Caddy, Traefik, Nginx). Les secrets ne sont jamais dans l'image
(`.dockerignore` exclut `.env`).

Validation effectuée : image construite, pile démarrée (`migrate` 0001→0004, API *healthy*,
401 sans clé, réponses avec clé, `/metrics`, compteurs de débit dans Redis, 12 tâches planifiées,
sauvegarde réussie, tâche `quality_check` exécutée dans le conteneur).

## Sauvegardes

```bash
DATABASE_URL=... scripts/backup_postgres.sh var/backups 14     # dump + vérification + rétention
DATABASE_URL=... scripts/restore_postgres.sh <fichier.dump> --yes
```

Restauration testée : base restaurée identique (matchs, modèles, révision Alembic). Le trigger
d'immuabilité des prédictions est restauré avec le schéma.

## Monitoring

### Métriques Prometheus (`GET /metrics`, protégeable par `METRICS_TOKEN`)

| Métrique | Sens |
|---|---|
| `football_ai_http_requests_total{method,route,status}`, `football_ai_http_request_seconds` | trafic et latence (routes = gabarits, cardinalité bornée) |
| `football_ai_api_quota_used`, `football_ai_api_quota_limit` | consommation du quota API-Football |
| `football_ai_job_last_success_age_seconds{job}`, `football_ai_job_failures_24h{job}` | santé des tâches planifiées |
| `football_ai_data_age_seconds` | fraîcheur des données collectées |
| `football_ai_predictions_stored`, `football_ai_predictions_evaluated` | volume |
| `football_ai_live_log_loss`, `football_ai_live_brier` | performance réelle (prédictions publiées) |
| `football_ai_model_info{name,version}` | modèle enregistré le plus récent |

Exemples de configuration : `deploy/prometheus.yml`, règles d'alerte `deploy/alerts.yml`
(API indisponible, taux d'erreurs, tâches en échec, synchronisation en retard, données
périmées, quota presque épuisé, dégradation des performances).

### Dérive des données et des performances

Tâche hebdomadaire `drift_check` (`monitoring/drift.py`) :
- **PSI** par variable entre une période de référence (2 ans) et la période récente (60 jours)
  + test de Kolmogorov-Smirnov ; PSI > 0,25 = dérive significative ;
- **performance** : log loss des dernières prédictions publiées comparée à celle du dernier
  backtest (tolérance 0,05, au moins 50 matchs) ;
- rapport `var/reports/drift_*.json` + alerte (logs / Telegram administrateur).
Une dérive signale qu'un ré-entraînement (`train-ensemble --register`) doit être envisagé.

## Intégration continue

`.github/workflows/ci.yml` : ruff (lint + format), mypy strict, tests sur SQLite **et**
PostgreSQL 16 (service), vérification que les migrations correspondent aux modèles, build de
l'image Docker et validation du fichier compose.

## Mise à jour / retour arrière

1. Sauvegarde (`backup` ou `scripts/backup_postgres.sh`).
2. `docker compose build && docker compose up -d` (`migrate` applique les nouvelles migrations).
3. Retour arrière : image précédente (`APP_VERSION`), `alembic downgrade <révision>` si une
   migration doit être annulée, ou restauration de la sauvegarde.
