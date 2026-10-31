# Base de données

PostgreSQL 16 (SQLite est accepté uniquement par les tests). Schéma défini dans
`src/football_ai/database/models.py`, migrations Alembic dans `migrations/`.

```bash
docker compose up -d postgres      # ou un PostgreSQL existant (DATABASE_URL)
football-ai db-upgrade             # = alembic upgrade head
football-ai db-stats               # nombre de lignes par table
```

## Conventions

- Clé primaire technique `id` partout ; identifiant source conservé (`provider`, `provider_id`) avec contrainte d'unicité.
- Dates `timestamptz` en UTC (`UTCDateTime` refuse les datetimes naïfs).
- JSON → `JSONB` sur PostgreSQL.
- Chargements **idempotents** : rejouer un import ne crée pas de doublon.

## Tables

| Table | Contenu | Unicité |
|---|---|---|
| `leagues` | compétitions (clé interne `key` + id fournisseur) | provider, provider_id |
| `seasons` | saisons et indicateurs de couverture (`coverage` JSONB) | league_id, year |
| `teams` | équipes | provider, provider_id |
| `players` | joueurs | provider, provider_id |
| `fixtures` | matchs : date UTC, statut, scores (mi-temps, 90 min, prolongation, tirs au but), arbitre, stade | provider, provider_id |
| `actual_results` | résultat final à 90 min des matchs terminés (`outcome` H/D/A), `revision` > 1 si corrigé | fixture_id |
| `data_change_log` | journal des corrections détectées (ancienne/nouvelle valeur) | — |
| `team_statistics` | statistiques d'équipe par match (tirs, possession, corners, **xG** si fourni) + JSON complet | fixture, team, provider |
| `player_statistics` | statistiques joueur par match (minutes, note, buts, passes...) | fixture, player, provider |
| `lineups` / `lineup_players` | compositions (formation, entraîneur, titulaires/remplaçants) | fixture, team, provider / lineup, player |
| `injuries` | blessures et suspensions par joueur, liées au match si connu | provider, player, league, season, fixture (NULLS NOT DISTINCT) |
| `standings` | **instantanés** de classement datés (`snapshot_date`) | provider, league, season, team, group, snapshot_date |
| `odds` | cotes normalisées (1x2, over_under_2.5, btts, double_chance), type pre_match / closing / market_average / market_max | fixture, provider, bookmaker, market, selection, type, source_updated_at |
| `model_versions` | versions de modèles, paramètres, métriques, run MLflow | name, version |
| `predictions` | prédictions **avant match**, **immuables** | fixture, model_version, created_at |
| `prediction_evaluations` | évaluation de chaque prédiction après le match (log loss, Brier...) | prediction_id |
| `ingestion_jobs` | avancement de l'import historique (reprise) | provider, job_type, scope |
| `canonical_teams` | clubs réels, toutes sources confondues | — |
| `team_links` | équipe source → équipe canonique (méthode, confiance, preuves) | team_id |
| `matches` | matchs canoniques (score 90 min, sources, conflit, `valid_for_training`, raison d'exclusion) | — |
| `match_sources` | matchs sources composant un match canonique | fixture_id |
| `data_quality_reports` | historique des rapports qualité | — |

## Garanties d'intégrité scientifique

- **Prédictions immuables** : un trigger PostgreSQL (`trg_predictions_immutable`) refuse tout `UPDATE`/`DELETE` sur `predictions`.
- `CHECK (created_at < kickoff_utc)` : une prédiction ne peut pas être enregistrée après le coup d'envoi.
- `CHECK` sur les probabilités 1N2 : positives et de somme 1 (±0,001).
- Classements stockés en **instantanés datés** : on peut toujours retrouver le classement connu à une date donnée (pas de fuite d'information future).
- Les corrections de résultats sont tracées (`data_change_log`, `actual_results.revision`).

## Sources multiples

Chaque fournisseur a ses propres lignes (`provider`). Un même match peut donc exister
une fois par source (ex. API-Football et football-data.co.uk). La couche canonique
(`canonical_teams`, `team_links`, `matches`, `match_sources`) les rapproche — voir
[data_quality.md](data_quality.md).
