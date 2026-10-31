# Automatisation (Phase 12)

```bash
football-ai scheduler            # exécute toutes les tâches selon configs/schedule.toml
football-ai run-job predictions  # exécute une tâche immédiatement
football-ai job-runs             # historique des exécutions
```

## Tâches (`automation/jobs.py`)

| Tâche | Rôle | Coût API |
|---|---|---|
| `sync_fixtures` | matchs d'hier à J+3 (programmes, résultats, reports) | 1 requête/jour couvert |
| `refresh_results` | résultats tardifs et **corrections** des derniers jours | 1/jour |
| `match_details` | statistiques, compositions, stats joueurs des matchs récents | 1 / 20 matchs, plafonné |
| `injuries` | blessures/suspensions de la saison en cours | 1 / compétition |
| `odds` | cotes avant-match des matchs proches | ≥ 1 / match, plafonné |
| `build_matches` | couche canonique multi-sources | 0 |
| `predictions` | prédiction + enregistrement immuable des matchs à venir | 0 |
| `daily_report` | rapport Telegram aux abonnés | 0 |
| `evaluate_predictions` | évaluation des prédictions dont le match est terminé | 0 |
| `quality_check` | rapport qualité + alertes (erreurs, résultats manquants, collectes et tâches en échec sur 24 h) | 0 |
| `historical_import` | dépense le quota restant sur l'historique (reprenable) | plafonné |

## Garanties

- **Horaires configurables** (`configs/schedule.toml`, syntaxe cron, fuseau horaire,
  `enabled = false` pour désactiver). Configuration validée au démarrage.
- **Quotas** : chaque tâche API vérifie le quota disponible (limite − réserve) et son
  `max_requests` **avant chaque appel** ; budget épuisé → arrêt propre (`partial`), reprise au
  prochain passage. Les horaires par défaut ordonnent les priorités : résultats et programmes
  d'abord, historique en dernier avec le reste du quota.
- **Pas d'exécutions concurrentes** d'une même tâche (`max_instances=1`), rattrapage fusionné
  après une interruption (`coalesce`), tolérance de retard de 30 min.
- **Une tâche en échec ne bloque jamais le planificateur** : l'erreur est enregistrée et
  déclenche une alerte.
- **Recalcul sur information importante** : chaque prédiction stocke l'empreinte des variables
  utilisées. La tâche `predictions` n'enregistre une nouvelle version que si l'information a
  changé (résultat d'un autre match, nouvelles cotes, blessures...) ou si le modèle a changé.
  Les versions précédentes restent (immuables) : l'historique complet est auditable.
- **Traçabilité** : chaque exécution est enregistrée dans `scheduled_job_runs` (statut,
  requêtes consommées, détails, erreur).

## Alertes

Toujours dans les logs (`football_ai.alerts`). Si `TELEGRAM_BOT_TOKEN` et
`TELEGRAM_ADMIN_CHAT_ID` sont définis, aussi dans le chat Telegram d'administration.
Déclencheurs : tâche en échec, erreurs du contrôle qualité, résultats non mis à jour,
collectes historiques en échec.
