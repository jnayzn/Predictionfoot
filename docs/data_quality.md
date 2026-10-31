# Nettoyage et qualité des données (Phase 3)

## 1. Validation à l'ingestion (par enregistrement)

`data/validators/` — appliquée à chaque objet avant chargement :
équipes identiques, buts négatifs, match terminé sans score, match terminé daté dans le
futur, date sans fuseau, statut inconnu, saison implausible, correspondance nom/pays des
compétitions. Un enregistrement en **erreur** n'est pas chargé (le JSON brut reste archivé),
un **avertissement** est chargé et signalé.

## 2. Couche canonique (multi-sources)

```bash
football-ai build-matches                       # toutes les compétitions
football-ai build-matches --competition ligue_1
```

### Rapprochement des équipes (piloté par les données)

Les sources n'ont ni identifiants ni orthographes communs. Deux équipes sont considérées
identiques lorsqu'elles **jouent les mêmes matchs** :

1. chaque match de la source secondaire vote pour les matchs de la source primaire joués
   à ±1 jour **avec le même score** ;
2. affectation un-pour-un par votes décroissants ;
3. **vérification** : confiance = part des matchs de l'équipe, sur la période couverte par
   les deux sources, retrouvés avec les deux équipes correspondantes.

Lien accepté si confiance ≥ 0,6 (`--min-confidence`) et au moins 3 matchs communs.
Sinon l'équipe reste **standalone** (jamais de rapprochement forcé) et les cas douteux sont
listés (`LOW CONFIDENCE`). La similarité des noms est conservée comme indice (`evidence`)
mais ne décide jamais seule. Overrides manuels : `configs/team_aliases.toml`.

| Méthode (`team_links.method`) | Signification |
|---|---|
| `primary` | équipe de la source prioritaire (API-Football) — 1 équipe canonique par équipe source |
| `cooccurrence` | rapprochée par les matchs communs (confiance + preuves stockées) |
| `alias` | imposée par `configs/team_aliases.toml` (jamais écrasée) |
| `standalone` | aucune correspondance fiable |

### Matchs canoniques (`matches`)

Les matchs des différentes sources ayant les mêmes équipes canoniques à ±1,5 jour sont
regroupés. La source prioritaire fournit les champs ; l'heure vient d'une source qui la
connaît. Le score retenu est celui des **90 minutes**.

`valid_for_training = false` avec `exclusion_reason` :

| Raison | Cas |
|---|---|
| `awarded` / `abandoned` / `cancelled` / `postponed` | match non joué normalement |
| `not_finished` | match à venir ou en cours (cible de prédiction) |
| `missing_score` | terminé sans score exploitable |
| `source_conflict` | les sources donnent des scores différents |

La construction est idempotente et peut être relancée après chaque import.

## 3. Contrôles au niveau du jeu de données

```bash
football-ai quality-report [--competition premier_league] [--json] [--fail-on-error]
```

| Contrôle | Sévérité |
|---|---|
| `duplicate_matches` — mêmes équipes le même jour | erreur |
| `team_double_booking` — une équipe joue deux fois en moins de 36 h | avertissement |
| `implausible_goals` — score > 12 ou négatif | erreur |
| `stale_results` — coup d'envoi passé de plus de 3 jours, résultat absent | avertissement |
| `source_conflicts` — scores divergents entre sources | avertissement |
| `incomplete_seasons` — saison terminée sans n×(n−1) matchs | avertissement |
| `shots_inconsistent` — tirs cadrés > tirs totaux | erreur |
| `possession_inconsistent` — possession ≠ ~100 % | avertissement |
| `odds_overround` — somme des probabilités implicites 1N2 hors [0,98 ; 1,35] | avertissement |

Le rapport inclut une **table de couverture** par compétition et saison : matchs terminés,
matchs valides, % avec statistiques, xG, compositions, cotes, cotes de clôture, % multi-sources.
Chaque rapport est enregistré dans `var/reports/quality_*.json` et dans la table
`data_quality_reports`, ce qui permet de suivre l'évolution de la qualité (monitoring, phase 13).
