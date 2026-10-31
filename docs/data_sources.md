# Sources de données

## 1. API-Football (api-sports.io) — source principale

| Élément | Valeur |
|---|---|
| Documentation | <https://www.api-football.com/documentation-v3> |
| URL de base | `https://v3.football.api-sports.io` |
| Authentification | en-tête `x-apisports-key` |
| Plan utilisé au démarrage | gratuit |
| Quota gratuit | 100 requêtes/jour (reset 00:00 UTC) ; limite par minute selon le plan (10/min configuré par défaut) |
| En-têtes de quota | `x-ratelimit-requests-limit`, `x-ratelimit-requests-remaining` (jour), `X-RateLimit-Limit`, `X-RateLimit-Remaining` (minute) |

### Identifiants des compétitions (configs/competitions.toml)

| Clé interne | Compétition | ID API-Football |
|---|---|---|
| `premier_league` | Premier League (Angleterre) | 39 |
| `la_liga` | La Liga (Espagne) | 140 |
| `bundesliga` | Bundesliga (Allemagne) | 78 |
| `serie_a` | Serie A (Italie) | 135 |
| `ligue_1` | Ligue 1 (France) | 61 |
| `champions_league` | UEFA Champions League | 2 |

Ces identifiants sont ceux publiés par le fournisseur ; ils sont **vérifiés automatiquement**
par `football-ai discover` (nom et pays) — à confirmer avec une vraie clé.

### Endpoints utilisés en phase 1

| Endpoint | Usage | Coût | Cache |
|---|---|---|---|
| `GET /status` | état du compte | non décompté (selon la doc) | aucun |
| `GET /leagues` | découverte des compétitions, saisons, couverture | 1 | 7 jours |
| `GET /fixtures?date=YYYY-MM-DD&timezone=UTC` | matchs d'un jour | 1 / jour | 15 min (récent) / 7 jours (passé) |

Endpoints prévus pour la phase 2 : `/teams`, `/standings`, `/fixtures?league&season`,
`/fixtures/statistics`, `/fixtures/lineups`, `/injuries`, `/players` (paginé), `/odds`.

### Couverture

La couverture réelle (statistiques, compositions, blessures, cotes...) varie selon la
compétition **et** la saison ; elle est fournie par l'API dans `seasons[].coverage` et
conservée dans `Season.coverage` (ex. `fixtures.statistics_fixtures`, `injuries`, `odds`).
La phase 2 s'appuiera sur ces indicateurs pour ne jamais demander une donnée non couverte.

### Limites connues / à vérifier avec une vraie clé

- **Restriction de saisons sur le plan gratuit** : historiquement, le plan gratuit
  n'autorise qu'un sous-ensemble de saisons ; l'API répond alors en HTTP 200 avec
  `errors.plan`. Le client lève `PlanRestrictionError`. La plage exacte doit être vérifiée
  sur le compte (cela conditionne la profondeur historique disponible gratuitement).
- 100 requêtes/jour ne suffisent pas pour importer l'historique complet de 6 compétitions
  avec statistiques détaillées par match : la phase 2 devra planifier l'import sur plusieurs
  jours ou envisager un plan payant / une source historique complémentaire.
- xG/xGA : non fournis de façon systématique par API-Football ; une source complémentaire
  sera évaluée en phase 2 (sous réserve de licence).

### Licence et conditions d'utilisation

Les données sont soumises aux conditions générales d'API-Sports. Avant toute
**commercialisation** ou **redistribution** (bot public, API publique), vérifier les
conditions en vigueur et le plan requis. Le projet conserve les données brutes uniquement
pour usage interne (traitement, audit).

## 2. Sources complémentaires (à évaluer en phase 2)

Aucune n'est intégrée à ce stade. Toute source ajoutée sera documentée ici avec sa
couverture, ses limites et sa licence avant intégration.
