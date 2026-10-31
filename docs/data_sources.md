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

### Endpoints ajoutés en phase 2

| Endpoint | Usage | Coût | Cache |
|---|---|---|---|
| `GET /fixtures?league&season` | tous les matchs d'une saison | 1 / saison | 30 j (saison terminée) / 6 h |
| `GET /teams?league&season` | équipes et stades | 1 / saison | idem |
| `GET /standings?league&season` | classement (stocké en instantané daté) | 1 / saison | idem |
| `GET /injuries?league&season` | blessures / suspensions | 1 / saison | idem |
| `GET /fixtures?ids=a-b-...` (max 20) | statistiques, compositions, stats joueurs | 1 / 20 matchs | aucun (chargé une seule fois) |
| `GET /odds?fixture=` | cotes avant-match d'un match (paginé) | ≥ 1 / match | 15 min |

**Budget estimé** pour une saison de championnat à 20 équipes (380 matchs) :
4 requêtes (matchs, équipes, classement, blessures) + 19 (détails) = **23 requêtes**.
Les 5 championnats + la Ligue des champions sur 10 saisons ≈ 1 300 requêtes, soit environ
14 jours d'import avec un budget de 95 requêtes/jour (plan gratuit), en supposant que le plan
donne accès à ces saisons. Les matchs (le plus utile) arrivent en premier : 60 requêtes
couvrent les 6 compétitions × 10 saisons.

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

## 2. football-data.co.uk — source historique complémentaire (gratuite)

| Élément | Valeur |
|---|---|
| Site | <https://www.football-data.co.uk> — notes des colonnes : <https://www.football-data.co.uk/notes.txt> |
| URL des fichiers | `https://www.football-data.co.uk/mmz4281/{AAAA}/{DIV}.csv` (ex. `2324/E0.csv`) |
| Divisions utilisées | E0 (Premier League), SP1 (La Liga), D1 (Bundesliga), I1 (Serie A), F1 (Ligue 1) — pas de Ligue des champions |
| Profondeur (approximative, à confirmer dans notes.txt et par championnat) | résultats depuis 1993/94 ; statistiques de match (tirs, corners, fautes, cartons) depuis ~2000 ; heure du match depuis 2019/20 ; cotes de clôture depuis 2019/20 |
| Authentification / quota | aucune ; chaque saison terminée n'est téléchargée qu'une fois (`var/raw/football_data_uk/`) |
| Commande | `football-ai import-csv --seasons 2005-2024` (ou `--file` pour un CSV téléchargé manuellement) |

### Hypothèses et limites

- **Pas d'identifiant de match** dans la source : un identifiant déterministe est calculé
  à partir de (division, date, équipe domicile, équipe extérieur) — stable d'un import à l'autre.
- Les équipes ne sont identifiées que par leur nom (rapprochement avec API-Football en phase 3).
- **Fuseau horaire** : la colonne `Time` est interprétée comme heure de Londres (hypothèse à
  confirmer). Sans colonne `Time` (saisons anciennes), `kickoff_time_known = false` : l'heure
  stockée (minuit) ne doit pas être utilisée.
- Les cotes « pre-closing » (ex. `B365H`) sont relevées par la source quelques jours avant
  le match (horodatage exact inconnu) ; elles sont stockées `pre_match` sans date de relevé.
  Les cotes de clôture (`B365CH`...) sont stockées `closing` : elles ne doivent **jamais**
  servir de variables explicatives avant le match (fuite d'information), seulement pour
  l'évaluation (comparaison au marché).
- Pas de xG, pas de compositions, pas de blessures.

### Licence

Les données sont mises à disposition gratuitement par le site. Avant tout usage commercial
ou redistribution, vérifier les conditions affichées sur le site et en citer la source.

## 3. Sources non retenues à ce stade

xG/xGA détaillés (ex. sites d'analytique publics) : non intégrés faute de licence claire
autorisant la réutilisation. À réévaluer avant la phase 4 ; toute nouvelle source sera
documentée ici avant intégration.
