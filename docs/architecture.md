# Architecture

## Vue d'ensemble (cible)

```
            ┌──────────────┐     ┌──────────────┐
            │ Bot Telegram │     │   FastAPI    │   (phases 10-11)
            └──────┬───────┘     └──────┬───────┘
                   └─────────┬──────────┘
                     services internes
        ┌────────────┬───────┴───────┬─────────────┐
        │ prediction │   analysis    │  training   │  (phases 5-9)
        └─────┬──────┴───────┬───────┴──────┬──────┘
              │          features (phase 4)  │
              └──────────────┬───────────────┘
                     database (PostgreSQL, phase 2)
                             │ ETL
                   data/collectors  ◄── PHASE 1 (ce livrable)
                             │
              FootballDataProvider (interface)
                 │                     │
          API-Football          autres fournisseurs
```

Règle de dépendance : les couches hautes ne dépendent que de `domain/` et des
interfaces, jamais d'un fournisseur concret.

## Phase 1 — composants

| Module | Rôle |
|---|---|
| `config/settings.py` | Paramètres typés depuis l'environnement ; la clé est un `SecretStr`. |
| `config/competitions.py` + `configs/competitions.toml` | Catalogue des compétitions : clé interne stable → identifiant par fournisseur. |
| `domain/models.py` | Modèles Pydantic immuables indépendants du fournisseur ; chaque entité garde `provider` + `provider_id`. |
| `data/collectors/base.py` | Protocole `FootballDataProvider`. |
| `data/collectors/api_football/client.py` | Transport HTTP : auth, cache, dédup, quotas, retry, mapping d'erreurs, archivage, pagination. |
| `data/collectors/api_football/quota.py` | Quota journalier persistant + fenêtre glissante par minute. |
| `data/collectors/api_football/parsers.py` | JSON API-Football → modèles de domaine (fonctions pures). |
| `data/collectors/api_football/provider.py` | Logique métier de collecte : découverte des compétitions, matchs par date. |
| `data/storage/` | Cache TTL et archive brute append-only. |
| `data/validators/` | Contrôles de qualité, issues typées `warning` / `error`. |
| `core/logging.py` | Logs structurés JSON/console, filtre de masquage des secrets. |

## Décisions techniques (ADR résumés)

### ADR-001 — Layout `src/football_ai/`
La structure proposée plaçait `data/`, `models/`, `backend/`... à la racine. Un package
racine nommé `data` ou `models` entre en collision avec des packages tiers et mélange code et
données. Tous les modules sont donc regroupés sous un package unique `football_ai`, en
conservant le même découpage (`data/collectors`, `data/validators`, plus tard `features/`,
`models/`, `prediction/`...). Les données d'exécution vont dans `var/` (ignoré par Git).

### ADR-002 — Client asynchrone
FastAPI et aiogram 3 sont asynchrones ; le client utilise `httpx.AsyncClient` pour être
réutilisable tel quel par l'API, le bot et le planificateur.

### ADR-003 — Minimisation des appels API
Budget de départ : plan gratuit (100 requêtes/jour).
- `GET /leagues` sans filtre renvoie toutes les ligues **avec toutes leurs saisons et indicateurs de couverture** : 1 requête au lieu de 6, mise en cache 7 jours.
- `GET /fixtures?date=` renvoie tous les matchs du jour dans le monde : 1 requête au lieu d'une par ligue ; filtrage local.
- TTL adaptatif : 7 jours pour une date passée de plus de 2 jours, 15 minutes sinon.
- Dédup des requêtes concurrentes identiques ; les réponses en erreur ne sont jamais mises en cache.
- `GET /status` est documenté comme non décompté du quota.

### ADR-004 — Le serveur fait foi pour le quota
Le compteur local empêche tout appel au-delà de `limite - réserve`. Dès qu'une réponse
contient `x-ratelimit-requests-limit/remaining`, le compteur local est réaligné. Une erreur
`errors.requests` (quota journalier atteint) bloque localement tout appel jusqu'au reset UTC.

### ADR-005 — Erreurs en HTTP 200
API-Football renvoie fréquemment HTTP 200 avec un champ `errors` non vide. Le client les
convertit en exceptions typées : `token` → `AuthenticationError`, `requests` →
`QuotaExceededError`, `rateLimit` → retry, `plan` → `PlanRestrictionError`, autre → `ApiRequestError`.

### ADR-006 — Archive brute immuable
Chaque réponse réussie est archivée (jamais écrasée). La phase 2 (ETL) pourra rejouer le
chargement sans consommer de quota, et l'archive sert de piste d'audit.

### ADR-007 — Aucune donnée inventée
Les parsers n'appliquent aucune valeur par défaut aux champs métier manquants (`None`).
Un champ obligatoire absent rejette l'enregistrement normalisé (l'original reste archivé) et
produit une `DataQualityIssue`.

### ADR-008 — Identifiants
Clé interne (`premier_league`) stable et indépendante du fournisseur ; identifiants
fournisseurs conservés sur chaque entité. Lors de la découverte, l'id configuré est vérifié
par nom/pays ; si l'id a disparu, un rapprochement par nom+pays est tenté et signalé.

### ADR-009 — Fuseaux horaires
Les dates sont stockées en UTC « aware ». Le fuseau envoyé à l'API (`API_FOOTBALL_TIMEZONE`,
UTC par défaut) détermine à quel jour calendaire un match est rattaché pour `fixtures?date=`.
