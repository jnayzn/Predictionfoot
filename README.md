# Football AI Predictor

Système d'analyse probabiliste des matchs de football (Premier League, La Liga,
Bundesliga, Serie A, Ligue 1, UEFA Champions League) basé sur des données réelles,
des modèles statistiques et du machine learning.

> **État actuel : PHASE 1 terminée** — initialisation du projet et connecteur API-Football.
> Aucune prédiction n'est encore produite. Voir la [feuille de route](#feuille-de-route).

## Sommaire

- [Fonctionnalités de la phase 1](#fonctionnalités-de-la-phase-1)
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

## Utilisation (CLI)

```bash
football-ai quota                                  # état local du quota (aucun appel réseau)
football-ai status                                 # statut du compte API (ne consomme pas de quota)
football-ai discover                               # 6 compétitions + saisons (1 requête, cache 7 jours)
football-ai discover --json
football-ai fixtures --date 2026-10-31             # matchs du jour (1 requête, cache 15 min / 7 jours)
football-ai fixtures --date 2026-10-31 --league premier_league --league ligue_1 --json
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
pytest                 # 74 tests, HTTP entièrement simulé (respx), aucun appel réseau
ruff check src tests
ruff format --check src tests
mypy                   # mode strict
```

Les payloads utilisés dans les tests sont **synthétiques** : ils reproduisent la
structure documentée de l'API mais leurs valeurs sont inventées (voir `tests/conftest.py`).

## Structure du projet

```
configs/competitions.toml          catalogue des compétitions (clé interne -> id par fournisseur)
src/football_ai/
  config/                          settings (pydantic-settings) et chargement des compétitions
  core/                            exceptions, logging structuré (JSON/console, masquage des secrets), I/O atomique
  domain/models.py                 modèles indépendants du fournisseur (Competition, Season, Fixture...)
  data/collectors/base.py          interface FootballDataProvider
  data/collectors/api_football/    client HTTP, quotas, parsers, provider
  data/storage/                    cache de réponses, archive JSON brute
  data/validators/                 contrôles de qualité des données
  bootstrap.py                     assemblage des dépendances
  cli.py                           interface en ligne de commande
tests/                             tests unitaires (pytest + respx)
docs/                              architecture, sources de données
```

Les dossiers des phases suivantes (`features/`, `models/`, `prediction/`, `backend/`, `bot/`...)
seront créés dans `src/football_ai/` au fil des phases ; voir [docs/architecture.md](docs/architecture.md).

## Documentation

- [docs/architecture.md](docs/architecture.md) — architecture et décisions techniques
- [docs/data_sources.md](docs/data_sources.md) — sources, couverture, quotas, licences

## Feuille de route

| Phase | Contenu | État |
|---|---|---|
| 1 | Initialisation + connecteur API-Football | ✅ terminé (validé par mocks, à valider avec une vraie clé) |
| 2 | Collecte historique + PostgreSQL | à venir |
| 3 | Nettoyage et validation | à venir |
| 4 | Feature engineering | à venir |
| 5 | Poisson / modèle de référence | à venir |
| 6 | XGBoost / LightGBM | à venir |
| 7 | Deep learning (PyTorch) | à venir |
| 8 | Ensemble + calibration | à venir |
| 9 | Backtesting et évaluation | à venir |
| 10 | API FastAPI | à venir |
| 11 | Bot Telegram | à venir |
| 12 | Automatisation | à venir |
| 13 | Docker, déploiement, monitoring | à venir |
| 14 | Optimisation, commercialisation | à venir |

## Avertissement

Ce projet produit des **probabilités**, jamais des certitudes. Il ne constitue pas un conseil
de pari. Respectez la législation de votre pays et les conditions d'utilisation des fournisseurs de données.
