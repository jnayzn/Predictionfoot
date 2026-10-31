# API FastAPI (Phase 10)

```bash
football-ai serve --host 0.0.0.0 --port 8000
# ou : uvicorn football_ai.backend.main:create_app --factory --port 8000
```

Documentation interactive : `http://localhost:8000/docs` (OpenAPI).

## Sécurité

- **Clé API** obligatoire sur `/v1/*` (en-tête `X-API-Key`, clés dans `API_KEYS`, comparaison
  à temps constant). Le démarrage **échoue en production** si aucune clé n'est configurée.
  En développement sans clé, l'API est ouverte et limitée par adresse IP.
- **Limitation de débit** par client (`API_RATE_LIMIT_PER_MINUTE`, réponse 429 + `Retry-After`).
- **Validation des entrées** (types, bornes, motifs) → 422 avec détail.
- Chaque réponse porte un `X-Request-ID` ; logs structurés (méthode, chemin, statut, durée) ;
  les clés API sont masquées dans les logs.
- Erreurs normalisées `{"error", "request_id"}` : 404 ressource inconnue, 503 aucun modèle
  en production, 422 données insuffisantes, 500 sans fuite d'information interne.

## Endpoints

| Méthode | Chemin | Description |
|---|---|---|
| GET | `/health` | état (base, modèle chargé) — public |
| GET | `/v1/competitions` | compétitions couvertes |
| GET | `/v1/fixtures?day=YYYY-MM-DD&competition=` | matchs d'un jour (UTC) |
| GET | `/v1/matches/{id}` | détail d'un match |
| GET | `/v1/matches/{id}/prediction?store=false` | probabilités 1N2, buts attendus, tous les marchés, confiance ; `store=true` enregistre la prédiction (immuable, avant match uniquement) |
| GET | `/v1/matches/{id}/analysis` | prédiction + analyse (favori, forces/faiblesses, forme, facteurs clés, incertitudes) + recommandation |
| GET | `/v1/teams?search=` | recherche d'équipes |
| GET | `/v1/teams/{id}/matches` | derniers matchs d'une équipe |
| GET | `/v1/compare?home=&away=` | confrontation hypothétique (même modèle, variables à la date du jour) |
| GET | `/v1/results?day=` | résultats + prédiction publiée et exactitude |
| GET | `/v1/performance` | track record (prédictions publiées) + dernier backtest |

## Couche de services

`backend/services/football_service.py` est le **seul** point d'accès métier : l'API, le bot
Telegram (phase 11) et le planificateur (phase 12) l'utilisent, aucune règle n'est dupliquée.

## Moteur de prédiction, analyse et recommandation

- `prediction/engine.py` : charge le dernier modèle enregistré (`PRODUCTION_MODEL_PRIORITY`),
  calcule les variables (cache de 5 min), produit probabilités, matrice de scores et marchés.
- `prediction/confidence.py` : **confiance ≠ probabilité**. Score 0–1 et niveau
  (faible/moyen/élevé) à partir de l'historique des équipes, du début de saison, de la
  disponibilité des xG et des blessures, et du **désaccord entre les modèles** de l'ensemble ;
  chaque incertitude est explicitée.
- `analysis/` : profil d'équipe (forme VND, attaque/défense rapportées à la moyenne de la ligue,
  écarts buts/xG), favori statistique (ou « match équilibré »), facteurs clés classés.
- `analysis/recommendation.py` — règles :
  1. sans cotes fiables → aucune évaluation de valeur (`no_odds`), seulement les scénarios les
     plus probables, présentés comme information ;
  2. confiance faible → **aucune recommandation** ;
  3. marché incohérent (marge < 0) ou trop margé (> 12 %) → ignoré ;
  4. probabilité du modèle **rétrécie vers le marché** selon la confiance
     (`p = λ·p_modèle + (1−λ)·p_marché`, λ = 0,4 ou 0,6) ;
  5. sélection retenue seulement si valeur attendue ajustée ≥ 3 %, valeur attendue du modèle
     seul ≥ 3 %, probabilité ≥ 25 %, cote ≤ 6 ;
  6. sinon « aucune recommandation suffisamment fiable », avec les raisons.
  Une sélection n'est jamais recommandée parce qu'elle est la plus probable. Les cotes utilisées
  sont la **moyenne** des bookmakers (jamais la meilleure cote) et uniquement des cotes avant-match.
