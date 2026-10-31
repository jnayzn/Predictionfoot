# Bot Telegram (Phase 11)

## Lancer

1. Créer un bot auprès de [@BotFather](https://t.me/BotFather) et copier le jeton dans `.env` :
   `TELEGRAM_BOT_TOKEN=...`
2. Avoir un modèle enregistré (`football-ai train-ensemble ... --register`).
3. `football-ai bot` (long polling).

Deux modes (`BOT_BACKEND`) :

| Mode | Usage |
|---|---|
| `service` (défaut) | le bot appelle directement la couche de services (même machine que la base) |
| `http` | le bot appelle l'API (`BOT_API_URL`, `BOT_API_KEY`) — conteneurs séparés (phase 13) |

Dans les deux cas le bot ne contient **aucune règle métier** : il passe par l'interface
`BotBackend` (`bot/backend.py`), qui renvoie les schémas de l'API.

## Commandes

| Commande | Effet |
|---|---|
| `/start` | présentation, avertissement, menu à boutons |
| `/today` | matchs du jour + boutons vers l'analyse de chaque match |
| `/leagues` | choix d'une compétition (boutons) → matchs → analyse |
| `/analyze <id>` | analyse complète : probabilités 1N2, buts attendus, confiance, résumé, profils des équipes (forme, forces, faiblesses), facteurs déterminants, incertitudes, recommandation ; boutons « Marchés détaillés » et « Actualiser » |
| `/predictions` | probabilités 1N2 compactes des matchs à venir du jour |
| `/team <nom>` | derniers matchs (boutons si plusieurs équipes correspondent) |
| `/compare <A> vs <B>` | confrontation hypothétique (avantage du terrain pour A) |
| `/results [AAAA-MM-JJ]` | résultats + ✅/❌ de la prédiction publiée avant le match |
| `/performance` | track record réel + résumé du dernier backtest |
| `/subscribe [compétitions]`, `/unsubscribe` | rapport quotidien automatique (consentement explicite) |
| `/help` | aide |

## Robustesse et sécurité

- **Anti-spam** par utilisateur (`TELEGRAM_RATE_LIMIT_PER_MINUTE`).
- **Liste blanche** optionnelle (`TELEGRAM_ALLOWED_USERS`) pour une bêta privée.
- **Erreurs** : élément introuvable / service indisponible / erreur inattendue → message clair,
  jamais de trace technique côté utilisateur ; journalisation côté serveur.
- Textes HTML **échappés**, messages tronqués à la limite Telegram (4096 caractères).
- Le jeton du bot est masqué dans les logs.
- Chaque analyse se termine par l'avertissement : probabilités, jamais des certitudes,
  pas de conseil de pari.

## Notifications

`bot/notifications.py` : `send_daily_report()` envoie aux abonnés actifs les probabilités des
matchs du jour (filtrés par leurs compétitions). Un utilisateur qui a bloqué le bot est
désabonné automatiquement. L'ordonnancement est assuré par la phase 12.

## Tests

`tests/integration/test_bot.py` exécute le vrai dispatcher aiogram (filtres, middlewares,
boutons) avec une session Telegram simulée qui capture les messages : aucun accès réseau.
Le mode HTTP est testé contre l'API réelle (FastAPI) via un transport en mémoire.
