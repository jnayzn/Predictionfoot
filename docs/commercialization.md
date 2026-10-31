# Optimisation et préparation à la commercialisation (Phase 14)

> Ce document recense ce qui est prêt techniquement et ce qui doit être validé avant une
> mise sur le marché. Les points juridiques sont des **pistes de vérification**, pas un
> avis juridique : faire valider par un juriste (droit des données, des jeux d'argent, RGPD).

## 1. Optimisations réalisées

| Sujet | Avant | Après | Moyen |
|---|---|---|---|
| Latence d'une prédiction (API/bot), ~28 500 matchs | ~570 ms à chaque requête | ~0 ms après la 1re (même fraîcheur) | cache des prédictions invalidé au rechargement des données (5 min) ou du modèle ; un seul calcul pour tous les matchs à venir |
| Taille de l'image Docker | 2,01 Go | 1,23 Go (−39 %) | `xgboost-cpu` sous Linux (pas de bibliothèques CUDA) |
| Calcul des variables, ~28 500 matchs | 1,7 s | — | déjà vectorisé ; mesuré, pas de goulot à traiter |
| Limitation de débit multi-workers | par processus | partagée (Redis) | phase 13 |

## 2. Offre API commerciale

- **Clients API** en base (`football-ai api-keys create --name ... --plan pro`) : la clé est
  affichée **une seule fois** ; seul son hachage SHA-256 est stocké. Révocation par préfixe.
- **Offres** (`configs/plans.toml`) : quota quotidien + débit par minute par client
  (exemple : free 100/j, pro 5 000/j, business 50 000/j). Dépassement → HTTP 429 explicite.
- **Comptage de l'usage** par client et par jour (`api_usage`) → base de facturation
  (`football-ai api-usage`, endpoint `/v1/usage` pour le client).
- Les clés d'administration (`API_KEYS`) restent pour l'usage interne (bot, supervision).

Étapes suivantes possibles : facturation (ex. Stripe) branchée sur `api_usage`, portail
client, rotation de clé en libre-service, quotas mensuels.

## 3. Licences des données — à vérifier avant tout usage commercial

| Source | Point à vérifier |
|---|---|
| API-Football (api-sports.io) | plan payant requis pour l'usage commercial et pour la profondeur historique ; autorisation de diffuser des **données dérivées** (probabilités, statistiques agrégées) ; interdiction éventuelle de redistribuer les données brutes ; attribution |
| football-data.co.uk | conditions d'usage du site pour un usage commercial ; citation de la source |
| Cotes des bookmakers | droits de réutilisation et d'affichage des cotes ; partenariats éventuels |
| Logos et noms d'équipes / compétitions | marques déposées : ne pas afficher de logos ni d'identité visuelle de compétition sans licence (le bot n'affiche que des noms en texte) |

## 4. Cadre réglementaire (paris sportifs) — à valider par un juriste

- Positionnement recommandé : **service d'analyse statistique**, pas d'opérateur de paris ni
  d'incitation au jeu. Le système affiche systématiquement que les sorties sont des
  probabilités, jamais des certitudes, et pas un conseil de pari.
- Vérifier, pays par pays, la réglementation des services de pronostics et de la publicité
  pour les jeux d'argent (en France : autorité de régulation des jeux en ligne), l'interdiction
  aux mineurs (mention 18+), les messages de jeu responsable obligatoires.
- Le moteur de recommandation est volontairement prudent : aucune recommandation sans cotes
  fiables, avec une confiance faible ou sous les seuils ; pas de suggestion de mise.

## 5. RGPD — inventaire des données personnelles

| Donnée | Où | Finalité | Durée / droits |
|---|---|---|---|
| Identifiant de chat et nom d'utilisateur Telegram, compétitions suivies | `telegram_subscribers` (seulement après `/subscribe`) | rapport quotidien | jusqu'à `/unsubscribe` ; `/forget` efface tout ; `/privacy` informe |
| Nom du client API, usage quotidien | `api_clients`, `api_usage` | facturation, quotas | durée du contrat + obligations comptables |
| Adresses IP (accès sans clé), identifiants de requête | logs | sécurité, débogage | à limiter (rotation des logs, p. ex. 30 jours) |

À prévoir : politique de confidentialité publiée, registre des traitements, contrats avec
l'hébergeur (sous-traitant), choix d'un hébergement dans l'UE si clientèle européenne.

## 6. Sécurité et exploitation

Clés API hachées, comparaison à temps constant, démarrage refusé en production sans clé,
secrets hors de Git et de l'image, utilisateur non-root, ports liés à `127.0.0.1` derrière un
reverse proxy TLS, sauvegardes quotidiennes vérifiées, alertes, métriques, CI.
Recommandations supplémentaires : utilisateur PostgreSQL applicatif à privilèges réduits,
rotation des clés, scan de vulnérabilités de l'image en CI, sauvegardes hors site.

## 7. Montée en charge

| Charge | Mesure |
|---|---|
| API | plusieurs workers (`API_WORKERS`) + Redis pour la limitation ; réplicas derrière un load balancer |
| Prédictions | cache en mémoire par worker ; les prédictions publiées sont aussi en base |
| Base | PostgreSQL managé, index existants sur les colonnes de recherche |
| Bot | passer du long polling aux webhooks au-delà de quelques milliers d'utilisateurs actifs |
| Quota de données | plan API-Football supérieur ; la collecte est déjà budgétée et reprenable |

## 8. Validation sur données réelles — prérequis au lancement

Rien n'a encore été évalué sur des données réelles (l'environnement de développement n'avait
accès à aucune source). Plan recommandé :

1. Obtenir une clé API-Football (vérifier le plan : saisons accessibles, usage commercial).
2. `football-ai import-csv --seasons 2005-2025` (historique gratuit), puis
   `football-ai import-history --seasons 2018-2025` sur plusieurs jours (quota).
3. `football-ai build-matches` puis `football-ai quality-report` : vérifier le rapprochement
   des équipes (liste `LOW CONFIDENCE`) et la couverture.
4. `football-ai build-features --check-leakage 200`.
5. `football-ai backtest --validation-seasons 2021,2022 --test-seasons 2023,2024` : comparer à
   la référence marché (cotes de clôture). **Ne pas commercialiser** un modèle qui ne bat pas
   significativement les références simples, et ne jamais prétendre battre le marché sans
   preuve statistique.
6. `football-ai train-ensemble --register`, puis **1 à 2 mois en mode « shadow »** :
   prédictions publiées en interne seulement (immuables), évaluées automatiquement ;
   comparer le track record au backtest (`drift_check`).
7. Lancement progressif (bêta privée via `TELEGRAM_ALLOWED_USERS`), suivi des métriques.

## 9. Indicateurs de pilotage

Log loss / RPS / Brier du track record vs backtest et vs marché ; calibration (ECE) ; part des
matchs avec « aucune recommandation fiable » ; disponibilité de l'API ; latence p95 ;
fraîcheur des données ; consommation du quota ; utilisateurs actifs du bot ; usage par offre.
