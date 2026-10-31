# Feature engineering (Phase 4)

```bash
football-ai build-features [--competition ligue_1] [--check-leakage 50]
```

Sortie : `var/features/features_<config>_<données>.parquet` + métadonnées JSON (configuration,
empreinte des données, liste des variables et taux de valeurs manquantes) — un même couple
(configuration, données) produit toujours le même fichier : **reproductibilité**.

## Principe anti-fuite

1. Chaque famille de variables calcule un **état après chaque match joué** (par équipe,
   par équipe et lieu, par couple d'équipes, par équipe-compétition-saison).
2. Les variables d'un match cible sont le **dernier état strictement antérieur** à son coup
   d'envoi (`merge_asof`, `allow_exact_matches=False`).
3. Le même code sert à l'entraînement et à la prédiction des matchs à venir.

**Vérification automatique** (`features/leakage.py`, `--check-leakage N`, tests) : pour N
matchs tirés au hasard, les variables sont recalculées à partir d'une base **tronquée au
coup d'envoi** (tous les matchs à partir de cette date redeviennent « non joués », sans
statistiques) et comparées au calcul complet. Toute différence = fuite. Un test vérifie
aussi que le détecteur attrape une fuite introduite volontairement.

> Ce contrôle a détecté pendant le développement un bug réel (perte de l'alignement des
> lignes après `merge_asof`) qui mélangeait les variables entre matchs.

### Données volontairement exclues des variables

| Donnée | Raison |
|---|---|
| Cotes de **clôture** | connues seulement au coup d'envoi → réservées à l'évaluation (comparaison au marché) |
| Classements stockés (`standings`) | instantanés pris à la date de collecte (souvent après les matchs) ; le bilan de saison est recalculé à partir des seuls résultats antérieurs |
| Compositions officielles | publiées ~1 h avant le match, indisponibles au moment habituel de la prédiction |
| Statistiques du match lui-même | évidemment postérieures |

## Variables (≈170 colonnes)

Préfixes `home_` / `away_` (équipe à domicile / à l'extérieur) et `diff_` (= domicile − extérieur).
Fenêtres par défaut : 5 et 10 matchs (`FeatureConfig`).

| Groupe | Variables |
|---|---|
| Forme | `ppg_w`, `win_rate_w`, `draw_rate_w`, `loss_rate_w`, séries (`win_streak`, `unbeaten_streak`, `loss_streak`, `winless_streak`), `matches_played` |
| Attaque | `gf_avg_w`, `xg_for_avg_w`, `shots_for_avg_w`, `sot_for_avg_w`, `conversion_w` (buts / tirs cadrés), `xg_overperf_w` (buts − xG), `failed_to_score_rate_w` |
| Défense | `ga_avg_w`, `xg_against_avg_w`, `shots_against_avg_w`, `sot_against_avg_w`, `clean_sheet_rate_w` |
| Domicile / extérieur | `venue_ppg_5`, `venue_gf_5`, `venue_ga_5` calculés uniquement sur les matchs à domicile (équipe recevante) ou à l'extérieur (équipe visiteuse) |
| Saison en cours | `season_played`, `season_ppg`, `season_gd_pg`, `season_gf_pg`, `season_ga_pg` (par compétition) |
| Calendrier | `rest_days`, `games_last_14d`, `games_last_30d` (toutes compétitions) |
| Confrontations directes | `h2h_ppg`, `h2h_gd_avg`, `h2h_count` (5 dernières) |
| Elo | `elo` (avant match), `diff_elo`, `elo_p_home` (probabilité attendue Elo) |
| Marché | `market_p_home/draw/away` (cotes avant-match sans marge, moyenne des bookmakers), `market_available` |
| Joueurs | `missing_players` (blessés/suspendus connus) — **NaN = non couvert**, jamais confondu avec 0 |
| Contexte | `competition_importance` (Ligue des champions = 2) |

Cibles (NaN pour les matchs non joués) : `outcome` (0 domicile, 1 nul, 2 extérieur),
`home_goals`, `away_goals`, `total_goals`, `btts`. Score = 90 minutes.

## Elo

Elo toutes compétitions confondues, chronologique : `K=20`, avantage du terrain 60 points,
multiplicateur d'écart de buts, régression de 20 % vers la moyenne à chaque nouvelle saison,
nouvelle équipe (promue) à 1450. Ce sont des **hyperparamètres** : ils seront ajustés sur
les données de validation (jamais sur le test) en phases 5 et 9.

## Valeurs manquantes

Elles sont conservées (NaN) : début d'historique, xG absent des saisons anciennes, cotes
absentes, blessures non couvertes. Les modèles à base d'arbres les gèrent nativement ; les
autres modèles utiliseront une imputation apprise sur le jeu d'entraînement uniquement.
Les taux de valeurs manquantes sont enregistrés dans les métadonnées du jeu de variables.

## Données synthétiques de test

`tests/synthetic.py` génère des ligues **synthétiques** (forces d'équipes latentes connues,
buts de Poisson) pour vérifier que les variables retrouvent un signal connu
(ex. corrélation Elo / force latente > 0,7). Ces données ne sont jamais utilisées hors tests.
