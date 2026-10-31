# Évaluation scientifique et backtesting (Phase 9)

## Principes

1. **Chronologie stricte** : entraînement < validation < test, sans chevauchement (contrôlé).
2. **Walk-forward** : sur la période évaluée, chaque modèle est ré-entraîné périodiquement
   (`--refit-days`, 30 j par défaut) sur les seuls matchs joués avant la fenêtre prédite.
3. **Test verrouillé** : poids d'ensemble, calibration, hyperparamètres et choix de modèles
   sont décidés sur la **validation** ; le test ne sert qu'au rapport final.
4. **Références** : fréquences historiques, Elo, Poisson, XGBoost, ensemble — et le
   **marché** (cotes de clôture sans marge), la référence la plus exigeante. Les cotes de clôture
   ne sont **jamais** des variables explicatives ; elles ne servent qu'à l'évaluation.
5. **Significativité** : pour chaque paire (modèle, référence), différence de perte match par
   match, intervalle de confiance à 95 % par **bootstrap par blocs de journées** (les matchs d'une
   même date ne sont pas indépendants) et p-value (test t apparié sur les moyennes journalières).
   Verdict : « better » seulement si l'IC est entièrement négatif.
6. **Même échantillon** : la comparaison au marché se fait sur le sous-ensemble des matchs pour
   lesquels des cotes de clôture existent, pour **tous** les modèles.

## Commande

```bash
football-ai backtest --validation-seasons 2021,2022 --test-seasons 2023,2024 \
    --models frequency,elo,poisson,dixon_coles,xgboost,lightgbm,lightgbm_goals,neural
```

Sortie : `var/reports/backtest_<date>.json` et `.md` avec
- métriques globales (log loss, Brier, RPS, accuracy, précision/rappel/F1 macro, ECE) ;
- tableau sur le sous-ensemble avec cotes de clôture (marché inclus) ;
- comparaisons appariées avec IC et p-values ;
- détails par compétition et par saison ;
- tables de calibration (fiabilité) par modèle ;
- configuration de l'ensemble (poids, élagage, calibration).

Un historique insuffisant (ex. saison de validation sans saison antérieure) produit une erreur
explicite plutôt qu'une évaluation biaisée.

## Historique réel des prédictions (« track record »)

Les performances annoncées du système en production ne proviennent **que** des prédictions
enregistrées avant les matchs :

- `PredictionRepository.record()` refuse toute prédiction au-delà du coup d'envoi ; la base
  l'interdit aussi (`CHECK created_at < kickoff_utc`) et un trigger PostgreSQL rend les
  prédictions **immuables** (ni mise à jour, ni suppression) ;
- `football-ai evaluate-predictions` évalue chaque prédiction dont le match est terminé
  (log loss, Brier, exactitude, erreur sur le total de buts) dans `prediction_evaluations`
  et affiche le bilan cumulé.

Ainsi, le backtest (simulation rigoureuse) et le track record (prédictions réellement publiées)
sont deux mesures distinctes et toutes deux vérifiables.

## Limites actuelles

Aucun résultat sur données réelles : l'environnement de développement n'accède pas aux sources.
Les tests utilisent des ligues **synthétiques** dont la vérité est connue (le « marché »
synthétique y est la vraie probabilité, qu'aucun modèle ne doit battre significativement — ce
que vérifient les tests).
