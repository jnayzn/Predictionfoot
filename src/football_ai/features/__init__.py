"""Leakage-free feature engineering.

Principle: every feature family produces a *state after each played match*
(per team, venue, pair...). Features of a target match are the last state
strictly before its kick-off (``merge_asof``, exact timestamps excluded).
The same code therefore serves training (past matches) and prediction
(upcoming matches) without any possibility of using the match's own result.
"""
