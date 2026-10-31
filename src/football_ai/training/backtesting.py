"""Walk-forward (expanding window) evaluation.

The evaluation period is cut into consecutive windows. Before each window the
model is re-fitted on every match played strictly before the window start,
then predicts the window. No prediction ever depends on a result it could not
have known.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_ai.models.base import OutcomeModel, played_rows


@dataclass(slots=True)
class WalkForwardResult:
    rows: pd.DataFrame  # evaluated rows, in prediction order
    proba: np.ndarray  # (n, 3)
    expected_goals: np.ndarray | None  # (n, 2) for goal models
    refits: int


def walk_forward(
    model_factory: Callable[[], OutcomeModel],
    history: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    refit_every_days: int = 30,
) -> WalkForwardResult:
    """``history`` = all rows available (any period); ``targets`` = rows to predict."""
    targets = targets.sort_values("kickoff_utc")
    if targets.empty:
        return WalkForwardResult(targets, np.zeros((0, 3)), None, 0)
    played = played_rows(history)
    start, end = targets["kickoff_utc"].min(), targets["kickoff_utc"].max()
    step = pd.Timedelta(days=refit_every_days)
    probas, goals, parts = [], [], []
    refits = 0
    window_start = start
    while window_start <= end:
        window_end = window_start + step
        window = targets[
            (targets["kickoff_utc"] >= window_start) & (targets["kickoff_utc"] < window_end)
        ]
        if not window.empty:
            train = played[played["kickoff_utc"] < window_start]
            model = model_factory().fit(train)
            refits += 1
            probas.append(model.predict_proba(window))
            if hasattr(model, "expected_goals"):
                goals.append(model.expected_goals(window))
            parts.append(window)
        window_start = window_end
    return WalkForwardResult(
        rows=pd.concat(parts),
        proba=np.vstack(probas),
        expected_goals=np.vstack(goals) if len(goals) == len(parts) else None,
        refits=refits,
    )
