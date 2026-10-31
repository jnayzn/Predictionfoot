"""Independent Poisson goal model (Maher / Dixon-Coles parameterization).

    log lambda_home = mu + home_adv + attack[home] - defence[away]
    log lambda_away = mu +            attack[away] - defence[home]

Fitted by weighted maximum likelihood (L-BFGS-B, analytic gradient):
* time decay ``w = exp(-xi * age_days)`` - recent matches weigh more;
* L2 penalty on attack/defence (identifiability + shrinkage of teams with
  little data towards the average).

Teams never seen in training get attack = defence = 0 (league average); the
``known_teams`` mask lets callers flag such predictions as less reliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from football_ai.models.base import check_probabilities, played_rows
from football_ai.prediction.score_prediction import (
    MAX_GOALS,
    outcome_probabilities,
    poisson_score_matrix,
)


@dataclass(slots=True)
class PoissonParams:
    mu: float
    home_advantage: float
    attack: dict[int, float]
    defence: dict[int, float]


def _decay_weights(kickoffs: pd.Series, reference: pd.Timestamp, xi: float) -> np.ndarray:
    age_days = (reference - kickoffs).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    weights: np.ndarray = np.exp(-xi * np.clip(age_days, 0, None))
    return weights


class PoissonModel:
    name = "poisson"

    def __init__(
        self,
        *,
        time_decay: float = 0.0019,
        l2: float = 1.0,
        max_history_days: int | None = 4 * 365,
        max_goals: int = MAX_GOALS,
    ) -> None:
        self.time_decay = time_decay
        self.l2 = l2
        self.max_history_days = max_history_days
        self.max_goals = max_goals
        self.params: PoissonParams | None = None
        self.fitted_until: pd.Timestamp | None = None

    # ------------------------------------------------------------------ fit
    def _training_data(self, train: pd.DataFrame) -> pd.DataFrame:
        data = played_rows(train)
        data = data[data["home_goals"].notna() & data["away_goals"].notna()]
        if data.empty:
            raise ValueError("no played match to fit on")
        if self.max_history_days is not None:
            start = data["kickoff_utc"].max() - pd.Timedelta(days=self.max_history_days)
            data = data[data["kickoff_utc"] >= start]
        return data

    def fit(self, train: pd.DataFrame) -> Self:
        data = self._training_data(train)
        teams = np.unique(np.concatenate([data["home_team_id"], data["away_team_id"]]))
        index = {int(t): i for i, t in enumerate(teams)}
        h = data["home_team_id"].map(index).to_numpy(dtype=np.int64)
        a = data["away_team_id"].map(index).to_numpy(dtype=np.int64)
        yh = data["home_goals"].to_numpy(dtype=float)
        ya = data["away_goals"].to_numpy(dtype=float)
        reference = data["kickoff_utc"].max()
        w = _decay_weights(data["kickoff_utc"], reference, self.time_decay)
        n_teams = len(teams)

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return poisson_nll(theta, h, a, yh, ya, w, n_teams, self.l2)

        theta0 = np.zeros(2 + 2 * n_teams)
        theta0[0] = np.log(max(np.average(np.concatenate([yh, ya])), 0.1))
        result = minimize(objective, theta0, jac=True, method="L-BFGS-B")
        theta = result.x
        att, dfn = theta[2 : 2 + n_teams], theta[2 + n_teams :]
        self.params = PoissonParams(
            mu=float(theta[0]),
            home_advantage=float(theta[1]),
            attack={int(t): float(att[i]) for t, i in index.items()},
            defence={int(t): float(dfn[i]) for t, i in index.items()},
        )
        self.fitted_until = reference
        self._after_fit(data, w)
        return self

    def _after_fit(self, data: pd.DataFrame, weights: np.ndarray) -> None:
        """Hook for subclasses (Dixon-Coles fits rho here)."""

    # -------------------------------------------------------------- predict
    def _require(self) -> PoissonParams:
        if self.params is None:
            raise RuntimeError(f"{self.name} model is not fitted")
        return self.params

    def expected_goals(self, rows: pd.DataFrame) -> np.ndarray:
        p = self._require()
        att_h = rows["home_team_id"].map(p.attack).fillna(0.0).to_numpy(dtype=float)
        att_a = rows["away_team_id"].map(p.attack).fillna(0.0).to_numpy(dtype=float)
        def_h = rows["home_team_id"].map(p.defence).fillna(0.0).to_numpy(dtype=float)
        def_a = rows["away_team_id"].map(p.defence).fillna(0.0).to_numpy(dtype=float)
        lam_h = np.exp(p.mu + p.home_advantage + att_h - def_a)
        lam_a = np.exp(p.mu + att_a - def_h)
        return np.stack([lam_h, lam_a], axis=1)

    def known_teams(self, rows: pd.DataFrame) -> np.ndarray:
        p = self._require()
        return (
            rows["home_team_id"].isin(p.attack) & rows["away_team_id"].isin(p.attack)
        ).to_numpy()

    def score_matrix(self, rows: pd.DataFrame) -> np.ndarray:
        lam = self.expected_goals(rows)
        return poisson_score_matrix(lam[:, 0], lam[:, 1], self.max_goals)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return check_probabilities(outcome_probabilities(self.score_matrix(rows)))


def poisson_nll(
    theta: np.ndarray,
    h: np.ndarray,
    a: np.ndarray,
    yh: np.ndarray,
    ya: np.ndarray,
    w: np.ndarray,
    n_teams: int,
    l2: float,
) -> tuple[float, np.ndarray]:
    """Weighted negative log-likelihood (constant terms dropped) and its gradient."""
    mu, home_adv = theta[0], theta[1]
    att = theta[2 : 2 + n_teams]
    dfn = theta[2 + n_teams :]
    eta_h = mu + home_adv + att[h] - dfn[a]
    eta_a = mu + att[a] - dfn[h]
    lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)
    nll = -np.sum(w * (yh * eta_h - lam_h + ya * eta_a - lam_a))
    nll += 0.5 * l2 * (att @ att + dfn @ dfn)

    r_h = -w * (yh - lam_h)  # d nll / d eta_h
    r_a = -w * (ya - lam_a)
    grad = np.empty_like(theta)
    grad[0] = r_h.sum() + r_a.sum()
    grad[1] = r_h.sum()
    grad[2 : 2 + n_teams] = np.bincount(h, r_h, n_teams) + np.bincount(a, r_a, n_teams) + l2 * att
    grad[2 + n_teams :] = -np.bincount(a, r_h, n_teams) - np.bincount(h, r_a, n_teams) + l2 * dfn
    return float(nll), grad
