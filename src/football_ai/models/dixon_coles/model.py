"""Dixon-Coles (1997): Poisson model with a dependence correction for low scores.

    tau(0,0) = 1 - lambda_h * lambda_a * rho     tau(0,1) = 1 + lambda_h * rho
    tau(1,0) = 1 + lambda_a * rho                tau(1,1) = 1 - rho

The attack/defence parameters are those of the Poisson fit; ``rho`` is then
estimated by weighted maximum likelihood (two-step estimation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from football_ai.models.poisson.model import PoissonModel
from football_ai.prediction.score_prediction import normalize, poisson_score_matrix


def tau(
    home: np.ndarray, away: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray, rho: float
) -> np.ndarray:
    out = np.ones_like(lam_h, dtype=float)
    out = np.where((home == 0) & (away == 0), 1 - lam_h * lam_a * rho, out)
    out = np.where((home == 0) & (away == 1), 1 + lam_h * rho, out)
    out = np.where((home == 1) & (away == 0), 1 + lam_a * rho, out)
    out = np.where((home == 1) & (away == 1), 1 - rho, out)
    return out


class DixonColesModel(PoissonModel):
    name = "dixon_coles"

    def __init__(
        self, *, rho_bounds: tuple[float, float] = (-0.25, 0.25), **kwargs: object
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.rho_bounds = rho_bounds
        self.rho = 0.0

    def _after_fit(self, data: pd.DataFrame, weights: np.ndarray) -> None:
        lam = self.expected_goals(data)
        yh = data["home_goals"].to_numpy(dtype=float)
        ya = data["away_goals"].to_numpy(dtype=float)
        low = (yh <= 1) & (ya <= 1)
        lam_h, lam_a, yh_l, ya_l, w_l = lam[low, 0], lam[low, 1], yh[low], ya[low], weights[low]

        def objective(rho: float) -> float:
            t = tau(yh_l, ya_l, lam_h, lam_a, rho)
            if (t <= 0).any():
                return float("inf")
            return float(-np.sum(w_l * np.log(t)))

        result = minimize_scalar(objective, bounds=self.rho_bounds, method="bounded")
        self.rho = float(result.x)

    def score_matrix(self, rows: pd.DataFrame) -> np.ndarray:
        lam = self.expected_goals(rows)
        matrix = poisson_score_matrix(lam[:, 0], lam[:, 1], self.max_goals)
        lam_h, lam_a = lam[:, 0], lam[:, 1]
        rho = self.rho
        matrix[:, 0, 0] *= np.clip(1 - lam_h * lam_a * rho, 1e-6, None)
        matrix[:, 0, 1] *= np.clip(1 + lam_h * rho, 1e-6, None)
        matrix[:, 1, 0] *= np.clip(1 + lam_a * rho, 1e-6, None)
        matrix[:, 1, 1] *= max(1 - rho, 1e-6)
        return normalize(matrix)
