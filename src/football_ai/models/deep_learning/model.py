"""Siamese multi-task neural network (PyTorch).

Architecture
------------
* a **shared team encoder** is applied to the home team's features and to the
  away team's features (same weights), optionally concatenated with a learned
  **team embedding** (index 0 = team unknown at training time);
* the match representation ``[h_home, h_away, h_home - h_away, context]``
  feeds a small MLP with two heads:
  - 3 logits for the 1X2 outcome (cross-entropy),
  - 2 log-intensities for home/away goals (Poisson negative log-likelihood),
  so the network also yields a score matrix and every goal market.

Preprocessing (median imputation + missing-value indicators + standardization)
is fitted on the training rows only. Early stopping uses the chronological
holdout carved from the training period. Training is seeded and runs on CPU.
"""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from torch import nn

from football_ai.models.base import check_probabilities
from football_ai.models.tabular.common import (
    chronological_holdout,
    select_features,
    training_rows,
)
from football_ai.prediction.score_prediction import (
    MAX_GOALS,
    outcome_probabilities,
    poisson_score_matrix,
)


@dataclass(slots=True)
class Preprocessor:
    """Column layout + train-fitted imputation/standardization statistics."""

    team_suffixes: list[str] = field(default_factory=list)  # home_<s> / away_<s> pairs
    context_columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    indicator_columns: list[str] = field(default_factory=list)
    team_index: dict[int, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, data: pd.DataFrame, columns: list[str], min_missing: float = 0.01) -> Preprocessor:
        suffixes = sorted(
            c[len("home_") :]
            for c in columns
            if c.startswith("home_") and f"away_{c[len('home_') :]}" in columns
        )
        paired = {f"home_{s}" for s in suffixes} | {f"away_{s}" for s in suffixes}
        context = [c for c in columns if c not in paired]
        prep = cls(team_suffixes=suffixes, context_columns=context)
        for col in [*paired, *context]:
            values = data[col].astype(float)
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            prep.medians[col] = median
            prep.means[col] = float(filled.mean())
            prep.stds[col] = float(filled.std()) or 1.0
            if values.isna().mean() >= min_missing:
                prep.indicator_columns.append(col)
        teams = pd.unique(pd.concat([data["home_team_id"], data["away_team_id"]]))
        prep.team_index = {int(t): i + 1 for i, t in enumerate(sorted(int(x) for x in teams))}
        return prep

    def _block(self, rows: pd.DataFrame, cols: list[str]) -> np.ndarray:
        out = np.empty((len(rows), len(cols)), dtype=np.float32)
        for j, col in enumerate(cols):
            values = rows[col].astype(float) if col in rows else pd.Series(np.nan, index=rows.index)
            out[:, j] = (
                (values.fillna(self.medians[col]) - self.means[col]) / self.stds[col]
            ).to_numpy()
        return out

    def transform(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        home = self._block(rows, [f"home_{s}" for s in self.team_suffixes])
        away = self._block(rows, [f"away_{s}" for s in self.team_suffixes])
        context = self._block(rows, self.context_columns)
        indicators = (
            np.stack(
                [
                    (rows[c].isna() if c in rows else pd.Series(True, index=rows.index)).to_numpy()
                    for c in self.indicator_columns
                ],
                axis=1,
            ).astype(np.float32)
            if self.indicator_columns
            else np.zeros((len(rows), 0), np.float32)
        )
        return {
            "home": home,
            "away": away,
            "context": np.concatenate([context, indicators], axis=1),
            "home_team": rows["home_team_id"].map(self.team_index).fillna(0).to_numpy(np.int64),
            "away_team": rows["away_team_id"].map(self.team_index).fillna(0).to_numpy(np.int64),
        }


class SiameseNet(nn.Module):
    def __init__(
        self,
        n_team_features: int,
        n_context: int,
        n_teams: int,
        *,
        hidden: int = 64,
        embedding_dim: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_teams + 1, embedding_dim) if embedding_dim else None
        enc_in = n_team_features + embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )  # fmt: skip
        self.head = nn.Sequential(
            nn.Linear(3 * hidden + n_context, hidden), nn.GELU(), nn.Dropout(dropout),
        )  # fmt: skip
        self.outcome = nn.Linear(hidden, 3)
        self.goals = nn.Linear(hidden, 2)

    def _encode(self, x: torch.Tensor, team: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = torch.cat([x, self.embedding(team)], dim=1)
        encoded: torch.Tensor = self.encoder(x)
        return encoded

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_home = self._encode(batch["home"], batch["home_team"])
        h_away = self._encode(batch["away"], batch["away_team"])
        z = self.head(torch.cat([h_home, h_away, h_home - h_away, batch["context"]], dim=1))
        log_rates = torch.clamp(self.goals(z), -3.0, 2.0)
        return self.outcome(z), log_rates


def _tensors(
    arrays: dict[str, np.ndarray], idx: np.ndarray | None = None
) -> dict[str, torch.Tensor]:
    sel = (lambda a: a[idx]) if idx is not None else (lambda a: a)
    return {k: torch.from_numpy(np.array(sel(v), copy=True)) for k, v in arrays.items()}


class NeuralModel:
    name = "neural"

    def __init__(
        self,
        *,
        use_market: bool = False,
        hidden: int = 64,
        embedding_dim: int = 8,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        goal_loss_weight: float = 0.5,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 15,
        seed: int = 42,
        max_goals: int = MAX_GOALS,
    ) -> None:
        self.use_market = use_market
        self.hidden = hidden
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.goal_loss_weight = goal_loss_weight
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.max_goals = max_goals
        self.prep: Preprocessor | None = None
        self.net: SiameseNet | None = None
        self.epochs_trained = 0
        self.history: list[dict[str, float]] = []

    # ---------------------------------------------------------------- training
    def _loss(self, net: SiameseNet, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits, log_rates = net(batch)
        ce = nn.functional.cross_entropy(logits, batch["outcome"])
        goals = torch.stack([batch["home_goals"], batch["away_goals"]], dim=1)
        poisson = nn.functional.poisson_nll_loss(log_rates, goals, log_input=True)
        loss: torch.Tensor = ce + self.goal_loss_weight * poisson
        return loss

    def _arrays(self, rows: pd.DataFrame, with_targets: bool) -> dict[str, np.ndarray]:
        assert self.prep is not None
        arrays = self.prep.transform(rows)
        if with_targets:
            arrays["outcome"] = rows["outcome"].to_numpy(dtype=np.int64)
            arrays["home_goals"] = rows["home_goals"].to_numpy(dtype=np.float32)
            arrays["away_goals"] = rows["away_goals"].to_numpy(dtype=np.float32)
        return arrays

    def fit(self, train: pd.DataFrame) -> Self:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        data = training_rows(train)
        data = data[data["home_goals"].notna() & data["away_goals"].notna()]
        columns = select_features(data, use_market=self.use_market)
        fit_part, holdout = chronological_holdout(data)
        self.prep = Preprocessor.fit(fit_part, columns)
        train_arrays = self._arrays(fit_part, True)
        hold_tensors = _tensors(self._arrays(holdout, True)) if holdout is not None else None
        net = SiameseNet(
            len(self.prep.team_suffixes),
            train_arrays["context"].shape[1],
            len(self.prep.team_index),
            hidden=self.hidden,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
        )
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        n = len(fit_part)
        best_state, best_loss, stale = copy.deepcopy(net.state_dict()), float("inf"), 0
        self.history = []
        for epoch in range(self.max_epochs):
            net.train()
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = _tensors(train_arrays, order[start : start + self.batch_size])
                optimizer.zero_grad()
                loss = self._loss(net, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()
            self.epochs_trained = epoch + 1
            if hold_tensors is None:
                continue
            net.eval()
            with torch.no_grad():
                hold_loss = float(self._loss(net, hold_tensors))
            self.history.append({"epoch": epoch + 1, "holdout_loss": hold_loss})
            if hold_loss < best_loss - 1e-4:
                best_loss, best_state, stale = hold_loss, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if hold_tensors is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net = net
        return self

    # -------------------------------------------------------------- prediction
    def _forward(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.net is None or self.prep is None:
            raise RuntimeError("model is not fitted")
        with torch.no_grad():
            logits, log_rates = self.net(_tensors(self._arrays(rows, False)))
            proba = torch.softmax(logits, dim=1).numpy().astype(float)
            rates = torch.exp(log_rates).numpy().astype(float)
        return proba, rates

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return check_probabilities(self._forward(rows)[0])

    def expected_goals(self, rows: pd.DataFrame) -> np.ndarray:
        return self._forward(rows)[1]

    def score_matrix(self, rows: pd.DataFrame) -> np.ndarray:
        rates = self.expected_goals(rows)
        return poisson_score_matrix(rates[:, 0], rates[:, 1], self.max_goals)

    def goal_head_outcome_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """1X2 implied by the goal head (consistency diagnostic)."""
        return check_probabilities(outcome_probabilities(self.score_matrix(rows)))

    def known_teams(self, rows: pd.DataFrame) -> np.ndarray:
        if self.prep is None:
            raise RuntimeError("model is not fitted")
        idx = self.prep.team_index
        return (rows["home_team_id"].isin(idx) & rows["away_team_id"].isin(idx)).to_numpy()

    # ------------------------------------------------------------ persistence
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        net = state.pop("net")
        if net is not None:
            buffer = io.BytesIO()
            torch.save(net.state_dict(), buffer)
            state["_net_state"] = buffer.getvalue()
            state["_net_shape"] = (
                len(self.prep.team_suffixes) if self.prep else 0,
                net.head[0].in_features - 3 * self.hidden,
                net.embedding.num_embeddings - 1 if net.embedding is not None else 0,
            )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        raw = state.pop("_net_state", None)
        shape = state.pop("_net_shape", None)
        self.__dict__.update(state)
        self.net = None
        if raw is not None and shape is not None:
            net = SiameseNet(
                shape[0], shape[1], shape[2],
                hidden=self.hidden, embedding_dim=self.embedding_dim, dropout=self.dropout,
            )  # fmt: skip
            net.load_state_dict(torch.load(io.BytesIO(raw), weights_only=True))
            net.eval()
            self.net = net
