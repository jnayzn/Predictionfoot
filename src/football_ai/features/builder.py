"""Assemble all feature families into one row per match.

Output columns
--------------
* metadata : match_id, competition_key, season_year, kickoff_utc, home_team_id,
  away_team_id, status_group, valid_for_training
* targets  : home_goals, away_goals, outcome (0 home / 1 draw / 2 away), total_goals, btts
  (NaN for matches not played yet)
* features : ``home_*`` / ``away_*`` team features, ``diff_*`` = home - away,
  Elo, market and context features. :func:`feature_columns` lists them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from football_ai.features.attack_features import attack_states
from football_ai.features.context_features import (
    COMPETITION_IMPORTANCE,
    schedule_features,
    season_features,
)
from football_ai.features.dataset import BaseFrames
from football_ai.features.defense_features import defense_states
from football_ai.features.elo_features import EloConfig, elo_states, expected_home, pre_match_elo
from football_ai.features.form_features import form_states, venue_states
from football_ai.features.h2h_features import h2h_features
from football_ai.features.history import asof, played_matches, targets_long, team_long
from football_ai.features.market_features import market_features
from football_ai.features.player_features import player_features

META_COLUMNS = [
    "match_id", "competition_key", "season_year", "kickoff_utc", "home_team_id",
    "away_team_id", "status_group", "valid_for_training",
]  # fmt: skip
TARGET_COLUMNS = ["home_goals", "away_goals", "outcome", "total_goals", "btts"]
FEATURE_SET_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    windows: tuple[int, ...] = (5, 10)
    venue_window: int = 5
    h2h_window: int = 5
    elo: EloConfig = field(default_factory=EloConfig)

    def fingerprint(self) -> str:
        payload = json.dumps({"version": FEATURE_SET_VERSION, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _team_side(team_rows: pd.DataFrame, side: str, n_matches: int) -> pd.DataFrame:
    """Pick home or away rows of the long target table and prefix their columns."""
    rows = team_rows.iloc[:n_matches] if side == "home" else team_rows.iloc[n_matches:]
    rows = rows.reset_index(drop=True)
    return rows.add_prefix(f"{side}_")


class FeatureBuilder:
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def build(
        self,
        frames: BaseFrames,
        *,
        competition_keys: Sequence[str] | None = None,
        match_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Features for target matches (default: all matches of the requested competitions).

        History always uses *every* played match (all competitions): a club's
        Champions League games inform its domestic form and vice versa.
        """
        cfg = self.config
        frames = frames.normalized()
        matches = frames.matches
        targets = matches
        if competition_keys:
            targets = targets[targets["competition_key"].isin(list(competition_keys))]
        if match_ids is not None:
            targets = targets[targets["match_id"].isin(list(match_ids))]
        targets = targets.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
        if targets.empty:
            return pd.DataFrame(columns=[*META_COLUMNS, *TARGET_COLUMNS])

        long = team_long(frames)
        tl = targets_long(targets)  # rows 0..n-1 = home side, n..2n-1 = away side
        n = len(targets)

        team_parts: list[pd.DataFrame] = []
        for states, by in (
            (form_states(long, cfg.windows), ["team_id"]),
            (attack_states(long, cfg.windows), ["team_id"]),
            (defense_states(long, cfg.windows), ["team_id"]),
        ):
            cols = [c for c in states.columns if c not in ("team_id", "kickoff_utc")]
            team_parts.append(asof(tl, states, by, cols)[cols])
        venue = venue_states(long, cfg.venue_window)
        venue_cols = [c for c in venue.columns if c not in ("team_id", "is_home", "kickoff_utc")]
        team_parts.append(asof(tl, venue, ["team_id", "is_home"], venue_cols)[venue_cols])
        team_parts.append(season_features(tl, long))
        team_parts.append(schedule_features(tl, long))
        team_parts.append(h2h_features(tl, long, cfg.h2h_window))

        elo = elo_states(played_matches(matches), cfg.elo)
        elo_merged = asof(tl, elo, ["team_id"], ["elo_post", "elo_season"])
        team_parts.append(
            pd.DataFrame({"elo": pre_match_elo(elo_merged, tl["season_year"], cfg.elo)})
        )

        team_features = pd.concat(team_parts, axis=1)
        team_features["matches_played"] = team_features["matches_played"].fillna(0.0)
        home = _team_side(team_features, "home", n)
        away = _team_side(team_features, "away", n)

        out = targets[META_COLUMNS].copy()
        out = pd.concat([out, home, away], axis=1)
        extra: dict[str, object] = {
            f"diff_{col}": out[f"home_{col}"] - out[f"away_{col}"]
            for col in team_features.columns
            if not col.startswith("h2h_")  # h2h is already expressed from each side
        }
        extra["elo_p_home"] = [
            expected_home(h, a, cfg.elo.home_advantage)
            for h, a in zip(out["home_elo"], out["away_elo"], strict=True)
        ]
        extra["competition_importance"] = (
            out["competition_key"].map(COMPETITION_IMPORTANCE).fillna(1.0).astype(float)
        )
        out = pd.concat(
            [
                out,
                pd.DataFrame(extra, index=out.index),
                market_features(targets, frames.odds),
                player_features(targets, frames.injuries),
            ],
            axis=1,
        )
        # targets
        hg, ag = targets["home_goals"].astype(float), targets["away_goals"].astype(float)
        played = (targets["status_group"] == "finished") & hg.notna() & ag.notna()
        outcome = pd.Series(
            np.select([hg > ag, hg == ag, hg < ag], [0.0, 1.0, 2.0], np.nan), index=out.index
        )
        target_frame = pd.DataFrame(
            {
                "home_goals": hg.where(played),
                "away_goals": ag.where(played),
                "outcome": outcome.where(played),
                "total_goals": (hg + ag).where(played),
                "btts": ((hg > 0) & (ag > 0)).astype(float).where(played),
                "diff_missing_players": out["home_missing_players"] - out["away_missing_players"],
            },
            index=out.index,
        )
        out = pd.concat([out, target_frame], axis=1)
        return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns (numeric features only - no ids, metadata or targets)."""
    excluded = set(META_COLUMNS) | set(TARGET_COLUMNS)
    return [
        c
        for c in frame.columns
        if c not in excluded
        and not c.endswith("_team_id")
        and pd.api.types.is_numeric_dtype(frame[c])
    ]


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "form": ("ppg_", "win_rate_", "draw_rate_", "loss_rate_", "streak", "matches_played"),
    "attack": ("gf_avg_", "xg_for_", "shots_for_", "sot_for_", "conversion_", "xg_overperf_",
               "failed_to_score"),
    "defense": ("ga_avg_", "xg_against_", "shots_against_", "sot_against_", "clean_sheet"),
    "venue": ("venue_",),
    "season": ("season_",),
    "schedule": ("rest_days", "games_last_"),
    "h2h": ("h2h_",),
    "elo": ("elo",),
    "market": ("market_",),
    "players": ("missing_players",),
    "context": ("competition_importance",),
}  # fmt: skip


def feature_group(column: str) -> str:
    for group, markers in FEATURE_GROUPS.items():
        if any(m in column for m in markers):
            return group
    return "other"
