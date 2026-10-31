"""Scientific backtest: every model + the ensemble on the locked test period.

* models are walk-forward re-fitted (never see the future);
* the ensemble weights/calibration are learned on the validation period only;
* the **market** (closing odds, overround removed) is the strongest benchmark;
  it is used for evaluation only, never as an input;
* results overall, per competition and per season, with paired significance
  tests (day-block bootstrap) against the references.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from football_ai.training.datasets import TemporalSplit
from football_ai.training.ensemble_pipeline import build_ensemble, predictions_frame
from football_ai.training.evaluate import calibration_table, evaluate_probabilities
from football_ai.training.significance import (
    PairedComparison,
    compare_losses,
    per_match_log_loss,
    per_match_rps,
)
from football_ai.training.train import ModelFactory, evaluate_models

OUTCOME_KEYS = ("home", "draw", "away")
MARKET = "market"
DEFAULT_REFERENCES = ("frequency", "elo", "poisson", MARKET)


@dataclass(slots=True)
class BacktestReport:
    period: dict[str, Any]
    overall: dict[str, dict[str, Any]]
    by_competition: dict[str, dict[str, dict[str, Any]]]
    by_season: dict[str, dict[str, dict[str, Any]]]
    comparisons: list[dict[str, Any]]
    calibration: dict[str, list[dict[str, float]]]
    ensemble: dict[str, Any] | None = None
    market_subset: dict[str, dict[str, Any]] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        cols = ["n", "log_loss", "brier", "rps", "accuracy", "ece"]
        lines = ["# Backtest report", ""]
        lines += [f"- {k}: {v}" for k, v in self.period.items()]
        lines += ["", "## Overall (test period)", "", _md_table(self.overall, cols)]
        if self.market_subset:
            lines += ["", "## Matches with closing odds (same subset for every model)", "",
                      _md_table(self.market_subset, cols)]  # fmt: skip
        lines += ["", "## Paired comparisons (day-block bootstrap, 95% CI)", ""]
        lines += ["| model | reference | metric | n | mean diff | CI | p-value | verdict |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for c in self.comparisons:
            verdict = (
                "better" if c["ci_high"] < 0 else "worse" if c["ci_low"] > 0 else "not significant"
            )
            lines.append(
                f"| {c['model']} | {c['reference']} | {c['metric']} | {c['n']} | "
                f"{c['mean_difference']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                f"{c['p_value']:.3f} | {verdict} |"
            )
        for title, block in (("competition", self.by_competition), ("season", self.by_season)):
            lines += ["", f"## By {title}", ""]
            for key, models in block.items():
                lines += [f"### {key}", "", _md_table(models, cols), ""]
        if self.ensemble:
            lines += ["## Ensemble", "", "```", str(self.ensemble), "```"]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def _md_table(models: Mapping[str, Mapping[str, Any]], cols: Sequence[str]) -> str:
    ordered = sorted(models.items(), key=lambda kv: kv[1].get("log_loss", np.inf))
    head = "| model | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    rows = [
        "| " + name + " | " + " | ".join(_fmt(m.get(c)) for c in cols) + " |" for name, m in ordered
    ]
    return "\n".join([head, sep, *rows])


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _probas(pred: pd.DataFrame, models: Sequence[str]) -> dict[str, np.ndarray]:
    return {m: pred[[f"{m}__{k}" for k in OUTCOME_KEYS]].to_numpy(dtype=float) for m in models}


def _metrics_by(pred: pd.DataFrame, models: Sequence[str], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for value, group in pred.groupby(key):
        y = group["outcome"].to_numpy(dtype=int)
        out[str(value)] = {
            m: evaluate_probabilities(y, p) for m, p in _probas(group, models).items()
        }
    return out


def run_backtest(
    frame: pd.DataFrame,
    split: TemporalSplit,
    factories: Mapping[str, ModelFactory],
    *,
    market: pd.DataFrame | None = None,
    with_ensemble: bool = True,
    references: Sequence[str] = DEFAULT_REFERENCES,
    refit_every_days: int = 30,
    n_boot: int = 2000,
) -> BacktestReport:
    if split.test.empty:
        raise ValueError("the backtest needs a test period (--test-seasons)")
    notes: list[str] = []
    ensemble_info: dict[str, Any] | None = None
    if with_ensemble and len(factories) > 1 and not split.validation.empty:
        _, ens_report = build_ensemble(
            frame, split, factories, refit_every_days=refit_every_days,
            evaluate_test=True, fit_production_members=False,
        )  # fmt: skip
        assert ens_report.test_predictions is not None
        pred = ens_report.test_predictions
        ensemble_info = ens_report.to_dict()
        ensemble_info.pop("test", None)
    else:
        results = evaluate_models(
            frame, split, factories, on="test", refit_every_days=refit_every_days
        )
        pred = predictions_frame(results[0].rows, {r.name: r.proba for r in results})
        if with_ensemble:
            notes.append("ensemble skipped (needs >= 2 models and a validation period)")

    models = [c[: -len("__home")] for c in pred.columns if c.endswith("__home")]
    market_subset = None
    if market is not None and not market.empty:
        pred = pred.merge(market, on="match_id", how="left")
        has_market = pred["market__home"].notna()
        notes.append(f"closing odds available for {int(has_market.sum())}/{len(pred)} test matches")
        if has_market.any():
            sub = pred[has_market]
            y_sub = sub["outcome"].to_numpy(dtype=int)
            market_subset = {
                m: evaluate_probabilities(y_sub, p)
                for m, p in _probas(sub, [*models, MARKET]).items()
            }
    else:
        notes.append("no closing odds: market benchmark unavailable")

    y = pred["outcome"].to_numpy(dtype=int)
    probas = _probas(pred, models)
    overall = {m: evaluate_probabilities(y, p) for m, p in probas.items()}
    calibration = {m: calibration_table(y, p) for m, p in probas.items()}
    blocks = pd.to_datetime(pred["kickoff_utc"], utc=True).dt.date.to_numpy()

    comparisons: list[PairedComparison] = []
    candidates = [m for m in models if m not in references or m == "ensemble"]
    for ref in references:
        if ref == MARKET:
            if market_subset is None:
                continue
            mask = pred["market__home"].notna().to_numpy()
            ref_p = _probas(pred[mask], [MARKET])[MARKET]
        elif ref in probas:
            mask = np.ones(len(pred), dtype=bool)
            ref_p = probas[ref]
        else:
            continue
        for model in models:
            if model == ref or (model not in candidates and ref != MARKET):
                continue
            p = probas[model][mask]
            for metric, fn in (("log_loss", per_match_log_loss), ("rps", per_match_rps)):
                comparisons.append(
                    compare_losses(
                        fn(y[mask], p),
                        fn(y[mask], ref_p),
                        blocks[mask],
                        model=model,
                        reference=ref,
                        metric=metric,
                        n_boot=n_boot,
                    )
                )

    test = split.test
    period = {
        "test_from": test["kickoff_utc"].min().isoformat(),
        "test_to": test["kickoff_utc"].max().isoformat(),
        "test_matches": len(pred),
        "competitions": sorted(pred["competition_key"].unique().tolist()),
        "refit_every_days": refit_every_days,
    }
    return BacktestReport(
        period=period,
        overall=overall,
        by_competition=_metrics_by(pred, models, "competition_key"),
        by_season=_metrics_by(pred, models, "season_year"),
        comparisons=[asdict(c) for c in comparisons],
        calibration=calibration,
        ensemble=ensemble_info,
        market_subset=market_subset,
        notes=notes,
    )
