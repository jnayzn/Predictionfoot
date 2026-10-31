"""Persist data-quality reports (JSON file + ``data_quality_reports`` history)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from football_ai.core.io import atomic_write_json
from football_ai.data.quality.checks import QualityReport
from football_ai.database import models as db


def save_report(report: QualityReport, session: Session, reports_dir: Path) -> Path:
    data = report.to_dict()
    path = reports_dir / f"quality_{report.created_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    atomic_write_json(path, data)
    session.add(
        db.DataQualityReport(
            created_at=report.created_at,
            scope=report.scope,
            errors=report.errors,
            warnings=report.warnings,
            summary={"findings": data["findings"], "coverage": data["coverage"]},
        )
    )
    return path


def format_report(report: QualityReport) -> str:
    lines = [f"Data quality [{report.scope}] errors={report.errors} warnings={report.warnings}"]
    for f in report.findings:
        lines.append(f"  [{f.severity}] {f.check}: {f.count} - {f.message}")
        lines.extend(f"      e.g. {example}" for example in f.examples[:3])
    if report.coverage:
        lines.append("  Coverage (finished matches):")
        lines.append(
            "    competition        season matches valid  stats%  xg%  lineups%  odds%  closing%"
        )
        for c in report.coverage:
            lines.append(
                f"    {c['competition']:<18} {c['season']:>6} {c['finished_matches']:>7} "
                f"{c['valid_for_training']:>5} {c['pct_stats']:>7} {c['pct_xg']:>5} "
                f"{c['pct_lineups']:>8} {c['pct_odds']:>6} {c['pct_closing_odds']:>8}"
            )
    return "\n".join(lines)
