"""Telegram message rendering (HTML parse mode, French).

All user-provided or data-provided strings are escaped. Every analysis ends
with the probabilistic disclaimer. Messages are truncated to Telegram's limit.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from html import escape
from zoneinfo import ZoneInfo

from football_ai.backend.schemas.api import (
    AnalysisOut,
    CompetitionOut,
    MatchOut,
    PerformanceOut,
    PredictionOut,
    ResultOut,
    TeamProfileOut,
)

TELEGRAM_LIMIT = 4096
DISPLAY_TZ = ZoneInfo("Europe/Paris")
DISCLAIMER = (
    "<i>Probabilités statistiques, jamais des certitudes. Pas un conseil de pari ; "
    "les paris comportent un risque de perte.</i>"
)
COMPETITION_EMOJI = {
    "premier_league": "🏴",
    "la_liga": "🇪🇸",
    "bundesliga": "🇩🇪",
    "serie_a": "🇮🇹",
    "ligue_1": "🇫🇷",
    "champions_league": "⭐",
}
LEVELS = {"low": "faible", "medium": "moyenne", "high": "élevée"}
STATUS = {
    "value": "valeur détectée",
    "none": "aucune recommandation fiable",
    "no_odds": "pas de cotes",
}


def pct(p: float | None) -> str:
    return "—" if p is None else f"{p * 100:.0f} %"


def truncate(text: str, limit: int = TELEGRAM_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20].rsplit("\n", 1)[0] + "\n…(tronqué)"


def kickoff(m: MatchOut) -> str:
    local = m.kickoff_utc.astimezone(DISPLAY_TZ)
    return local.strftime("%d/%m %H:%M") if m.kickoff_time_known else local.strftime("%d/%m")


def match_title(m: MatchOut) -> str:
    return f"{escape(m.home.name)} – {escape(m.away.name)}"


def match_line(m: MatchOut) -> str:
    emoji = COMPETITION_EMOJI.get(m.competition, "⚽")
    if m.score is not None:
        return f"{emoji} {match_title(m)} <b>{m.score['home']}-{m.score['away']}</b>"
    return f"{emoji} {kickoff(m)} · {match_title(m)}"


def format_competitions(items: Iterable[CompetitionOut]) -> str:
    lines = ["<b>Compétitions couvertes</b>", ""]
    lines += [
        f"{COMPETITION_EMOJI.get(c.key, '⚽')} {escape(c.name)} ({escape(c.country)})"
        for c in items
    ]
    lines += ["", "Choisissez une compétition :"]
    return "\n".join(lines)


def format_fixtures(matches: list[MatchOut], day: date, title: str = "Matchs") -> str:
    head = f"<b>{escape(title)} du {day:%d/%m/%Y}</b>"
    if not matches:
        return f"{head}\n\nAucun match des compétitions couvertes ce jour-là."
    lines = [head, ""] + [match_line(m) for m in matches]
    lines += ["", "Sélectionnez un match pour l'analyse complète."]
    return truncate("\n".join(lines))


def probability_bar(p: float, width: int = 10) -> str:
    filled = round(p * width)
    return "▰" * filled + "▱" * (width - filled)


def format_prediction_block(p: PredictionOut) -> list[str]:
    probs = p.probabilities
    home, away = escape(p.match.home.name), escape(p.match.away.name)
    lines = [
        f"🏠 {home} : <b>{pct(probs['home'])}</b> {probability_bar(probs['home'])}",
        f"🤝 Nul : <b>{pct(probs['draw'])}</b> {probability_bar(probs['draw'])}",
        f"✈️ {away} : <b>{pct(probs['away'])}</b> {probability_bar(probs['away'])}",
    ]
    if p.expected_goals:
        xg = p.expected_goals
        lines.append(f"⚽ Buts attendus : {xg['home']:.2f} – {xg['away']:.2f}")
    lines.append(
        f"🎯 Confiance : {LEVELS.get(p.confidence.level, p.confidence.level)} "
        f"({p.confidence.score:.2f})"
    )
    return lines


def format_prediction_short(p: PredictionOut) -> str:
    probs = p.probabilities
    return (
        f"{match_line(p.match)}\n   1 {pct(probs['home'])} · N {pct(probs['draw'])} · "
        f"2 {pct(probs['away'])}"
    )


def format_markets(p: PredictionOut) -> str:
    lines = [f"<b>Marchés — {match_title(p.match)}</b>", ""]
    m = p.markets
    if not m:
        lines.append("Marchés détaillés indisponibles pour ce modèle.")
        return "\n".join(lines)
    dc = m["double_chance"]
    lines += [
        f"Double chance : 1N {pct(dc['1x'])} · 12 {pct(dc['12'])} · N2 {pct(dc['x2'])}",
        "",
        "<b>Nombre de buts</b>",
    ]
    for line, probs in m["over_under"].items():
        lines.append(f"+{line} : {pct(probs['over'])} · −{line} : {pct(probs['under'])}")
    lines += [
        f"Les deux équipes marquent : oui {pct(m['btts']['yes'])} · non {pct(m['btts']['no'])}",
        "",
        "<b>Scores les plus probables</b>",
    ]
    lines += [f"{s['score']} : {pct(s['probability'])}" for s in m["top_scores"]]
    lines += ["", "<b>Buts par équipe</b>"]
    for side, name in (("home", p.match.home.name), ("away", p.match.away.name)):
        tg = m["team_goals"][side]
        lines.append(
            f"{escape(name)} : ≥1 {pct(tg['1+'])} · ≥2 {pct(tg['2+'])} · ≥3 {pct(tg['3+'])}"
        )
    lines += ["", DISCLAIMER]
    return truncate("\n".join(lines))


def _team_block(t: TeamProfileOut) -> list[str]:
    lines = [f"<b>{escape(t.team.name)}</b> — forme : {escape(t.form_last5) or '—'}"]
    stats = []
    if t.goals_for_avg is not None and t.goals_against_avg is not None:
        stats.append(f"buts {t.goals_for_avg:.2f} / {t.goals_against_avg:.2f} par match")
    if t.xg_for_avg is not None and t.xg_against_avg is not None:
        stats.append(f"xG {t.xg_for_avg:.2f} / {t.xg_against_avg:.2f}")
    if t.elo is not None:
        stats.append(f"Elo {t.elo:.0f}")
    if stats:
        lines.append("   " + " · ".join(stats))
    lines += [f"   ✅ {escape(s)}" for s in t.strengths]
    lines += [f"   ⚠️ {escape(w)}" for w in t.weaknesses]
    return lines


def format_analysis(a: AnalysisOut) -> str:
    p = a.prediction
    lines = [
        f"<b>{match_title(p.match)}</b>",
        f"{COMPETITION_EMOJI.get(p.match.competition, '⚽')} {kickoff(p.match)}",
        "",
        *format_prediction_block(p),
        "",
        escape(a.summary),
        "",
        *_team_block(a.home),
        *_team_block(a.away),
    ]
    if a.key_factors:
        lines += ["", "<b>Facteurs déterminants</b>"]
        lines += [f"• {escape(f.text)}" for f in a.key_factors]
    if a.uncertainties:
        lines += ["", "<b>Incertitudes</b>"]
        lines += [f"• {escape(u)}" for u in a.uncertainties]
    rec = a.recommendation
    lines += ["", f"<b>Recommandation : {STATUS.get(rec.status, rec.status)}</b>"]
    for s in rec.selections:
        lines.append(
            f"• {escape(s.label)} — cote {s.odds:.2f}, "
            f"proba ajustée {pct(s.adjusted_probability)}, "
            f"valeur attendue {s.expected_value * 100:+.1f} %"
        )
    if not rec.selections:
        lines += [f"• {escape(r)}" for r in rec.reasons[:3]]
        if rec.most_likely:
            lines.append("Scénarios les plus probables (information, pas une recommandation) :")
            lines += [
                f"   – {escape(x['label'])} : {pct(x['probability'])}" for x in rec.most_likely
            ]
    lines += ["", DISCLAIMER]
    return truncate("\n".join(lines))


def format_compare(p: PredictionOut) -> str:
    lines = [
        f"<b>Confrontation hypothétique : {match_title(p.match)}</b>",
        f"(avantage du terrain pour {escape(p.match.home.name)}, données à ce jour)",
        "",
        *format_prediction_block(p),
    ]
    if p.markets:
        lines.append(f"Score le plus probable : {p.markets['most_likely_score']}")
    if p.confidence.uncertainties:
        lines += ["", "<b>Incertitudes</b>"] + [
            f"• {escape(u)}" for u in p.confidence.uncertainties
        ]
    lines += ["", DISCLAIMER]
    return truncate("\n".join(lines))


def format_team(team_name: str, matches: list[MatchOut]) -> str:
    lines = [f"<b>{escape(team_name)}</b> — derniers matchs", ""]
    lines += [f"{kickoff(m)} {match_line(m)}" for m in matches] or ["Aucun match joué."]
    return truncate("\n".join(lines))


def format_results(results: list[ResultOut], day: date) -> str:
    head = f"<b>Résultats du {day:%d/%m/%Y}</b>"
    if not results:
        return f"{head}\n\nAucun résultat disponible."
    lines = [head, ""]
    for r in results:
        mark = "" if r.prediction_correct is None else (" ✅" if r.prediction_correct else " ❌")
        lines.append(match_line(r.match) + mark)
    lines += ["", "✅/❌ : issue la plus probable de la prédiction publiée avant le match."]
    return truncate("\n".join(lines))


def format_performance(perf: PerformanceOut) -> str:
    live = perf.live
    lines = ["<b>Performances du système</b>", "", "<b>Prédictions publiées avant les matchs</b>"]
    if live.get("n"):
        lines += [
            f"Matchs évalués : {live['n']}",
            f"Issue la plus probable correcte : {pct(live.get('accuracy'))}",
            f"Log loss : {live['log_loss']:.3f} · Brier : {live['brier']:.3f}",
        ]
    else:
        lines.append("Pas encore de prédiction évaluée.")
    if perf.backtest and perf.backtest.get("overall"):
        lines += ["", "<b>Backtest (simulation sur saisons passées)</b>"]
        ranked = sorted(perf.backtest["overall"].items(), key=lambda kv: kv[1]["log_loss"])
        for name, m in ranked[:5]:
            lines.append(
                f"{escape(name)} : log loss {m['log_loss']:.3f}, exactitude {pct(m['accuracy'])}"
            )
    lines += ["", "<i>Plus la log loss est basse, meilleures sont les probabilités.</i>"]
    return "\n".join(lines)


HELP = (
    """<b>Football AI Predictor</b> — analyses statistiques probabilistes

/today — matchs du jour
/leagues — choisir une compétition
/analyze &lt;id&gt; — analyse complète d'un match
/predictions — probabilités des matchs du jour
/team &lt;nom&gt; — derniers matchs d'une équipe
/compare &lt;équipe A&gt; vs &lt;équipe B&gt; — confrontation hypothétique
/results [AAAA-MM-JJ] — résultats et vérification des prédictions
/performance — performances historiques du système
/subscribe, /unsubscribe — rapport quotidien automatique
/help — cette aide

"""
    + DISCLAIMER
)
