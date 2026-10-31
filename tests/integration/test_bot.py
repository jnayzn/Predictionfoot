"""Telegram bot: real aiogram dispatching with a fake Telegram session (no network)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.base import BaseSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramForbiddenError
from aiogram.methods import AnswerCallbackQuery, SendMessage, TelegramMethod
from aiogram.types import CallbackQuery, Chat, Message, Update, User
from fastapi.testclient import TestClient

from football_ai.backend.main import create_app
from football_ai.backend.services.football_service import FootballService
from football_ai.bot.backend import BackendNotFoundError, HttpBackend, ServiceBackend
from football_ai.bot.formatting import TELEGRAM_LIMIT, format_analysis, truncate
from football_ai.bot.keyboards import CompetitionCB, MatchCB, MenuCB
from football_ai.bot.main import build_dispatcher
from football_ai.bot.notifications import send_daily_report
from football_ai.config.competitions import load_competitions
from football_ai.config.settings import Settings
from football_ai.prediction.engine import PredictionEngine
from tests.conftest import ROOT

TOKEN = "123456789:AAEhBP0av28h1BMKnUm7XNy7aQZzOLs6yS4"
USER = User(id=42, is_bot=False, first_name="Testeur", username="testeur")
CHAT = Chat(id=42, type="private")


class FakeSession(BaseSession):
    """Captures every Bot API call and returns plausible responses."""

    def __init__(self, forbidden_chats: set[int] | None = None) -> None:
        super().__init__()
        self.calls: list[TelegramMethod[Any]] = []
        self.forbidden = forbidden_chats or set()

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[Any],
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> Any:
        self.calls.append(method)
        if isinstance(method, SendMessage):
            if method.chat_id in self.forbidden:
                raise TelegramForbiddenError(method=method, message="bot was blocked by the user")
            return Message(
                message_id=len(self.calls),
                date=datetime.now(UTC),
                chat=Chat(id=int(method.chat_id), type="private"),
                text=method.text,
            )
        return True

    async def stream_content(self, *args: Any, **kwargs: Any) -> AsyncGenerator[bytes, None]:
        if False:  # pragma: no cover
            yield b""

    async def close(self) -> None:
        return None

    def texts(self) -> list[str]:
        return [c.text for c in self.calls if isinstance(c, SendMessage)]


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "competitions_file": ROOT / "configs" / "competitions.toml",
        "telegram_rate_limit_per_minute": 1000,
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def service(world: dict[str, Any], engine: PredictionEngine) -> FootballService:
    return FootballService(
        world["factory"], engine, load_competitions(ROOT / "configs" / "competitions.toml")
    )


@pytest.fixture
def harness(service: FootballService) -> tuple[Any, Bot, FakeSession]:
    session = FakeSession()
    bot = Bot(TOKEN, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    return build_dispatcher(ServiceBackend(service), settings()), bot, session


def message(text: str, update_id: int = 1, user: User = USER) -> Update:
    return Update(
        update_id=update_id,
        message=Message(
            message_id=update_id, date=datetime.now(UTC), chat=CHAT, from_user=user, text=text
        ),
    )


def callback(data: str, update_id: int = 1) -> Update:
    origin = Message(message_id=99, date=datetime.now(UTC), chat=CHAT, text="menu")
    return Update(
        update_id=update_id,
        callback_query=CallbackQuery(
            id=str(update_id), from_user=USER, chat_instance="ci", data=data, message=origin
        ),
    )


async def feed(harness: tuple[Any, Bot, FakeSession], update: Update) -> list[str]:
    dp, bot, session = harness
    before = len(session.texts())
    await dp.feed_update(bot, update)
    return session.texts()[before:]


async def test_start_help_and_menu(harness: tuple[Any, Bot, FakeSession]) -> None:
    [welcome] = await feed(harness, message("/start"))
    assert "Bienvenue" in welcome and "jamais des certitudes" in welcome
    sent = [c for c in harness[2].calls if isinstance(c, SendMessage)][-1]
    assert sent.reply_markup is not None  # main menu buttons
    [help_text] = await feed(harness, message("/help"))
    assert "/compare" in help_text and "/analyze" in help_text
    [perf] = await feed(harness, callback(MenuCB(action="performance").pack()))
    assert "Performances du système" in perf
    assert any(isinstance(c, AnswerCallbackQuery) for c in harness[2].calls)


async def test_today_leagues_and_analysis_flow(
    harness: tuple[Any, Bot, FakeSession], world: dict[str, Any], service: FootballService
) -> None:
    match_id = world["seeded"].upcoming_match_ids[0]
    day = service.match(match_id).kickoff_utc.date()
    leagues = await feed(harness, message("/leagues"))
    assert "Compétitions couvertes" in leagues[0]
    # competition button -> today's matches of that competition (may be empty today)
    listing = await feed(harness, callback(CompetitionCB(key="premier_league").pack()))
    assert "Matchs du" in listing[0]
    [analysis] = await feed(harness, callback(MatchCB(id=match_id).pack()))
    assert "Confiance" in analysis and "Recommandation" in analysis
    assert "jamais des certitudes" in analysis and len(analysis) <= TELEGRAM_LIMIT
    [markets] = await feed(harness, callback(MatchCB(id=match_id, view="markets").pack()))
    assert "Scores les plus probables" in markets and "+2.5" in markets
    [via_command] = await feed(harness, message(f"/analyze {match_id}"))
    assert "Facteurs" in via_command or "Incertitudes" in via_command
    assert day  # the seeded fixture date is consistent


async def test_bad_inputs_are_explained(harness: tuple[Any, Bot, FakeSession]) -> None:
    [usage] = await feed(harness, message("/analyze abc"))
    assert "Usage" in usage
    [missing] = await feed(harness, message("/analyze 999999"))
    assert "introuvable" in missing
    [bad_date] = await feed(harness, message("/results 31-10-2026"))
    assert "AAAA-MM-JJ" in bad_date
    [compare_usage] = await feed(harness, message("/compare Alpha"))
    assert "Usage" in compare_usage


async def test_team_and_compare(harness: tuple[Any, Bot, FakeSession]) -> None:
    [ambiguous] = await feed(harness, message("/team synthetic"))
    assert "Plusieurs équipes" in ambiguous
    [one] = await feed(harness, message("/team Synthetic FC 3"))
    assert "Synthetic FC 3" in one and "derniers matchs" in one
    [cmp] = await feed(harness, message("/compare Synthetic FC 1 vs Synthetic FC 2"))
    assert "Confrontation hypothétique" in cmp and "Synthetic FC 1" in cmp
    [unknown] = await feed(harness, message("/compare Synthetic FC 1 vs Nowhere United"))
    assert "inconnue" in unknown


async def test_results_predictions_and_subscriptions(
    harness: tuple[Any, Bot, FakeSession], world: dict[str, Any], service: FootballService
) -> None:
    played = service.match(world["seeded"].played_match_ids[-1])
    [results] = await feed(harness, message(f"/results {played.kickoff_utc.date().isoformat()}"))
    assert "Résultats du" in results
    [preds] = await feed(harness, message("/predictions"))
    assert "Probabilités du" in preds or "Aucun match" in preds
    [sub] = await feed(harness, message("/subscribe ligue_1"))
    assert "Rapport quotidien activé (ligue_1)" in sub
    [unsub] = await feed(harness, message("/unsubscribe"))
    assert "désactivé" in unsub
    [again] = await feed(harness, message("/unsubscribe"))
    assert "pas abonné" in again


async def test_throttling_and_allow_list(service: FootballService) -> None:
    session = FakeSession()
    bot = Bot(TOKEN, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = build_dispatcher(ServiceBackend(service), settings(telegram_rate_limit_per_minute=2))
    for i in range(3):
        await dp.feed_update(bot, message("/help", update_id=i + 1))
    assert "Trop de requêtes" in session.texts()[-1]

    private = build_dispatcher(ServiceBackend(service), settings(telegram_allowed_users="7"))
    session2 = FakeSession()
    bot2 = Bot(TOKEN, session=session2)
    await private.feed_update(bot2, message("/help"))
    assert "Accès réservé" in session2.texts()[0]


async def test_daily_report_and_blocked_users(
    world: dict[str, Any], service: FootballService
) -> None:
    backend = ServiceBackend(service)
    match = service.match(world["seeded"].upcoming_match_ids[0])
    await backend.subscribe(100, "a", [])
    await backend.subscribe(200, "b", ["premier_league"])
    await backend.subscribe(300, "c", ["serie_a"])  # nothing to report
    session = FakeSession(forbidden_chats={200})
    bot = Bot(TOKEN, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    stats = await send_daily_report(bot, backend, match.kickoff_utc.date())
    assert stats.subscribers >= 3 and stats.sent >= 1 and stats.deactivated == 1
    assert "Rapport du" in session.texts()[0]
    assert 200 not in {s.chat_id for s in await backend.subscribers()}


async def test_http_backend_against_the_api(
    service: FootballService, world: dict[str, Any]
) -> None:
    app = create_app(settings(api_keys="bot-key-123456", api_rate_limit_per_minute=1000), service)
    client = TestClient(app)
    transport = httpx.MockTransport(lambda req: _forward(client, req))
    http = HttpBackend(
        "http://api",
        "bot-key-123456",
        client=httpx.AsyncClient(
            base_url="http://api", transport=transport, headers={"X-API-Key": "bot-key-123456"}
        ),
    )
    comps = await http.competitions()
    assert len(comps) == 6
    match_id = world["seeded"].upcoming_match_ids[0]
    analysis = await http.analysis(match_id)
    assert analysis.prediction.match.id == match_id
    assert "Recommandation" in format_analysis(analysis)
    with pytest.raises(BackendNotFoundError):
        await http.analysis(999999)
    sub = await http.subscribe(555, None, [])
    assert sub.active and await http.unsubscribe(555)
    await http.aclose()


def _forward(client: TestClient, request: httpx.Request) -> httpx.Response:
    resp = client.request(
        request.method,
        request.url.raw_path.decode(),
        headers=dict(request.headers),
        content=request.content,
    )
    return httpx.Response(resp.status_code, headers=resp.headers, content=resp.content)


def test_truncate() -> None:
    long = "\n".join("ligne " * 20 for _ in range(100))
    out = truncate(long)
    assert len(out) <= TELEGRAM_LIMIT and out.endswith("(tronqué)")
    assert date.today()
