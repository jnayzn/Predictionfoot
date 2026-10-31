from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from football_ai.cli import EXIT_CONFIG, EXIT_OK, main
from football_ai.config.competitions import load_competitions
from football_ai.config.settings import ApiFootballSettings
from football_ai.core.exceptions import ConfigurationError
from football_ai.core.logging import JsonFormatter, SecretRedactionFilter
from football_ai.data.storage.response_cache import ResponseCache, request_fingerprint
from tests.conftest import ROOT, TEST_KEY, FakeClock


# ------------------------------------------------------------------- settings
def test_key_is_secret_and_empty_key_means_unconfigured() -> None:
    s = ApiFootballSettings(_env_file=None, key=TEST_KEY)  # type: ignore[call-arg]
    assert s.is_configured
    assert TEST_KEY not in repr(s) and TEST_KEY not in str(s.model_dump())
    empty = ApiFootballSettings(_env_file=None, key="   ")  # type: ignore[call-arg]
    assert empty.key is None and not empty.is_configured


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_FOOTBALL_DAILY_LIMIT", "7500")
    monkeypatch.setenv("API_FOOTBALL_KEY", "abcdef123456")
    s = ApiFootballSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.daily_limit == 7500 and s.is_configured


# ---------------------------------------------------------------- competitions
def test_competitions_file_has_the_six_targets() -> None:
    comps = load_competitions(ROOT / "configs" / "competitions.toml")
    assert {c.key: c.provider_ids["api_football"] for c in comps} == {
        "premier_league": 39,
        "la_liga": 140,
        "bundesliga": 78,
        "serie_a": 135,
        "ligue_1": 61,
        "champions_league": 2,
    }


def test_competitions_duplicate_keys_rejected(tmp_path: Path) -> None:
    block = (
        '[[competitions]]\nkey="x"\nname="X"\ncountry="Y"\ntype="League"\n'
        "[competitions.provider_ids]\napi_football=1\n"
    )
    path = tmp_path / "c.toml"
    path.write_text(block * 2)
    with pytest.raises(ConfigurationError):
        load_competitions(path)


def test_competitions_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_competitions(tmp_path / "missing.toml")


# ---------------------------------------------------------------------- cache
def test_cache_roundtrip_and_ttl(tmp_path: Path) -> None:
    clock = FakeClock()
    cache = ResponseCache(tmp_path, clock=clock.time)
    fp = request_fingerprint("p", "fixtures", {"date": "2026-10-31"})
    assert cache.get("p", "fixtures", fp, 60) is None
    cache.set("p", "fixtures", fp, {"date": "2026-10-31"}, {"response": [1]})
    hit = cache.get("p", "fixtures", fp, 60)
    assert hit is not None and hit.payload == {"response": [1]}
    clock.now += 61
    assert cache.get("p", "fixtures", fp, 60) is None


def test_cache_corrupted_entry_is_a_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    fp = request_fingerprint("p", "leagues", {})
    path = tmp_path / "p" / "leagues" / f"{fp}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    assert cache.get("p", "leagues", fp, 60) is None
    assert not path.exists()


def test_fingerprint_is_order_independent() -> None:
    a = request_fingerprint("p", "e", {"a": "1", "b": "2"})
    b = request_fingerprint("p", "e", dict([("b", "2"), ("a", "1")]))
    assert a == b


# -------------------------------------------------------------------- logging
def test_json_logging_and_secret_redaction() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    redactor = SecretRedactionFilter()
    redactor.register(TEST_KEY)
    handler.addFilter(redactor)
    logger = logging.getLogger("test.redaction")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info("leak %s", TEST_KEY, extra={"header": f"key={TEST_KEY}", "n": 3})
    record = json.loads(stream.getvalue())
    assert TEST_KEY not in stream.getvalue()
    assert record["event"] == "leak ***REDACTED***"
    assert record["n"] == 3


# ------------------------------------------------------------------------ CLI
def test_cli_without_key_exits_with_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setenv("COMPETITIONS_FILE", str(ROOT / "configs" / "competitions.toml"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "var"))
    assert main(["fixtures", "--date", "2026-10-31"]) == EXIT_CONFIG
    assert "API_FOOTBALL_KEY" in capsys.readouterr().err


def test_cli_quota_works_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "var"))
    assert main(["quota"]) == EXIT_OK
    assert "used 0/100" in capsys.readouterr().out
