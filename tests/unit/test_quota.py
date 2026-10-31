from __future__ import annotations

from pathlib import Path

import pytest

from football_ai.core.exceptions import QuotaExceededError
from football_ai.data.collectors.api_football.quota import QuotaManager
from tests.conftest import FakeClock


def make_quota(clock: FakeClock, tmp_path: Path, **kwargs: int) -> QuotaManager:
    params = {"daily_limit": 10, "per_minute_limit": 3, "reserve": 2, **kwargs}
    return QuotaManager(
        state_file=tmp_path / "quota.json",
        clock=clock.time,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        **params,
    )


async def test_daily_limit_respects_reserve(clock: FakeClock, tmp_path: Path) -> None:
    quota = make_quota(clock, tmp_path, per_minute_limit=100)
    for _ in range(8):
        await quota.acquire()
    with pytest.raises(QuotaExceededError):
        await quota.acquire()
    snap = quota.snapshot()
    assert snap.used == 8 and snap.available == 0 and snap.remaining == 2


async def test_quota_free_requests_do_not_count(clock: FakeClock, tmp_path: Path) -> None:
    quota = make_quota(clock, tmp_path)
    await quota.acquire(counts_against_daily=False)
    assert quota.snapshot().used == 0


async def test_per_minute_window_waits(clock: FakeClock, tmp_path: Path) -> None:
    quota = make_quota(clock, tmp_path, daily_limit=100)
    for _ in range(3):
        await quota.acquire()
    assert clock.sleeps == []
    await quota.acquire()  # 4th request in the same minute -> must wait
    assert len(clock.sleeps) == 1
    assert 59.9 < clock.sleeps[0] <= 60.1


async def test_state_persists_across_instances(clock: FakeClock, tmp_path: Path) -> None:
    quota = make_quota(clock, tmp_path, per_minute_limit=100)
    for _ in range(4):
        await quota.acquire()
    reloaded = make_quota(clock, tmp_path, per_minute_limit=100)
    assert reloaded.snapshot().used == 4


async def test_daily_reset_at_utc_midnight(clock: FakeClock, tmp_path: Path) -> None:
    quota = make_quota(clock, tmp_path, per_minute_limit=100)
    for _ in range(8):
        await quota.acquire()
    clock.now += 12 * 3600  # 2026-11-01T00:00Z
    assert quota.snapshot().used == 0
    await quota.acquire()
    # A new instance on the new day does not inherit yesterday's counter either
    assert make_quota(clock, tmp_path).snapshot().used == 1


def test_headers_resync_counter_and_limit(clock: FakeClock, tmp_path: Path) -> None:
    quota = make_quota(clock, tmp_path, daily_limit=100, reserve=5)
    quota.update_from_headers(
        {"X-RateLimit-Requests-Limit": "75", "X-RateLimit-Requests-Remaining": "70"}
    )
    snap = quota.snapshot()
    assert snap.daily_limit == 75  # server limit lower than configured -> used
    assert snap.used == 5
    assert snap.server_remaining == 70


def test_garbage_headers_are_ignored(clock: FakeClock, tmp_path: Path) -> None:
    quota = make_quota(clock, tmp_path)
    quota.update_from_headers({"x-ratelimit-requests-remaining": "abc"})
    assert quota.snapshot().used == 0


def test_corrupted_state_file_is_tolerated(clock: FakeClock, tmp_path: Path) -> None:
    (tmp_path / "quota.json").write_text("{not json")
    assert make_quota(clock, tmp_path).snapshot().used == 0


def test_reserve_must_be_below_limit(clock: FakeClock, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_quota(clock, tmp_path, daily_limit=5, reserve=5)
