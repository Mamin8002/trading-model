"""Ретраи транзиентных ошибок Bybit (retCode 10016 и др.) и сохранение/продолжение
частичной загрузки (download_history --resume). Всё офлайн: фейковые сессии/источники.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import btc_bot.data_sources.bybit as bybit_mod  # noqa: E402
from btc_bot.data.finders import find_latest_clean_csv  # noqa: E402
from btc_bot.data.prep import clean_klines  # noqa: E402
from btc_bot.data.store import dedupe_klines, klines_to_dataframe  # noqa: E402
from btc_bot.data_sources.base import DataSourceAPIError, Kline  # noqa: E402
from btc_bot.data_sources.bybit import BybitSource, period_end_ms  # noqa: E402

import download_history as dh  # noqa: E402

STEP = 300_000
START = 1_700_000_100_000
N_TOTAL = 250
END = START + (N_TOTAL - 1) * STEP
SRV = END + STEP


def bar(t: int, closed: bool = True) -> Kline:
    return Kline(t, 1.0, 2.0, 0.5, 1.5, 1.0, 1.5, closed)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.status_code = status_code
        self.text = json.dumps(payload)
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FlakySession:
    """Фейковая requests.Session: первые fail_times ответов - заданный retCode."""

    def __init__(self, fail_times: int, code: int = 10016):
        self.fail_left = fail_times
        self.code = code
        self.headers: dict = {}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.fail_left > 0:
            self.fail_left -= 1
            return FakeResponse({"retCode": self.code, "retMsg": "internal error",
                                 "result": None, "time": str(SRV)})
        start = int((params or {}).get("start", START))
        limit = int((params or {}).get("limit", 100))
        bars = [t for t in range(max(start, START), END + STEP, STEP)][:limit]
        rows = [[str(t), "1.0", "2.0", "0.5", "1.5", "1.0", "1.5"]
                for t in sorted(bars, reverse=True)]
        return FakeResponse({"retCode": 0, "retMsg": "OK", "time": str(SRV),
                             "result": {"list": rows}})


def test_retry_on_transient_10016_then_success():
    sess = FlakySession(fail_times=2)
    src = BybitSource(base_url="https://fake.local", min_request_interval_s=0.0,
                      max_retries=5, session=sess)
    with mock.patch.object(bybit_mod.time, "sleep") as fake_sleep:
        kl = src.fetch_klines("BTCUSDT", "spot", "5", start_ms=START, end_ms=END, limit=100)
    assert [k.start_ms for k in kl] == list(range(START, END + STEP, STEP))
    assert len(kl) == N_TOTAL
    assert fake_sleep.call_count >= 2  # backoff между ретраями
    assert all(k.is_closed for k in kl)


@pytest.mark.parametrize("code", [10006, 10016, 10018])
def test_persistent_transient_code_raises_after_retries(code):
    sess = FlakySession(fail_times=999, code=code)
    src = BybitSource(base_url="https://fake.local", min_request_interval_s=0.0,
                      max_retries=2, session=sess)
    with mock.patch.object(bybit_mod.time, "sleep"):
        with pytest.raises(DataSourceAPIError) as ei:
            src.fetch_klines("BTCUSDT", "spot", "5", start_ms=START, end_ms=END, limit=100)
    assert ei.value.code == code
    assert sess.calls == 3  # 1 + max_retries, без бесконечного цикла


def test_fatal_error_carries_partial_klines():
    fail_from = START + 200 * STEP

    class FatalSource(BybitSource):
        def __init__(self):
            super().__init__(min_request_interval_s=0.0)

        def fetch_kline_page(self, symbol, category, interval, start_ms=None,
                             end_ms=None, limit=100):
            start = start_ms if start_ms is not None else START
            if start >= fail_from:
                raise DataSourceAPIError("Bybit retCode 10016: internal error", code=10016)
            return [bar(t) for t in range(start, min(start + limit * STEP, fail_from), STEP)], SRV

    with pytest.raises(DataSourceAPIError) as ei:
        FatalSource().fetch_klines("BTCUSDT", "spot", "5", start_ms=START, end_ms=END, limit=100)
    partial = getattr(ei.value, "partial_klines", None)
    assert partial is not None and len(partial) == 200
    assert partial[-1].start_ms == fail_from - STEP
    assert ei.value.pages_fetched == 2


def test_dedupe_prefers_closed_bar():
    old = [bar(START, closed=False), bar(START + STEP)]
    new = [bar(START), bar(START + STEP), bar(START + 2 * STEP)]
    merged = dedupe_klines(old + new)
    assert len(merged) == 3 and all(k.is_closed for k in merged)
    df, rep = clean_klines(klines_to_dataframe(merged), "5")
    assert rep.n_output == 3 and rep.n_unconfirmed_excluded == 0


def test_partial_save_resume_merge_and_finder(tmp_path):
    raw_dir, proc_dir = tmp_path / "raw", tmp_path / "proc"
    base_name = dh.make_base_name("spot", "BTCUSDT", "5", START, END)

    err = DataSourceAPIError("Bybit retCode 10016: internal error", code=10016)
    err.partial_klines = [bar(t) for t in range(START, START + 200 * STEP, STEP)]
    err.pages_fetched = 2

    saved = dh.save_partial_and_report_error(
        err, [], base_name, raw_dir, proc_dir,
        category="spot", symbol="BTCUSDT", interval="5",
        requested_start_ms=START, requested_end_ms=END,
    )
    assert saved is not None and "_PARTIAL" in saved.name
    meta = json.loads(saved.with_suffix(".csv.metadata.json").read_text(encoding="utf-8"))
    assert meta["complete"] is False
    assert meta["requested_start_ms"] == START and meta["requested_end_ms"] == END
    assert "10016" in meta["interrupted_by_error"]

    # частичный файл не выбирается автоматически для обучения/реплея
    assert find_latest_clean_csv(proc_dir) is None

    # resume: загрузка partial -> докачка -> склейка -> полный датасет
    part_kl, part_meta = dh.load_partial_for_resume(saved)
    assert len(part_kl) == 200
    resume_start = period_end_ms(part_kl[-1].start_ms, "5")
    assert resume_start == START + 200 * STEP
    assert part_meta["requested_end_ms"] == END

    class HealthySource(BybitSource):
        def __init__(self):
            super().__init__(min_request_interval_s=0.0)

        def fetch_kline_page(self, symbol, category, interval, start_ms=None,
                             end_ms=None, limit=100):
            start = start_ms if start_ms is not None else START
            return [bar(t) for t in range(start, min(start + limit * STEP, END + STEP, ), STEP)], SRV

    rest = HealthySource().fetch_klines("BTCUSDT", "spot", "5",
                                        start_ms=resume_start, end_ms=END, limit=100)
    assert len(rest) == 50
    merged = dedupe_klines(part_kl + rest)
    df, rep = clean_klines(klines_to_dataframe(merged), "5")
    assert rep.n_output == N_TOTAL and rep.total_missing_bars == 0 and rep.n_duplicates_removed == 0

    full_base = dh.make_base_name("spot", "BTCUSDT", "5", START, END)
    out = dh.persist_outputs(merged, full_base, raw_dir, proc_dir,
                             category="spot", symbol="BTCUSDT", interval="5",
                             extra_meta={"resumed_from": saved.name}, complete=True)
    latest = find_latest_clean_csv(proc_dir)
    assert latest is not None and latest.name == out["clean_path"].name
    assert "_PARTIAL" not in latest.name


def test_resume_error_accumulates_progress(tmp_path):
    """Повторный сбой во время --resume: старый partial + новый partial склеиваются."""
    raw_dir, proc_dir = tmp_path / "raw", tmp_path / "proc"
    base_name = dh.make_base_name("spot", "BTCUSDT", "5", START, END)

    before = [bar(t) for t in range(START, START + 200 * STEP, STEP)]
    err1 = DataSourceAPIError("Bybit retCode 10016: internal error", code=10016)
    err1.partial_klines = []
    p1 = dh.save_partial_and_report_error(err1, before, base_name, raw_dir, proc_dir,
                                          category="spot", symbol="BTCUSDT", interval="5",
                                          requested_start_ms=START, requested_end_ms=END)
    assert p1 is not None

    err2 = DataSourceAPIError("Bybit retCode 10016: internal error", code=10016)
    err2.partial_klines = [bar(t) for t in range(START + 200 * STEP, START + 230 * STEP, STEP)]
    p2 = dh.save_partial_and_report_error(err2, before, base_name, raw_dir, proc_dir,
                                          category="spot", symbol="BTCUSDT", interval="5",
                                          requested_start_ms=START, requested_end_ms=END)
    assert p2 is not None
    kl2, _ = dh.load_partial_for_resume(p2)
    assert len(kl2) == 230 and kl2[-1].start_ms == START + 229 * STEP
