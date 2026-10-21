"""Тесты ресэмплинга: агрегация OHLCV на известном маленьком наборе для всех
требуемых таймфреймов (5m, 10m, 15m, 1h), неполные интервалы, pending-хвост,
календарь сессий, эквивалентность 1m->10m и 1m->5m->10m (decisions D3).

Эталонный набор: N минутных баров, open_i = 100 + i, high = open+2, low = open-1,
close = open+1, volume = 10*(i+1). Все ожидаемые значения посчитаны вручную.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.data.bar import BarSource, Timeframe
from tradebot.data.resampler import ResampleIssueCode, resample
from tradebot.data.sessions import SessionCalendar
from conftest import make_bar

UTC = timezone.utc
START = datetime(2024, 1, 4, 7, 0, tzinfo=UTC)


def known_series(count: int, skip: set[int] | None = None):
    skip = skip or set()
    bars = []
    for i in range(count):
        if i in skip:
            continue
        o = 100.0 + i
        bars.append(
            make_bar(
                START + timedelta(minutes=i),
                open_=o,
                high=o + 2.0,
                low=o - 1.0,
                close=o + 1.0,
                volume=10 * (i + 1),
            )
        )
    return bars


class TestAggregationKnownValues:
    def test_5m_first_and_last(self):
        bars = known_series(30)
        result = resample(bars, "1m", "5m")
        assert result.ok
        assert result.pending is None
        assert len(result.bars) == 6
        first = result.bars[0]
        assert first.ts_open == START
        assert first.open == pytest.approx(100.0)   # open первой свечи группы
        assert first.high == pytest.approx(106.0)   # max high (i=4: 104+2)
        assert first.low == pytest.approx(99.0)     # min low (i=0: 100-1)
        assert first.close == pytest.approx(105.0)  # close последней (i=4: 104+1)
        assert first.volume == 150                  # 10*(1+2+3+4+5)
        last = result.bars[-1]
        assert last.ts_open == START + timedelta(minutes=25)
        assert last.open == pytest.approx(125.0)
        assert last.close == pytest.approx(130.0)
        assert last.volume == 10 * sum(range(26, 31))

    def test_10m_known_values(self):
        bars = known_series(30)
        result = resample(bars, "1m", "10m")
        assert result.ok and result.pending is None
        assert len(result.bars) == 3
        first = result.bars[0]
        assert first.open == pytest.approx(100.0)
        assert first.high == pytest.approx(111.0)   # i=9: 109+2
        assert first.low == pytest.approx(99.0)
        assert first.close == pytest.approx(110.0)  # i=9: 109+1
        assert first.volume == 550                  # 10*sum(1..10)
        assert first.source is BarSource.DERIVED
        assert first.is_final is True

    def test_15m_known_values(self):
        bars = known_series(30)
        result = resample(bars, "1m", "15m")
        assert result.ok and result.pending is None
        assert len(result.bars) == 2
        first = result.bars[0]
        assert first.open == pytest.approx(100.0)
        assert first.high == pytest.approx(116.0)   # i=14: 114+2
        assert first.low == pytest.approx(99.0)
        assert first.close == pytest.approx(115.0)  # i=14: 114+1
        assert first.volume == 1200                 # 10*sum(1..15)

    def test_1h_known_values(self):
        bars = known_series(60)
        result = resample(bars, "1m", "1h")
        assert result.ok and result.pending is None
        assert len(result.bars) == 1
        bar = result.bars[0]
        assert bar.ts_open == START
        assert bar.open == pytest.approx(100.0)
        assert bar.high == pytest.approx(161.0)     # i=59: 159+2
        assert bar.low == pytest.approx(99.0)
        assert bar.close == pytest.approx(160.0)    # i=59: 159+1
        assert bar.volume == 18300                  # 10*sum(1..60)

    def test_10m_from_5m_equals_10m_from_1m(self):
        """Эквивалентность пути ресэмплинга (decisions D3): 1m->10m == 1m->5m->10m."""
        bars = known_series(60)
        direct = resample(bars, "1m", "10m")
        intermediate = resample(bars, "1m", "5m")
        two_step = resample(intermediate.bars, "5m", "10m")
        assert direct.ok and two_step.ok
        assert [b.ts_open for b in direct.bars] == [b.ts_open for b in two_step.bars]
        for a, b in zip(direct.bars, two_step.bars):
            assert (a.open, a.high, a.low, a.close, a.volume) == (b.open, b.high, b.low, b.close, b.volume)

    def test_symbol_propagation(self):
        bars = known_series(10)
        result = resample(bars, "1m", "5m", symbol="BTCUSDT")
        assert all(b.symbol == "BTCUSDT" for b in result.bars)


class TestIncompleteHandling:
    def test_pending_tail_not_in_bars(self):
        bars = known_series(65)  # 60 полных минут + 5 минут хвоста
        result = resample(bars, "1m", "1h")
        assert len(result.bars) == 1
        pending = result.pending
        assert pending is not None
        assert pending.is_final is False
        assert pending.ts_open == START + timedelta(minutes=60)
        # значения pending собраны только из имеющихся 5 баров (i=60..64)
        assert pending.open == pytest.approx(160.0)
        assert pending.high == pytest.approx(166.0)   # i=64: 164+2
        assert pending.low == pytest.approx(159.0)    # i=60: 160-1
        assert pending.close == pytest.approx(165.0)  # i=64: 164+1
        assert pending.volume == 3150                 # 10*(61+62+63+64+65)
        assert pending not in result.bars

    def test_10m_pending_with_odd_5m_count(self):
        bars = known_series(35)  # 3 полных 10m + хвост 5 минут
        result = resample(bars, "1m", "10m")
        assert len(result.bars) == 3
        assert result.pending is not None and result.pending.is_final is False

    def test_coverage_gap_not_emitted_by_default(self):
        bars = known_series(12, skip={3})  # дыра внутри первого 10m-интервала
        result = resample(bars, "1m", "10m")
        assert result.bars == []  # неполный интервал НЕ выдаётся за полный
        codes = [i.code for i in result.issues]
        assert ResampleIssueCode.COVERAGE_INCOMPLETE in codes
        issue = next(i for i in result.issues if i.code is ResampleIssueCode.COVERAGE_INCOMPLETE)
        assert issue.expected == 10 and issue.found == 9

    def test_coverage_gap_with_allow_incomplete_marked_non_final(self):
        bars = known_series(12, skip={3})
        result = resample(bars, "1m", "10m", allow_incomplete=True)
        assert len(result.bars) == 1
        assert result.bars[0].is_final is False  # явная пометка неполноты
        assert result.issues  # issue всё равно зафиксирован

    def test_missing_whole_interval_reported(self):
        # бары 07:00..07:04, затем сразу 07:20..07:29: интервал 07:10-07:20 пуст
        head = known_series(5)
        tail = [
            make_bar(
                START + timedelta(minutes=i),
                open_=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=10 * (i + 1),
            )
            for i in range(20, 30)
        ]
        result = resample(head + tail, "1m", "10m")
        codes = [i.code for i in result.issues]
        assert ResampleIssueCode.NO_SOURCE_BARS in codes  # пустой интервал 07:10 отмечен
        assert ResampleIssueCode.COVERAGE_INCOMPLETE in codes  # интервал 07:00 неполный (5 из 10)
        # интервал 07:20-07:30 собран полностью (10 баров, закрыт данными)
        assert any(b.ts_open == START + timedelta(minutes=20) for b in result.bars)
        # пустой интервал НЕ сфабрикован
        assert not any(b.ts_open == START + timedelta(minutes=10) for b in result.bars)

    def test_non_final_source_bars_skipped(self):
        bars = known_series(10)
        bars[-1] = make_bar(bars[-1].ts_open, is_final=False)  # незакрытая свеча из потока
        result = resample(bars, "1m", "5m")
        assert any(i.code is ResampleIssueCode.NON_FINAL_SKIPPED for i in result.issues)
        # второй 5m-интервал теперь неполный (4 из 5), и данные ещё не закрыты -> pending
        assert len(result.bars) == 1
        assert result.pending is not None and result.pending.is_final is False

    def test_empty_source(self):
        result = resample([], "1m", "5m")
        assert result.bars == [] and result.pending is None and result.ok


class TestValidation:
    def test_target_must_be_larger(self):
        with pytest.raises(ValueError):
            resample(known_series(5), "5m", "1m")
        with pytest.raises(ValueError):
            resample(known_series(5), "5m", "5m")

    def test_target_must_be_multiple(self):
        with pytest.raises(ValueError):
            resample(known_series(5), "10m", "15m")


class TestSessionCalendar:
    def test_break_interval_complete_with_calendar(self):
        # сессии 07:00-07:05 и 07:10-07:15 UTC; бары только в окнах
        calendar = SessionCalendar.from_spec(
            "UTC",
            [(range(0, 7), [("07:00", "07:05"), ("07:10", "07:15")])],
        )
        bars = []
        for minute in list(range(0, 5)) + list(range(10, 15)):
            i = minute
            o = 100.0 + i
            bars.append(
                make_bar(
                    START + timedelta(minutes=i),
                    open_=o,
                    high=o + 2.0,
                    low=o - 1.0,
                    close=o + 1.0,
                    volume=10 * (i + 1),
                )
            )
        result = resample(bars, "1m", "10m", calendar=calendar)
        # интервал 10:00-10:10: торговых слотов 5 (10:00..10:04), найдено 5 -> полный
        complete = [b for b in result.bars if b.ts_open == START]
        assert len(complete) == 1
        coverage_issues = [i for i in result.issues if i.code is ResampleIssueCode.COVERAGE_INCOMPLETE]
        assert coverage_issues == []
        # интервал 10:10-10:20 ещё не закрыт данными -> pending
        assert result.pending is not None and result.pending.ts_open == START + timedelta(minutes=10)

    def test_non_trading_interval_not_fabricated(self):
        calendar = SessionCalendar.from_spec(
            "UTC",
            [(range(0, 7), [("07:00", "07:05")])],
        )
        bars = known_series(5)  # только 10:00..10:04
        # продлим данными до 10:20, но вне сессии баров нет
        later = make_bar(START + timedelta(minutes=19), open_=200.0, high=202.0, low=199.0, close=201.0, volume=1)
        result = resample(bars + [later], "1m", "10m", calendar=calendar)
        # интервал 10:00-10:10 полный (5 торговых слотов, 5 баров)
        assert any(b.ts_open == START for b in result.bars)
        # интервал 10:10-10:20 вне сессии: бар НЕ фабрикуется, а исходный бар 10:19
        # не отбрасывается молча — фиксируется issue
        assert not any(b.ts_open == START + timedelta(minutes=10) for b in result.bars)
        from tradebot.data.resampler import ResampleIssueCode

        assert any(i.code is ResampleIssueCode.OUT_OF_SESSION_SOURCE for i in result.issues)
