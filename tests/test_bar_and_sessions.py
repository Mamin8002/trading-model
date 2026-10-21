"""Тесты контракта Bar/Timeframe и календаря сессий."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.data.bar import UTC, Bar, BarSource, Timeframe, floor_to_grid
from tradebot.data.sessions import SessionCalendar, SessionWindow
from conftest import make_bar


class TestTimeframe:
    def test_supported_set(self):
        assert [t.value for t in Timeframe] == ["1m", "5m", "10m", "15m", "1h"]

    @pytest.mark.parametrize(
        ("code", "minutes"),
        [("1m", 1), ("5m", 5), ("10m", 10), ("15m", 15), ("1h", 60)],
    )
    def test_minutes(self, code, minutes):
        tf = Timeframe.parse(code)
        assert tf.minutes == minutes
        assert tf.duration == timedelta(minutes=minutes)

    def test_parse_rejects_unknown(self):
        for bad in ("2m", "M10", "10min", "", "30m", None, 5):
            with pytest.raises(ValueError):
                Timeframe.parse(bad)

    def test_parse_case_insensitive(self):
        assert Timeframe.parse("1H") is Timeframe.H1


class TestBar:
    def test_requires_aware_utc(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            Bar(
                ts_open=datetime(2024, 1, 4, 10, 0),  # naive
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=10,
            )

    def test_normalizes_non_utc_to_utc(self):
        jst = timezone(timedelta(hours=9))  # произвольная зона UTC+9 (без DST)
        bar = make_bar(datetime(2024, 1, 4, 10, 0, tzinfo=jst))
        assert bar.ts_open == datetime(2024, 1, 4, 1, 0, tzinfo=UTC)
        assert bar.ts_open.tzinfo == UTC

    def test_rejects_nan_and_negative_volume(self):
        with pytest.raises(ValueError):
            make_bar(datetime(2024, 1, 4, tzinfo=UTC), open_=float("nan"))
        with pytest.raises(ValueError):
            Bar(
                ts_open=datetime(2024, 1, 4, tzinfo=UTC),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=-5,
            )

    def test_accepts_fractional_volume(self):
        # крипторынок: объём в базовом активе дробный (docs/decisions.md D13)
        bar = Bar(
            ts_open=datetime(2024, 1, 4, tzinfo=UTC),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=0.12345,
        )
        assert bar.volume == pytest.approx(0.12345)

    def test_rejects_bool_volume(self):
        with pytest.raises(TypeError):
            Bar(
                ts_open=datetime(2024, 1, 4, tzinfo=UTC),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=True,  # type: ignore[arg-type]
            )

    def test_ts_close(self):
        bar = make_bar(datetime(2024, 1, 4, 7, 0, tzinfo=UTC))
        assert bar.ts_close("5m") == datetime(2024, 1, 4, 7, 5, tzinfo=UTC)


class TestFloorToGrid:
    def test_floors_to_utc_grid(self):
        ts = datetime(2024, 1, 4, 7, 37, tzinfo=UTC)
        assert floor_to_grid(ts, 10) == datetime(2024, 1, 4, 7, 30, tzinfo=UTC)
        assert floor_to_grid(ts, 60) == datetime(2024, 1, 4, 7, 0, tzinfo=UTC)
        assert floor_to_grid(ts, 5) == datetime(2024, 1, 4, 7, 35, tzinfo=UTC)

    def test_integer_hour_offset_keeps_alignment(self):
        # любая целочасовая зона (например Asia/Tokyo = UTC+9): 10-минутная сетка
        # совпадает в обеих зонах
        from zoneinfo import ZoneInfo

        local = datetime(2024, 1, 4, 10, 37, tzinfo=ZoneInfo("Asia/Tokyo"))
        floored_utc = floor_to_grid(local.astimezone(UTC), 10)
        assert floored_utc.astimezone(ZoneInfo("Asia/Tokyo")).minute == 30

    def test_rejects_naive(self):
        with pytest.raises(ValueError):
            floor_to_grid(datetime(2024, 1, 4, 7, 37), 10)


class TestSessionCalendar:
    @pytest.fixture()
    def calendar(self) -> SessionCalendar:
        # пн-пт 10:00-12:00 UTC (условное демо-окно)
        return SessionCalendar.from_spec(
            "UTC",
            [(range(0, 5), [("10:00", "12:00")])],
        )

    def test_trading_time_boundaries(self, calendar):
        assert calendar.is_trading_time(datetime(2024, 1, 4, 10, 0, tzinfo=UTC))  # четверг, старт включительно
        assert not calendar.is_trading_time(datetime(2024, 1, 4, 12, 0, tzinfo=UTC))  # конец исключительно
        assert not calendar.is_trading_time(datetime(2024, 1, 4, 9, 59, tzinfo=UTC))
        assert not calendar.is_trading_time(datetime(2024, 1, 6, 10, 30, tzinfo=UTC))  # суббота

    def test_trading_slots_counts_expected(self, calendar):
        start = datetime(2024, 1, 4, 10, 0, tzinfo=UTC)
        end = datetime(2024, 1, 4, 12, 0, tzinfo=UTC)
        slots = calendar.trading_slots(start, end, "1m")
        assert len(slots) == 120
        slots_5m = calendar.trading_slots(start, end, "5m")
        assert len(slots_5m) == 24

    def test_trading_slots_excludes_break(self, calendar):
        # окно 11:30-12:30 UTC: только 11:30-11:59 торговые (30 минут)
        start = datetime(2024, 1, 4, 11, 30, tzinfo=UTC)
        end = datetime(2024, 1, 4, 12, 30, tzinfo=UTC)
        assert len(calendar.trading_slots(start, end, "1m")) == 30

    def test_always_on_calendar_is_24_7(self):
        # крипторынок: календарь 24/7 (always_on_calendar) — торговое время всегда,
        # включая выходные и переход через полночь
        from tradebot.data.sessions import always_on_calendar

        cal = always_on_calendar()
        assert cal.is_trading_time(datetime(2024, 1, 6, 23, 59, tzinfo=UTC))  # суббота
        assert cal.is_trading_time(datetime(2024, 1, 7, 0, 0, tzinfo=UTC))    # воскресенье, полночь
        slots = cal.trading_slots(datetime(2024, 1, 5, tzinfo=UTC), datetime(2024, 1, 7, tzinfo=UTC), "1h")
        assert len(slots) == 48  # двое суток без выходных

    def test_window_parse_validation(self):
        with pytest.raises(ValueError):
            SessionWindow.parse("25:00", "26:00")
        with pytest.raises(ValueError):
            SessionWindow.parse("12:00", "12:00")
        with pytest.raises(ValueError):
            SessionWindow.parse("13:00", "12:00")
