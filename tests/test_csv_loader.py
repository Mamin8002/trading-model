"""Тесты загрузчика и валидатора CSV (критерии приёмки этапа: сортировка, дубликаты,
пустой файл, пропуски, корректность OHLC, нечисловые значения)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.data.bar import UTC
from tradebot.data.csv_loader import REQUIRED_COLUMNS, DataValidationError, load_csv
from tradebot.data.sessions import SessionCalendar
from tradebot.data.validator import IssueCode, Severity, validate_bars
from conftest import make_bar, minute_series, write_csv


def msk(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone(timedelta(hours=3)))


def row(ts: datetime, o=100.0, h=101.0, l=99.0, c=100.5, v=1000) -> str:
    return f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{o},{h},{l},{c},{v}"


class TestHappyPath:
    def test_loads_demo_csv(self):
        result = load_csv(
            Path("data/demo/DEMO_M1.csv"),
            timeframe="1m",
            symbol="DEMO@DEMO",
            source_timezone="Europe/Moscow",
            timestamp_semantics="bar_open",
        )
        assert result.ok
        assert len(result.bars) == 120
        # 10:00 MSK -> 07:00 UTC, нормализация к bar_open
        assert result.bars[0].ts_open == datetime(2024, 1, 4, 7, 0, tzinfo=UTC)
        assert result.bars[0].open == pytest.approx(100.00)
        assert result.bars[0].volume == 1000
        assert result.bars[-1].ts_open == datetime(2024, 1, 4, 8, 59, tzinfo=UTC)
        assert all(b.symbol == "DEMO@DEMO" for b in result.bars)

    def test_bar_close_semantics_shifts_by_timeframe(self, tmp_path):
        # те же бары, но метки = конец бара (10:01..10:05 для 5m)
        rows = []
        for i in range(5):
            ts = msk(2024, 1, 4, 10, 0) + timedelta(minutes=5 * (i + 1))
            rows.append(row(ts, o=100 + i, h=101 + i, l=99 + i, c=100.5 + i, v=10 * (i + 1)))
        path = write_csv(tmp_path / "close_stamped.csv", rows)
        result = load_csv(path, timeframe="5m", timestamp_semantics="bar_close")
        opens = [b.ts_open for b in result.bars]
        assert opens == [datetime(2024, 1, 4, 7, 0, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(5)]

    def test_iso_datetime_with_offset(self, tmp_path):
        rows = [row(msk(2024, 1, 4, 10, 0)), "2024-01-04T10:01:00+03:00,100,101,99,100.5,500"]
        path = write_csv(tmp_path / "iso.csv", rows)
        result = load_csv(path, timeframe="1m")
        assert len(result.bars) == 2

    def test_column_order_irrelevant(self, tmp_path):
        path = tmp_path / "reordered.csv"
        path.write_text(
            "close,datetime,volume,open,high,low\n"
            "100.5,2024-01-04 10:00:00,1000,100.0,101.0,99.0\n",
            encoding="utf-8",
        )
        result = load_csv(path, timeframe="1m")
        assert len(result.bars) == 1
        assert result.bars[0].close == pytest.approx(100.5)


class TestErrors:
    def _expect_error(self, tmp_path, rows, code: IssueCode, header=None):
        path = write_csv(tmp_path / "bad.csv", rows, header=header or ",".join(REQUIRED_COLUMNS))
        with pytest.raises(DataValidationError) as exc_info:
            load_csv(path, timeframe="1m")
        codes = [i.code for i in exc_info.value.report.errors]
        assert code in codes, f"ожидался {code.value}, получены {[c.value for c in codes]}"
        return exc_info.value

    def test_empty_file_zero_bytes(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_bytes(b"")
        with pytest.raises(DataValidationError) as exc:
            load_csv(path, timeframe="1m")
        assert IssueCode.E_EMPTY_FILE in [i.code for i in exc.value.report.errors]

    def test_header_only_file(self, tmp_path):
        path = write_csv(tmp_path / "header_only.csv", [])
        with pytest.raises(DataValidationError) as exc:
            load_csv(path, timeframe="1m")
        codes = [i.code for i in exc.value.report.errors]
        assert IssueCode.E_EMPTY_FILE in codes

    def test_missing_column(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,100.0,101.0,99.0,1000"],
            IssueCode.E_HEADER_MISSING_COLS,
            header="datetime,open,high,low,volume",
        )

    def test_unknown_column(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,100.0,101.0,99.0,100.5,1000,extra"],
            IssueCode.E_HEADER_UNKNOWN_COLS,
            header="datetime,open,high,low,close,volume,extra",
        )

    def test_non_numeric_price(self, tmp_path):
        err = self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,abc,101.0,99.0,100.5,1000"],
            IssueCode.E_NUMBER_PARSE,
        )
        assert "нечисловое" in str(err)

    def test_comma_decimal_rejected(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,100,5,101.0,99.0,100.5,1000"],  # запятая ломает число полей
            IssueCode.E_NUMBER_PARSE,
        )

    def test_nan_inf_rejected(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,nan,101.0,99.0,100.5,1000"],
            IssueCode.E_NUMBER_PARSE,
        )

    def test_bad_datetime(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["04.01.2024 10:00,100.0,101.0,99.0,100.5,1000"],
            IssueCode.E_DATETIME_PARSE,
        )

    def test_negative_volume(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,100.0,101.0,99.0,100.5,-5"],
            IssueCode.E_VOLUME_INVALID,
        )

    def test_fractional_volume_rejected(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,100.0,101.0,99.0,100.5,10.5"],
            IssueCode.E_VOLUME_INVALID,
        )

    def test_ohlc_violation_high_below_low(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,100.0,98.0,99.0,100.5,1000"],  # high < low и high < close
            IssueCode.E_OHLC,
        )

    def test_ohlc_close_above_high(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,100.0,100.2,99.0,101.5,1000"],
            IssueCode.E_OHLC,
        )

    def test_zero_price_rejected(self, tmp_path):
        self._expect_error(
            tmp_path,
            ["2024-01-04 10:00:00,0,1.0,0,0.5,1000"],
            IssueCode.E_PRICE_INVALID,
        )

    def test_duplicate_timestamps(self, tmp_path):
        ts = msk(2024, 1, 4, 10, 0)
        self._expect_error(
            tmp_path,
            [row(ts), row(ts), row(ts + timedelta(minutes=1))],
            IssueCode.E_DUPLICATE_TS,
        )

    def test_unsorted_rows(self, tmp_path):
        ts = msk(2024, 1, 4, 10, 0)
        err = self._expect_error(
            tmp_path,
            [row(ts + timedelta(minutes=1)), row(ts)],
            IssueCode.E_UNSORTED,
        )
        assert "порядок времени" in str(err)

    def test_off_grid_timestamp_for_5m(self, tmp_path):
        rows = [row(msk(2024, 1, 4, 10, 3))]  # 10:03 не на 5-минутной сетке
        path = write_csv(tmp_path / "offgrid.csv", rows)
        with pytest.raises(DataValidationError) as exc:
            load_csv(path, timeframe="5m")
        assert IssueCode.E_OFF_GRID in [i.code for i in exc.value.report.errors]

    def test_missing_file(self, tmp_path):
        with pytest.raises(DataValidationError):
            load_csv(tmp_path / "no_such_file.csv", timeframe="1m")

    def test_error_message_is_readable(self, tmp_path):
        ts = msk(2024, 1, 4, 10, 0)
        path = write_csv(tmp_path / "bad.csv", [row(ts), row(ts)])
        with pytest.raises(DataValidationError) as exc:
            load_csv(path, timeframe="1m")
        message = str(exc.value)
        assert "не прошёл проверку данных" in message
        assert "E_DUPLICATE_TS" in message
        assert "строка" in message


class TestWarnings:
    def test_gap_warning_no_calendar(self, tmp_path):
        ts = msk(2024, 1, 4, 10, 0)
        rows = [row(ts), row(ts + timedelta(minutes=3))]  # пропуск 2 минут
        path = write_csv(tmp_path / "gap.csv", rows)
        result = load_csv(path, timeframe="1m")
        gaps = [i for i in result.report.warnings if i.code is IssueCode.W_GAP]
        assert len(gaps) == 1
        assert "2 бар(ов)" in gaps[0].message

    def test_gap_explained_by_calendar_is_not_warning(self, tmp_path):
        # окно сессии 10:00-10:01 и 10:03-10:05: пропуск 10:01-10:03 законный
        calendar = SessionCalendar.from_spec(
            "Europe/Moscow",
            [(range(0, 7), [("10:00", "10:01"), ("10:03", "10:05")])],
        )
        ts = msk(2024, 1, 4, 10, 0)
        rows = [row(ts), row(ts + timedelta(minutes=3))]
        path = write_csv(tmp_path / "gap_cal.csv", rows)
        result = load_csv(path, timeframe="1m", calendar=calendar)
        gaps = [i for i in result.report.warnings if i.code is IssueCode.W_GAP]
        assert gaps == []

    def test_zero_volume_warning(self, tmp_path):
        rows = [row(msk(2024, 1, 4, 10, 0), v=0)]
        path = write_csv(tmp_path / "zerovol.csv", rows)
        result = load_csv(path, timeframe="1m")
        assert any(i.code is IssueCode.W_ZERO_VOLUME for i in result.report.warnings)

    def test_out_of_session_warning(self, tmp_path):
        calendar = SessionCalendar.from_spec("Europe/Moscow", [(range(0, 5), [("10:00", "10:30")])])
        rows = [row(msk(2024, 1, 4, 10, 0)), row(msk(2024, 1, 4, 10, 31))]
        path = write_csv(tmp_path / "oos.csv", rows)
        result = load_csv(path, timeframe="1m", calendar=calendar)
        assert any(i.code is IssueCode.W_OUT_OF_SESSION for i in result.report.warnings)

    def test_warnings_do_not_fail_load(self, tmp_path):
        rows = [row(msk(2024, 1, 4, 10, 0), v=0)]
        path = write_csv(tmp_path / "w.csv", rows)
        result = load_csv(path, timeframe="1m")
        assert result.ok
        assert len(result.bars) == 1


class TestValidateBarsDirect:
    def test_valid_series_ok(self):
        bars = minute_series(datetime(2024, 1, 4, 7, 0, tzinfo=UTC), 30)
        report = validate_bars(bars, "1m")
        assert report.ok
        assert report.warnings == []

    def test_detects_unsorted(self):
        bars = minute_series(datetime(2024, 1, 4, 7, 0, tzinfo=UTC), 5)
        bars[2], bars[3] = bars[3], bars[2]
        report = validate_bars(bars, "1m")
        assert any(i.code is IssueCode.E_UNSORTED for i in report.errors)

    def test_report_summary_has_counts(self):
        bars = minute_series(datetime(2024, 1, 4, 7, 0, tzinfo=UTC), 5)
        report = validate_bars(bars, "1m")
        text = report.summary_text()
        assert "ОШИБОК: 0" in text
