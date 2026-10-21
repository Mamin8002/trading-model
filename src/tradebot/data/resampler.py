"""Ресэмплинг таймфреймов (docs/architecture.md §3.3, docs/decisions.md D3).

Правила агрегации OHLCV из базового таймфрейма в целевой (требование ТЗ):
- open  = open первой свечи группы;
- high  = максимум high;
- low   = минимум low;
- close = close последней свечи группы;
- volume = сумма объёмов.

Полнота интервала (защита «не выдавать неполную свечу за полную»):
1. интервал закрыт данными: последний ФИНАЛЬНЫЙ исходный бар покрывает конец интервала
   (ts_open + base_tf >= конец интервала); иначе группа считается pending;
2. покрытие: число исходных баров равно ожидаемому числу слотов сетки
   (при заданном календаре сессий — только торговых слотов).

Поведение:
- полные группы -> ``bars`` (is_final=True, source=DERIVED);
- последняя незакрытая группа -> ``pending`` (is_final=False), НИКОГДА не в ``bars``;
- закрытая группа с неполным покрытием -> issue ``coverage_incomplete`` и НЕ входит
  в ``bars`` (пропущенные свечи не фабрикуются и не заполняются); при
  ``allow_incomplete=True`` такой бар отдаётся с is_final=False (явная пометка);
- пустые интервалы (нет исходных баров) не фабрикуются.

Один и тот же код используется в TEST (из CSV) и в будущем LIVE (из потока),
что исключает расхождение построения 10m между режимами (decisions D3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence

from .bar import Bar, BarSource, Timeframe, floor_to_grid
from .sessions import SessionCalendar

__all__ = ["ResampleIssue", "ResampleIssueCode", "ResampleResult", "resample"]


class ResampleIssueCode(str, Enum):
    COVERAGE_INCOMPLETE = "coverage_incomplete"
    NO_SOURCE_BARS = "no_source_bars"
    OFF_GRID_SOURCE = "off_grid_source"
    NON_FINAL_SKIPPED = "non_final_skipped"
    OUT_OF_SESSION_SOURCE = "out_of_session_source"


@dataclass(frozen=True)
class ResampleIssue:
    code: ResampleIssueCode
    interval_start: datetime
    message: str
    expected: int | None = None
    found: int | None = None


@dataclass
class ResampleResult:
    bars: list[Bar] = field(default_factory=list)
    pending: Bar | None = None
    issues: list[ResampleIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Нет проблем покрытия (pending — нормальное состояние конца данных)."""
        return not self.issues


def resample(
    source_bars: Sequence[Bar],
    source_timeframe: "Timeframe | str",
    target_timeframe: "Timeframe | str",
    *,
    calendar: SessionCalendar | None = None,
    symbol: str | None = None,
    allow_incomplete: bool = False,
) -> ResampleResult:
    """Агрегирует финальные бары ``source_timeframe`` в ``target_timeframe``.

    Args:
        source_bars: отсортированные по ts_open финальные бары одного символа.
        source_timeframe: базовый таймфрейм (например "1m" или "5m").
        target_timeframe: целевой таймфрейм (например "10m" или "1h").
        calendar: календарь сессий; при None ожидаемое покрытие = равномерная сетка.
        symbol: символ для результирующих баров (None — брать из первого исходного).
        allow_incomplete: включать ли закрытые группы с неполным покрытием в выдачу
            с is_final=False (по умолчанию False — неполное не выдаётся вовсе).
    """
    src_tf = Timeframe.parse(source_timeframe)
    tgt_tf = Timeframe.parse(target_timeframe)
    if tgt_tf.minutes <= src_tf.minutes:
        raise ValueError(
            f"Целевой таймфрейм {tgt_tf.value} должен быть строго крупнее базового {src_tf.value}"
        )
    if tgt_tf.minutes % src_tf.minutes != 0:
        raise ValueError(
            f"Целевой таймфрейм {tgt_tf.value} не кратен базовому {src_tf.value}: "
            "агрегация по сетке невозможна"
        )

    result = ResampleResult()
    if not source_bars:
        return result

    sym = symbol if symbol is not None else (source_bars[0].symbol or "")
    tgt_step = timedelta(minutes=tgt_tf.minutes)

    # Группируем только финальные бары; нефинальные (pending из потока) игнорируются
    # с явной пометкой — решение принимается по завершённым свечам (architecture §7, D9).
    groups: dict[datetime, list[Bar]] = {}
    non_final_count = 0
    off_grid_count = 0
    for bar in source_bars:
        if not bar.is_final:
            non_final_count += 1
            continue
        interval_start = floor_to_grid(bar.ts_open, tgt_tf.minutes)
        if floor_to_grid(bar.ts_open, src_tf.minutes) != bar.ts_open:
            off_grid_count += 1
            continue
        groups.setdefault(interval_start, []).append(bar)

    if non_final_count:
        result.issues.append(
            ResampleIssue(
                ResampleIssueCode.NON_FINAL_SKIPPED,
                source_bars[0].ts_open,
                f"Пропущено нефинальных исходных баров: {non_final_count} "
                "(незавершённые свечи не участвуют в агрегации)",
            )
        )
    if off_grid_count:
        result.issues.append(
            ResampleIssue(
                ResampleIssueCode.OFF_GRID_SOURCE,
                source_bars[0].ts_open,
                f"Пропущено исходных баров вне сетки {src_tf.value}: {off_grid_count}",
            )
        )

    last_final = max((b.ts_open for b in source_bars if b.is_final), default=None)
    if last_final is None:
        return result
    data_coverage_end = last_final + src_tf.duration  # до какого момента данные точно закрыты

    for interval_start in sorted(groups.keys()):
        group = sorted(groups[interval_start], key=lambda b: b.ts_open)
        interval_end = interval_start + tgt_step

        # ожидаемое число слотов базовой сетки в интервале
        if calendar is not None:
            expected_slots = len(calendar.trading_slots(interval_start, interval_end, src_tf))
        else:
            expected_slots = tgt_tf.minutes // src_tf.minutes
        if expected_slots == 0:
            # неторговый интервал (например, клиринг/перерыв по календарю) — не фабрикаем бар;
            # если в нём тем не менее ЕСТЬ исходные бары — явно отмечаем (данные вне сессии
            # не отбрасываются молча)
            if group:
                result.issues.append(
                    ResampleIssue(
                        ResampleIssueCode.OUT_OF_SESSION_SOURCE,
                        interval_start,
                        f"Интервал {interval_start.isoformat()} вне сессий по календарю, но содержит "
                        f"{len(group)} исходных бар(ов); агрегат не выпущен. Проверьте календарь "
                        "или данные (бары не отброшены молча)",
                        expected=0,
                        found=len(group),
                    )
                )
            continue

        found = len(group)
        closed_by_data = data_coverage_end >= interval_end

        aggregated = Bar(
            ts_open=interval_start,
            open=group[0].open,
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,
            volume=sum(b.volume for b in group),
            source=BarSource.DERIVED,
            is_final=True,
            symbol=sym,
        )

        if not closed_by_data:
            # хвост данных: интервал ещё не закрыт — только pending, is_final=False
            pending_bar = Bar(
                ts_open=interval_start,
                open=aggregated.open,
                high=aggregated.high,
                low=aggregated.low,
                close=aggregated.close,
                volume=aggregated.volume,
                source=BarSource.DERIVED,
                is_final=False,
                symbol=sym,
            )
            if result.pending is not None:
                # более ранний незакрытый интервал с барами после разрыва данных — аномалия
                result.issues.append(
                    ResampleIssue(
                        ResampleIssueCode.COVERAGE_INCOMPLETE,
                        interval_start,
                        "Незакрытый интервал не в конце серии (разрыв данных); "
                        f"pending заменён на более поздний {pending_bar.ts_open.isoformat()}",
                        expected=expected_slots,
                        found=found,
                    )
                )
            result.pending = pending_bar
            continue

        if found != expected_slots:
            issue = ResampleIssue(
                ResampleIssueCode.COVERAGE_INCOMPLETE,
                interval_start,
                f"Неполное покрытие интервала {interval_start.isoformat()}: "
                f"ожидалось {expected_slots} бар(ов) {src_tf.value}, найдено {found}. "
                "Пропущенные свечи НЕ заполняются (docs/architecture.md §3.3)",
                expected=expected_slots,
                found=found,
            )
            result.issues.append(issue)
            if allow_incomplete:
                result.bars.append(
                    Bar(
                        ts_open=interval_start,
                        open=aggregated.open,
                        high=aggregated.high,
                        low=aggregated.low,
                        close=aggregated.close,
                        volume=aggregated.volume,
                        source=BarSource.DERIVED,
                        is_final=False,
                        symbol=sym,
                    )
                )
            continue

        result.bars.append(aggregated)

    # интервалы без исходных баров между первым и последним (дыры/праздники) — отмечаем
    if len(groups) >= 1:
        first_interval = floor_to_grid(min(b.ts_open for b in source_bars if b.is_final), tgt_tf.minutes)
        last_interval = floor_to_grid(last_final, tgt_tf.minutes)
        cursor = first_interval
        while cursor < last_interval:
            if cursor not in groups:
                if calendar is None or calendar.trading_slots(cursor, cursor + tgt_step, src_tf):
                    result.issues.append(
                        ResampleIssue(
                            ResampleIssueCode.NO_SOURCE_BARS,
                            cursor,
                            f"Интервал {cursor.isoformat()} не содержит исходных баров; "
                            "свеча не сфабрикована",
                        )
                    )
            cursor += tgt_step

    return result
