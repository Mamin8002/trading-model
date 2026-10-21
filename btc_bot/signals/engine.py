"""Сигнальный движок: закрытый бар -> сигнал BUY/SELL/HOLD с gating по свежести.

Правила (по спецификации):
  * сигнал формируется ТОЛЬКО по закрытому бару (confirm=true из WS либо
    закрытый бар из REST/backfill/реплея);
  * если данные устарели/неполны или соединение потеряно - сигнал помечается
    data_fresh=False с перечислением причин и НЕ выдаётся как актуальный;
  * HOLD при уверенности ниже порога = воздержание (причина фиксируется);
  * каждый сигнал содержит: время выдачи, время бара, цену на момент выдачи,
    уверенность, инструмент, версию модели, горизонт прогноза.

Семантика SELL для spot (закрытие/уменьшение длинной позиции) и учёт позиций -
уровень журнала/сопровождения (Этап 4); здесь - только вывод модели.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..data_sources.bybit import step_ms_approx
from ..model.predictor import Predictor
from ..streaming.base import ConnState, FreshnessChange, GapDetected, KlineBar

ClockMs = Callable[[], int]


@dataclass(frozen=True)
class Signal:
    signal_id: str
    created_at_ms: int        # время выдачи сигнала (когда движок обработал бар)
    bar_start_ms: int
    bar_end_ms: int           # момент закрытия бара (время данных сигнала)
    symbol: str
    category: str
    interval: str
    action: str               # BUY | SELL | HOLD
    confidence: float
    prob_up: float
    raw_direction: int        # +1 / -1
    price_at_signal: float
    price_source: str         # "ticker" | "bar_close"
    model_version: str
    horizon_bars: int
    horizon_minutes: int
    feature_set: str
    data_fresh: bool          # False -> сигнал НЕ актуален (устарел/неполон/нет соединения)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    origin: str = "ws"        # ws | backfill | replay
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["reasons"] = list(self.reasons)
        return d


class SignalEngine:
    def __init__(
        self,
        predictor: Predictor,
        *,
        symbol: str,
        category: str,
        interval: str,
        horizon_bars: int,
        stale_after_intervals: int = 3,
        clock_ms: Optional[ClockMs] = None,
        last_price_fn: Optional[Callable[[], Optional[float]]] = None,
        stream_fresh_fn: Optional[Callable[[], bool]] = None,
    ):
        predictor.ensure_compatible(symbol=symbol, category=category,
                                    interval=interval, horizon_bars=horizon_bars)
        self.predictor = predictor
        self.symbol = symbol
        self.category = category
        self.interval = str(interval)
        self.horizon_bars = horizon_bars
        self.stale_after_ms = stale_after_intervals * step_ms_approx(self.interval)
        self._clock = clock_ms or (lambda: int(time.time() * 1000))
        self._last_price_fn = last_price_fn
        self._stream_fresh_fn = stream_fresh_fn
        self.signals: deque[Signal] = deque(maxlen=500)
        self._ids = itertools.count(1)
        self._processed_starts: set[int] = set()
        self._last_processed_start: Optional[int] = None
        self._pending_gap = False
        self._connected = False
        self._stream_fresh = False

    # -- stream state hooks ---------------------------------------------------
    def on_conn_state(self, state: ConnState) -> None:
        self._connected = state == ConnState.CONNECTED

    def on_freshness(self, ev: FreshnessChange) -> None:
        self._stream_fresh = ev.fresh

    def on_gap(self, ev: GapDetected) -> None:
        self._pending_gap = True

    # -- signals ----------------------------------------------------------------
    def on_confirmed_bar(self, bar: KlineBar) -> Optional[Signal]:
        """Обработка закрытого бара. Возвращает Signal или None (дубликат)."""
        if not bar.confirmed:
            raise ValueError("SignalEngine принимает только закрытые бары (confirmed=True)")
        if bar.start_ms in self._processed_starts:
            return None  # дубликат (WS + backfill/реплей)

        now = self._clock()
        reasons: list[str] = []
        data_fresh = True

        age_ms = now - bar.end_ms
        if age_ms > self.stale_after_ms:
            data_fresh = False
            reasons.append(
                f"устаревший бар: возраст {age_ms / 1000:.0f} c > порога {self.stale_after_ms / 1000:.0f} c"
            )
        if self._pending_gap:
            data_fresh = False
            reasons.append("разрыв истории баров (данные неполны)")
            self._pending_gap = False
        if not self._connected and bar.origin == "ws":
            data_fresh = False
            reasons.append("нет подтверждения соединения")
        # свежесть потока оценивается НА МОМЕНТ выдачи сигнала (live-запрос),
        # а не по последнему transition-событию - иначе первый бар сразу после
        # подключения ошибочно помечался бы устаревшим
        stream_fresh = self._stream_fresh_fn() if self._stream_fresh_fn is not None else self._stream_fresh
        if not stream_fresh and bar.origin == "ws":
            data_fresh = False
            reasons.append("поток данных устарел (нет обновлений)")
        if bar.origin == "backfill":
            reasons.append("бар доставлен REST-догрузкой после разрыва")

        pred = self.predictor.predict_bar(bar.as_feature_mapping())
        if pred.action == "HOLD":
            reasons.append(
                f"воздержание: уверенность {pred.confidence:.3f} < порога "
                f"{self.predictor.confidence_threshold:.2f}"
            )

        price = None
        price_source = "bar_close"
        if self._last_price_fn is not None:
            try:
                price = self._last_price_fn()
            except Exception:  # noqa: BLE001 - цена тикера не критична
                price = None
        if price is None:
            price = bar.close
        else:
            price_source = "ticker"

        sig = Signal(
            signal_id=f"sig-{next(self._ids):06d}",
            created_at_ms=now,
            bar_start_ms=bar.start_ms,
            bar_end_ms=bar.end_ms,
            symbol=self.symbol,
            category=self.category,
            interval=self.interval,
            action=pred.action,
            confidence=pred.confidence,
            prob_up=pred.prob_up,
            raw_direction=pred.raw_direction,
            price_at_signal=float(price),
            price_source=price_source,
            model_version=pred.model_version,
            horizon_bars=pred.horizon_bars,
            horizon_minutes=pred.horizon_minutes,
            feature_set=pred.feature_set,
            data_fresh=data_fresh,
            reasons=tuple(reasons),
            origin=bar.origin,
            features=pred.features,
        )
        self._processed_starts.add(bar.start_ms)
        self._last_processed_start = max(self._last_processed_start or bar.start_ms, bar.start_ms)
        self.signals.append(sig)
        return sig

    # -- read-only views -------------------------------------------------------
    @property
    def last_signal(self) -> Optional[Signal]:
        return self.signals[-1] if self.signals else None

    def recent_signals(self, n: int = 5) -> list[Signal]:
        return list(self.signals)[-n:]
