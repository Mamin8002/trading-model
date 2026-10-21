"""StreamProcessor: преобразование сырых сообщений публичного WS в события.

Логика отделена от сети, чтобы её можно было полностью проверить офлайн
(реплей сообщений, фейковые часы). Правила:
  * kline-сообщение с confirm=false -> обновляет "формирующуюся" свечу (только отображение);
  * kline-сообщение с confirm=true  -> закрытый бар (единственная основа для сигналов);
  * дубликаты закрытых баров отбрасываются;
  * разрыв сетки закрытых баров -> GapDetected (данные считаются неполными);
  * свежесть: FRESH, пока возраст последних данных <= stale_after_intervals * interval.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional

from ..data_sources.bybit import period_end_ms, step_ms_approx
from .base import (
    ConnState,
    ErrorEvent,
    FreshnessChange,
    GapDetected,
    KlineBar,
    StateChange,
    StreamSnapshot,
    TickerUpdate,
)

ClockMs = Callable[[], int]


def default_clock_ms() -> int:
    return int(time.time() * 1000)


class StreamProcessor:
    def __init__(
        self,
        symbol: str,
        interval: str,
        *,
        stale_after_intervals: int = 3,
        history_bars: int = 300,
        clock_ms: Optional[ClockMs] = None,
    ):
        self.symbol = symbol
        self.interval = str(interval)
        self.stale_after_intervals = stale_after_intervals
        self._clock = clock_ms or default_clock_ms
        self._kline_topic = f"kline.{self.interval}.{self.symbol}"
        self._ticker_topic = f"tickers.{self.symbol}"

        self.conn_state = ConnState.DISCONNECTED
        self.conn_detail = ""
        self.reconnect_count = 0
        self.last_data_ms: Optional[int] = None
        self.server_ts_ms: Optional[int] = None
        self.fresh = False
        self._fresh_emitted: Optional[bool] = None

        self.last_price: Optional[TickerUpdate] = None
        self.live_candle: Optional[KlineBar] = None
        self.confirmed: deque[KlineBar] = deque(maxlen=history_bars)
        self._confirmed_starts: set[int] = set()
        self._expected_next_start_ms: Optional[int] = None
        self.recent_gaps: deque[GapDetected] = deque(maxlen=20)

    # -- helpers -----------------------------------------------------------
    @property
    def stale_after_ms(self) -> int:
        return self.stale_after_intervals * step_ms_approx(self.interval)

    def age_ms(self, now_ms: Optional[int] = None) -> Optional[int]:
        if self.last_data_ms is None:
            return None
        now = now_ms if now_ms is not None else self._clock()
        return max(0, now - self.last_data_ms)

    def is_fresh(self, now_ms: Optional[int] = None) -> bool:
        age = self.age_ms(now_ms)
        return age is not None and age <= self.stale_after_ms

    def _mark_data(self, msg_ts: Optional[int], received_ms: int) -> None:
        self.last_data_ms = received_ms
        if isinstance(msg_ts, (int, float)) and msg_ts > 0:
            self.server_ts_ms = int(msg_ts)

    # -- message handling ---------------------------------------------------
    def on_ws_message(self, msg: dict) -> list:
        """Обработка одного сообщения публичного WS Bybit (уже распарсенного JSON)."""
        received = self._clock()
        events: list = []
        self._mark_data(msg.get("ts"), received)

        op = msg.get("op")
        if op == "subscribe":
            if msg.get("success"):
                self.conn_state = ConnState.CONNECTED
                self.conn_detail = "подписка подтверждена"
                events.append(StateChange(ConnState.CONNECTED, self.conn_detail, received))
            else:
                events.append(ErrorEvent(f"подписка отклонена: {msg.get('ret_msg')}", received))
            return events
        if op == "pong":
            return events  # свежесть уже обновлена
        if "success" in msg and op is None:
            return events  # приветственное/служебное сообщение

        topic = msg.get("topic", "")
        data = msg.get("data")
        if topic == self._ticker_topic and data:
            row = data[0] if isinstance(data, list) else data
            price = row.get("lastPrice")
            if price is not None:
                self.last_price = TickerUpdate(self.symbol, float(price),
                                               int(msg.get("ts") or received), received)
                events.append(self.last_price)
            return events

        if topic == self._kline_topic and data:
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                ev = self._on_kline_row(row, int(msg.get("ts") or received), received)
                if ev is not None:
                    events.extend(ev)
            return events

        if topic:
            events.append(ErrorEvent(f"неожиданный топик: {topic}", received))
        return events

    def _on_kline_row(self, row: dict, server_ts: int, received: int) -> Optional[list]:
        start = int(row["start"])
        end = int(row.get("end") or period_end_ms(start, self.interval) - 1)
        bar = KlineBar(
            symbol=self.symbol, interval=self.interval,
            start_ms=start, end_ms=period_end_ms(start, self.interval),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row["volume"]), turnover=float(row["turnover"]),
            confirmed=bool(row.get("confirm", False)),
            server_ts_ms=server_ts, received_ms=received,
        )
        if not bar.confirmed:
            self.live_candle = bar
            return None
        return self._register_confirmed(bar)

    def _register_confirmed(self, bar: KlineBar) -> list:
        events: list = []
        if bar.start_ms in self._confirmed_starts:
            return events  # дубликат (WS + backfill) - игнорируем
        if self._expected_next_start_ms is not None and bar.start_ms > self._expected_next_start_ms:
            gap = GapDetected(self._expected_next_start_ms, bar.start_ms, bar.received_ms,
                              detail="разрыв сетки закрытых баров")
            self.recent_gaps.append(gap)
            events.append(gap)
        if self.confirmed and bar.start_ms < self.confirmed[-1].start_ms:
            # out-of-order (например, поздний backfill): вставляем с сохранением порядка
            pos = len(self.confirmed)
            for i in range(len(self.confirmed) - 1, -1, -1):
                if self.confirmed[i].start_ms < bar.start_ms:
                    pos = i + 1
                    break
            else:
                pos = 0
            self.confirmed.insert(pos, bar)
        else:
            self.confirmed.append(bar)
        self._confirmed_starts.add(bar.start_ms)
        if len(self._confirmed_starts) > 4096:  # не растим множество бесконечно
            self._confirmed_starts = {b.start_ms for b in self.confirmed}
        self._expected_next_start_ms = period_end_ms(bar.start_ms, self.interval)
        if self.live_candle is not None and self.live_candle.start_ms == bar.start_ms:
            self.live_candle = None
        events.append(bar)
        return events

    def ingest_backfilled(self, klines: list) -> list:
        """Ввод закрытых баров из REST-догрузки (Kline из data_sources.base)."""
        events: list = []
        for k in klines:
            if not k.is_closed:
                continue
            bar = KlineBar(
                symbol=self.symbol, interval=self.interval, start_ms=k.start_ms,
                end_ms=period_end_ms(k.start_ms, self.interval),
                open=k.open, high=k.high, low=k.low, close=k.close,
                volume=k.volume, turnover=k.turnover,
                confirmed=True, server_ts_ms=0, received_ms=self._clock(),
                origin="backfill",
            )
            events.extend(self._register_confirmed(bar))
        return events

    # -- connection / freshness ---------------------------------------------
    def on_connection_state(self, state: ConnState, detail: str = "") -> list:
        received = self._clock()
        prev = self.conn_state
        self.conn_state = state
        self.conn_detail = detail
        if state == ConnState.RECONNECTING and prev == ConnState.CONNECTED:
            self.reconnect_count += 1
        return [StateChange(state, detail, received)]

    def tick(self, now_ms: Optional[int] = None) -> list:
        """Периодическая проверка свежести (вызывается из основного цикла)."""
        now = now_ms if now_ms is not None else self._clock()
        fresh = self.is_fresh(now)
        events: list = []
        if self._fresh_emitted != fresh:
            age = self.age_ms(now)
            self.fresh = fresh
            self._fresh_emitted = fresh
            events.append(FreshnessChange(
                fresh, age if age is not None else -1,
                "данные свежие" if fresh else f"нет обновлений {age} мс (порог {self.stale_after_ms} мс)",
                now,
            ))
        else:
            self.fresh = fresh
        return events

    # -- snapshot ------------------------------------------------------------
    def snapshot(self) -> StreamSnapshot:
        return StreamSnapshot(
            conn_state=self.conn_state,
            conn_detail=self.conn_detail,
            fresh=self.fresh,
            last_data_ms=self.last_data_ms,
            age_ms=self.age_ms(),
            last_price=self.last_price,
            live_candle=self.live_candle,
            confirmed_bars=list(self.confirmed),
            reconnect_count=self.reconnect_count,
            gaps=list(self.recent_gaps),
            server_ts_ms=self.server_ts_ms,
        )
