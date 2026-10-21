"""Bybit V5 public WebSocket client (spot/linear/inverse - public only, без аутентификации).

Документация:
  * Подключение: https://bybit-exchange.github.io/docs/v5/ws/connect
  * Kline-стрим: https://bybit-exchange.github.io/docs/v5/websocket/public/kline

Учтено:
  * app-level ping каждые ~20 с (рекомендация Bybit); сервер рвёт соединение
    при отсутствии трафика ~10 мин;
  * reconnect с экспоненциальным backoff (1..60 c) - в пределах IP-лимитов
    (<=500 соединений / 5 мин, не частить connect/disconnect);
  * закрытие свечи определяется полем confirm=true сообщения kline-топика;
  * клиент работает в отдельном потоке и кладёт в очередь сырые сообщения
    ("msg", dict) и смены состояния ("state", ConnState, detail) - всю семантику
    разбирает бирже-независимый StreamProcessor (тестируется офлайн).
"""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
import time
from typing import Callable, Optional

from websockets.sync.client import connect as ws_connect

from .base import ConnState, KlineStreamBase

logger = logging.getLogger(__name__)


def backoff_delay_s(attempt: int, base_s: float = 1.0, max_s: float = 60.0) -> float:
    """Экспоненциальная задержка переподключения (без джиттера) - чистая функция для тестов."""
    attempt = max(0, int(attempt))
    return float(min(max_s, base_s * (2 ** attempt)))


class BybitWsClient(KlineStreamBase):
    def __init__(
        self,
        ws_url: str,
        symbol: str,
        interval: str,
        out_queue: "queue.Queue",
        *,
        ping_interval_s: float = 20.0,
        connect_timeout_s: float = 10.0,
        recv_timeout_s: float = 1.0,
        reconnect_base_s: float = 1.0,
        reconnect_max_s: float = 60.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_ms: Optional[Callable[[], int]] = None,
        subscribe_topics: Optional[list[str]] = None,
    ):
        self.ws_url = ws_url
        self.symbol = symbol
        self.interval = str(interval)
        self.out = out_queue
        self.ping_interval_ms = int(ping_interval_s * 1000)
        self.connect_timeout_s = connect_timeout_s
        self.recv_timeout_s = recv_timeout_s
        self.reconnect_base_s = reconnect_base_s
        self.reconnect_max_s = reconnect_max_s
        self._sleep = sleep_fn
        self._clock = clock_ms or (lambda: int(time.time() * 1000))
        self.topics = subscribe_topics or [
            f"kline.{self.interval}.{self.symbol}",
            f"tickers.{self.symbol}",
        ]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="bybit-ws", daemon=True)
        self._thread.start()

    def stop(self, join_timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=join_timeout_s)
        self._put_state(ConnState.STOPPED, "остановлено пользователем")

    # -- internals -------------------------------------------------------------
    def _put(self, item: tuple) -> None:
        try:
            self.out.put_nowait(item)
        except queue.Full:
            logger.warning("очередь событий переполнена, сообщение пропущено")

    def _put_state(self, state: ConnState, detail: str) -> None:
        self._put(("state", state, detail))

    def _interruptible_sleep(self, seconds: float) -> None:
        self._stop.wait(timeout=seconds)

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                self._put_state(ConnState.CONNECTING if attempt == 0 else ConnState.RECONNECTING,
                                f"подключение к {self.ws_url}" + (f" (попытка {attempt + 1})" if attempt else ""))
                with ws_connect(
                    self.ws_url,
                    open_timeout=self.connect_timeout_s,
                    close_timeout=5.0,
                    ping_interval=self.ping_interval_s_prop,
                    ping_timeout=self.ping_interval_s_prop + 10.0,
                    max_size=2 ** 22,
                ) as ws:
                    ws.send(json.dumps({
                        "op": "subscribe",
                        "args": self.topics,
                        "req_id": f"sub-{self._clock()}",
                    }))
                    attempt = 0
                    last_ping = self._clock()
                    while not self._stop.is_set():
                        try:
                            raw = ws.recv(timeout=self.recv_timeout_s)
                        except TimeoutError:
                            raw = None
                        if raw is not None:
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError as e:
                                self._put(("error", f"не-JSON сообщение: {e}"))
                                continue
                            self._put(("msg", msg))
                        now = self._clock()
                        if now - last_ping >= self.ping_interval_ms:
                            ws.send(json.dumps({"op": "ping", "req_id": f"ping-{now}"}))
                            last_ping = now
            except Exception as e:  # noqa: BLE001 - любой сбой сети -> reconnect
                if self._stop.is_set():
                    break
                detail = f"{type(e).__name__}: {e}"
                logger.warning("WS-соединение потеряно: %s", detail)
                self._put_state(ConnState.RECONNECTING, detail)
                delay = backoff_delay_s(attempt, self.reconnect_base_s, self.reconnect_max_s)
                delay += random.uniform(0, 0.5)
                attempt += 1
                self._interruptible_sleep(delay)
        self._put_state(ConnState.STOPPED, "поток завершён")

    @property
    def ping_interval_s_prop(self) -> float:
        return max(5.0, self.ping_interval_ms / 1000.0)
