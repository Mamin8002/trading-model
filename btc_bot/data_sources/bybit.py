"""Bybit V5 public REST client (market data only, no auth).

Документация:
  - GET /v5/market/kline:  https://bybit-exchange.github.io/docs/v5/market/kline
  - Rate limits:           https://bybit-exchange.github.io/docs/v5/rate-limit

Ключевые факты, учтённые в реализации:
  * category по умолчанию linear -> spot всегда передаём явно;
  * result.list отсортирован ПО УБЫВАНИЮ startTime (новые первые) -> разворачиваем;
  * close незакрытой свечи = цена последней сделки -> помечаем is_closed
    по серверному времени ответа (поле time);
  * limit <= 1000; при заданном start сервер возвращает свечи вперёд от start;
  * IP-лимит 600 req / 5 s; при 403 "access too frequent" бан ~10 минут ->
    троттлинг (min_request_interval) и backoff, 403 не ретраим агрессивно.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

import pandas as pd
import requests

from .base import (
    DataSourceAPIError,
    DataSourceNetworkError,
    Kline,
    LastPrice,
    MarketDataSource,
    ProgressCallback,
)

logger = logging.getLogger(__name__)

INTERVAL_MS = {
    "1": 60_000, "3": 180_000, "5": 300_000, "15": 900_000, "30": 1_800_000,
    "60": 3_600_000, "120": 7_200_000, "240": 14_400_000, "360": 21_600_000,
    "720": 43_200_000,
}

MAX_LIMIT = 1000


def interval_to_ms(interval: str) -> int:
    """Точная длительность интервала в ms (для D/W/M используется pandas-офсет)."""
    if interval in INTERVAL_MS:
        return INTERVAL_MS[interval]
    if interval in ("D", "W", "M"):
        # для дневных и старше длительность переменная - вычисляем через period_end_ms
        return -1
    raise DataSourceAPIError(f"Неподдерживаемый interval: {interval!r}")


def step_ms_approx(interval: str) -> int:
    """Шаг сетки в ms: точный для внутридневных, приблизительный для D/W/M."""
    if interval in INTERVAL_MS:
        return INTERVAL_MS[interval]
    if interval == "D":
        return 86_400_000
    if interval == "W":
        return 7 * 86_400_000
    if interval == "M":
        return 30 * 86_400_000
    raise DataSourceAPIError(f"Неподдерживаемый interval: {interval!r}")


def period_end_ms(start_ms: int, interval: str) -> int:
    """Момент закрытия свечи (exclusive) для start_ms и интервала."""
    if interval in INTERVAL_MS:
        return start_ms + INTERVAL_MS[interval]
    ts = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    if interval == "D":
        end = ts + pd.offsets.Day(1)
    elif interval == "W":
        end = ts + pd.offsets.Week(1)
    else:  # M
        end = ts.normalize() + pd.offsets.MonthBegin(1)
    return int(end.timestamp() * 1000)


class BybitSource(MarketDataSource):
    """Публичный REST-источник данных Bybit V5 (без ключей)."""

    name = "bybit"

    def __init__(
        self,
        base_url: str = "https://api.bybit.com",
        request_timeout_s: float = 15.0,
        min_request_interval_s: float = 0.35,
        max_retries: int = 5,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = request_timeout_s
        self.min_request_interval = min_request_interval_s
        self.max_retries = max_retries
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "btc-bybit-monitor-bot/0.1 (public market data)"})
        self._last_request_ts = 0.0

    # -- low-level ------------------------------------------------------
    def _throttle(self) -> None:
        wait = self.min_request_interval - (time.monotonic() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _get(self, path: str, params: dict) -> dict:
        """GET с троттлингом и retry/backoff. Возвращает полный JSON-ответ."""
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = DataSourceNetworkError(f"Сетевая ошибка {url}: {e}")
                logger.warning("network error (attempt %d): %s", attempt + 1, e)
            else:
                if resp.status_code == 403:
                    # IP-бан Bybit ~10 минут; агрессивный ретрай только ухудшит ситуацию
                    raise DataSourceAPIError(
                        "HTTP 403 от Bybit (IP rate limit). Прекратите запросы примерно на 10 минут.",
                        code=403,
                    )
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = DataSourceAPIError(
                        f"HTTP {resp.status_code} от {url}", code=resp.status_code
                    )
                    logger.warning("HTTP %d (attempt %d)", resp.status_code, attempt + 1)
                elif resp.status_code != 200:
                    raise DataSourceAPIError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}", code=resp.status_code
                    )
                else:
                    try:
                        payload = resp.json()
                    except ValueError as e:
                        raise DataSourceAPIError(f"Не-JSON ответ от {url}: {e}")
                    ret_code = payload.get("retCode")
                    if ret_code == 0:
                        return payload
                    if ret_code == 10006:  # Too many visits -> backoff и повтор
                        last_err = DataSourceAPIError(
                            f"Bybit retCode 10006: {payload.get('retMsg')}", code=10006
                        )
                        logger.warning("rate limited (10006), backing off")
                    else:
                        raise DataSourceAPIError(
                            f"Bybit retCode {ret_code}: {payload.get('retMsg')}", code=ret_code
                        )
            if attempt < self.max_retries:
                backoff = min(30.0, (2 ** attempt)) + random.uniform(0, 0.5)
                time.sleep(backoff)
        assert last_err is not None
        raise last_err

    # -- public API -----------------------------------------------------
    def get_server_time_ms(self) -> int:
        payload = self._get("/v5/market/time", {})
        result = payload.get("result") or {}
        # формат ответа: result.timeSecond/timeNano + top-level "time" (ms)
        if isinstance(result.get("time"), (int, str)):
            return int(result["time"])
        if "time" in payload:
            return int(payload["time"])
        if "timeSecond" in result:
            return int(float(result["timeSecond"]) * 1000)
        raise DataSourceAPIError("Не удалось разобрать ответ /v5/market/time")

    def get_last_price(self, symbol: str, category: str = "spot") -> LastPrice:
        payload = self._get(
            "/v5/market/tickers", {"category": category, "symbol": symbol}
        )
        lst = payload["result"].get("list") or []
        if not lst:
            raise DataSourceAPIError(f"Пустой ответ tickers для {category}/{symbol}")
        row = lst[0]
        return LastPrice(
            price=float(row["lastPrice"]),
            ts_ms=int(payload["time"]),
            source=self.name,
            symbol=symbol,
        )

    def fetch_kline_page(
        self,
        symbol: str,
        category: str,
        interval: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = MAX_LIMIT,
    ) -> tuple[list[Kline], int]:
        """Одна страница свечей. Возвращает (список по возрастанию start_ms, server_time_ms)."""
        params: dict = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(int(limit), MAX_LIMIT),
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)
        payload = self._get("/v5/market/kline", params)
        result = payload.get("result") or {}
        server_time_ms = int(payload.get("time") or 0)
        rows = result.get("list") or []
        klines: list[Kline] = []
        for row in rows:  # ответ по убыванию startTime -> развернём ниже сортировкой
            start = int(row[0])
            klines.append(
                Kline(
                    start_ms=start,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    turnover=float(row[6]),
                    is_closed=period_end_ms(start, interval) <= server_time_ms,
                )
            )
        klines.sort(key=lambda k: k.start_ms)  # гарантируем возрастание
        return klines, server_time_ms

    def fetch_klines(
        self,
        symbol: str,
        category: str,
        interval: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = MAX_LIMIT,
        max_pages: Optional[int] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> list[Kline]:
        """Полная загрузка истории с пагинацией, дедупликацией и сортировкой.

        Важно (эмпирически + документация): при передаче И start, И end сервер
        возвращает САМЫЕ СВЕЖИЕ limit свечей окна (от конца), поэтому пагинация
        выполняется запросами только со start (сервер идёт вперёд от start),
        а граница end применяется клиентской фильтрацией.
        Первая свеча может начинаться чуть раньше запрошенного start - сервер
        возвращает свечу, СОДЕРЖАЩУЮ момент start (выравнивание по сетке).

        Останов: next_start > end_ms / получили меньше limit свечей / нет прогресса.
        """
        # для внутридневных интервалов шаг точный; для D/W/M - через period_end_ms (переменная длина)
        fixed_step_ms = None if interval in ("D", "W", "M") else interval_to_ms(interval)
        by_start: dict[int, Kline] = {}
        cursor = start_ms
        page_size = min(int(limit), MAX_LIMIT)
        pages = 0

        while True:
            batch, _srv = self.fetch_kline_page(
                symbol, category, interval, start_ms=cursor, end_ms=None, limit=page_size
            )
            pages += 1
            if not batch:
                break
            for k in batch:
                if end_ms is not None and k.start_ms > end_ms:
                    continue
                # при дубликатах предпочтительнее закрытая свеча (более полная)
                prev = by_start.get(k.start_ms)
                if prev is None or (k.is_closed and not prev.is_closed):
                    by_start[k.start_ms] = k
            last_start = batch[-1].start_ms
            if on_progress:
                on_progress(pages, len(by_start), last_start)
            next_cursor = (period_end_ms(last_start, interval) if fixed_step_ms is None
                           else last_start + fixed_step_ms)
            if end_ms is not None and next_cursor > end_ms:
                break
            if len(batch) < page_size:
                # сервер вернул меньше, чем просили -> данные до конца окна/истории исчерпаны
                break
            if cursor is not None and next_cursor <= cursor:
                logger.warning("Пагинация не продвигается (next=%s <= cursor=%s), останавливаюсь", next_cursor, cursor)
                break
            if max_pages is not None and pages >= max_pages:
                break
            cursor = next_cursor

        out = [by_start[s] for s in sorted(by_start)]
        logger.info("fetch_klines: pages=%d unique=%d", pages, len(out))
        return out

    def describe(self) -> dict:
        return {"source": self.name, "rest_base_url": self.base_url, "api": "v5/market"}
