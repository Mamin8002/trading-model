"""
data_loader.py — Загрузка рыночных данных.

Два источника с единым интерфейсом (возвращают pandas.DataFrame с колонками
datetime, open, high, low, close, volume, отсортированные по времени):

  * CsvDataLoader   — TEST-режим: история Мосбиржи из CSV.
  * FinamDataLoader — LIVE-режим: официальный Finam Trade API (REST, async).

Также здесь lives генерация синтетической истории (для тестов/демо без файла).
"""
from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import BotConfig, TIMEFRAME_FINAM, _tf_minutes

log = logging.getLogger("data")

REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит произвольный OHLCV-датафрейм к каноническому виду."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # частые алиасы колонок
    alias = {"time": "datetime", "date": "datetime", "ts": "datetime",
             "vol": "volume", "qty": "volume"}
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют колонки {missing}. Ожидается: {REQUIRED_COLS}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLS)
    df = df.sort_values("datetime").drop_duplicates(subset="datetime").reset_index(drop=True)
    # базовая целостность свечей
    bad = (df["high"] < df["low"]) | (df["close"] <= 0)
    if bad.any():
        log.warning("Отброшено %d неконсистентных баров", int(bad.sum()))
        df = df[~bad].reset_index(drop=True)
    return df[REQUIRED_COLS]


class BaseDataLoader(ABC):
    """Интерфейс источника OHLCV-данных."""

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    @abstractmethod
    async def fetch_history(self, n_bars: Optional[int] = None) -> pd.DataFrame:
        """Вернуть не менее n_bars последних баров (или всю доступную историю)."""

    async def fetch_latest_price(self) -> float:
        """Последняя закрытая цена (для mark-to-market в LIVE)."""
        df = await self.fetch_history(n_bars=2)
        return float(df["close"].iloc[-1])


# ---------------------------------------------------------------------------
# CSV loader (TEST mode)
# ---------------------------------------------------------------------------
class CsvDataLoader(BaseDataLoader):
    """Читает историю из CSV формата: datetime,open,high,low,close,volume."""

    def __init__(self, cfg: BotConfig, csv_path: Optional[str] = None):
        super().__init__(cfg)
        self.csv_path = csv_path or cfg.csv_path
        self._df: Optional[pd.DataFrame] = None
        self._cursor = 0  # для пошаговой "live-имитации" в бэктесте

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(
                    f"CSV с историей не найден: {self.csv_path}. "
                    f"Скачайте данные или сгенерируйте синтетику: python main.py --gen-data"
                )
            raw = pd.read_csv(self.csv_path)
            self._df = normalize_ohlcv(raw)
            log.info("Загружено %d баров из %s (%s ... %s)",
                     len(self._df), self.csv_path,
                     self._df['datetime'].iloc[0], self._df['datetime'].iloc[-1])
        return self._df

    async def fetch_history(self, n_bars: Optional[int] = None) -> pd.DataFrame:
        df = self._load()
        if n_bars is not None:
            # эмулируем "окно свежих данных" относительно текущего курсора
            end = self._cursor if self._cursor > 0 else len(df)
            start = max(0, end - n_bars)
            return df.iloc[start:end].copy()
        return df.copy()

    # ---- вспомогательное для симуляции последовательного потока баров ----
    def total_bars(self) -> int:
        return len(self._load())

    def set_cursor(self, i: int) -> None:
        self._cursor = i


# ---------------------------------------------------------------------------
# Синтетический генератор (демо/тесты)
# ---------------------------------------------------------------------------
def generate_synthetic_csv(path: str, n_bars: int = 8000, tf_minutes: int = 5,
                           start_price: float = 274.0, seed: int = 42) -> pd.DataFrame:
    """Генерирует правдоподобную intraday-историю (GBM + кластеризация волатильности)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=datetime.now(timezone.utc).replace(second=0, microsecond=0),
                       periods=n_bars, freq=f"{tf_minutes}min")
    # режимы волатильности (GARCH-подобная кластеризация)
    vol_state = np.zeros(n_bars)
    cur = 0.0006
    for i in range(n_bars):
        cur = 0.985 * cur + 0.015 * abs(rng.normal(0, 0.0012)) + 0.0002
        vol_state[i] = cur
    drift = 0.0
    rets = []
    for i in range(n_bars):
        # лёгкая автокорреляция доходностей → индикаторы учат сигнал
        momentum = 0.0
        if i >= 5:
            r = np.array(rets[-5:])
            momentum = 0.15 * r.mean() / max(vol_state[i], 1e-9)
        base = rng.normal(momentum * vol_state[i], vol_state[i])
        rets.append(base)
    rets = np.array(rets)
    close = start_price * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, vol_state)) * close
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = (rng.lognormal(mean=np.log(3e5), sigma=0.6, size=n_bars)).astype(int)
    df = pd.DataFrame({"datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                       "open": np.round(open_, 2), "high": np.round(high, 2),
                       "low": np.round(low, 2), "close": np.round(close, 2),
                       "volume": volume})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Синтетическая история сохранена: %s (%d баров)", path, n_bars)
    return df


# ---------------------------------------------------------------------------
# Finam Trade API loader (LIVE mode)
# ---------------------------------------------------------------------------
class FinamClient:
    """
    Минимальный асинхронный клиент официального Finam Trade API
    (https://finamrus.github.io/trade-api, REST-эндпоинты api.open-api.finam.ru).

    Аутентификация: Bearer access_token, полученным через POST /v1/connect/token
    (client_id + api_key / логин+пароль).
    """

    def __init__(self, cfg: BotConfig, session=None):
        self.cfg = cfg
        self._session = session
        self._own_session = session is None
        self.access_token: Optional[str] = None
        self.client_id: Optional[str] = None  # resolved after auth

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def authenticate(self) -> None:
        """Получение access token. Поддержаны оба способа Finam: api-key и login/password."""
        import aiohttp
        sess = await self._ensure_session()
        url = f"{self.cfg.finam_base_url}/v1/connect/token"
        payload = {"clientId": self.cfg.finam_client_id}
        if self.cfg.finam_api_key:
            payload["apiKey"] = self.cfg.finam_api_key
        else:
            payload.update(relogin=self.cfg.finam_login, password=self.cfg.finam_password)
        async with sess.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Finam auth failed ({resp.status}): {text[:300]}")
            data = await resp.json()
            self.access_token = data.get("accessToken") or data.get("access_token")
            self.client_id = data.get("clientInfo", {}).get("clientId") or self.cfg.finam_client_id
            if not self.access_token:
                raise RuntimeError("Finam auth: accessToken отсутствует в ответе")
        log.info("Finam API: авторизация успешна (clientId=%s...)", str(self.client_id)[:8])

    def _headers(self) -> dict:
        if not self.access_token:
            raise RuntimeError("Не пройдена авторизация Finam (authenticate())")
        return {"Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"}

    async def candles_get(self, exchange: str, symbol: str, tf_finam: str,
                          from_ms: int, to_ms: int, limit: int = 1500) -> list[dict]:
        """Candles/get — список свечей (REST). Возвращает 'result' массив."""
        sess = await self._ensure_session()
        body = {
            "exchange": exchange,
            "code": symbol,
            "interval": tf_finam,
            "from": str(from_ms),
            "to": str(to_ms),
            "limit": limit,
        }
        url = f"{self.cfg.finam_base_url}/v1/marketdata/candles/get"
        async with sess.post(url, json=body, headers=self._headers()) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Finam candles/get failed ({resp.status}): {text[:300]}")
            data = await resp.json()
            return data.get("result", []) or []

    async def get_portfolio(self) -> dict:
        sess = await self._ensure_session()
        url = f"{self.cfg.finam_base_url}/v1/statement/commissions/portfolio/get"
        params = {"clientId": self.client_id or self.cfg.finam_client_id}
        async with sess.get(url, params=params, headers=self._headers()) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Finam portfolio failed ({resp.status}): {await resp.text()}")
            return await resp.json()

    async def place_order(self, exchange: str, symbol: str, side: str,
                          qty: int, price: Optional[float] = None,
                          order_type: str = "MARKET") -> dict:
        """Trade/send — выставление ордер-заявки (side: BUY|SELL)."""
        sess = await self._ensure_session()
        body = {
            "clientId": self.client_id or self.cfg.finam_client_id,
            "orderRequestType": order_type,          # MARKET | LIMIT
            "tradeDirection": side,                  # BUY | SELL
            "price": price,
            "quantity": qty,
            "conditionalPrice": 0,
            "instrument": {"exchange": exchange, "code": symbol,
                           "classCode": self.cfg.finam_exchange},
        }
        url = f"{self.cfg.finam_base_url}/v1/order/execute"
        async with sess.post(url, json={"executeOrderRequest": body},
                             headers=self._headers()) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Finam order execute failed ({resp.status}): {text[:400]}")
            return await resp.json()

    async def close(self):
        if self._own_session and self._session is not None:
            await self._session.close()


class FinamDataLoader(BaseDataLoader):
    """LIVE-коннектор: свежие свечи через Finam Trade API (async)."""

    def __init__(self, cfg: BotConfig, client: Optional[FinamClient] = None):
        super().__init__(cfg)
        self.client = client or FinamClient(cfg)
        self._cache: Optional[pd.DataFrame] = None
        self._cache_ts: float = 0.0
        self._cache_ttl = max(5, cfg.poll_seconds // 2)

    async def connect(self) -> None:
        await self.client.authenticate()

    async def fetch_history(self, n_bars: Optional[int] = None) -> pd.DataFrame:
        n = n_bars or (self.cfg.tf.lookback_bars + self.cfg.tf.warmup_bars + 50)
        now = datetime.now(timezone.utc)
        if self._cache is not None and (now.timestamp() - self._cache_ts) < self._cache_ttl \
                and len(self._cache) >= n:
            return self._cache.tail(n).reset_index(drop=True)

        tf_code = TIMEFRAME_FINAM[self.cfg.timeframe]
        minutes = _tf_minutes(self.cfg.timeframe)
        span_min = n * minutes * 2 + 60 * 24  # запас на нерабочие часы/выходные
        frm = int((now - timedelta(minutes=min(span_min, self.cfg.live_history_days * 1440))).timestamp() * 1000)
        to = int(now.timestamp() * 1000)

        raw = await self.client.candles_get(
            exchange=self.cfg.finam_market_name,
            symbol=self.cfg.symbol,
            tf_finam=tf_code, from_ms=frm, to_ms=to, limit=min(n, 1500),
        )
        if not raw:
            raise RuntimeError(f"Finam вернул пустой набор свечей для {self.cfg.symbol}/{tf_code}")
        recs = []
        for c in raw:
            recs.append({
                "datetime": pd.to_datetime(int(c["msDate"]), unit="ms", utc=True),
                "open": float(c["open"]), "high": float(c["high"]),
                "low": float(c["low"]), "close": float(c["close"]),
                "volume": float(c.get("volume", 0)),
            })
        df = normalize_ohlcv(pd.DataFrame(recs))
        self._cache, self._cache_ts = df, now.timestamp()
        log.debug("Finam: получено %d баров %s %s", len(df), self.cfg.symbol, tf_code)
        return df.tail(n).reset_index(drop=True)

    async def close(self):
        await self.client.close()
