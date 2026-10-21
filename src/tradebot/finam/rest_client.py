"""Read-only REST-клиент Finam Trade API (docs/research.md §2, подтверждено S3/S5).

Реализованы ТОЛЬКО методы, чьи пути И схемы ответов подтверждены официальной
документацией на дату исследования (2026-10-21):

- POST /v1/sessions                      — обмен secret_key на access_key (~15 мин);
- POST /v1/sessions/details              — account_ids/readonly/expires_at/md_permissions;
- GET  /v1/instruments/{symbol}/bars     — история свечей (enum TimeFrame, глубины);
- GET  /v1/instruments/{symbol}/quotes/latest — снимок котировки;
- GET  /v1/instruments/{symbol}/orderbook     — снимок стакана;
- GET  /v1/assets/{symbol}               — метаданные инструмента;
- GET  /v1/assets/{symbol}/params        — торговые параметры (нужен account_id).

НЕ реализовано в этой фазе (осознанно, без угадывания):
- Clock/Schedule/Usage/Exchanges/AllAssets — пути подтверждены swagger-ом, но схемы
  ответов здесь не проверялись; будут добавлены на этапе L1 roadmap после сверки
  с документацией (research: «не угадывать детали API»);
- любые операции выставления/отмены заявок — запрещены задачей этой фазы
  (read-only клиент физически не содержит торговых методов);
- потоковые подписки (WS/gRPC) — контракт определён в marketdata.interfaces.StreamSource,
  реализация на этапе L2 (decisions D7).

Надёжность (research §2.8):
- GET ретраятся при транспортных ошибках и 500/503/504 с backoff 1→2→4→8→16→30 с + jitter;
- 429: немедленный повтор запрещён — ждём следующее минутное окно (+jitter), одна попытка;
- 401 на GET: один refresh access_key и повтор; на auth-запросе — FinamAuthError;
- POST не ретраится (в этой фазе POST — только аутентификация);
- клиентский лимитер ≤ 200 запросов/мин на метод;
- все сообщения ошибок и логи маскируют secret_key/access_key.
"""

from __future__ import annotations

import random
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from ..data.bar import UTC, Timeframe
from ..marketdata.interfaces import (
    OrderBookRow,
    OrderBookRowAction,
    OrderBookLevelSide,
    OrderBookSnapshot,
    QuoteSnapshot,
)
from .errors import (
    FinamAuthError,
    FinamError,
    FinamInvalidRequestError,
    FinamInvalidResponseError,
    FinamNotFoundError,
    FinamPermissionDeniedError,
    FinamRateLimitError,
    FinamServerError,
    FinamTransportError,
    HistoryDepthError,
    UnsupportedTimeframeError,
    get_logger,
)
from .ratelimit import SlidingWindowLimiter
from .transport import HttpResponse, HttpTransport, UrllibTransport

__all__ = [
    "AssetInfo",
    "AssetParams",
    "FinamBarRecord",
    "FinamRestClient",
    "TokenDetails",
    "API_TIMEFRAME_BY_CODE",
    "HISTORY_DEPTH_LIMITS",
]

logger = get_logger(__name__)

#: Подтверждённый документацией enum TimeFrame (research §2.3). M10 ОТСУТСТВУЕТ.
API_TIMEFRAME_BY_CODE: dict[str, str] = {
    "1m": "TIME_FRAME_M1",
    "5m": "TIME_FRAME_M5",
    "15m": "TIME_FRAME_M15",
    "1h": "TIME_FRAME_H1",
}

#: Документированная глубина истории (research §2.3). Ключи — наши коды ТФ.
HISTORY_DEPTH_LIMITS: dict[str, timedelta] = {
    "1m": timedelta(days=7),
    "5m": timedelta(days=30),
    "15m": timedelta(days=30),
    "1h": timedelta(days=30),
}

_BACKOFF_SCHEDULE_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_TOKEN_REFRESH_MARGIN = timedelta(seconds=30)
_FALLBACK_TOKEN_TTL = timedelta(seconds=60)  # если expires_at не распознан — перевыпуск каждые 60 с


@dataclass(frozen=True)
class TokenDetails:
    """Детали access_key (POST /v1/sessions/details).

    account_ids хранятся, но НИКОГДА не логируются и не попадают в repr/str
    (account id относится к чувствительным данным по требованию задачи).
    """

    expires_at: datetime
    readonly: bool
    account_ids: tuple[str, ...] = field(repr=False)
    md_permissions: tuple[dict[str, Any], ...] = ()
    created_at: datetime | None = None
    expires_at_parsed: bool = True

    def __str__(self) -> str:  # pragma: no cover - защитный вывод без id
        return (
            f"TokenDetails(expires_at={self.expires_at.isoformat()}, readonly={self.readonly}, "
            f"account_count={len(self.account_ids)})"
        )


@dataclass(frozen=True)
class FinamBarRecord:
    """Сырая свеча API (значения в Decimal — без потерь на бинарном float).

    Семантика timestamp (bar_open/bar_close) документацией НЕ зафиксирована
    (research Q3): нормализация к внутреннему Bar выполняется отдельно и требует
    ЯВНОГО указания семантики (finam/source.py).
    """

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_data_snapshot: bool = False


@dataclass(frozen=True)
class AssetInfo:
    symbol: str
    name: str | None
    ticker: str | None
    mic: str | None
    quote_currency: str | None
    lot_size: Decimal | None
    min_step: int | None
    decimals: int | None
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class AssetParams:
    trade_lot_size: int | None
    is_tradable: bool | None
    raw: dict[str, Any] = field(repr=False)


def _format_api_time(ts: datetime) -> str:
    """RFC3339 UTC c суффиксом Z (формат query-параметров из документации)."""
    if ts.tzinfo is None:
        raise ValueError("Ожидается timezone-aware datetime")
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_api_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise FinamInvalidResponseError(f"Ожидалась строковая метка времени, получено: {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinamInvalidResponseError(f"Не удалось разобрать метку времени API {value!r}") from exc
    if parsed.tzinfo is None:
        # Документация показывает метки с часовым поясом (Z). Naive-ответ неоднозначен —
        # не угадываем (research Q3), а сообщаем.
        raise FinamInvalidResponseError(
            f"Метка времени API без часового пояса: {value!r}. "
            "Семантика не документирована — ответ отклонён вместо угадывания"
        )
    return parsed.astimezone(UTC)


def _parse_decimal(value: Any, field_name: str) -> Decimal:
    """Decimal-формат REST: {"value": "123.45"} (research §2.7). Строго."""
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise FinamInvalidResponseError(
                f"Поле {field_name!r}: некорректное decimal-значение {value!r}"
            ) from exc
    raise FinamInvalidResponseError(
        f"Поле {field_name!r}: ожидался decimal-объект или строка, получено {type(value).__name__}"
    )


def _optional_decimal(payload: dict[str, Any], field_name: str) -> Decimal | None:
    if field_name not in payload or payload[field_name] is None:
        return None
    return _parse_decimal(payload[field_name], field_name)


class FinamRestClient:
    """Минимальный безопасный read-only клиент (аутентификация + рыночные данные)."""

    def __init__(
        self,
        secret: str,
        *,
        base_url: str = "https://api.finam.ru",
        timeout_seconds: float = 15.0,
        rate_limit_per_minute: int = 200,
        max_get_attempts: int = 4,
        transport: HttpTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not secret:
            raise ValueError("secret не может быть пустым")
        if not base_url.startswith("https://"):
            raise ValueError("base_url должен быть https:// (см. FinamConfig)")
        from ..config.env import GLOBAL_MASKER

        self._masker = GLOBAL_MASKER
        self._masker.register(secret, label="SECRET_KEY")
        self._secret = secret
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_get_attempts = max(1, max_get_attempts)
        self._transport = transport or UrllibTransport()
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._limiter = SlidingWindowLimiter(
            rate_limit_per_minute, clock=clock, sleep=sleep, rng=self._rng
        )
        self._token: str | None = None
        self._token_details: TokenDetails | None = None

    # ------------------------------------------------------------------ auth

    def authenticate(self) -> TokenDetails:
        """POST /v1/sessions -> POST /v1/sessions/details (документированный поток)."""
        payload = self._post_json("/v1/sessions", {"secret": self._secret}, auth=False)
        token = payload.get("token")
        if not isinstance(token, str) or not token:
            raise FinamInvalidResponseError("Ответ /v1/sessions не содержит строкового поля 'token'")
        self._masker.register(token, label="ACCESS_KEY")
        self._token = token
        details_payload = self._post_json("/v1/sessions/details", {"token": token}, auth=False)
        self._token_details = self._parse_token_details(details_payload)
        logger.info(
            "Аутентификация выполнена: readonly=%s, счетов=%d, expires_at=%s",
            self._token_details.readonly,
            len(self._token_details.account_ids),
            self._token_details.expires_at.isoformat(),
        )
        return self._token_details

    def _parse_token_details(self, payload: dict[str, Any]) -> TokenDetails:
        account_ids = payload.get("account_ids") or []
        if not isinstance(account_ids, list) or not all(isinstance(a, str) for a in account_ids):
            raise FinamInvalidResponseError("Поле account_ids должно быть списком строк")
        expires_raw = payload.get("expires_at")
        expires_parsed = True
        try:
            expires_at = _parse_api_time(expires_raw)
        except FinamInvalidResponseError:
            # Не угадываем срок жизни: консервативно считаем токен короткоживущим.
            expires_at = datetime.now(timezone.utc) + _FALLBACK_TOKEN_TTL
            expires_parsed = False
        created_raw = payload.get("created_at")
        try:
            created_at = _parse_api_time(created_raw) if created_raw else None
        except FinamInvalidResponseError:
            created_at = None
        md_permissions = payload.get("md_permissions") or []
        if not isinstance(md_permissions, list):
            md_permissions = []
        return TokenDetails(
            expires_at=expires_at,
            readonly=bool(payload.get("readonly", True)),
            account_ids=tuple(account_ids),
            md_permissions=tuple(p for p in md_permissions if isinstance(p, dict)),
            created_at=created_at,
            expires_at_parsed=expires_parsed,
        )

    def _ensure_token(self) -> str:
        if self._token is None or self._token_details is None:
            self.authenticate()
            assert self._token is not None
            return self._token
        now = datetime.now(timezone.utc)
        if now >= self._token_details.expires_at - _TOKEN_REFRESH_MARGIN:
            logger.info("access_key истекает (expires_at) — перевыпуск")
            self.authenticate()
            assert self._token is not None
        return self._token

    @property
    def token_details(self) -> TokenDetails | None:
        return self._token_details

    # ------------------------------------------------------------- market data

    def fetch_bars(
        self,
        symbol: str,
        timeframe: "Timeframe | str",
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[FinamBarRecord]:
        """GET /v1/instruments/{symbol}/bars. [start, end) — как в документации."""
        tf = Timeframe.parse(timeframe)
        api_tf = API_TIMEFRAME_BY_CODE.get(tf.value)
        if api_tf is None:
            raise UnsupportedTimeframeError(
                f"Таймфрейм {tf.value} не поддерживается Finam API: в документированном enum "
                "TimeFrame нет TIME_FRAME_M10 (research §2.3). Стройте 10m локальным "
                "ресэмплингом из 5m/1m (data.resampler, decisions D3)"
            )
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("start_utc/end_utc должны быть timezone-aware (UTC)")
        if end_utc <= start_utc:
            raise ValueError("end_utc должен быть строго позже start_utc")
        depth = HISTORY_DEPTH_LIMITS.get(tf.value)
        if depth is not None and (end_utc - start_utc) > depth:
            raise HistoryDepthError(
                f"Запрошенное окно {end_utc - start_utc} превышает документированную глубину "
                f"{depth} для {tf.value} (research §2.3). Уменьшите окно или используйте CSV-архив"
            )
        params = {
            "timeframe": api_tf,
            "interval.start_time": _format_api_time(start_utc),
            "interval.end_time": _format_api_time(end_utc),
        }
        payload = self._get_json(f"/v1/instruments/{_enc(symbol)}/bars", params)
        raw_bars = payload.get("bars")
        if not isinstance(raw_bars, list):
            raise FinamInvalidResponseError("Ответ bars не содержит списка 'bars'")
        records: list[FinamBarRecord] = []
        for item in raw_bars:
            if not isinstance(item, dict):
                raise FinamInvalidResponseError(f"Элемент bars не является объектом: {item!r}")
            records.append(
                FinamBarRecord(
                    timestamp=_parse_api_time(item.get("timestamp")),
                    open=_parse_decimal(item.get("open"), "open"),
                    high=_parse_decimal(item.get("high"), "high"),
                    low=_parse_decimal(item.get("low"), "low"),
                    close=_parse_decimal(item.get("close"), "close"),
                    volume=_parse_decimal(item.get("volume"), "volume"),
                    is_data_snapshot=bool(item.get("is_data_snapshot", False)),
                )
            )
        return records

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        """GET /v1/instruments/{symbol}/quotes/latest."""
        payload = self._get_json(f"/v1/instruments/{_enc(symbol)}/quotes/latest", None)
        quote = payload.get("quote")
        if not isinstance(quote, dict):
            raise FinamInvalidResponseError("Ответ quotes/latest не содержит объекта 'quote'")
        ts_raw = quote.get("timestamp")
        return QuoteSnapshot(
            symbol=str(payload.get("symbol", symbol)),
            timestamp=_parse_api_time(ts_raw) if ts_raw else None,
            last=_optional_decimal(quote, "last"),
            bid=_optional_decimal(quote, "bid"),
            ask=_optional_decimal(quote, "ask"),
            bid_size=_optional_decimal(quote, "bid_size"),
            ask_size=_optional_decimal(quote, "ask_size"),
            volume=_optional_decimal(quote, "volume"),
        )

    def get_order_book(self, symbol: str) -> OrderBookSnapshot:
        """GET /v1/instruments/{symbol}/orderbook (снимок; формат подтверждён research §2.5).

        Ограничение (research Q4): гарантированная глубина стакана документацией
        не задана — возвращаем столько уровней, сколько прислал сервер.
        """
        payload = self._get_json(f"/v1/instruments/{_enc(symbol)}/orderbook", None)
        orderbook = payload.get("orderbook")
        if not isinstance(orderbook, dict):
            raise FinamInvalidResponseError("Ответ orderbook не содержит объекта 'orderbook'")
        raw_rows = orderbook.get("rows")
        if not isinstance(raw_rows, list):
            raise FinamInvalidResponseError("orderbook не содержит списка 'rows'")
        rows: list[OrderBookRow] = []
        for item in raw_rows:
            if not isinstance(item, dict):
                raise FinamInvalidResponseError(f"Строка стакана не является объектом: {item!r}")
            price = _parse_decimal(item.get("price"), "price")
            buy_size = _optional_decimal(item, "buy_size")
            sell_size = _optional_decimal(item, "sell_size")
            if buy_size is not None and sell_size is not None:
                # Документация описывает oneof buy_size/sell_size — оба значения
                # одновременно не ожидаются; не угадываем приоритет.
                raise FinamInvalidResponseError(
                    "Строка стакана содержит одновременно buy_size и sell_size "
                    "(документирован oneof) — ответ отклонён"
                )
            if buy_size is not None:
                side, size = OrderBookLevelSide.BUY, buy_size
            elif sell_size is not None:
                side, size = OrderBookLevelSide.SELL, sell_size
            else:
                raise FinamInvalidResponseError("Строка стакана не содержит ни buy_size, ни sell_size")
            action_raw = item.get("action", "ACTION_UNSPECIFIED")
            try:
                action = OrderBookRowAction(str(action_raw))
            except ValueError:
                action = OrderBookRowAction.UNSPECIFIED
            ts_raw = item.get("timestamp")
            rows.append(
                OrderBookRow(
                    price=price,
                    size=size,
                    side=side,
                    action=action,
                    mpid=item.get("mpid") if isinstance(item.get("mpid"), str) else None,
                    timestamp=_parse_api_time(ts_raw) if ts_raw else None,
                )
            )
        return OrderBookSnapshot(
            symbol=str(payload.get("symbol", symbol)),
            rows=tuple(rows),
            received_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------- instruments

    def get_asset(self, symbol: str, *, account_id: str | None = None) -> AssetInfo:
        """GET /v1/assets/{symbol} (метаданные; поля подтверждены S3d)."""
        params = {"account_id": account_id} if account_id else None
        payload = self._get_json(f"/v1/assets/{_enc(symbol)}", params)
        if not isinstance(payload, dict):
            raise FinamInvalidResponseError("Ответ GetAsset не является объектом")
        lot_size = _optional_decimal(payload, "lot_size")
        min_step_raw = payload.get("min_step")
        decimals_raw = payload.get("decimals")
        return AssetInfo(
            symbol=str(payload.get("symbol", symbol)),
            name=payload.get("name") if isinstance(payload.get("name"), str) else None,
            ticker=payload.get("ticker") if isinstance(payload.get("ticker"), str) else None,
            mic=payload.get("mic") if isinstance(payload.get("mic"), str) else None,
            quote_currency=(
                payload.get("quote_currency") if isinstance(payload.get("quote_currency"), str) else None
            ),
            lot_size=lot_size,
            min_step=int(min_step_raw) if isinstance(min_step_raw, (int, str)) and _is_int(min_step_raw) else None,
            decimals=int(decimals_raw) if isinstance(decimals_raw, (int, str)) and _is_int(decimals_raw) else None,
            raw=payload,
        )

    def get_asset_params(self, symbol: str, *, account_id: str) -> AssetParams:
        """GET /v1/assets/{symbol}/params?account_id=... (S3d: параметры в привязке к счёту).

        trade_lot_size == 0 означает «значение недоступно» — НЕ «один лот = одна штука»
        (research §2.7, S3j Рецепт 4): потребитель обязан остановить торговую операцию.
        """
        if not account_id:
            raise ValueError("account_id обязателен для GetAssetParams (чувствительное значение, передаётся из env)")
        payload = self._get_json(f"/v1/assets/{_enc(symbol)}/params", {"account_id": account_id})
        if not isinstance(payload, dict):
            raise FinamInvalidResponseError("Ответ GetAssetParams не является объектом")
        tls_raw = payload.get("trade_lot_size")
        tradable_raw = payload.get("is_tradable")
        return AssetParams(
            trade_lot_size=int(tls_raw) if _is_int(tls_raw) else None,
            is_tradable=bool(tradable_raw) if isinstance(tradable_raw, bool) else None,
            raw=payload,
        )

    # ----------------------------------------------------------------- plumbing

    def close(self) -> None:
        """Состояния соединения нет (запрос-ответ); метод для симметрии интерфейсов."""
        self._token = None
        self._token_details = None

    def _headers(self, auth: bool) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self._ensure_token()}"
        return headers

    def _get_json(self, path: str, params: dict[str, str] | None) -> dict[str, Any]:
        url = self._base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        method_key = f"GET {_path_template(path)}"
        attempts = 0
        reauth_used = False
        rate_limit_wait_used = False
        while True:
            attempts += 1
            self._limiter.acquire(method_key)
            started = self._clock()
            try:
                response = self._transport.request(
                    "GET", url, headers=self._headers(auth=True), timeout=self._timeout
                )
            except FinamTransportError as exc:
                if attempts >= self._max_get_attempts:
                    logger.warning("%s: транспортная ошибка, попытки исчерпаны", method_key)
                    raise
                self._backoff_sleep(attempts, f"транспортная ошибка: {exc}")
                continue
            elapsed_ms = (self._clock() - started) * 1000.0
            logger.info("%s -> %d (%.0f ms)", method_key, response.status, elapsed_ms)

            if response.status == 200:
                return self._parse_json_body(response)
            if response.status == 401:
                if not reauth_used:
                    # Документированное действие: получить новый access_key и повторить
                    # безопасный (GET) запрос один раз (research §2.8).
                    reauth_used = True
                    logger.info("%s -> 401: перевыпуск access_key и один повтор", method_key)
                    self._token = None
                    self._token_details = None
                    continue
                raise FinamAuthError(self._error_message(response, "401 после перевыпуска access_key"))
            if response.status == 429:
                if not rate_limit_wait_used:
                    # Не повторять немедленно: дождаться следующего минутного окна (research §2.8).
                    rate_limit_wait_used = True
                    wait = 60.0 + self._rng.uniform(0.0, 1.0)
                    logger.warning("%s -> 429: ожидание %.0f с до следующего окна", method_key, wait)
                    self._sleep(wait)
                    continue
                raise FinamRateLimitError(self._error_message(response, "лимит 200 запросов/мин на метод исчерпан"))
            if response.status == 400:
                raise FinamInvalidRequestError(self._error_message(response, "неверный запрос"))
            if response.status == 403:
                raise FinamPermissionDeniedError(self._error_message(response, "недостаточно прав"))
            if response.status == 404:
                raise FinamNotFoundError(self._error_message(response, "не найдено"))
            if response.status in (500, 503, 504):
                if attempts >= self._max_get_attempts:
                    raise FinamServerError(
                        self._error_message(response, f"серверная ошибка {response.status}"),
                        status_code=response.status,
                    )
                self._backoff_sleep(attempts, f"HTTP {response.status}")
                continue
            raise FinamError(self._error_message(response, f"неожиданный HTTP {response.status}"))

    def _post_json(self, path: str, body: dict[str, Any], *, auth: bool) -> dict[str, Any]:
        """POST без автоповторов (в этой фазе — только аутентификация)."""
        url = self._base_url + path
        method_key = f"POST {_path_template(path)}"
        self._limiter.acquire(method_key)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self._ensure_token()}"
        try:
            response = self._transport.request(method="POST", url=url, headers=headers, json_body=body, timeout=self._timeout)
        except FinamTransportError:
            logger.warning("POST %s: транспортная ошибка (без автоповтора торговых/auth POST)", method_key)
            raise
        logger.info("POST %s -> %d", method_key, response.status)
        if response.status == 200:
            return self._parse_json_body(response)
        if response.status == 401:
            raise FinamAuthError(self._error_message(response, "аутентификация отклонена (проверьте secret_key)"))
        if response.status == 403:
            raise FinamPermissionDeniedError(self._error_message(response, "недостаточно прав"))
        if response.status == 429:
            raise FinamRateLimitError(self._error_message(response, "лимит запросов исчерпан"))
        if response.status == 400:
            raise FinamInvalidRequestError(self._error_message(response, "неверный запрос"))
        if response.status == 404:
            raise FinamNotFoundError(self._error_message(response, "не найдено"))
        raise FinamServerError(
            self._error_message(response, f"HTTP {response.status}"), status_code=response.status
        )

    def _parse_json_body(self, response: HttpResponse) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception as exc:
            raise FinamInvalidResponseError(f"Ответ не является валидным JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise FinamInvalidResponseError("Ожидался JSON-объект верхнего уровня")
        return payload

    def _error_message(self, response: HttpResponse, context: str) -> str:
        """Сообщение об ошибке из тела {code, message, details} (research §2.8) + контекст."""
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                code = payload.get("code")
                message = payload.get("message")
                if message is not None or code is not None:
                    detail = f": code={code!r} message={message!r}"
        except Exception:
            body_text = response.body[:200].decode("utf-8", errors="replace")
            if body_text.strip():
                detail = f": {body_text!r}"
        return f"{context} (HTTP {response.status}){detail}"

    def _backoff_sleep(self, attempt: int, reason: str) -> None:
        index = min(attempt - 1, len(_BACKOFF_SCHEDULE_SECONDS) - 1)
        base = _BACKOFF_SCHEDULE_SECONDS[index]
        delay = base + self._rng.uniform(0.0, base * 0.1 + 0.05)
        logger.info("Повтор через %.1f с (попытка %d, причина: %s)", delay, attempt, reason)
        self._sleep(delay)


def _enc(symbol: str) -> str:
    """URL-кодирование символа ticker@mic (encodeURIComponent-эквивалент)."""
    return urllib.parse.quote(symbol, safe="")


def _path_template(path: str) -> str:
    """Ключ лимитера/логов — шаблон пути без конкретных значений (research S3j Рецепт 8)."""
    parts = path.split("/")
    for i, part in enumerate(parts):
        if "@" in part or urllib.parse.quote(part, safe="") != part:
            parts[i] = "{symbol}"
    return "/".join(parts)


def _is_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        try:
            int(value)
            return True
        except ValueError:
            return False
    return False
