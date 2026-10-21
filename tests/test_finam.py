"""Тесты read-only адаптера Finam: подменный транспорт (без сети), документированные
методы и форматы, ошибки/ретраи по официальной таблице, маскирование секретов.

Все токены/счета — фиктивные строки (FAKE-*), сеть не используется.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from tradebot.data.bar import UTC
from tradebot.finam.errors import (
    FinamAuthError,
    FinamInvalidRequestError,
    FinamInvalidResponseError,
    FinamNotFoundError,
    FinamRateLimitError,
    FinamServerError,
    HistoryDepthError,
    UnsupportedTimeframeError,
)
from tradebot.finam.ratelimit import SlidingWindowLimiter
from tradebot.finam.rest_client import FinamRestClient
from tradebot.finam.source import FinamHistorySource, FinamSnapshotSource
from tradebot.marketdata.interfaces import (
    HistorySource,
    OrderBookLevelSide,
    OrderBookSnapshotSource,
    QuoteSnapshotSource,
)
from conftest import (
    FAKE_ACCESS_KEY,
    FAKE_ACCOUNT_ID,
    FAKE_SECRET,
    bar_payload,
    json_response,
    make_bars_payload,
)

TS0 = datetime(2026, 10, 20, 7, 0, tzinfo=timezone.utc)


def auth_responses(expires_in_minutes: int = 15):
    """Стандартная пара ответов аутентификации (документированные схемы)."""
    expires = (datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return [
        json_response(200, {"token": FAKE_ACCESS_KEY}),
        json_response(
            200,
            {
                "created_at": "2026-10-20T07:00:00Z",
                "expires_at": expires,
                "md_permissions": [{"quote_level": "QUOTE_LEVEL_1", "delay_minutes": 0, "mic": "MISX"}],
                "account_ids": [FAKE_ACCOUNT_ID],
                "readonly": False,
            },
        ),
    ]


def make_client(transport, **kwargs):
    defaults = dict(
        base_url="https://api.finam.ru",
        timeout_seconds=5.0,
        max_get_attempts=3,
        clock=kwargs.pop("clock", lambda: 0.0),
        sleep=kwargs.pop("sleep", lambda s: None),
        rng=random.Random(42),
    )
    defaults.update(kwargs)
    return FinamRestClient(FAKE_SECRET, transport=transport, **defaults)


class TestAuth:
    def test_auth_flow_and_token_details(self):
        from conftest import FakeTransport

        transport = FakeTransport(responses=auth_responses())
        client = make_client(transport)
        details = client.authenticate()
        assert details.readonly is False
        assert details.account_ids == (FAKE_ACCOUNT_ID,)
        assert details.expires_at_parsed is True
        # POST /v1/sessions содержит secret в теле (документировано) и НЕ содержит его в URL
        first = transport.requests[0]
        assert first.method == "POST" and first.url.endswith("/v1/sessions")
        assert first.json_body == {"secret": FAKE_SECRET}
        assert FAKE_SECRET not in first.url
        # details: токен в теле, не в URL
        second = transport.requests[1]
        assert second.url.endswith("/v1/sessions/details")
        assert second.json_body == {"token": FAKE_ACCESS_KEY}

    def test_unparsable_expires_at_falls_back_conservative(self):
        from conftest import FakeTransport

        responses = [
            json_response(200, {"token": FAKE_ACCESS_KEY}),
            json_response(200, {"account_ids": [], "readonly": True, "expires_at": "не-дата"}),
        ]
        client = make_client(FakeTransport(responses=responses))
        details = client.authenticate()
        assert details.expires_at_parsed is False
        # консервативный TTL: не позже ~60 c от текущего момента
        assert details.expires_at <= datetime.now(timezone.utc) + timedelta(seconds=61)

    def test_auth_401_raises_masked(self):
        from conftest import FakeTransport

        transport = FakeTransport(
            responses=[json_response(401, {"code": 16, "message": f"bad token {FAKE_SECRET}"})]
        )
        client = make_client(transport)
        with pytest.raises(FinamAuthError) as exc:
            client.authenticate()
        assert FAKE_SECRET not in str(exc.value)
        assert "***SECRET_KEY***" in str(exc.value)

    def test_account_ids_not_in_repr_or_str(self):
        from conftest import FakeTransport

        client = make_client(FakeTransport(responses=auth_responses()))
        details = client.authenticate()
        assert FAKE_ACCOUNT_ID not in repr(details)
        assert FAKE_ACCOUNT_ID not in str(details)


class TestBars:
    def _client_with_bars(self, bars, extra_responses=None):
        from conftest import FakeTransport

        responses = auth_responses() + [json_response(200, make_bars_payload("SBER@MISX", bars))]
        if extra_responses:
            responses.extend(extra_responses)
        return make_client(FakeTransport(responses=responses))

    def test_fetch_bars_parses_documented_shape(self):
        bars = [
            bar_payload("2026-10-20T07:00:00Z", "313.58", "314.10", "313.01", "313.90", "123456"),
            bar_payload("2026-10-20T07:05:00Z", "313.90", "314.55", "313.70", "314.40", "98765"),
        ]
        client = self._client_with_bars(bars)
        records = client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))
        assert len(records) == 2
        assert records[0].open == Decimal("313.58")
        assert records[0].volume == Decimal("123456")
        assert records[0].timestamp == datetime(2026, 10, 20, 7, 0, tzinfo=UTC)

    def test_request_url_and_params(self):
        from conftest import FakeTransport

        transport = FakeTransport(
            responses=auth_responses() + [json_response(200, make_bars_payload("SBER@MISX", []))]
        )
        client = make_client(transport)
        client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))
        bars_request = transport.requests[-1]
        assert bars_request.method == "GET"
        assert "/v1/instruments/SBER%40MISX/bars" in bars_request.url
        assert "timeframe=TIME_FRAME_M5" in bars_request.url
        assert "interval.start_time=2026-10-20T07%3A00%3A00Z" in bars_request.url
        assert "interval.end_time=2026-10-20T08%3A00%3A00Z" in bars_request.url
        assert bars_request.headers["Authorization"] == f"Bearer {FAKE_ACCESS_KEY}"
        # secret не фигурирует в GET-запросах
        for req in transport.requests[2:]:
            assert FAKE_SECRET not in req.url
            assert FAKE_SECRET not in str(req.headers)

    def test_10m_unsupported(self):
        from conftest import FakeTransport

        client = make_client(FakeTransport(responses=auth_responses()))
        client.authenticate()
        with pytest.raises(UnsupportedTimeframeError, match="TIME_FRAME_M10"):
            client.fetch_bars("SBER@MISX", "10m", TS0, TS0 + timedelta(hours=1))

    def test_history_depth_guard(self):
        from conftest import FakeTransport

        client = make_client(FakeTransport(responses=auth_responses()))
        client.authenticate()
        # M1: документированная глубина 7 дней (research §2.3)
        with pytest.raises(HistoryDepthError):
            client.fetch_bars("SBER@MISX", "1m", TS0, TS0 + timedelta(days=8))
        with pytest.raises(HistoryDepthError):
            client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(days=31))
        # в пределах глубины — без ошибки
        client_ok = make_client(
            FakeTransport(responses=auth_responses() + [json_response(200, make_bars_payload("SBER@MISX", []))])
        )
        client_ok.fetch_bars("SBER@MISX", "1m", TS0, TS0 + timedelta(days=6))

    def test_naive_timestamp_rejected(self):
        bars = [bar_payload("2026-10-20T07:00:00", "1", "2", "0.5", "1.5", "10")]  # без Z
        client = self._client_with_bars(bars)
        with pytest.raises(FinamInvalidResponseError, match="часового пояса"):
            client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))

    def test_bad_decimal_rejected(self):
        bars = [{"timestamp": "2026-10-20T07:00:00Z", "open": {"value": "abc"}, "high": {"value": "2"},
                 "low": {"value": "0.5"}, "close": {"value": "1.5"}, "volume": {"value": "10"}}]
        client = self._client_with_bars(bars)
        with pytest.raises(FinamInvalidResponseError):
            client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))


class TestSnapshots:
    def test_latest_quote(self):
        from conftest import FakeTransport

        quote = {
            "symbol": "SBER@MISX",
            "quote": {
                "timestamp": "2026-10-20T07:00:01Z",
                "last": {"value": "313.58"},
                "bid": {"value": "313.57"},
                "ask": {"value": "313.59"},
                "bid_size": {"value": "120"},
                "ask_size": {"value": "90"},
                "volume": {"value": "1234567"},
            },
        }
        transport = FakeTransport(responses=auth_responses() + [json_response(200, quote)])
        client = make_client(transport)
        snapshot_source = FinamSnapshotSource(client)
        q = snapshot_source.get_latest_quote("SBER@MISX")
        assert q.last == Decimal("313.58")
        assert q.bid == Decimal("313.57") and q.ask == Decimal("313.59")
        assert q.bid_size == Decimal("120") and q.ask_size == Decimal("90")

    def test_order_book_snapshot_documented_shape(self):
        from conftest import FakeTransport

        payload = {
            "symbol": "SBER@MISX",
            "orderbook": {
                "rows": [
                    {"price": {"value": "313.57"}, "buy_size": {"value": "120"}, "action": "ACTION_UPDATE",
                     "mpid": "MC0003500000", "timestamp": "2026-10-20T07:00:01Z"},
                    {"price": {"value": "313.59"}, "sell_size": {"value": "90"}, "action": "ACTION_ADD"},
                ]
            },
        }
        transport = FakeTransport(responses=auth_responses() + [json_response(200, payload)])
        client = make_client(transport)
        source = FinamSnapshotSource(client)
        book = source.get_order_book("SBER@MISX")
        assert book.depth == 2
        assert book.rows[0].side is OrderBookLevelSide.BUY
        assert book.rows[0].price == Decimal("313.57")
        assert book.rows[1].side is OrderBookLevelSide.SELL
        assert book.rows[1].size == Decimal("90")

    def test_order_book_oneof_violation_rejected(self):
        from conftest import FakeTransport

        payload = {
            "symbol": "SBER@MISX",
            "orderbook": {
                "rows": [
                    {"price": {"value": "1"}, "buy_size": {"value": "10"}, "sell_size": {"value": "5"}}
                ]
            },
        }
        client = make_client(FakeTransport(responses=auth_responses() + [json_response(200, payload)]))
        with pytest.raises(FinamInvalidResponseError, match="oneof"):
            FinamSnapshotSource(client).get_order_book("SBER@MISX")

    def test_snapshot_source_implements_interfaces(self):
        from conftest import FakeTransport

        source = FinamSnapshotSource(make_client(FakeTransport(responses=auth_responses())))
        assert isinstance(source, QuoteSnapshotSource)
        assert isinstance(source, OrderBookSnapshotSource)


class TestAssets:
    def test_get_asset_fields(self):
        from conftest import FakeTransport

        payload = {
            "symbol": "SBER@MISX",
            "name": "Сбербанк",
            "ticker": "SBER",
            "mic": "MISX",
            "quote_currency": "RUB",
            "lot_size": {"value": "10"},
            "min_step": 1,
            "decimals": 2,
        }
        transport = FakeTransport(responses=auth_responses() + [json_response(200, payload)])
        client = make_client(transport)
        asset = client.get_asset("SBER@MISX", account_id=FAKE_ACCOUNT_ID)
        assert asset.ticker == "SBER" and asset.mic == "MISX"
        assert asset.lot_size == Decimal("10")
        assert asset.min_step == 1 and asset.decimals == 2
        assert f"account_id={FAKE_ACCOUNT_ID}" in transport.requests[-1].url

    def test_get_asset_params_requires_account(self):
        from conftest import FakeTransport

        client = make_client(FakeTransport(responses=auth_responses()))
        client.authenticate()
        with pytest.raises(ValueError, match="account_id"):
            client.get_asset_params("SBER@MISX", account_id="")

    def test_get_asset_params_zero_lot_size_preserved(self):
        from conftest import FakeTransport

        payload = {"trade_lot_size": 0, "is_tradable": True}
        client = make_client(FakeTransport(responses=auth_responses() + [json_response(200, payload)]))
        params = client.get_asset_params("SBER@MISX", account_id=FAKE_ACCOUNT_ID)
        # 0 = «значение недоступно» (research §2.7) — передаём как есть, не подменяем на 1
        assert params.trade_lot_size == 0
        assert params.is_tradable is True


class TestErrorHandling:
    def _client_after_auth(self, responses):
        from conftest import FakeTransport

        transport = FakeTransport(responses=auth_responses() + responses)
        return make_client(transport), transport

    def test_400_no_retry(self):
        client, transport = self._client_after_auth([json_response(400, {"code": 3, "message": "bad symbol"})])
        with pytest.raises(FinamInvalidRequestError, match="bad symbol"):
            client.fetch_bars("NOPE@NOPE", "5m", TS0, TS0 + timedelta(hours=1))
        get_calls = [r for r in transport.requests if r.method == "GET" and "/bars" in r.url]
        assert len(get_calls) == 1  # без повторов

    def test_404(self):
        client, _ = self._client_after_auth([json_response(404, {"code": 5, "message": "not found"})])
        with pytest.raises(FinamNotFoundError):
            client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))

    def test_503_retries_with_backoff_then_raises(self):
        sleeps: list[float] = []
        from conftest import FakeTransport

        responses = [json_response(503, {"message": "unavailable"})] * 5
        transport = FakeTransport(responses=auth_responses() + responses)
        client = make_client(transport, sleep=sleeps.append)
        with pytest.raises(FinamServerError) as exc:
            client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))
        assert exc.value.status_code == 503
        # max_get_attempts=3: две паузы backoff перед третьей (последней) попыткой
        assert len(sleeps) == 2
        assert sleeps[0] < sleeps[1]  # возрастающие паузы

    def test_429_waits_for_window_and_retries_once(self):
        sleeps: list[float] = []
        from conftest import FakeTransport

        responses = [
            json_response(429, {"message": "rate limit"}),
            json_response(200, make_bars_payload("SBER@MISX", [])),
        ]
        transport = FakeTransport(responses=auth_responses() + responses)
        client = make_client(transport, sleep=sleeps.append)
        records = client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))
        assert records == []
        # ожидание следующего минутного окна (~60 c), не немедленный повтор
        assert any(s >= 60.0 for s in sleeps)

    def test_429_twice_raises(self):
        from conftest import FakeTransport

        responses = [json_response(429, {"message": "limit"})] * 3
        client = make_client(FakeTransport(responses=auth_responses() + responses))
        with pytest.raises(FinamRateLimitError):
            client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))

    def test_401_triggers_single_reauth_and_retry(self):
        from conftest import FakeTransport

        responses = [
            json_response(401, {"message": "expired"}),
            # повторная аутентификация
            json_response(200, {"token": FAKE_ACCESS_KEY}),
            json_response(200, {"account_ids": [], "readonly": True, "expires_at": "2026-10-20T09:00:00Z"}),
            # успешный повтор GET
            json_response(200, make_bars_payload("SBER@MISX", [])),
        ]
        transport = FakeTransport(responses=auth_responses() + responses)
        client = make_client(transport)
        records = client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))
        assert records == []
        auth_posts = [r for r in transport.requests if r.url.endswith("/v1/sessions")]
        assert len(auth_posts) == 2  # первичная + один перевыпуск


class TestMasking:
    def test_secrets_masked_in_errors_and_logs(self, caplog):
        from conftest import FakeTransport

        # сервер «эхом» возвращает в тексте ошибки и secret, и access_key;
        # оба значения к этому моменту зарегистрированы в маскировщике
        body = {"code": 3, "message": f"invalid request: secret={FAKE_SECRET} token={FAKE_ACCESS_KEY}"}
        transport = FakeTransport(responses=auth_responses() + [json_response(400, body)])
        client = make_client(transport)
        with caplog.at_level(logging.INFO, logger="tradebot"):
            with pytest.raises(FinamInvalidRequestError) as exc:
                client.fetch_bars("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))
        message = str(exc.value)
        assert FAKE_SECRET not in message
        assert FAKE_ACCESS_KEY not in message
        assert "***SECRET_KEY***" in message
        assert "***ACCESS_KEY***" in message
        # в логах — тоже маскировано (и токенов там быть не должно вовсе)
        full_log = caplog.text
        assert FAKE_SECRET not in full_log
        assert FAKE_ACCESS_KEY not in full_log

    def test_masker_unit(self):
        from tradebot.config.env import SecretMasker

        masker = SecretMasker()
        masker.register("abc123", label="SECRET_KEY")
        assert masker.mask("token abc123 leaked") == "token ***SECRET_KEY*** leaked"
        assert masker.mask("no secrets here") == "no secrets here"


class TestRateLimiter:
    def test_limiter_waits_when_exhausted(self):
        now = [0.0]
        sleeps: list[float] = []

        def clock():
            return now[0]

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        limiter = SlidingWindowLimiter(2, clock=clock, sleep=sleep, jitter_seconds=0.0)
        limiter.acquire("GET /test")
        limiter.acquire("GET /test")
        limiter.acquire("GET /test")  # третий — ждёт окно
        assert len(sleeps) == 1
        assert sleeps[0] >= 59.9

    def test_limiter_per_method(self):
        now = [0.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        limiter = SlidingWindowLimiter(
            1, clock=lambda: now[0], sleep=sleep, jitter_seconds=0.0
        )
        limiter.acquire("GET /a")
        # лимит считается НА МЕТОД (research §2.7): другой метод не блокируется
        limiter.acquire("GET /b")
        assert sleeps == []
        # повтор того же метода при лимите 1 — уже с ожиданием
        limiter.acquire("GET /a")
        assert len(sleeps) == 1
        assert sleeps[0] >= 59.9

    def test_limit_above_documented_rejected(self):
        with pytest.raises(ValueError):
            SlidingWindowLimiter(201)


class TestHistorySource:
    def test_requires_explicit_semantics(self):
        from conftest import FakeTransport

        client = make_client(FakeTransport(responses=auth_responses()))
        with pytest.raises(ValueError, match="bar_timestamp_semantics"):
            FinamHistorySource(client, bar_timestamp_semantics=None)

    def test_normalization_bar_open_and_bar_close(self):
        from conftest import FakeTransport

        bars = [bar_payload("2026-10-20T07:05:00Z", "100", "101", "99", "100.5", "10")]
        for semantics, expected in (("bar_open", datetime(2026, 10, 20, 7, 5, tzinfo=UTC)),
                                    ("bar_close", datetime(2026, 10, 20, 7, 0, tzinfo=UTC))):
            transport = FakeTransport(responses=auth_responses() + [json_response(200, make_bars_payload("SBER@MISX", bars))])
            source = FinamHistorySource(make_client(transport), bar_timestamp_semantics=semantics)
            result = source.fetch_history("SBER@MISX", "5m", TS0, TS0 + timedelta(hours=1))
            assert result[0].ts_open == expected
            assert result[0].volume == 10
            assert isinstance(source, HistorySource)

    def test_10m_raises_with_guidance(self):
        from conftest import FakeTransport

        source = FinamHistorySource(
            make_client(FakeTransport(responses=auth_responses())),
            bar_timestamp_semantics="bar_open",
        )
        with pytest.raises(UnsupportedTimeframeError, match="ресэмплинг"):
            source.fetch_history("SBER@MISX", "10m", TS0, TS0 + timedelta(hours=1))

    def test_fractional_volume_rejected_not_rounded(self):
        from conftest import FakeTransport

        bars = [bar_payload("2026-10-20T07:00:00Z", "100", "101", "99", "100.5", "10.5")]
        transport = FakeTransport(responses=auth_responses() + [json_response(200, make_bars_payload("SBER@MISX", bars))])
        source = FinamHistorySource(make_client(transport), bar_timestamp_semantics="bar_open")
        with pytest.raises(FinamInvalidResponseError, match="целым"):
            source.fetch_history("SBER@MISX", "1m", TS0, TS0 + timedelta(hours=1))
