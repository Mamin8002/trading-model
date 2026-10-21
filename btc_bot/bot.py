"""Оркестратор бота: LIVE (публичный WS Bybit) и TEST (офлайн-реплей сохранённых данных).

LIVE:  WS-поток -> StreamProcessor -> SignalEngine -> ConsoleUI;
       при (пере)подключении - разовая REST-догрузка истории и пропущенных баров
       (публичные read-only эндпоинты; торговые вызовы отсутствуют).
TEST:  реплей закрытых баров из сохранённого clean-CSV через тот же тракт
       (фейковые часы), плюс демонстрационный сценарий устаревших данных.
"""

from __future__ import annotations

import logging
import queue
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import AppConfig, load_config
from .data.prep import clean_klines
from .data.store import load_klines_csv
from .data_sources.base import DataSourceError, Kline
from .data_sources.bybit import BybitSource, period_end_ms, step_ms_approx
from .model.predictor import Predictor
from .model.train import find_latest_model_dir
from .signals.engine import Signal, SignalEngine
from .streaming.base import (
    ConnState,
    ErrorEvent,
    FreshnessChange,
    GapDetected,
    KlineBar,
    StateChange,
    TickerUpdate,
)
from .streaming.bybit_ws import BybitWsClient
from .streaming.processor import StreamProcessor
from .ui.console import ConsoleUI, format_snapshot_text

logger = logging.getLogger(__name__)


def load_predictor(cfg: AppConfig) -> tuple[Predictor, Path]:
    if cfg.model.artifact_dir:
        model_dir = cfg.resolve(cfg.model.artifact_dir)
    else:
        found = find_latest_model_dir(cfg.resolve(cfg.data.models_dir))
        if found is None:
            raise FileNotFoundError(
                "Артефакт модели не найден. Сначала выполните: python scripts/train_model.py"
            )
        model_dir = found
    predictor = Predictor.load(model_dir, confidence_threshold=cfg.model.confidence_threshold)
    predictor.ensure_compatible(symbol=cfg.exchange.symbol, category=cfg.exchange.category,
                                interval=cfg.exchange.interval, horizon_bars=cfg.model.horizon_bars)
    return predictor, model_dir


class MonitorBot:
    """Общая логика LIVE/TEST: события -> процессор -> движок -> журнал/советник -> UI."""

    def __init__(self, cfg: AppConfig, predictor: Predictor, ui: ConsoleUI,
                 clock_ms: Optional[Callable[[], int]] = None,
                 rest_source: Optional[BybitSource] = None,
                 journal=None, advisor=None):
        self.cfg = cfg
        self.predictor = predictor
        self.ui = ui
        self.journal = journal          # portfolio.journal.Journal | None
        self.advisor = advisor          # portfolio.advisor.PositionAdvisor | None
        self.last_advice = None
        self.processor = StreamProcessor(
            cfg.exchange.symbol, cfg.exchange.interval,
            stale_after_intervals=cfg.signals.stale_after_intervals,
            history_bars=cfg.stream.history_bars,
            clock_ms=clock_ms,
        )
        self.engine = SignalEngine(
            predictor,
            symbol=cfg.exchange.symbol, category=cfg.exchange.category,
            interval=cfg.exchange.interval, horizon_bars=cfg.model.horizon_bars,
            stale_after_intervals=cfg.signals.stale_after_intervals,
            clock_ms=clock_ms,
            last_price_fn=lambda: self.processor.last_price.price if self.processor.last_price else None,
            stream_fresh_fn=lambda: self.processor.is_fresh(),
        )
        self.rest_source = rest_source
        self.mode = cfg.mode
        self._seeded = False
        self.on_signal: Optional[Callable[[Signal], None]] = None

    # -- event plumbing ----------------------------------------------------
    def dispatch(self, events: list) -> list[Signal]:
        new_signals: list[Signal] = []
        for ev in events:
            if isinstance(ev, KlineBar) and ev.confirmed:
                sig = self.engine.on_confirmed_bar(ev)
                if sig is not None:
                    new_signals.append(sig)
            elif isinstance(ev, StateChange):
                self.engine.on_conn_state(ev.state)
                if ev.state == ConnState.CONNECTED:
                    self._on_connected()
            elif isinstance(ev, FreshnessChange):
                self.engine.on_freshness(ev)
            elif isinstance(ev, GapDetected):
                self.engine.on_gap(ev)
            elif isinstance(ev, ErrorEvent):
                logger.warning("Ошибка стрима: %s", ev.detail)
        for sig in new_signals:
            self._journal_and_advise(sig)
            if self.on_signal:
                self.on_signal(sig)
        return new_signals

    def _journal_and_advise(self, sig: Signal) -> None:
        """Журналирование сигнала + оценка советником (Этап 4)."""
        if self.journal is not None:
            try:
                self.journal.record_signal(sig)
            except Exception as e:  # noqa: BLE001 - журнал не должен ронять мониторинг
                logger.error("Не удалось записать сигнал в журнал: %s", e)
        if self.advisor is not None:
            try:
                price_now = (self.processor.last_price.price
                             if self.processor.last_price else sig.price_at_signal)
                advice = self.advisor.evaluate(sig, price_now=price_now)
                self.last_advice = advice
                if self.journal is not None:
                    self.journal.record_advice(advice.to_dict(), signal_id=sig.signal_id)
            except Exception as e:  # noqa: BLE001
                logger.error("Ошибка советника: %s", e)

    def _on_connected(self) -> None:
        """После (пере)подключения: seed/backfill истории через публичный REST.

        Seed (первое подключение) заполняет ТОЛЬКО отображаемую историю - сигналы
        по давно закрытым барам не выдаются. Backfill после обрыва догружает бары,
        закрывшиеся во время разрыва, - по ним сигналы формируются (помечаются
        origin="backfill" и проходят обычный gating по свежести).
        """
        if self.rest_source is None or not self.cfg.stream.backfill_on_reconnect:
            return
        try:
            now_ms = self.rest_source.get_server_time_ms()
            step = step_ms_approx(self.cfg.exchange.interval)
            if not self._seeded:
                start = now_ms - self.cfg.stream.history_bars * step
                klines = self.rest_source.fetch_klines(
                    self.cfg.exchange.symbol, self.cfg.exchange.category,
                    self.cfg.exchange.interval, start_ms=start, end_ms=now_ms)
                self._seeded = True
                if klines:
                    self.processor.ingest_backfilled(klines)  # без dispatch: история для UI
                logger.info("Seed истории через REST: %d баров (сигналы по ним не выдаются)",
                            len(klines))
            else:
                last = self.processor.confirmed[-1] if self.processor.confirmed else None
                if last is not None:
                    expected_next = period_end_ms(last.start_ms, self.cfg.exchange.interval)
                    if expected_next < now_ms:
                        klines = self.rest_source.fetch_klines(
                            self.cfg.exchange.symbol, self.cfg.exchange.category,
                            self.cfg.exchange.interval, start_ms=expected_next, end_ms=now_ms)
                        logger.info("Backfill после reconnect: %d баров", len(klines))
                        if klines:
                            self.dispatch(self.processor.ingest_backfilled(klines))
        except DataSourceError as e:
            logger.warning("REST-догрузка не удалась (продолжаем на WS): %s", e)

    # -- view ----------------------------------------------------------------
    def build_view(self, extra_warnings: Optional[list[str]] = None) -> dict:
        snap = self.processor.snapshot()
        now_ms = self.processor._clock()
        age = snap.age_ms
        warnings: list[str] = list(extra_warnings or [])
        if snap.reconnect_count:
            warnings.append(f"переподключений: {snap.reconnect_count}")
        if snap.gaps:
            g = snap.gaps[-1]
            warnings.append(f"разрыв истории баров: ожидался бар {g.expected_start_ms}, "
                            f"получен {g.got_start_ms} (ms)")
        card = self.predictor.card
        price_now = snap.last_price.price if snap.last_price else (
            snap.confirmed_bars[-1].close if snap.confirmed_bars else None)

        # -- Этап 4: позиции из журнала + последняя рекомендация советника --
        positions_view: list[dict] = []
        stats = None
        journal_path = None
        if self.journal is not None:
            try:
                journal_path = str(self.journal.path)
                stats = self.journal.stats()
                for p in self.journal.open_positions(self.cfg.exchange.symbol):
                    rules = p.exit_rules
                    if not rules and self.advisor is not None:
                        rules = self.advisor.default_exit_rules
                    bar_end = snap.confirmed_bars[-1].end_ms if snap.confirmed_bars else None
                    rules_txt = (self.advisor.rules_text(rules or {}, p, price_now, bar_end)
                                 if self.advisor is not None else str(rules))
                    src = p.data.get("source_signal") or {}
                    positions_view.append({
                        "position_id": p.position_id,
                        "direction": p.data["direction"],
                        "symbol": p.data["symbol"],
                        "entry_price": p.entry_price,
                        "opened_at_utc": p.data.get("opened_at_utc"),
                        "quantity_open": p.quantity_open,
                        "quantity_initial": float(p.data["quantity_initial"]),
                        "price_now": price_now,
                        "unrealized_pct": (p.unrealized_pct(price_now) if price_now else None),
                        "rules_text": rules_txt,
                        "source_signal_id": src.get("signal_id"),
                        "source_signal_action": src.get("action"),
                        "source_signal_conf": src.get("confidence"),
                        "horizon_minutes": src.get("horizon_minutes"),
                        "model_version": src.get("model_version"),
                    })
            except Exception as e:  # noqa: BLE001 - проблемы журнала не роняют мониторинг
                warnings.append(f"журнал недоступен: {e}")

        return {
            "mode": self.mode,
            "exchange": {"category": self.cfg.exchange.category,
                         "symbol": self.cfg.exchange.symbol,
                         "interval": self.cfg.exchange.interval},
            "model": {"version": card.get("model_version", "?"),
                      "horizon_minutes": card.get("horizon_minutes", "?"),
                      "threshold": self.predictor.confidence_threshold,
                      "feature_set": card.get("feature_set", "?")},
            "conn_state": snap.conn_state.value if isinstance(snap.conn_state, ConnState) else str(snap.conn_state),
            "conn_detail": snap.conn_detail,
            "reconnect_count": snap.reconnect_count,
            "fresh": snap.fresh,
            "age_s": (age / 1000.0) if age is not None else None,
            "stale_threshold_s": self.processor.stale_after_ms / 1000.0,
            "server_ts_ms": snap.server_ts_ms,
            "price": snap.last_price.price if snap.last_price else None,
            "price_age_s": ((now_ms - snap.last_price.received_ms) / 1000.0) if snap.last_price else None,
            "live_candle": snap.live_candle,
            "bars": snap.confirmed_bars,
            "display_bars": self.cfg.signals.history_display_bars,
            "last_signal": self.engine.last_signal,
            "signal_history": self.engine.recent_signals(5),
            "positions": positions_view,
            "portfolio_stats": stats,
            "journal_path": journal_path,
            "last_advice": (self.last_advice.to_dict() if self.last_advice else None),
            "warnings": warnings,
            "rendered_at_ms": now_ms,
        }

    # -- LIVE ------------------------------------------------------------------
    def run_live(self, duration_s: float = 0.0) -> int:
        q: "queue.Queue" = queue.Queue(maxsize=10000)
        client = BybitWsClient(
            self.cfg.exchange.ws_public_url, self.cfg.exchange.symbol,
            self.cfg.exchange.interval, q,
            ping_interval_s=self.cfg.stream.ping_interval_s,
            connect_timeout_s=self.cfg.stream.connect_timeout_s,
            recv_timeout_s=self.cfg.stream.recv_timeout_s,
            reconnect_base_s=self.cfg.stream.reconnect_base_s,
            reconnect_max_s=self.cfg.stream.reconnect_max_s,
        )
        client.start()
        t_start = time.monotonic()
        try:
            while True:
                drained = 0
                while drained < 500:
                    try:
                        item = q.get(timeout=0.1 if drained == 0 else 0.0)
                    except queue.Empty:
                        break
                    drained += 1
                    kind = item[0]
                    if kind == "msg":
                        self.dispatch(self.processor.on_ws_message(item[1]))
                    elif kind == "state":
                        self.dispatch(self.processor.on_connection_state(item[1], item[2]))
                    elif kind == "error":
                        logger.warning("WS: %s", item[1])
                self.dispatch(self.processor.tick())
                self.ui.render(self.build_view())
                if duration_s and (time.monotonic() - t_start) >= duration_s:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nОстановка по Ctrl+C...")
        finally:
            client.stop()
            self.ui.render(self.build_view(extra_warnings=["остановлено"]), force=True)
        return 0

    # -- TEST (реплей) ------------------------------------------------------------
    def run_test_replay(self, csv_path: Path, n_bars: int = 60, speed_bps: float = 2.0,
                        demo_stale: bool = True) -> int:
        df_raw = load_klines_csv(csv_path)
        df_clean, _ = clean_klines(df_raw, self.cfg.exchange.interval)
        if len(df_clean) < n_bars:
            n_bars = len(df_clean)
        df = df_clean.tail(n_bars).reset_index(drop=True)

        interval_ms = step_ms_approx(self.cfg.exchange.interval)
        fake_now = {"ms": int(df["start_ms"].iloc[0])}
        # фейковые часы: весь тракт (processor/engine) работает от них - детерминированно
        self.processor._clock = lambda: fake_now["ms"]
        self.engine._clock = lambda: fake_now["ms"]
        self.mode = "TEST"

        def emit(msg: dict) -> list[Signal]:
            return self.dispatch(self.processor.on_ws_message(msg))

        self.dispatch(self.processor.on_connection_state(ConnState.CONNECTING, "TEST replay"))
        emit({"op": "subscribe", "success": True, "ret_msg": "subscribe", "ts": fake_now["ms"]})
        # первичный tick: установить FRESH до первого бара (иначе первый сигнал
        # ошибочно помечается неактуальным)
        self.dispatch(self.processor.tick(now_ms=fake_now["ms"]))
        all_signals: list[Signal] = []
        for i, row in df.iterrows():
            start = int(row["start_ms"])
            base = {"topic": f"kline.{self.cfg.exchange.interval}.{self.cfg.exchange.symbol}",
                    "ts": start}
            # формирующаяся свеча (сигналов не даёт)
            fake_now["ms"] = start + interval_ms // 2
            emit({**base, "data": [{"start": start, "end": start + interval_ms - 1,
                                    "interval": self.cfg.exchange.interval,
                                    "open": str(row["open"]), "close": str((row["open"] + row["close"]) / 2),
                                    "high": str(row["high"]), "low": str(row["low"]),
                                    "volume": str(row["volume"] / 2), "turnover": str(row["turnover"] / 2),
                                    "confirm": False, "timestamp": fake_now["ms"]}]})
            emit({"topic": f"tickers.{self.cfg.exchange.symbol}", "ts": fake_now["ms"],
                  "data": [{"symbol": self.cfg.exchange.symbol, "lastPrice": str(row["close"])}]})
            # закрытие свечи -> сигнал
            fake_now["ms"] = start + interval_ms
            sigs = emit({**base, "ts": fake_now["ms"],
                         "data": [{"start": start, "end": start + interval_ms - 1,
                                   "interval": self.cfg.exchange.interval,
                                   "open": str(row["open"]), "close": str(row["close"]),
                                   "high": str(row["high"]), "low": str(row["low"]),
                                   "volume": str(row["volume"]), "turnover": str(row["turnover"]),
                                   "confirm": True, "timestamp": fake_now["ms"]}]})
            all_signals.extend(sigs)
            self.dispatch(self.processor.tick(now_ms=fake_now["ms"]))
            self.ui.render(self.build_view())
            if speed_bps > 0:
                time.sleep(1.0 / speed_bps)

        if demo_stale:
            # демонстрация: данные устарели -> STALE, новых актуальных сигналов нет
            fake_now["ms"] += self.processor.stale_after_ms + 60_000
            self.dispatch(self.processor.tick(now_ms=fake_now["ms"]))
            view = self.build_view(extra_warnings=["ДЕМО: данные устарели (STALE) - новые сигналы не выдаются как актуальные"])
            self.ui.render(view, force=True)
            if not self.ui.ansi:
                self.ui.render_full_once(view)

        n_fresh = sum(1 for s in all_signals if s.data_fresh)
        n_hold = sum(1 for s in all_signals if s.action == "HOLD")
        print()
        print(f"TEST-реплей завершён: баров {len(df)}, сигналов {len(all_signals)} "
              f"(fresh {n_fresh}, HOLD/воздержание {n_hold})")
        return 0
