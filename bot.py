"""
bot.py — Главный асинхронный цикл торгового робота.

Два исполнения жизненного цикла:

  TEST (backtest/sandbox):
     Пошаговая имитация биржевого потока по CSV: на каждом новом закрытом баре
        1) пересчёт признаков, 2) инференс модели, 3) вход при уверенности,
        4) сопровождение позиции (SL/TP/таймаут), 5) периодическая walk-forward
           ретренировка модели на расширяющемся окне истории.
     Все издержки (комиссия + проскальзывание) учитываются.

  LIVE:
     Тот же алгоритм поверх Finam Trade API: периодический async-опрос свечей,
     отправка рыночных ордеров через FinamClient.place_order, mark-to-market.

Уведомления: лог-файл + терминал; Telegram опционально (telegram_enabled=false по умолчанию).
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import BotConfig
from data_loader import BaseDataLoader, CsvDataLoader, FinamDataLoader, FinamClient
from features import build_features, build_dataset, feature_columns
from model import TradeModel, ModelPrediction
from risk_manager import RiskManager, TradePlan
from portfolio import Portfolio

log = logging.getLogger("bot")


# ---------------------------------------------------------------------------
# Telegram notifier (structure-ready, disabled by default)
# ---------------------------------------------------------------------------
class Notifier:
    """
    Единая точка оповещений. При telegram_enabled=False всё пишется только
    в лог/терминал. Структура сообщений уже готова к отправке в TG.
    """

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.enabled = cfg.telegram_enabled and bool(cfg.telegram_bot_token and cfg.telegram_chat_id)
        if cfg.telegram_enabled and not self.enabled:
            log.warning("Telegram включён флагом, но токен/chat_id не заданы — уведомления отключены")

    async def send(self, text: str, level: int = logging.INFO) -> None:
        log.log(level, "[NOTIFY] %s", text)
        if not self.enabled:
            return  # режим TERMINAL-only, как запрошено
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
            payload = {"chat_id": self.cfg.telegram_chat_id, "text": text,
                       "parse_mode": "Markdown"}
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        log.warning("Telegram send failed: %s", await r.text())
        except Exception as e:  # уведомления не должны ронять бота
            log.warning("Telegram error: %s", e)

    # ---- convenience formatters ----
    def trade_opened(self, plan: TradePlan, conf: float) -> str:
        return (f"📈 <b>OPEN {plan.side}</b> {self.cfg.symbol} x{plan.qty} @ {plan.entry_price:.4f}\n"
                f"SL {plan.stop_loss:.4f} | TP {plan.take_profit:.4f} | R {plan.rr_ratio:.2f}\n"
                f"Confidence {conf:.2%} | Risk {plan.risk_amount:,.0f} RUB")

    def trade_closed(self, t) -> str:
        emoji = "✅" if t.pnl_rub >= 0 else "❌"
        return (f"{emoji} <b>CLOSE {t.side}</b> {t.symbol} @ {t.exit_price:.4f} "
                f"({t.exit_reason})\nPnL: {t.pnl_rub:,.2f} RUB ({t.pnl_pct:.3f}%)\n"
                f"R: {t.r_multiple} | Balance: {t.exit_time}")


# ---------------------------------------------------------------------------
# Ядро стратегии (общее для TEST и LIVE)
# ---------------------------------------------------------------------------
class TradingEngine:
    """Бизнес-логика: прогноз -> план -> вход -> сопровождение -> выход."""

    def __init__(self, cfg: BotConfig, portfolio: Portfolio, risk: RiskManager,
                 notifier: Notifier, lot_size: int = 1):
        self.cfg = cfg
        self.pf = portfolio
        self.risk = risk
        self.notifier = notifier
        self.lot_size = lot_size
        self.model: Optional[TradeModel] = None
        self.bars_seen = 0
        self.last_retrain_bar = -10**9
        self.last_signal_conf = 0.0

    # ------------------------------------------------------------- training
    def train_or_retrain(self, df_hist: pd.DataFrame, force: bool = False,
                         run_wf: bool = True) -> None:
        """Walk-forward обучение/ретренировка на историческом окне."""
        tfp = self.cfg.tf
        need = max(tfp.min_history_bars, tfp.warmup_bars + tfp.lookback_bars + 500)
        if len(df_hist) < need:
            # в LIVE разрешаем обучиться раньше (нет смысла ждать недели истории),
            # но не ниже разумного минимума для устойчивой статистики индикаторов
            live_min = tfp.warmup_bars + 600
            if not (force and self.cfg.mode == "LIVE" and len(df_hist) >= live_min):
                log.warning("Истории мало (%d < %d баров) — обучение отложено", len(df_hist), need)
                return
            log.warning("LIVE bootstrap: обучение на уменьшенном окне (%d баров)", len(df_hist))
        X, y, feats = build_dataset(df_hist, horizon_bars=tfp.horizon_bars)
        if len(X) < 500:
            log.warning("Валидных обучающих строк мало (%d) — пропуск обучения", len(X))
            return
        model = TradeModel(self.cfg.model_type)
        if run_wf and len(X) > 1500:
            acc, f1, windows = model.walk_forward_cv(X, y, n_splits=self.cfg.wf_n_splits)
            log.info("Walk-Forward CV: accuracy=%.3f f1=%.3f", acc, f1)
        else:
            model.fit(X, y)
        self.model = model
        self.last_retrain_bar = self.bars_seen
        tag = "первичное" if force else "ретренировка"
        log.info("Модель (%s): обучена на %d строках, %d фич", tag, len(X), len(feats))
        try:
            path = f"{self.cfg.model_dir}/{self.cfg.symbol}_{self.cfg.timeframe}.joblib"
            model.save(path)
        except Exception as e:
            log.warning("Не удалось сохранить модель: %s", e)

    def maybe_retrain(self, df_hist: pd.DataFrame) -> None:
        if self.model is None:
            self.train_or_retrain(df_hist, force=True)
        elif self.bars_seen - self.last_retrain_bar >= self.cfg.retrain_every_bars:
            log.info("Ретренировка по таймеру (%d баров с прошлого раза)",
                     self.bars_seen - self.last_retrain_bar)
            self.train_or_retrain(df_hist)

    # ------------------------------------------------------------- signal
    def predict_on_window(self, df_recent: pd.DataFrame) -> Optional[ModelPrediction]:
        if self.model is None or not self.model.trained:
            return None
        fx = build_features(df_recent)
        feats = feature_columns(fx)
        row = fx[feats].dropna()
        if row.empty:
            return None
        return self.model.predict(row.tail(1))

    # ------------------------------------------------------ open / manage
    async def try_enter(self, bar: pd.Series, atr_abs: float, pred: ModelPrediction) -> None:
        if self.pf.has_position:
            return
        side = None
        if pred.direction == 1 and pred.confidence >= self.cfg.signal_threshold:
            side = "LONG"
        elif pred.direction == -1 and pred.confidence >= self.cfg.signal_threshold:
            side = "SHORT"
        if side is None:
            return
        entry = float(bar["close"]) * (1 + (-1 if side == "LONG" else 1)
                                       * self.cfg.slippage_pct / 100.0)
        plan = self.risk.make_plan(self.pf.cash, entry, atr_abs, side, self.lot_size)
        if plan is None:
            return
        # время входа — по бару сигнала (в бэктесте это исторический timestamp;
        # в LIVE бар закрывается «сейчас», поэтому wall-clock совпадает)
        entry_ts = self._bar_time(bar)
        self.pf.open_position(side=side, qty=plan.qty, entry_price=entry,
                              stop=plan.stop_loss, target=plan.take_profit,
                              now=entry_ts, bar_index=self.bars_seen,
                              risk_amount=plan.risk_amount)
        self.last_signal_conf = pred.confidence
        await self.notifier.send(self.notifier.trade_opened(plan, pred.confidence))

    @staticmethod
    def _bar_time(bar: pd.Series) -> datetime:
        """Timestamp бара как aware-datetime (UTC), фолбэк — текущее время."""
        try:
            ts = pd.Timestamp(bar["datetime"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts.to_pydatetime()
        except Exception:
            return datetime.now(timezone.utc)

    async def manage_position(self, bar: pd.Series, bar_ts: datetime) -> None:
        """Сопровождение: SL -> TP -> TIMEOUT. Возвращает ClosedTrade или None."""
        p = self.pf.position
        if p is None:
            return None
        reason = RiskManager.check_exit(p.side, float(bar["high"]), float(bar["low"]),
                                        float(bar["close"]), p.stop_loss, p.take_profit)
        bars_held = self.bars_seen - p.entry_bar_index
        if reason is None and bars_held >= self.cfg.tf.timeout_bars:
            reason = "TIMEOUT"
        if reason is None:
            return None
        exit_price = RiskManager.exit_price_for(
            reason, p.stop_loss, p.take_profit, float(bar["close"]),
            self.cfg.slippage_pct, p.side)
        t = self.pf.close_position(exit_price, bar_ts, reason,
                                   confidence=self.last_signal_conf, mode=self.cfg.mode)
        await self.notifier.send(self.notifier.trade_closed(t))
        await self.notifier.send("💰 " + self.pf.summary_line(float(bar["close"])))
        return t


# ---------------------------------------------------------------------------
# TEST-режим: event-driven бэктест по CSV
# ---------------------------------------------------------------------------
class BacktestRunner:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.loader = CsvDataLoader(cfg)
        self.pf = Portfolio(cfg.initial_balance, cfg.symbol, cfg.trade_journal_csv,
                            commission_pct=cfg.commission_pct)
        self.risk = RiskManager(cfg)
        self.notifier = Notifier(cfg)
        self.engine = TradingEngine(cfg, self.pf, self.risk, self.notifier)

    async def run(self) -> dict:
        cfg = self.cfg
        tfp = cfg.tf
        full = await self.loader.fetch_history()          # вся история
        total = len(full)
        warmup = tfp.warmup_bars
        min_train = max(tfp.min_history_bars, warmup + 800)
        start_i = min(min_train, total - 50)
        log.info("BACKTEST: всего баров=%d, торговля со старта %d (после обучения на %d)",
                 total, start_i, start_i)

        # первичное walk-forward обучение
        self.loader.set_cursor(start_i)
        train_df = await self.loader.fetch_history()      # [0:start_i]
        self.engine.train_or_retrain(train_df, force=True)

        closed_trades = []
        for i in range(start_i, total):
            self.loader.set_cursor(i + 1)                  # видимые данные: [..:i+1]
            bar = full.iloc[i]
            self.engine.bars_seen += 1

            # сопровождение открытой позиции по текущему бару
            bar_ts = pd.Timestamp(bar["datetime"]).to_pydatetime().replace(tzinfo=timezone.utc)
            t = await self.engine.manage_position(bar, bar_ts)
            if t is not None:
                closed_trades.append(t)

            # ретренировка по расписанию
            self.engine.maybe_retrain(await self.loader.fetch_history())

            # новый сигнал только если нет позиции
            if not self.pf.has_position and self.engine.model is not None:
                window = await self.loader.fetch_history(n_bars=tfp.lookback_bars)
                pred = self.engine.predict_on_window(window)
                if pred is not None:
                    fx = build_features(window)
                    atr_abs = float(fx["_atr_abs"].iloc[-1])
                    await self.engine.try_enter(bar, atr_abs, pred)

        # закрыть висящую позицию в конце теста
        if self.pf.has_position:
            last = full.iloc[-1]
            ts = pd.Timestamp(last["datetime"]).to_pydatetime().replace(tzinfo=timezone.utc)
            px = RiskManager.exit_price_for("EOD", self.pf.position.stop_loss,
                                            self.pf.position.take_profit,
                                            float(last["close"]), cfg.slippage_pct,
                                            self.pf.position.side)
            closed_trades.append(self.pf.close_position(px, ts, "EOD", mode="TEST"))

        m = self.pf.metrics()
        log.info("=" * 70)
        log.info("BACKTEST COMPLETE. %s", self.pf.summary_line())
        log.info("Метрики: %s", m)
        reasons = {}
        for t in closed_trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        log.info("Причины выхода: %s", reasons)
        log.info("=" * 70)
        return m


# ---------------------------------------------------------------------------
# LIVE-режим: боевой цикл поверх Finam Trade API
# ---------------------------------------------------------------------------
class LiveRunner:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.client = FinamClient(cfg)
        self.loader = FinamDataLoader(cfg, client=self.client)
        self.pf = Portfolio(cfg.initial_balance, cfg.symbol, cfg.trade_journal_csv,
                            commission_pct=cfg.commission_pct)
        self.risk = RiskManager(cfg)
        self.notifier = Notifier(cfg)
        self.engine = TradingEngine(cfg, self.pf, self.risk, self.notifier)
        self._last_bar_dt: Optional[pd.Timestamp] = None

    async def _execute_side(self, side_action: str, qty: int) -> None:
        """Отправка ордера в Finam с retry."""
        for attempt in range(3):
            try:
                res = await self.client.place_order(
                    exchange=self.cfg.finam_market_name, symbol=self.cfg.symbol,
                    side=side_action, qty=qty, order_type="MARKET")
                log.info("Finam order accepted: %s", str(res)[:200])
                return
            except Exception as e:
                log.error("Order send failed (attempt %d): %s", attempt + 1, e)
                await asyncio.sleep(1 + attempt)
        await self.notifier.send(f"🚨 Ошибка выставления ордера {side_action} {qty} "
                                 f"{self.cfg.symbol} после 3 попыток", logging.CRITICAL)

    async def bootstrap(self) -> pd.DataFrame:
        await self.client.authenticate()
        # просим у API столько баров, сколько нужно для первичного обучения модели
        train_bars = max(self.cfg.tf.min_history_bars,
                         self.cfg.tf.warmup_bars + 1500)
        hist = await self.loader.fetch_history(n_bars=train_bars)
        log.info("LIVE bootstrap: %d баров, последняя свеча %s",
                 len(hist), hist['datetime'].iloc[-1])
        return hist

    async def run(self) -> None:
        cfg = self.cfg
        tfp = cfg.tf
        hist = await self.bootstrap()

        # первичное обучение на собранной истории (для live лучше заранее загрузить больше баров)
        self.engine.train_or_retrain(hist, force=True)

        await self.notifier.send(f"🤖 Bot LIVE started: {cfg.symbol} {cfg.timeframe}, "
                                 f"balance {cfg.initial_balance:,.0f} RUB")

        while True:
            try:
                recent = await self.loader.fetch_history(n_bars=tfp.lookback_bars + 2)
                # работаем только по ЗАКРЫТЫМ барам: последний incomplete bar игнорируем
                bars = recent.iloc[:-1] if len(recent) > 1 else recent
                last_dt = pd.Timestamp(bars["datetime"].iloc[-1])

                if self._last_bar_dt is None or last_dt > self._last_bar_dt:
                    self._last_bar_dt = last_dt
                    self.engine.bars_seen += 1
                    bar = bars.iloc[-1]
                    bar_ts = last_dt.to_pydatetime().replace(tzinfo=timezone.utc)

                    # --- сопровождение позиции реальными данными ---
                    p = self.pf.position
                    if p is not None:
                        prev_len = len(self.pf.closed)
                        t = await self.engine.manage_position(bar, bar_ts)
                        if t is not None and len(self.pf.closed) > prev_len:
                            close_side = "SELL" if t.side == "LONG" else "BUY"
                            await self._execute_side(close_side, t.qty)

                    # --- ретренировка ---
                    self.engine.maybe_retrain(await self.loader.fetch_history())

                    # --- новый сигнал ---
                    if not self.pf.has_position and self.engine.model is not None:
                        pred = self.engine.predict_on_window(bars)
                        if pred is not None:
                            fx = build_features(bars)
                            atr_abs = float(fx["_atr_abs"].iloc[-1])
                            had_pos_before = self.pf.has_position
                            await self.engine.try_enter(bar, atr_abs, pred)
                            new_p = self.pf.position
                            if new_p is not None and not had_pos_before:
                                order_side = "BUY" if new_p.side == "LONG" else "SELL"
                                await self._execute_side(order_side, new_p.qty)

                    log.debug("bar %s close=%.4f | %s", last_dt, float(bar["close"]),
                              self.pf.summary_line(float(bar["close"])))
                await asyncio.sleep(cfg.poll_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("LIVE loop error: %s", e)
                await self.notifier.send(f"⚠️ Ошибка цикла: {e}", logging.ERROR)
                await asyncio.sleep(max(5, cfg.poll_seconds))

    async def shutdown(self):
        await self.client.close()


def build_runner(cfg: BotConfig):
    if cfg.mode == "TEST":
        return BacktestRunner(cfg)
    return LiveRunner(cfg)
