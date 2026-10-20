"""
config.py — Общие настройки торгового робота Finam NN-Trade.

Все параметры можно задавать:
  1) напрямую в этом файле;
  2) через переменные окружения (префикс FNBOT_);
  3) через .env файл в корне проекта (если установлен pydantic-settings).

Пример:  FNBOT_MODE=LIVE FNBOT_TIMEFRAME=5m python main.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Таймфреймы: маппинг -> минуты и параметры адаптации под интервал
# ---------------------------------------------------------------------------
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "1h": 60,
}

# Finam API internally maps timeframes to these codes
TIMEFRAME_FINAM: dict[str, str] = {
    "1m": "M1",
    "5m": "M5",
    "10m": "M10",
    "15m": "M15",
    "1h": "H1",
}


def _tf_minutes(tf: str) -> int:
    if tf not in TIMEFRAME_MINUTES:
        raise ValueError(f"Неподдерживаемый таймфрейм: {tf}. Доступны: {list(TIMEFRAME_MINUTES)}")
    return TIMEFRAME_MINUTES[tf]


@dataclass
class TimeframeParams:
    """Адаптивные параметры под конкретный таймфрейм."""
    lookback_bars: int          # глубина истории для признаков/инференса
    horizon_bars: int           # горизонт прогноза вперёд (бары)
    timeout_bars: int           # максимум удержания позиции (бары), затем принудительный выход
    min_history_bars: int       # минимум баров для обучения модели
    warmup_bars: int            # bars lost to indicator warm-up


# Готовые пресеты: чем крупнее ТФ — тем меньше баров нужно, но длиннее горизонт
TF_PRESETS: dict[str, TimeframeParams] = {
    "1m":  TimeframeParams(lookback_bars=180, horizon_bars=10, timeout_bars=30, min_history_bars=6000, warmup_bars=120),
    "5m":  TimeframeParams(lookback_bars=144, horizon_bars=6,  timeout_bars=24, min_history_bars=4000, warmup_bars=120),
    "10m": TimeframeParams(lookback_bars=120, horizon_bars=5,  timeout_bars=18, min_history_bars=3000, warmup_bars=120),
    "15m": TimeframeParams(lookback_bars=120, horizon_bars=4,  timeout_bars=16, min_history_bars=2500, warmup_bars=120),
    "1h":  TimeframeParams(lookback_bars=96,  horizon_bars=3,  timeout_bars=12, min_history_bars=1500, warmup_bars=120),
}


def get_tf_params(tf: str) -> TimeframeParams:
    return TF_PRESETS[tf]


# ---------------------------------------------------------------------------
# Главный конфиг
# ---------------------------------------------------------------------------
@dataclass
class BotConfig:
    # ------------------ Режим работы ------------------
    mode: str = os.getenv("FNBOT_MODE", "TEST")            # TEST | LIVE
    timeframe: str = os.getenv("FNBOT_TIMEFRAME", "5m")    # 1m | 5m | 10m | 15m | 1h

    # ------------------ Инструмент / данные ------------------
    symbol: str = os.getenv("FNBOT_SYMBOL", "SBER")        # тикер Мосбиржи
    csv_path: str = os.getenv("FNBOT_CSV_PATH", "data/sber_5m.csv")  # история для TEST-режима
    live_history_days: int = int(os.getenv("FNBOT_LIVE_HISTORY_DAYS", "60"))

    # ------------------ Капитал и риск ------------------
    initial_balance: float = float(os.getenv("FNBOT_INITIAL_BALANCE", "1000000"))  # RUB
    risk_per_trade_pct: float = float(os.getenv("FNBOT_RISK_PER_TRADE", "1.0"))    # 1-2% депозита на сделку
    max_position_pct: float = float(os.getenv("FNBOT_MAX_POSITION_PCT", "95.0"))   # потолок доли депозита в позиции
    atr_sl_mult: float = float(os.getenv("FNBOT_ATR_SL_MULT", "1.5"))              # стоп-лосс = N * ATR
    atr_tp_mult: float = float(os.getenv("FNBOT_ATR_TP_MULT", "2.5"))              # тейк-профит = M * ATR
    trailing_atr_mult: Optional[float] = None              # если задан — трейлинг-стоп по ATR

    # ------------------ Сигнал / модель ------------------
    signal_threshold: float = float(os.getenv("FNBOT_SIGNAL_THRESHOLD", "0.62"))  # порог уверенности (prob up/down)
    model_type: str = os.getenv("FNBOT_MODEL_TYPE", "lightgbm")                    # lightgbm | logreg
    retrain_every_bars: int = int(os.getenv("FNBOT_RETRAIN_EVERY_BARS", "500"))    # walk-forward период ретренировки
    wf_n_splits: int = int(os.getenv("FNBOT_WF_N_SPLITS", "4"))                    # число окон walk-forward валидации
    model_dir: str = os.getenv("FNBOT_MODEL_DIR", "models")                        # куда складывать сериализованные модели

    # ------------------ Издержки ------------------
    commission_pct: float = float(os.getenv("FNBOT_COMMISSION_PCT", "0.05"))       # % за сторону сделки (брокер+биржа)
    slippage_pct: float = float(os.getenv("FNBOT_SLIPPAGE_PCT", "0.02"))           # % проскальзывания при исполнении

    # ------------------ Цикл ------------------
    poll_seconds: int = int(os.getenv("FNBOT_POLL_SECONDS", "20"))                 # частота опроса данных в LIVE
    # ------------------ Логи / уведомления ------------------
    log_file: str = os.getenv("FNBOT_LOG_FILE", "logs/bot.log")
    trade_journal_csv: str = os.getenv("FNBOT_TRADE_JOURNAL", "logs/trades.csv")
    telegram_enabled: bool = os.getenv("FNBOT_TELEGRAM_ENABLED", "false").lower() == "true"  # ВЫКЛЮЧЕНО по умолчанию
    telegram_bot_token: str = os.getenv("FNBOT_TG_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("FNBOT_TG_CHAT_ID", "")

    # ------------------ Finam API (LIVE) ------------------
    finam_client_id: str = os.getenv("FNBOT_FINAM_CLIENT_ID", "")
    finam_api_key: str = os.getenv("FNBOT_FINAM_API_KEY", "")          # api key из ЛК Finam (Trade API key)
    finam_login: str = os.getenv("FNBOT_FINAM_LOGIN", "")
    finam_password: str = os.getenv("FNBOT_FINAM_PASSWORD", "")
    finam_base_url: str = os.getenv("FNBOT_FINAM_BASE_URL", "https://api.open-api.finam.ru")
    finam_market_name: str = os.getenv("FNBOT_FINAM_MARKET", "MOEX")   # рынок
    finam_exchange: str = os.getenv("FNBOT_FINAM_EXCHANGE", "TSBS")    # TSBS — основная площадка TIS

    def validate(self) -> None:
        _tf_minutes(self.timeframe)  # падает, если ТФ неизвестен
        assert self.mode in ("TEST", "LIVE"), "mode must be TEST or LIVE"
        assert 0 < self.risk_per_trade_pct <= 5, "risk_per_trade_pct должен быть в диапазоне (0, 5]"
        assert self.atr_tp_mult > 0 and self.atr_sl_mult > 0
        assert 0 < self.signal_threshold <= 1
        if self.mode == "LIVE":
            if not (self.finam_api_key or (self.finam_login and self.finam_password)):
                raise RuntimeError("Для LIVE режима нужны FNBOT_FINAM_API_KEY или логин/пароль Finam")

    @property
    def tf(self) -> TimeframeParams:
        return get_tf_params(self.timeframe)

    def to_dict(self) -> dict:
        d = asdict(self)
        # не светим секреты в логах
        for k in ("finam_api_key", "finam_password", "telegram_bot_token"):
            if d.get(k):
                d[k] = "***"
        return d


CONFIG = BotConfig()
CONFIG.validate()
