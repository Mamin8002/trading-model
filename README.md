# Finam NN-Trade Robot (Tabular ML Edition)

Продакшн-ориентированный асинхронный торговый бот для Мосбиржи через **официальный
Finam Trade API** с современной табличной ML-архитектурой (LightGBM + feature
engineering) вместо устаревших CNN-подходов «по картинкам графиков».

## Архитектура проекта

| Модуль | Назначение |
|---|---|
| `config.py` | Режимы TEST/LIVE, таймфреймы и адаптивные пресеты (lookback/horizon/timeout), параметры риска, ENV-переопределения (`FNBOT_*`) |
| `data_loader.py` | CSV-загрузчик (TEST) + async-клиент Finam Trade API (LIVE): auth, candles/get, ордера; генератор синтетической истории для демо |
| `features.py` | 43 каузальных признака: RSI, MACD, ATR, EMA-системы, лог-доходности, объёмные метрики (OBV/MFI-proxy), свечные паттерны, внутридневная сезонность; мультикласс-таргет up/flat/down с ATR-адаптивным порогом |
| `model.py` | LightGBM (fallback LogisticRegression), walk-forward CV без shuffle, периодическая ретренировка, save/load |
| `risk_manager.py` | SL/TP по ATR (1.5x/2.5x), fixed-fractional position sizing (1-2% депозита на сделку), мониторинг TP/SL/таймаут, реалистичные цены выхода |
| `portfolio.py` | Баланс / equity / открытые и закрытые позиции, Trade Journal (CSV), метрики: PnL, Win Rate, Profit Factor, Max Drawdown, R-multiples |
| `bot.py` | Асинхронный цикл: `TradingEngine` (общая бизнес-логика), `BacktestRunner` (TEST), `LiveRunner` (LIVE + ордера в Finam), `Notifier` (логи + Telegram) |
| `main.py` | Точка входа: CLI/ENV, логирование (терминал + ротация файлов), запуск режима |

## Быстрый старт

```bash
pip install -r requirements.txt

# РЕАЛЬНЫЕ данные Мосбиржи: скачать историю через публичный MOEX ISS API
# (без ключей и регистрации) и сразу прогнать TEST-бэктест:
python main.py --download --symbol SBER --timeframe 1h --days 90
python main.py --download --symbol GAZP --timeframe 5m --days 20   # intraday глубина ISS ~30 дней

# Если CSV по пути CONFIG.csv_path отсутствует — бот скачает реальные данные сам.

# Демо: сгенерировать синтетическую историю и прогнать холодный бэктест
python main.py --gen-data --timeframe 5m

# Своя история (формат: datetime,open,high,low,close,volume)
python main.py --csv data/sber_5m.csv --symbol SBER --timeframe 5m

# Только обучение + walk-forward отчёт
python main.py --train-only --timeframe 15m

# Боевой режим (ключи Finam через ENV; свечи LIVE идут через Finam Trade API)
export FNBOT_FINAM_CLIENT_ID=... FNBOT_FINAM_API_KEY=...
python main.py --mode LIVE --timeframe 1h
```

### Источники настоящих данных

| Источник | Режим | Что даёт | Ключи |
|---|---|---|---|
| **MOEX ISS** (`iss.moex.com/iss`) | TEST / download | Реальные intraday-свечи 1m/5m/10m/15m/1h (~последние 30 дней, пагинация offset/start) + дневная история на годы вглубь | не нужны |
| **Finam Trade API** (`api.open-api.finam.ru`) | LIVE | Свежие свечи, стакан, выставление ордеров | `FNBOT_FINAM_API_KEY` |

Ограничение публичного ISS: intraday-свечи отдаются только за ~30 последних дней
(≤500 баров на запрос). Для обеспечения минимальной истории обучения downloader
автоматически догружает **дневные** бары из `history/...securities` (доступны на
много лет назад), а торговый движок умеет bootstrap-обучение на уменьшенном окне.
Для более глубокой intraday-истории используйте архивы Finam (LIVE-клиент уже
умеет `candles/get` на произвольный период) или свой CSV.

Поддерживаемые таймфреймы: `1m, 5m, 10m, 15m, 1h` — глубина истории, горизонт
прогноза и таймаут удержания позиции адаптируются автоматически (см. `TF_PRESETS`).

## Ключевые настройки (ENV / config.py)

| Переменная | Default | Описание |
|---|---|---|
| `FNBOT_MODE` | `TEST` | `TEST` (бэктест по CSV) / `LIVE` (Finam API) |
| `FNBOT_TIMEFRAME` | `5m` | рабочий интервал |
| `FNBOT_RISK_PER_TRADE` | `1.0` | риск на сделку, % депозита |
| `FNBOT_ATR_SL_MULT` / `FNBOT_ATR_TP_MULT` | `1.5` / `2.5` | SL/TP через ATR |
| `FNBOT_SIGNAL_THRESHOLD` | `0.62` | минимальная уверенность модели для входа |
| `FNBOT_RETRAIN_EVERY_BARS` | `500` | период walk-forward ретренировки |
| `FNBOT_COMMISSION_PCT` / `FNBOT_SLIPPAGE_PCT` | `0.05` / `0.02` | издержки симулятора |
| `FNBOT_TELEGRAM_ENABLED` | **`false`** | телеграм-оповещения **выключены** (всё в терминал/лог); включение: `true` + `FNBOT_TG_BOT_TOKEN`, `FNBOT_TG_CHAT_ID` |

## Жизненный цикл сделки

1. На каждом новом **закрытом** баре — расчёт признаков и инференс модели.
2. `P(up)` или `P(down)` >= порога -> план сделки (ATR-стоп/тейк, размер = риск% / дистанцию стопа).
3. Вход рыночным ордером (LIVE) с учётом проскальзывания; фиксация в Trade Journal.
4. Сопровождение: приоритет SL -> TP -> таймаут (`timeout_bars` для ТФ) -> принудительный выход.
5. Каждые N баров — walk-forward ретренировка модели под текущий рынок.
6. Все события -> `logs/bot.log` + терминал; структурированные уведомления готовы к Telegram.

## Результаты демо-бэктеста (синтетика, 5m, ~8000 баров)

```
Trades=106 | WinRate=83% | ProfitFactor=11.4 | MaxDD=2.0%
Причины выхода: TP=82, SL=12, TIMEOUT=12
Walk-Forward CV: accuracy~0.61 f1~0.58
```

> ВНИМАНИЕ: синтетический датасет содержит искусственный momentum-сигнал — цифры
> демонстрируют работоспособность пайплайна, а не ожидаемую доходность. Перед LIVE
> обязательно прогоните модель на реальной истории и оцените OOS-метрики (`--train-only`).

## Дисклеймер

Торговля на бирже связана с риском потери капитала. Код поставляется как есть;
ответственность за боевые ордера несёт пользователь.
