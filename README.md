# BTC Bybit Monitor Bot

Бот мониторинга BTC и генерации торговых сигналов **BUY / SELL / HOLD** на публичных
рыночных данных Bybit (V5 API). Модель — переобученный подход из репозитория
[Sumit2514/Bitcoin-Price-Prediction](https://github.com/Sumit2514/Bitcoin-Price-Prediction)
(5-минутный ноутбук: признаки OHLCV, цель — направление следующего бара, XGBoost).

> **Дисклеймер.** Сигналы — вывод статистической модели, а не гарантия дохода и не
> персональная финансовая рекомендация. Бот **не** размещает ордера, **не** использует
> приватные API и **не** требует ключей. Позиции пользователя учитываются только
> после ручной регистрации.

## Статус по этапам

| Этап | Содержание | Статус |
|---|---|---|
| 1 | Изучение репозитория модели + документации Bybit, план | ✅ `docs/stage1_research_and_plan.md` |
| 2a | Загрузка истории Bybit, очистка, сохранение | ✅ готов, ждёт проверки пользователем |
| 2b | Переобучение модели, walk-forward-валидация, отчёт | ⏳ следующий |
| 3 | WebSocket-мониторинг, сигналы в реальном времени, консольный UI | ⏳ |
| 4 | Журнал сигналов, ручные позиции, сопровождение | ⏳ |

## Установка

Требуется Python 3.10+ (проверено на 3.11). Из корня проекта:

```powershell
cd btc_bybit_bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Конфигурация — `config/default.yaml`

Ключевые параметры:
- `mode`: `TEST` (офлайн на сохранённых данных) или `LIVE` (публичный WebSocket Bybit);
- `exchange.symbol` / `exchange.category` / `exchange.interval`: по умолчанию `BTCUSDT` / `spot` / `"5"`;
- `model.horizon_bars`: на сколько баров вперёд смотрит модель (1 = 5 минут; больше — дольше горизонт, требуется переобучение);
- `model.confidence_threshold`: ниже порога модель воздерживается (HOLD);
- `data.history_start`: начало загружаемой истории для обучения.

Секреты в конфигурации не используются и не должны появляться.

## Этап 2a: данные

Загрузка истории (публичный REST, ~2 минуты для 2024-01-01 → сегодня, ≈296 запросов):

```powershell
python scripts\download_history.py
```

Создаёт:
- `data\raw\bybit_spot_BTCUSDT_5m_<период>_fetched<UTC>.csv` + `.metadata.json`
  (инструмент, категория, интервал, период, дата загрузки);
- `data\processed\...clean.csv` — очищенные закрытые бары;
- `data\processed\...clean.report.json` — отчёт очистки (дубликаты, пропуски, незакрытые).

Пропуски баров **не заполняются** — только детектируются и логируются.

Ручная проверка этапа:

```powershell
python scripts\manual_checks\check_stage2_data.py
```

Ожидаемый вывод: 9 строк `[PASS]` (часть A — офлайн-логика на синтетике, часть B —
read-only запросы к публичному API Bybit) и итог `RESULT: PASS`, код возврата 0.
Без сети: `--no-network` (проверяется только часть A).

## Тесты

```powershell
python -m pytest tests -q
```

## Структура

```
btc_bot/                  # пакет: config, data_sources (интерфейс + Bybit), data (store/prep)
scripts/                  # download_history.py, manual_checks/
config/default.yaml       # конфигурация
data/                     # raw/, processed/, models/, journal/ (не коммитится)
docs/                     # план этапа 1, отчёты валидации
reference_model/          # копия ноутбуков исходного репозитория (для сверки подхода)
tests/                    # офлайн unit-тесты
```

Интерфейс `btc_bot/data_sources/base.MarketDataSource` позволяет добавить другую
биржу без изменения остального кода.
