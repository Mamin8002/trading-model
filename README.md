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
| 2a | Загрузка истории Bybit, очистка, сохранение | ✅ проверен пользователем (PASS 9/9, 2026-10-21) |
| 2b | Переобучение модели, walk-forward-валидация, отчёт | ✅ проверен пользователем (PASS 6/6, 2026-10-21) |
| 3 | WebSocket-мониторинг, сигналы в реальном времени, консольный UI | ✅ готов, ждёт проверки пользователем |
| 4 | Журнал сигналов, ручные позиции, сопровождение | ⏳ следующий |

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

## Этап 2b: модель

Подход исходного 5-минутного ноутбука (признаки OHLCV, цель — знак изменения close
через `horizon_bars` баров, XGBoost) перенесён на свежие данные Bybit:

- наборы признаков: `raw` (точно как в оригинале) и `ratio` (цены относительно close —
  минимально необходимая правка: деревья не экстраполируют сырые уровни при дрейфе цены);
- walk-forward-валидация: expanding window, 8 окон, переобучение на каждом окне,
  **embargo** по времени разрешения метки (утечка будущих данных исключена);
- сравнение с baseline (majority, momentum) и с протоколом оригинала (single split 75/25);
- артефакт: `data\models\<model_version>\` (model.joblib + model_card.json + walk_forward.json);
- HOLD/воздержание: уверенность = max(predict_proba); ниже `model.confidence_threshold` → HOLD.

Обучение (офлайн, ~35 с, сеть не используется):

```powershell
python scripts\train_model.py            # полная валидация + финальная модель
python scripts\train_model.py --quick    # быстрый смоук-прогон
```

Создаёт: `data\models\<model_version>\*`, `docs\validation_report.md`.

Ручная проверка этапа (офлайн, ~10–60 с; нужны данные из `data\processed\` или образец
из проверки 2a):

```powershell
python scripts\manual_checks\check_stage2_model.py
```

Ожидаемый вывод: `[PASS] A1…A4`, `[PASS] B1…B2`, итог `RESULT: PASS (проверок пройдено: 6)`,
код возврата 0. Создаёт тестовый артефакт в `data\models\_manual_check_stage2b\`.

Результаты валидации на данных 2024-01-01 → 2026-10-21 (295 120 баров, 8 окон по ~14.8k):
выбран набор `ratio`; агрегированная accuracy **0.5059** против majority **0.4981** и
momentum **0.4894**; acc при уверенности ≥ 0.55 — 0.5142 (покрытие 21.3%). Отличие от
baseline статистически заметно (~5σ), но мало: ожидаемый валовой выигрыш ≈ 0.001% на
сигнал против ~0.2% комиссий round-trip — **для реальной торговли недостаточно**.
Подробности: `docs/validation_report.md`.

Смена `model.horizon_bars` (например, 12 баров = 1 час) требует переобучения:
`python scripts\train_model.py` — артефакт зафиксирует новый горизонт, а predictor
отклонит несовпадение конфига и модели.

## Этап 3: мониторинг в реальном времени

Режимы запуска:

```powershell
# TEST: офлайн-реплей последних баров из data\processed\ (без сети) + демо-сценарий STALE
python scripts\run_bot.py --mode TEST --replay-bars 40 --speed 4

# LIVE: публичный WebSocket Bybit (wss://stream.bybit.com/v5/public/spot), до Ctrl+C
python scripts\run_bot.py --mode LIVE
python scripts\run_bot.py --mode LIVE --duration 660   # или выйти через 11 минут
```

Что показывает консольный дашборд: текущую цену (ticker), последние закрытые свечи и
формирующуюся свечу, состояние соединения (CONNECTED/RECONNECTING/...), свежесть данных
(FRESH/STALE), текущий сигнал (действие, уверенность, время выдачи, время бара, цена,
версия модели, горизонт) и историю сигналов.

Ключевые правила:
- сигнал формируется **только после закрытия свечи** (`confirm=true` в WS-сообщении);
- app-level ping каждые 20 с; при обрыве — reconnect с backoff 1→60 с и REST-догрузка
  пропущенных закрытых баров (публичные эндпоинты, read-only);
- если данные устарели (нет обновлений дольше `signals.stale_after_intervals` интервалов),
  история неполна или соединение потеряно — это явно показывается, а новый сигнал
  помечается **не актуальным** (data_fresh=false с перечислением причин);
- при уверенности ниже `model.confidence_threshold` выдаётся HOLD (воздержание) с причиной.

Ручная проверка этапа (по умолчанию офлайн — детерминированный реплей с фейковыми часами;
A10 поднимает локальный fake-WS-сервер на 127.0.0.1 — внешняя сеть не нужна):

```powershell
python scripts\manual_checks\check_stage3_stream.py
python scripts\manual_checks\check_stage3_stream.py --live              # + read-only WS Bybit (~45 с)
python scripts\manual_checks\check_stage3_stream.py --live --wait-confirm  # + ждать закрытия реальной свечи (до ~7 мин)
```

Ожидаемый вывод: `[PASS] A1…A10`, итог `RESULT: PASS (проверок пройдено: 10)` (с `--live`
добавляется B1 → 11), код возврата 0. Файлы не создаются. A9 — структурный скан кода на
отсутствие торговых/приватных эндпоинтов и ключей.

### Известные нюансы (Этап 3)

- **UserWarning от XGBoost при загрузке модели** (pickle/serialize): возникает, если
  артефакт сохранён другой версией XGBoost (например, взят из готового workspace).
  Предупреждение безвредно — модель работает; чтобы его убрать, переобучите локально:
  `python scripts\train_model.py`.
- **Seed истории при старте LIVE**: бот разово догружает ~300 последних закрытых баров
  через публичный REST для отображения; **сигналы по ним не выдаются** (сигнал — только
  по барам, закрывшимся при вас, либо по backfill после реального обрыва).
- **Windows Terminal**: очистка экрана при перерисовке отправляет старые кадры в
  scrollback — при копировании текста вы можете увидеть несколько кадров подряд; это
  не дублирование вывода. При проблемах с ANSI используйте `--no-ansi` (или `NO_ANSI=1`).

## Тесты

```powershell
python -m pytest tests -q
```

## Структура

```
btc_bot/                  # пакет
  config.py               # конфигурация (dataclasses + YAML)
  data_sources/           # интерфейс MarketDataSource + реализация Bybit (REST)
  streaming/              # события потока, StreamProcessor (confirm/gaps/freshness), WS-клиент Bybit
  data/                   # store (raw CSV + metadata), prep (очистка, датасет)
  model/                  # features, train (артефакт+card), walk_forward, predictor
  signals/engine.py       # BUY/SELL/HOLD + уверенность + gating по свежести
  ui/console.py           # консольный дашборд
  bot.py                  # оркестратор TEST (реплей) / LIVE (WS)
scripts/                  # download_history.py, train_model.py, run_bot.py, manual_checks/
config/default.yaml       # конфигурация
data/                     # raw/, processed/, models/, journal/, manual_checks/ (не коммитится)
docs/                     # план этапа 1, отчёт валидации
reference_model/          # копия ноутбуков исходного репозитория (для сверки подхода)
tests/                    # офлайн unit-тесты
```

Интерфейс `btc_bot/data_sources/base.MarketDataSource` (и событийная модель
`btc_bot/streaming/base`) позволяют добавить другую биржу без изменения остального кода.
