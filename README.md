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
| 2b | Переобучение модели, walk-forward-валидация, отчёт | ✅ готов, ждёт проверки пользователем |
| 3 | WebSocket-мониторинг, сигналы в реальном времени, консольный UI | ⏳ следующий |
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
