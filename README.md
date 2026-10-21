# tradebot — торговый бот для криптовалютного рынка (TEST-каркас на локальных CSV)

Офлайн-каркас исследовательского торгового бота для крипторынка: локальные
OHLCV-CSV (UTC, рынок 24/7), ресэмплинг таймфреймов, 31 причинный признак,
LightGBM с честной walk-forward-валидацией, риск-контроль, событийный
TEST-симулятор исполнения и JSONL-журнал решений.

> **Статус (честно):**
> - **TEST-режим реализован и проверен офлайн** (тесты + ручные сценарии приёмки);
> - **LIVE-торговля НЕДОСТУПНА**: подключение к криптобирже не реализовано —
>   нет API-клиента, сетевых запросов, аутентификации и отправки ордеров;
>   многофакторный LIVE-гейт блокирует ордерный контур всегда (`docs/live.md`);
> - проект **не production-ready** и **не обещает доходность**: все торговые
>   цифры — диагностические (`docs/decisions.md` D16);
> - тип рынка (спот/маржа/фьючерсы) НЕ предполагается — симулятор long-only без
>   плеча; решения отложены до подключения биржи (`docs/decisions.md` D15).

Прежняя интеграция с брокерским API и биржевыми акциями полностью удалена из
кода, конфигов, зависимостей и документации (решения D8, D12).

---

## Что реализовано

| Блок | Состав | Статус |
|---|---|---|
| Конфигурация | строгая pydantic-схема (extra=forbid), YAML, хэш конфига, env-инфраструктура секретов | проверено тестами |
| Данные | CSV-загрузчик+валидатор (`datetime,open,high,low,close,volume`; метки UTC; дробный объём), ресэмплинг 1m/5m→5m/10m/15m/1h, календарь сессий 24/7, синтетические демо-данные | проверено тестами и ручными сценариями |
| Признаки/ML | 31 причинный признак v1, метки forward-return с нейтральной полосой, purged walk-forward + embargo, leakage-аудит, LightGBM + базейны, реестр моделей и отчёты | проверено офлайн |
| Риск/симуляция | RiskGuards (лимиты позиции/капитала/дня/просадки/свежести), sizing, EventSimulator (open следующего бара, слиппедж/комиссия, стоп/цель, консервативный конфликт), JSONL-журнал | проверено офлайн |
| TEST-пайплайн | `python -m tradebot test` одной командой: данные → обучение → симуляция → отчёт/журнал | проверено офлайн |
| LIVE-граница | многофакторный гейт + kill-switch; контракт биржевого адаптера (`exchange/`) БЕЗ реализации; LIVE всегда заблокирован | проверено тестами |

НЕ реализовано: биржевой адаптер (данные/ордера), LIVE-оркестратор (потоки,
BarFinalizer, reconciliation), CatBoost, triple-barrier-метки, мульти-символ,
маржа/фьючерсы/шорты. План — `docs/roadmap.md` (этапы E1–E3).

---

## Установка и проверка (PowerShell)

```powershell
# установка (Python >= 3.10; для Windows-таймзон ставится tzdata автоматически)
python -m pip install -e ".[dev,ml]"

# тесты (офлайн)
python -m pytest

# ручные офлайн-сценарии приёмки (ожидаемый итог каждого — «RESULT: PASS (N/N checks)»):
python scripts\manual_checks\check_crypto_mode.py        # крипторежим: CSV/UTC/24-7/TEST-путь/LIVE-блокировка
python scripts\manual_checks\check_phase_02_data.py      # слой данных: конфиг, CSV, агрегация
python scripts\manual_checks\check_phase_03_ml.py        # ML-контур (обучение на синтетике)
python scripts\manual_checks\check_phase_04_simulation.py # риск-контроль и симулятор
python scripts\manual_checks\check_phase_05_test_run.py  # полный TEST-путь + защита LIVE

# валидация демо-данных и агрегации
python -m tradebot.cli.validate_data --config configs\demo.yaml --resample 10m

# полный TEST-прогон на демо-данных (обучение в этом же запуске)
python -m tradebot test --config configs\sim_demo.yaml --timeframe 5m --train
```

Свои данные: положите CSV криптопар в `data\raw`, скопируйте
`configs\test_pipeline_example.yaml`, заполните файлы/параметры и запустите:

```powershell
python -m tradebot test --config configs\my.yaml --timeframe 5m --train
```

Коды возврата `tradebot test`: 0 — успех; 2 — ошибка конфигурации/гейта;
3 — недостаточно истории; 4 — ошибка данных/модели/симуляции.

---

## Данные и время (криптосоглашения)

- формат CSV: `datetime,open,high,low,close,volume`;
- метки — **UTC** (по умолчанию; `data.source_timezone`), семантика метки —
  **явно** (`timestamp_semantics: bar_open|bar_close`);
- рынок **24/7**: календарь `00:00–24:00 UTC` во все дни (или без секции
  `sessions` — равномерная сетка); годовой пересчёт — 365 дней;
- объём — дробный в единицах базового актива (`volume_units: base_asset`);
- подробности: `docs/data.md`.

## Безопасность

- LIVE выключен по умолчанию и заблокирован: даже `mode: LIVE` + `--allow-live` +
  env-подтверждение + `live.allow_orders: true` НЕ открывают ордера — условие №0
  гейта (интеграция с биржей) невыполнимо до её реализации;
- сетевых вызовов в проекте нет вообще; гигиенические тесты блокируют появление
  торговых эндпоинтов в коде;
- секреты не требуются; инфраструктура для будущих биржевых ключей — только
  окружение/`.env` (вне git) с маскированием в логах;
- kill-switch-файл (`.kill`) запрещает новые заявки (для будущего LIVE).

## Структура

```
src/tradebot/
├── config/      # схема, загрузка, env-секреты
├── data/        # Bar, CSV loader/validator, resampler, sessions (24/7), synthetic
├── features/    # признаки v1 + builder
├── labels/      # forward-return метки
├── ml/          # dataset, splits, train, metrics, registry, report
├── strategy/    # модель → сигналы
├── execution/   # интерфейсы (включая BrokerGateway), risk, sizing, simulator, journal
├── backtest/    # прогон симуляции, результаты, отчёты
├── marketdata/  # бирженезависимые порты (History/Quote/OrderBook/Stream) + CsvHistorySource
├── exchange/    # точка подключения биржи: контракт + статус «НЕ реализовано»
├── live/        # многофакторный LIVE-гейт (сейчас всегда отказывает)
└── cli/         # python -m tradebot {test, live-preflight}; train/simulate/validate_data
configs/         # demo.yaml, example.yaml, ml_example.yaml, sim_demo.yaml, test_pipeline_example.yaml
docs/            # architecture, data, ml, simulation, live, decisions, roadmap, руководства
scripts/manual_checks/  # офлайн-сценарии приёмки (PASS/FAIL)
data/demo/       # СИНТЕТИЧЕСКИЕ демо-данные (не рыночные)
```

## Документация

- [docs/user-guide.md](docs/user-guide.md) — простое руководство для новичка;
- [docs/user_guide.md](docs/user_guide.md) — техническое руководство TEST-запуска;
- [docs/data.md](docs/data.md) — данные/время/таймфреймы;
- [docs/ml.md](docs/ml.md) — признаки, метки, обучение, валидация;
- [docs/simulation.md](docs/simulation.md) — риск-контроль и TEST-симулятор;
- [docs/live.md](docs/live.md) — статус LIVE и путь к подключению биржи;
- [docs/architecture.md](docs/architecture.md), [docs/decisions.md](docs/decisions.md), [docs/roadmap.md](docs/roadmap.md).
