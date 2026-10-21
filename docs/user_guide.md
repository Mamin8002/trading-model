# Руководство пользователя: TEST-запуск от установки до отчёта (крипторежим)

Для кого: пользователь, который хочет прогнать бота на своих криптоданных.
Простое руководство для новичка — [user-guide.md](user-guide.md).

> Статус: работает ТОЛЬКО TEST-режим на локальных CSV. LIVE заблокирован:
> подключение к криптобирже не реализовано ([live.md](live.md)).

---

## 1. Установка

Требования: Python 3.10+ (проверено на 3.11), Windows/Linux/macOS.

```powershell
# из корня репозитория
python -m pip install -e ".[dev,ml]"
python -m pytest        # ожидается: все тесты зелёные (офлайн)
```

Примечания:
- `tzdata` ставится автоматически на Windows (IANA-таймзоны через zoneinfo);
- LightGBM на Linux может потребовать системный `libgomp1` (на Windows обычно не нужно);
- предупреждения OpenBLAS в выводе безвредны.

## 2. Настройка CSV-данных (формат — [data.md](data.md))

Положите выгрузку свечей криптопары в `data\raw` (каталог вне git). Формат:

```
datetime,open,high,low,close,volume
2024-01-03 18:00:00,42150.50,42210.00,42100.10,42180.20,123.456
```

- метки — UTC (стандарт криптоэкспортов); если ваша выгрузка в другой таймзоне —
  укажите `data.source_timezone` явно;
- семантику метки подтвердите явно: `timestamp_semantics: bar_open|bar_close`
  (у большинства биржевых klines — bar_open; у некоторых экспортов — время
  закрытия; ошибка на один бар ломает причинность сигналов);
- объём дробный допустим (`volume_units: base_asset` — объём в BTC для BTCUSDT);
- проверка файла:

```powershell
python -m tradebot.cli.validate_data --csv data\raw\BTCUSDT_M1.csv --timeframe 1m --timestamp-semantics bar_open
# или весь конфиг целиком (+ агрегация 10m из 1m/5m):
python -m tradebot.cli.validate_data --config configs\my.yaml --resample 10m
```

## 3. Конфигурация

Скопируйте `configs\test_pipeline_example.yaml` в `configs\my.yaml` и заполните:

- `data.symbols[].ticker` (например `BTCUSDT`) и `files` (ваши CSV по таймфреймам;
  достаточно 1m или 5m — остальные ТФ строятся ресэмплингом через `base_timeframe`);
- `timeframes` — для каждого включённого ТФ обязательны `lookback_bars` и
  `horizon_bars` (явные значения, молча ничего не подбирается);
- `risk`/`simulation` — капитал, лимиты, комиссии и слиппедж (ЯВНЫЕ допущения;
  без них запуск отказывает с кодом 2); `instrument.price_step/lot_size_units` —
  из спецификации пары вашей биржи (или оставьте `null` — проверки шага безопасно
  отложатся с предупреждением в журнал);
- `sessions` — для крипто по умолчанию 24/7 (окно 00:00–24:00 UTC) или удалите
  секцию (равномерная сетка);
- `evaluation.trading_days_per_year: 365` — крипторынок без выходных.

Секреты в конфиге запрещены схемой; сеть не используется.

## 4. Обучение модели и walk-forward-отчёт

```powershell
python -m tradebot.cli.train --config configs\my.yaml --timeframe 5m
```

Результат: артефакт модели в `models\<experiment>_<tf>_<hash>\` (model.lgb.txt +
metadata.json + feature_manifest.json) и markdown-отчёт в `reports\`. Обучение
детерминировано (seed), метрики — out-of-sample по purged walk-forward фолдам;
вердикт по порогам — диагностический, не гарантия доходности.

## 5. Полный TEST-прогон единой командой

```powershell
# вариант A: обучение в этом же прогоне
python -m tradebot test --config configs\my.yaml --timeframe 5m --train

# вариант B: ранее обученный артефакт
python -m tradebot test --config configs\my.yaml --timeframe 5m --model-dir models\<артефакт>
```

Пайплайн: CSV → (обучение ИЛИ загрузка модели) → сигналы → риск-контроль →
симулятор → результаты. Артефакты: отчёт `reports\test_run_<experiment>_<tf>.md`
и журнал `reports\journal_<experiment>_<tf>.jsonl`.

Коды возврата: 0 — успех; 2 — ошибка конфигурации/гейта; 3 — недостаточно
истории; 4 — ошибка данных/модели/симуляции.

Отдельно только симуляция с готовой моделью:

```powershell
python -m tradebot.cli.simulate --config configs\my.yaml --timeframe 5m --model-dir models\<артефакт>
```

## 6. Что проверялось и чем является TEST-результат

Проверено офлайн (юнит-тесты + сценарии `scripts\manual_checks\`): слой данных
(UTC/24-7/ресэмплинг), причинность признаков, сплиты/embargo, детерминизм
обучения, риск-лимиты, исполнения симулятора, журнал, блокировка LIVE.

TEST-результат — диагностика на ИСТОРИЧЕСКИХ данных с ВАШИМИ допущениями
(комиссии/слиппедж/лимиты). Он НЕ является: прогнозом доходности, проверкой в
реальной биржевой среде, подтверждением ликвидности/исполнения. Симулятор не
моделирует частичные исполнения, стакан, маржу/плечо/funding (long-only,
decisions D15).

## 7. Секреты и чувствительные данные

- сейчас секреты НЕ нужны: сетевых вызовов в проекте нет;
- инфраструктура для будущих биржевых ключей: только переменные окружения или
  `.env` (файл вне git; шаблон `.env.example`); значения маскируются в логах
  и журнале (`config/env.py`);
- никогда не коммитьте ключи и не вставляйте их в код/конфиги/чат.

## 8. LIVE: почему он не включится

```powershell
python -m tradebot live-preflight --config configs\my.yaml --allow-live
# всегда: «LIVE-гейт НЕ пройден … интеграция с криптобиржей НЕ реализована», exit 2
```

- условие №0 гейта (`exchange.base.EXCHANGE_INTEGRATION_AVAILABLE`) сейчас
  `False` — гейт отказывает при ЛЮБОЙ комбинации флагов/конфигов/env;
- команда `test` отклоняет `run.mode: LIVE` (код 2);
- торговых эндпоинтов в кодовой базе нет — гигиенические тесты блокируют их появление;
- путь к реальному LIVE описан в [live.md](live.md) §5 и [roadmap.md](roadmap.md) E1–E3.

## 9. Ручные проверки (репозиторий)

```powershell
python scripts\manual_checks\check_crypto_mode.py         # крипторежим (основной сценарий)
python scripts\manual_checks\check_phase_02_data.py       # слой данных
python scripts\manual_checks\check_phase_03_ml.py         # ML-контур
python scripts\manual_checks\check_phase_04_simulation.py # риск/симулятор
python scripts\manual_checks\check_phase_05_test_run.py   # полный TEST-путь + LIVE-гейт
```

Каждый сценарий офлайн и печатает `RESULT: PASS (N/N checks)` либо `RESULT: FAIL`.

## 10. Если что-то не работает — что присылать

1. ПОЛНЫЙ вывод команды/сценария;
2. `python --version`, ОС;
3. `pip show tradebot pydantic PyYAML lightgbm`;
4. хвост `python -m pytest` (последние ~20 строк);
5. для ошибок данных — вывод `validate_data` и первые/последние 3 строки CSV
   (без личных данных).
