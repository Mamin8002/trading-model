# Architecture: торговый бот на табличных признаках (MOEX CSV → Finam Trade API)

Статус: проектное решение до реализации. Дата: 2026-10-21. Код на этом этапе не пишется.
Основание: факты и открытые вопросы — в [research.md](research.md); аргументация выборов — в [decisions.md](decisions.md); этапы — в [roadmap.md](roadmap.md).

---

## 1. Границы системы и режимы

```
                ┌────────────────────────────── РЕЖИМ TEST ──────────────────────────────┐
 CSV (MOEX)  →  │ DataLayer → Features → Labels → ML(train/validate) → Backtest/Simulator │
                └─────────────────────────────────────────────────────────────────────────┘
                                     │  (общие модули: Features, Labels, Strategy, Model artifacts)
                ┌────────────────────────────── РЕЖИМ LIVE ──────────────────────────────┐
 Finam API   →  │ FinamMarketData (warmup+WS) → BarFinalizer → Features → Strategy        │
 (REST/WS/gRPC) │        → RiskGuards → BrokerGateway (dry-run | демо | real*)            │
                └─────────────────────────────────────────────────────────────────────────┘
                                            * real — вне области текущего этапа, только через явное решение
```

Принцип TEST/LIVE-инвариантности: **один и тот же код** признаков, меток, модели и стратегии в обоих режимах. Различаются только два подключаемых слоя:

| Слой | TEST | LIVE |
|---|---|---|
| `HistorySource` / `StreamSource` | CSV-replay (детерминированный) | Finam REST `Bars` (warmup) + WS/gRPC подписки |
| `BrokerGateway` | `SimulatedBroker` (филлы по правилам бэктеста) | `FinamBroker` через `finam-sdk`, по умолчанию `DryRunBroker`-обёртка |

Режимы задаются конфигурацией, а не ветвлением внутри модулей; переключение = замена фабрики шлюзов.

---

## 2. Модули и их границы

Каталог (предложение; имена финализируются на этапе 0):

```
src/tradebot/
├── config/        # схема конфига (pydantic), загрузка, валидация, хэш конфигурации
├── data/          # контракт данных: Bar-типы, CSV loader+validator, resampler, session calendar, хранилище (parquet)
├── features/      # библиотеки индикаторов и FeatureBuilder (окно lookback → вектор признаков)
├── labels/        # построение целевой переменной (горизонт H)
├── ml/            # DatasetAssembler, PurgedWalkForwardSplit, обучение (LightGBM/CatBoost), модельный реестр, отчёты оценки
├── strategy/      # SignalPolicy (модель → торговое намерение), PositionManager (состояние позиции)
├── backtest/      # EventSimulator, модель издержек/филлов, метрики, отчёты
├── execution/     # интерфейсы BrokerGateway, OrderRequest/OrderState; DryRunBroker; SimulatedBroker; RiskGuards; KillSwitch
├── marketdata/    # интерфейсы HistorySource/StreamSource/InstrumentsSource; CsvReplaySource; FinamHistorySource; FinamStreamSource; BarFinalizer
├── finam/         # адаптер finam-sdk: AuthSession, клиенты, маппинг ошибок, rate limiter, reconciliation — ЕДИНСТВЕННОЕ место, знающее про SDK
├── live/          # LiveOrchestrator: warmup → подписки → цикл сигналов → execution; журналирование состояния
├── observability/ # структурированные логи (без секретов), метрики, алерты
└── cli/           # точки входа: validate-data, build-dataset, train, backtest, live --dry-run
tests/             # unit, property-based (причинность), контрактные (фейк-API), golden-фикстуры
configs/           # YAML-конфиги (base + per-timeframe + per-mode)
docs/              # этот набор документов
data/              # raw CSV, артефакты (вне git)
```

Правила границ:
1. `features/`, `labels/`, `strategy/`, `ml/` **не знают** ни о CSV, ни о Finam, ни о режиме — работают только с внутренними типами `data/`.
2. Всё знание о Finam Trade API (эндпоинты, enum-ы, decimal-конверсии, retry, reconciliation) изолировано в `finam/` + реализациях интерфейсов `marketdata/` и `execution/`. Открытые вопросы Q1–Q9 (research.md) закрыты этими интерфейсами.
3. Стратегический расчёт отделён от транспорта — как в официальном примере SMA 9/30 (S8i) и Рецепте 6 (S3j): тестируется без токена и сети.
4. Зависимости направлены внутрь: cli/live/backtest → strategy → features/labels → data. Обратных зависимостей нет.

---

## 3. Контракт данных

### 3.1 Входной CSV (внешний, от пользователя/MOEX)

```
datetime,open,high,low,close,volume
```

- Формат полностью совпадает с CSV референсного проекта WISEPLAT (S9h) — это подтверждённый прецедент формата.
- Обязательные метаданные (в конфиге, не в файле): `timestamp_semantics` (`bar_open` | `bar_close`), `timezone` (ожидание: naive Europe/Moscow для MOEX-выгрузок), `volume_units` (штуки), символ инструмента и таймфрейм файла.
- **[РАСХОЖДЕНИЕ/Q3]**: у WISEPLAT скрипт генерации сдвигает метку на конец бара, а опубликованные CSV выглядят как метки начала сессии — семантика неоднозначна. Валидатор (этап D1) детектирует семантику эмпирически (границы сессий/аукционов, монотонность, отсутствие перекрытий) и требует подтверждения в конфиге. Внутренне всё нормализуется к `bar_open`, tz-aware UTC.
- Валидация при загрузке: схема и типы колонок; монотонность и уникальность datetime; OHLC-соотношения (low ≤ open/close ≤ high); volume ≥ 0; пропуски против торгового календаря; дубликаты; аномалии (нулевые объёмы, разрывы цены) — отчёт, не тихая починка.

### 3.2 Внутреннее представление

- `Bar`: `ts_open: datetime (UTC, tz-aware)`, `open/high/low/close: Decimal-совместимые float64`, `volume: int64`, `source: enum{csv,finam_rest,finam_ws,derived}` (derived — локальный ресэмплинг, добавлено при реализации фазы 0-2, см. docs/data.md §3), `is_final: bool`, `symbol: "TICKER@MIC"`.
- Цены/объёмы в признаках — float64 (для ML); **в контуре заявок — только Decimal/строки** (требование API: бинарный float запрещён для цены заявки, S3c/S4). Конвертация — на границе `execution/`.
- Свечи Finam API (REST decimal-объекты `{"value": "..."}`, WS-строки, gRPC `google.type.Decimal`) нормализуются к одному типу в `finam/` — модели ответов между протоколами не переносимы (S3f).
- Хранилище обработанных данных: parquet-партиции по symbol/timeframe; каждый артефакт сопровождается манифестом (хэш конфига, границы окна, версия кода, семантика метки).

### 3.3 Таймфреймы и ресэмплинг

Поддержка: **1m, 5m, 10m, 15m, 1h** (ТЗ).

| ТФ | Источник TEST | Источник LIVE |
|---|---|---|
| 1m | CSV 1m | API TIME_FRAME_M1 (глубина 7 дней) / WS BARS M1 |
| 5m | CSV 5m | TIME_FRAME_M5 (30 дней) / WS BARS M5 |
| **10m** | CSV 10m **или ресэмплинг 5m→10m** | **только ресэмплинг** (M10 в API нет — R1/Q, подтверждено enum-ом TimeFrame) |
| 15m | CSV 15m | TIME_FRAME_M15 / WS BARS M15 |
| 1h | CSV 1h | TIME_FRAME_H1 / WS BARS H1 |

- Единственный модуль `Resampler`: агрегация базового ТФ в целевой по календарной сетке (open=первый, high=max, low=min, close=последний, volume=sum); выпускает **только полные группы**; неполный хвост остаётся pending (согласуется с паттерном незакрытой свечи S3j Рецепт 6).
- Один и тот же Resampler работает в TEST (из CSV) и LIVE (из потока M5) — расхождение TEST/LIVE по построению 10m исключается конструкцией; проверяется тестом эквивалентности на общем окне.
- Сессии MOEX (утро/вечер, аукционы, клиринг) учитываются `SessionCalendar`: бары вне сессий не фабрикуются, границы 10m-баров не пересекают разрывы сессий (поведение на стыках — фикстуры этапа D2; Q5).

### 3.4 Конфигурация (схема)

Один YAML на запуск; схема валидируется pydantic-моделью; хэш конфига входит в имена всех артефактов (датасеты, модели, отчёты).

```yaml
# configs/example.yaml  (ИЛЛЮСТРАТИВНАЯ СХЕМА, не код)
run:
  mode: TEST                # TEST | LIVE
  experiment_id: sber-5m-v1

data:
  csv_dir: data/raw
  symbols:
    - ticker: SBER
      mic: MISX             # полный symbol = ticker@mic (требование API)
      files:                # файлы истории по ТФ
        "1m":  SBER_M1.csv
        "5m":  SBER_M5.csv
        "10m": SBER_M10.csv # опционально: иначе строится из 5m
  timestamp_semantics: bar_open   # bar_open | bar_close — подтверждается валидатором
  source_timezone: Europe/Moscow

timeframes:                 # для КАЖДОГО ТФ свои lookback и horizon (требование ТЗ)
  "5m":
    enabled: true
    lookback_bars: 240      # глубина истории для признаков (баров этого ТФ)
    horizon_bars: 6         # горизонт прогноза (баров этого ТФ)
    label:
      kind: fwd_return_class    # fwd_return_class | fwd_return_reg | triple_barrier(позже)
      neutral_band_bps: 3.0
  "10m":
    enabled: true
    base_timeframe: "5m"    # 10m строится ресэмплингом из 5m
    lookback_bars: 120
    horizon_bars: 3
    label: { kind: fwd_return_class, neutral_band_bps: 4.0 }

features:
  set: v1                   # версионируемый набор: returns, SMA/EMA, RSI, ATR, volume-норм., внутридневные эффекты
  rolling_norm: expanding   # только причинные преобразования

model:
  kind: lightgbm            # lightgbm | catboost
  params_file: configs/models/lgbm_v1.yaml
  seed: 42

validation:
  scheme: purged_walk_forward
  n_splits: 5
  embargo_bars: 8           # ≥ horizon — защита от пересечения меток

backtest:
  initial_capital: 100000
  commission_bps: 3         # ОЦЕНОЧНЫЕ параметры — реальные тарифы неизвестны (Q9)
  slippage_bps: 2
  fill: next_bar_open

live:                       # используется только в mode: LIVE
  account_id: ""            # демо-счёт; обязательное поле для недry-run
  allow_orders: false       # явный флаг; true — только после этапа L3-приёмки
  dry_run: true             # по умолчанию ВСЕГДА
  max_order_quantity: 1
  max_position: 1
  max_orders_per_hour: 4
  kill_switch_file: .kill   # наличие файла запрещает новые заявки
  transport: grpc-sdk       # grpc-sdk (finam-sdk) | rest | ws  (интерфейс один)
```

Секреты (`TRADE_API_SECRET`) — только переменные окружения/.env вне git (требование S3b); в логах не появляются (S4 chego-ne-delat).

> Примечание (фаза 3): фактическая реализация ML-секций конфига — `features:` (set+params),
> `model:` (kind/seed/params), `validation.walk_forward:` и `validation.thresholds:`,
> `evaluation:`, `run.model_dir/report_dir`; схема и значения описаны в
> [docs/ml.md](ml.md) и `configs/ml_example.yaml`. Эскиз выше сохраняет смысл как целевая схема.

---

## 4. Потоки данных

### 4.1 TEST (обучение и симуляция)

```
CSV → Loader/Validator → (Resampler при base_timeframe) → BarStore(parquet)
BarStore → FeatureBuilder(lookback) → Labels(horizon) → DatasetAssembler
DatasetAssembler → PurgedWalkForwardSplit → Train(LightGBM/CatBoost) → ModelRegistry(+манифест)
ModelRegistry + BarStore(replay) → EventSimulator(Strategy, SimulatedBroker, издержки) → Отчёт/метрики
```

- Replay детерминирован: сигналы рассчитываются на закрытии бара t, исполнение — на открытии t+1 (fill=next_bar_open), издержки из конфига.
- Датасет и сплиты версионируются хэшем конфига; «золотые» фикстуры для регрессии.

### 4.2 LIVE

```
startup: AuthSession (SDK: авто-JWT + SubscribeJwtRenewal) → TokenDetails (account_ids, readonly, md_permissions → лог)
         → GetAssetParams (trade_lot_size, is_tradable), Schedule, Clock  [pre-flight; при trade_lot_size=0 — стоп (S3j Рецепт 4)]
warmup:  Bars (REST) за окно ≥ lookback+horizon → BarStore (локальный кэш закрытых свечей)
stream:  WS/gRPC SubscribeBars(5m и/или 1m) + SubscribeQuote → BarFinalizer → Resampler(10m при выборе) → FeatureBuilder
         (решение ТОЛЬКО по завершённым свечам; pending-бар мутирует — S3f/S3j Рецепт 6)
signal:  Strategy(Model) → OrderIntent → RiskGuards (лимиты, расписание, kill switch, dry_run)
execution: DryRunBroker(журнал намерений) | FinamBroker:
           OrderIntent → валидация (шаг цены min_step/10^decimals, штуки = лоты × trade_lot_size, client_order_id ≤20, детерминированный из сигнала)
           → PlaceOrder → журнал «неизвестного результата» → сверка GetOrders/ORDERS-поток → OrderState-машина (executed/remaining)
state:   ORDERS/TRADES-потоки + периодический REST-снимок GetAccount → PositionManager → reconciliation (расхождения → алерт + стоп торговли)
recovery: обрыв → backoff 1→2→4→8→16→30+jitter → re-handshake → переподписки → REST-снимки Bars/GetOrders/GetAccount → сверка → продолжение
rate limit: клиентский лимитер ≤ 200 зап/мин на метод + GetUsageMetrics-наблюдение; 429 → пауза, не немедленный повтор
```

### 4.3 Точки интеграции с Finam Trade API (все — через интерфейсы)

| Интерфейс | Реализация LIVE | Методы API (подтверждено S5/S8f/S8h) |
|---|---|---|
| `AuthSession` | `finam-sdk` FinamClient/AsyncFinamClient | POST /v1/sessions, /v1/sessions/details, SubscribeJwtRenewal |
| `InstrumentsSource` | FinamInstruments | GET /v1/assets/{symbol}, /params, /schedule, /v1/assets/clock, /v1/exchanges, /v1/assets/all (кэш каталога) |
| `HistorySource` | FinamHistory | GET /v1/instruments/{symbol}/bars (окна, кэш закрытых свечей) |
| `StreamSource` | FinamStream (WS или gRPC — за одним интерфейсом) | BARS, QUOTES; опционально ORDER_BOOK, INSTRUMENT_TRADES |
| `AccountSource` | FinamAccount | GET /v1/accounts/{id}, /trades, /transactions; WS ORDERS/TRADES |
| `BrokerGateway` | FinamBroker | POST /orders (PlaceOrder), POST /sltp-orders, DELETE /orders/{id}, GET /orders, GET /orders/{id} |
| `QuotaMonitor` | FinamMetrics | GET /v1/usage |

Тесты без сети: фейковые реализации тех же интерфейсов + `for_testing()`-канал SDK к локальному фейк-серверу (S8b).

---

## 5. Модель и целевая переменная

### 5.1 Выбор модели

- **Основной кандидат: LightGBM** (градиентный бустинг над таблицей): быстрые итерации, нативная работа с числовыми признаками свечей, importance/SHAP для аудита, устойчивость к масштабу.
- **Второй кандидат: CatBoost** (ordered boosting — меньше риск оптимистичного смещения на временных рядах при корректных сплитах; удобен, если появятся категориальные признаки — сессия/день недели уже есть как ordinal).
- Сравнение — строго на purged walk-forward, по заранее зафиксированным метрикам (см. 5.3). Победитель фиксируется в decisions.md после этапа M2.
- **Transformer/LSTM — не на первом этапе.** Обоснование допуска в будущем: только если (а) табличная модель упрётся в подтверждённый потолок, (б) последовательностная архитектура даст воспроизводимое улучшение на walk-forward с учётом издержек и (в) появится бюджет на инфраструктуру/обслуживание. Сейчас ни одно условие не подтверждено — см. decisions.md D5. Никаких моделей по изображениям графиков (подход WISEPLAT) — исключено ТЗ.

### 5.2 Целевая переменная (базовый вариант)

Для бара t таймфрейма TF с горизонтом H = `horizon_bars`:

- Основная: `fwd_return(t) = close[t+H] / close[t] − 1` — рассчитывается по **закрытым** свечам того же ТФ.
- Базовая задача: **классификация направления с нейтральной полосой** (учитывает издержки):
  - `y = 1`, если `fwd_return > +band`; `y = 0`, если `< −band`; класс `neutral` (или исключение из обучения) в полосе `±band`, где band ≈ комиссия+слиппедж из конфига (neutral_band_bps).
- Альтернативы, оцениваемые на этапе M1 как ablation: регрессия `fwd_return`; тройной барьер (TP/SL/времени) — откладывается до работающего базового контура (усложняет разметку и симулятор).
- Важно: WISEPLAT-подобная схема «метка = закрытие старшего ТФ выше предыдущего» учтена как частный случай (H баров младшего ТФ ≈ 1 бар старшего); отдельный «старший ТФ» в контракте не требуется — горизонт задаётся в барах своего ТФ (конфиг per-timeframe, требование ТЗ).

### 5.3 Метрики оценки (фиксируются ДО обучения)

- ML: ROC-AUC / log-loss (классификация), MAE/R² (регрессия) — по каждому walk-forward-фолду и агрегированно.
- Торговые (бэктест, с издержками): число сделок, hit-rate, profit factor, max drawdown, распределение PnL — **как диагностические метрики симуляции**. Никаких обещаний доходности; результаты на истории не гарантируют будущее. Система не объявляется production-ready.
- Порог «модель проходит дальше» — заранее записанный в конфиг валидации (например, AUC OOS > 0.52 стабильно по фолдам + неотрицательный PnL после издержек в симуляции); иначе этап не принимается.

### 5.4 Признаки (набор v1, табличный)

Для окна `lookback_bars` на закрытии бара t (все — причинные, только ≤ t):
- лагированные доходности (1..k баров), log-return;
- скользящие (SMA/EMA нескольких периодов) и отношения close/SMA;
- волатильность: rolling std, ATR, range-статистики;
- объёмные: volume/rolling_mean(volume), дельты объёма;
- позиционные: (close−low)/(high−low), gap к предыдущему open;
- внутридневные: минута дня/сессия (ordinal), близость к границам сессии;
- опционально (после проверки Q4): агрегаты стакана/спреда из QUOTES — только если эмпирически подтверждены на демо.

Версионирование: набор признаков — `features.set: v1/v2/...`, манифест колонок сохраняется вместе с моделью; рассинхрон «модель ↔ признаки» недопустим (проверяется загрузчик модели).

---

## 6. План защиты от утечки данных (leakage)

1. **Причинность признаков**: FeatureBuilder видит только бары с `ts_open + tf ≤ decision_time`. Property-based тест: возмущение любых будущих баров не меняет вектор признаков в t (hypothesis-тест, этап F2).
2. **Метки только в обучении**: y(t) использует close[t+H] — строится исключительно в DatasetAssembler; в inference-пути кода меток нет физически (раздельные модули, линтер-правило на импорт `labels` из `strategy/live`).
3. **Purged Walk-Forward + embargo**: фолды идут строго по времени; между train и test удаляется зазор ≥ `horizon_bars` (embargo), чтобы перекрывающиеся метки не протекали. Никакого случайного KFold.
4. **Никаких глобальных фитов**: нормализация/скейлинг — только expanding/rolling статистики, фитованные на прошлом фолда; стандартизация на всём датасете запрещена (проверяется тестом).
5. **Временные метки**: нормализация bar_open/bar_close и таймзон — до сборки датасета (ошибка сдвига на 1 бар = системная утечка); валидатор D1 + фикстуры на стыках сессий.
6. **Индексация «решение ↔ исполнение»**: сигнал на закрытии t → исполнение не раньше открытия t+1 в симуляторе; в LIVE решение отправляется после финализации бара (BarFinalizer), а не по pending-бару.
7. **Версионирование артефактов**: датасет/сплиты/модель/отчёт связаны хэшем конфига и версии кода; повторный прогон воспроизводим (seed фиксируется).
8. **Аудит-отчёт утечек** (часть этапа M1): сравнение «наивного» random-split и walk-forward — большая разница трактуется как признак утечки/нестационарности и блокирует промоушен модели.
9. **Дубли свечей**: в LIVE поток может повторно присылать тот же timestamp (S4) — BarFinalizer дедуплицирует по (symbol, tf, ts_open); иначе признаки «увидят будущее» собственного финального значения.

---

## 7. Обработка ошибок, надёжность, безопасность (LIVE)

- Retry-политика — по официальной таблице (research.md §2.8): GET — backoff+jitter; 429 — пауза и снижение частоты; PlaceOrder без ответа — NEVER blind retry, только сверка по `client_order_id` + GetOrders; CancelOrder без ответа — GetOrder. SDK уже ретраит UNAVAILABLE и server-approved rate-limit pushback; стримы переподключает приложение.
- Reconciliation: при старте, после каждого обрыва и периодически — снимки GetAccount/GetOrders vs локальный журнал; расхождение → алерт + приостановка новых заявок.
- RiskGuards (обязательный слой перед BrokerGateway): dry_run по умолчанию; allowlist account_id (демо); лимиты количества/позиции/частоты; проверка расписания (Schedule) и is_tradable; запрет новой команды при неизвестном результате предыдущей; kill switch; всё из Рецепта 7 (S3j) и примера SMA (S8i: «ровно один счёт у токена, иначе не торговать»).
- Журналирование: структурированные логи (auth-результат и expires_at без токена, подписки, REST-статусы и длительности, заявки client_order_id/order_id/status, сделки, сверки) — состав полей по S3j Рецепт 8; секреты не логируются никогда.
- Деньги/цены: Decimal/целые тики в контуре заявок; проверка кратности шагу `min_step/10^decimals`; лоты→штуки через `trade_lot_size` (0 = стоп операции).

---

## 8. Технологический стек (предложение)

- Python **3.11** (SDK требует ≥3.10; 3.11 — стабильная совместимость с lightgbm/catboost).
- Ядро: `pandas`, `numpy`, `pyarrow` (parquet), `pydantic` (конфиг/контракты), `PyYAML`.
- ML: `lightgbm` (основной), `catboost` (опциональный extra), `scikit-learn` (метрики/сплиты-утилиты), `shap` (аудит, опционально).
- LIVE extra: `finam-sdk==2.23.0` (pin; changelog читать перед обновлением — S3k), `websockets` (если выбирается WS-транспорт вместо gRPC).
- Качество: `pytest`, `hypothesis` (property-тесты причинности), `ruff`, `mypy`, pre-commit; CI — линт+тесты на каждый этап.
- Хранилище: файловая система (parquet + JSON-манифесты); БД не требуется на этом масштабе.

---

## 9. Что сознательно НЕ входит в текущую архитектуру

- Реальная торговля на боевом счёте (решение не принимается; контур закрыт dry-run/демо).
- Модели по изображениям графиков (запрет ТЗ).
- Портфельная торговля несколькими инструментами одновременно (v1 — один символ на конфигурацию; мульти-символ — расширение PositionManager позже).
- Стакан как обязательный источник признаков (Q4 не закрыт — только опциональный адаптер).
- Собственная реализация gRPC-транспорта (используем официальный SDK; «самописный транспорт» прямо не рекомендован S8i).
