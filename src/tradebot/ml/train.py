"""Оркестратор обучения и walk-forward оценки (docs/ml.md §4-§6, roadmap M1/M2).

Порядок (все этапы детерминированы при фиксированном seed):
1. загрузка/построение баров таймфрейма (CSV или ресэмплинг из базового ТФ);
2. сборка датасета (причины -> X, будущее -> y только здесь);
3. purged walk-forward разбиение (embargo >= horizon);
4. для каждого фолда: LightGBM (параметры ФИКСИРОВАНЫ конфигом до обучения;
   early stopping — только на holdout внутри обучающего окна), базовые ориентиры
   (majority / momentum / logreg), метрики ML и торговые метрики с издержками;
5. OOS-агрегация (объединение валидационных окон — хронологически, без перекрытий);
6. аудит «random-split vs walk-forward» (architecture §6.8) — только диагностика;
7. финальная модель на всех строках (метрики ей НЕ присваиваются — все оценки
   из фолдов), сохранение артефакта с метаданными и отчёта.

Подбор гиперпараметров на валидационных фолдах ЗАПРЕЩЁН (decisions D6):
конфиг фиксирует параметры заранее; финальный тест (последний фолд) в подборе
не участвует.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config.loader import build_calendar, config_hash
from ..config.schema import AppConfig, LabelConfig
from ..data.bar import Bar, Timeframe
from ..data.csv_loader import DataValidationError, load_csv
from ..data.resampler import resample
from ..data.validator import validate_bars
from .baselines import LogisticBaseline, MajorityBaseline, MomentumBaseline
from .dataset import DatasetInfo, MlDataset, assemble_dataset, dataset_fingerprint, dataset_period
from .errors import InsufficientHistoryError, TrainingError
from .metrics import (
    TradingAssumptions,
    TradingStats,
    aggregate_trading,
    annualized_cagr,
    classification_metrics,
    derive_bars_per_year,
    positions_from_proba,
    positions_from_regression,
    regression_metrics,
    sharpe_ratio,
    simulate_positions,
)
from .registry import library_versions, save_model_bundle
from .splits import WalkForwardFold, assert_no_label_overlap, purged_walk_forward

__all__ = ["FoldOutcome", "TrainingRunResult", "load_bars_for_timeframe", "run_training"]

_AUDIT_DELTA_FLAG = 0.05  # порог flags аудита (architecture §6.8), задокументирован в docs/ml.md


@dataclass
class FoldOutcome:
    fold_id: int
    n_train: int
    n_val: int
    train_range: tuple[str, str]
    val_range: tuple[str, str]
    embargo_bars: int
    model_metrics: dict[str, Any]
    model_train_metrics_diagnostic: dict[str, Any]
    baseline_metrics: dict[str, dict[str, Any]]
    model_trading: dict[str, Any]
    baseline_trading: dict[str, dict[str, Any]]
    best_iteration: int | None
    val_proba: np.ndarray = field(repr=False)
    val_y: np.ndarray = field(repr=False)
    val_positions: np.ndarray = field(repr=False)
    baseline_val_proba: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    baseline_val_positions: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


@dataclass
class TrainingRunResult:
    experiment_id: str
    symbol: str
    timeframe: str
    horizon_bars: int
    lookback_bars: int
    label: dict[str, Any]
    seed: int
    config_hash: str
    dataset_fingerprint: str
    dataset_info: dict[str, Any]
    training_period: tuple[str, str]
    folds: list[FoldOutcome]
    oos_metrics: dict[str, Any]
    oos_trading: dict[str, Any]
    oos_trading_annualized: dict[str, Any]
    baselines_oos_metrics: dict[str, dict[str, Any]]
    baselines_oos_trading: dict[str, dict[str, Any]]
    assumptions: dict[str, Any]
    audit: dict[str, Any] | None
    verdict: dict[str, Any]
    versions: dict[str, str]
    model_dir: str
    report_path: str
    final_n_estimators: int


# ---------------------------------------------------------------------------
# Загрузка баров
# ---------------------------------------------------------------------------

def load_bars_for_timeframe(
    config: AppConfig,
    timeframe: "Timeframe | str",
    *,
    symbol: str | None = None,
) -> tuple[list[Bar], str]:
    """Загружает бары ТФ из CSV (или строит ресэмплингом из base_timeframe).

    Возвращает (bars, symbol). Ошибки данных/конфига — явные исключения,
    пропуски НЕ заполняются (decisions D1-D3).
    """
    tf = Timeframe.parse(timeframe)
    tf_cfg = config.timeframes.get(tf.value)
    if tf_cfg is None:
        raise TrainingError(
            f"Таймфрейм {tf.value} не описан в конфиге (секция timeframes). "
            f"Доступны: {sorted(config.timeframes)}"
        )
    if not tf_cfg.enabled:
        raise TrainingError(f"Таймфрейм {tf.value} отключён в конфиге (enabled: false)")

    symbol_cfg = _resolve_symbol(config, symbol)
    calendar = build_calendar(config)
    files = symbol_cfg.files

    if tf.value in files:
        result = load_csv(
            config.data.csv_dir / files[tf.value],
            timeframe=tf,
            symbol=symbol_cfg.symbol,
            source_timezone=config.data.source_timezone,
            timestamp_semantics=config.data.timestamp_semantics,
            calendar=calendar,
            warn_zero_volume=config.validation.warn_zero_volume,
        )
        bars = result.bars
        if result.report.warnings:
            # предупреждения (пропуски и пр.) не блокируют обучение, но фиксируются
            warnings.warn(
                f"CSV {files[tf.value]}: {len(result.report.warnings)} предупреждений валидации "
                f"(первое: {result.report.warnings[0].format()})",
                stacklevel=2,
            )
        return bars, symbol_cfg.symbol

    base_code = tf_cfg.base_timeframe
    if base_code is None or base_code not in files:
        raise TrainingError(
            f"Для таймфрейма {tf.value} нет файла в data.symbols[].files и нет доступного "
            f"base_timeframe ({base_code!r}) с файлом. Укажите CSV или базовый таймфрейм"
        )
    base_result = load_csv(
        config.data.csv_dir / files[base_code],
        timeframe=base_code,
        symbol=symbol_cfg.symbol,
        source_timezone=config.data.source_timezone,
        timestamp_semantics=config.data.timestamp_semantics,
        calendar=calendar,
        warn_zero_volume=config.validation.warn_zero_volume,
    )
    resampled = resample(base_result.bars, base_code, tf, calendar=calendar, symbol=symbol_cfg.symbol)
    blocking = [i for i in resampled.issues if i.code.value != "non_final_skipped"]
    if blocking:
        preview = "; ".join(i.message for i in blocking[:3])
        raise TrainingError(
            f"Ресэмплинг {base_code}->{tf.value} дал {len(blocking)} проблем покрытия "
            f"(неполные интервалы НЕ выдаются за полные): {preview}. "
            "Настройте календарь сессий (sessions:) или очистите данные"
        )
    return resampled.bars, symbol_cfg.symbol


def _resolve_symbol(config: AppConfig, symbol: str | None):
    symbols = config.data.symbols
    if symbol is None:
        if len(symbols) != 1:
            raise TrainingError(
                "В конфиге несколько символов — укажите --symbol (например BTCUSDT)"
            )
        return symbols[0]
    for cfg in symbols:
        if symbol in (cfg.symbol, cfg.ticker):
            return cfg
    raise TrainingError(
        f"Символ {symbol!r} не найден в data.symbols: {[c.symbol for c in symbols]}"
    )


# ---------------------------------------------------------------------------
# Обучение фолда
# ---------------------------------------------------------------------------

def _lightgbm_params(config: AppConfig, objective: str) -> dict[str, Any]:
    p = config.model.params
    params: dict[str, Any] = {
        "objective": objective,
        "learning_rate": p.learning_rate,
        "num_leaves": p.num_leaves,
        "max_depth": p.max_depth,
        "min_child_samples": p.min_child_samples,
        "subsample": p.subsample,
        "subsample_freq": p.subsample_freq,
        "colsample_bytree": p.colsample_bytree,
        "reg_alpha": p.reg_alpha,
        "reg_lambda": p.reg_lambda,
        "random_state": config.model.seed,
        "n_jobs": 1,               # детерминированность важнее скорости (docs/ml.md §7)
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    return params


def _fit_lightgbm(config: AppConfig, dataset: MlDataset, train_idx: np.ndarray, objective: str, metric: str):
    """Фит LightGBM на train-окне; early stopping — только на holdout ВНУТРИ train."""
    import lightgbm as lgb

    p = config.model.params
    holdout = p.early_stopping_holdout_bars
    X_tr, y_tr = dataset.X[train_idx], dataset.y[train_idx]
    best_iteration: int | None = None

    if objective == "binary":
        model: Any = lgb.LGBMClassifier(n_estimators=p.n_estimators, metric=metric, **_lightgbm_params(config, objective))
    else:
        model = lgb.LGBMRegressor(n_estimators=p.n_estimators, metric=metric, **_lightgbm_params(config, objective))

    fit_kwargs: dict[str, Any] = {}
    if holdout > 0 and len(train_idx) > holdout * 3:
        fit_part, es_part = train_idx[:-holdout], train_idx[-holdout:]
        fit_kwargs = {
            "eval_set": [(dataset.X[es_part], dataset.y[es_part])],
            "callbacks": [lgb.early_stopping(p.early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
        }
        # lightgbm 4.7 помечает eval_set deprecated (замена появится в 5.x);
        # мы закрепляем lightgbm>=4,<5 и глушим deprecation-предупреждение
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(dataset.X[fit_part], dataset.y[fit_part], **fit_kwargs)
        best_iteration = int(getattr(model, "best_iteration_", 0)) or None
    else:
        model.fit(X_tr, y_tr)
    return model, best_iteration


def _predict(model: Any, X: np.ndarray, classification: bool) -> np.ndarray:
    if classification:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _train_fold(
    config: AppConfig,
    dataset: MlDataset,
    fold: WalkForwardFold,
    threshold: float,
    objective: str,
    metric: str,
) -> FoldOutcome:
    classification = dataset.is_classification
    model, best_iteration = _fit_lightgbm(config, dataset, fold.train_idx, objective, metric)
    val_pred = _predict(model, dataset.X[fold.val_idx], classification)
    train_pred = _predict(model, dataset.X[fold.train_idx], classification)

    y_val = dataset.y[fold.val_idx]
    y_train = dataset.y[fold.train_idx]
    if classification:
        model_metrics = classification_metrics(y_val, val_pred, threshold).as_dict()
        train_metrics = classification_metrics(y_train, train_pred, threshold).as_dict()
        positions = positions_from_proba(val_pred, threshold)
    else:
        model_metrics = regression_metrics(y_val, val_pred).__dict__
        train_metrics = regression_metrics(y_train, train_pred).__dict__
        positions = positions_from_regression(val_pred)

    # базовые ориентиры — те же фолды, те же метрики
    majority = MajorityBaseline()
    majority.fit(y_train)
    majority_res = majority.predict_proba(len(fold.val_idx)) if classification else None

    momentum = MomentumBaseline()
    momentum_res = momentum.predict_proba(dataset.prev_ret[fold.val_idx]) if classification else None

    logreg = LogisticBaseline(seed=config.model.seed)
    logreg_res = None
    if classification:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logreg.fit(dataset.X[fold.train_idx], y_train)
            logreg_res = logreg.predict_proba(dataset.X[fold.val_idx])

    baseline_metrics: dict[str, dict[str, Any]] = {}
    baseline_trading: dict[str, dict[str, Any]] = {}
    baseline_val_proba: dict[str, np.ndarray] = {}
    baseline_val_positions: dict[str, np.ndarray] = {}

    def _register_baseline(name: str, proba: np.ndarray | None, note: str, degenerate: bool) -> None:
        if proba is None:
            baseline_metrics[name] = {"note": "не применяется к регрессии"}
            return
        baseline_val_proba[name] = proba
        if classification:
            baseline_metrics[name] = classification_metrics(y_val, proba, threshold).as_dict()
            pos = positions_from_proba(proba, threshold)
        else:
            baseline_metrics[name] = regression_metrics(y_val, proba).__dict__
            pos = positions_from_regression(proba)
        baseline_val_positions[name] = pos
        stats = _simulate(dataset, fold.val_idx, pos, config)
        baseline_trading[name] = stats.as_dict()
        if degenerate or note:
            baseline_metrics[name]["note"] = note

    _register_baseline("majority", majority_res.proba if majority_res else None,
                       majority_res.note if majority_res else "", bool(majority_res and majority_res.degenerate))
    _register_baseline("momentum", momentum_res.proba if momentum_res else None,
                       momentum_res.note if momentum_res else "", False)
    _register_baseline("logreg", logreg_res.proba if logreg_res else None,
                       logreg_res.note if logreg_res else "", bool(logreg_res and logreg_res.degenerate))
    if not classification:
        # для регрессии momentum/majority не определены — отмечаем явно
        for name in ("majority", "momentum"):
            baseline_metrics.setdefault(name, {"note": "не применяется к регрессии"})

    model_stats = _simulate(dataset, fold.val_idx, positions, config)

    ts = dataset.ts
    return FoldOutcome(
        fold_id=fold.fold_id,
        n_train=fold.n_train,
        n_val=fold.n_val,
        train_range=(str(pd.Timestamp(ts[fold.train_idx[0]])), str(pd.Timestamp(ts[fold.train_idx[-1]]))),
        val_range=(str(pd.Timestamp(ts[fold.val_idx[0]])), str(pd.Timestamp(ts[fold.val_idx[-1]]))),
        embargo_bars=fold.embargo_bars,
        model_metrics=model_metrics,
        model_train_metrics_diagnostic=train_metrics,
        baseline_metrics=baseline_metrics,
        model_trading=model_stats.as_dict(),
        baseline_trading=baseline_trading,
        best_iteration=best_iteration,
        val_proba=val_pred,
        val_y=y_val,
        val_positions=positions,
        baseline_val_proba=baseline_val_proba,
        baseline_val_positions=baseline_val_positions,
    )


def _simulate(dataset: MlDataset, val_idx: np.ndarray, positions: np.ndarray, config: AppConfig) -> TradingStats:
    return simulate_positions(
        dataset.open_t1[val_idx],
        dataset.open_t2[val_idx],
        positions,
        commission_bps=config.evaluation.commission_bps,
        slippage_bps=config.evaluation.slippage_bps,
    )


# ---------------------------------------------------------------------------
# Аудит random-split (только диагностика, architecture §6.8)
# ---------------------------------------------------------------------------

def _audit_random_split(
    config: AppConfig,
    dataset: MlDataset,
    objective: str,
    metric: str,
    oos_auc: float | None,
) -> dict[str, Any] | None:
    if not config.validation.walk_forward.audit_random_split:
        return {"enabled": False}
    if not dataset.is_classification:
        return {"enabled": True, "applicable": False, "note": "аудит определён для классификации"}
    n = dataset.n_samples
    rng = np.random.RandomState(config.model.seed)
    perm = rng.permutation(n)  # НАМЕРЕННО нехронологический сплит — только для аудита утечки
    cut = int(n * 0.8)
    train_idx, test_idx = perm[:cut], perm[cut:]
    import lightgbm as lgb

    p = config.model.params
    model = lgb.LGBMClassifier(n_estimators=p.n_estimators, metric=metric, **_lightgbm_params(config, objective))
    model.fit(dataset.X[train_idx], dataset.y[train_idx])
    proba = model.predict_proba(dataset.X[test_idx])[:, 1]
    metrics = classification_metrics(dataset.y[test_idx], proba, config.evaluation.decision_threshold)
    auc_random = metrics.roc_auc
    delta = None
    flagged = False
    if auc_random is not None and oos_auc is not None:
        delta = auc_random - oos_auc
        flagged = delta > _AUDIT_DELTA_FLAG
    return {
        "enabled": True,
        "applicable": True,
        "auc_random_split": auc_random,
        "auc_walk_forward_oos": oos_auc,
        "delta": delta,
        "flag_threshold": _AUDIT_DELTA_FLAG,
        "flagged": flagged,
        "interpretation": (
            "большая разница в пользу random-split — индикатор утечки будущего или "
            "сильной нестационарности; такие результаты блокируют промоушен модели"
            if flagged
            else "разница в пределах порога — явных признаков утечки не выявлено"
        ),
    }


# ---------------------------------------------------------------------------
# Основной прогон
# ---------------------------------------------------------------------------

def run_training(
    config: AppConfig,
    timeframe: "Timeframe | str",
    *,
    symbol: str | None = None,
    bars: Sequence[Bar] | None = None,
    model_dir: "str | Path | None" = None,
    report_dir: "str | Path | None" = None,
    local_timezone: str | None = None,
) -> TrainingRunResult:
    """Полный цикл обучения и walk-forward оценки. См. docstring модуля."""
    tf = Timeframe.parse(timeframe)
    if config.model.kind != "lightgbm":
        raise NotImplementedError(
            f"Модель {config.model.kind!r} не подключена в фазе 3: LightGBM — основной кандидат "
            "(decisions D4), CatBoost — контрольный, подключается отдельной задачей (P1). "
            "Transformer/LSTM требуют явного обоснования (architecture §5.1)"
        )
    tf_cfg = config.timeframes.get(tf.value)
    if tf_cfg is None or not tf_cfg.enabled:
        raise TrainingError(
            f"Таймфрейм {tf.value} отсутствует или отключён в конфиге; "
            f"доступны: {sorted(k for k, v in config.timeframes.items() if v.enabled)}"
        )
    label_cfg = tf_cfg.label or LabelConfig()
    horizon = tf_cfg.horizon_bars
    lookback = tf_cfg.lookback_bars

    if bars is None:
        bars_list, resolved_symbol = load_bars_for_timeframe(config, tf, symbol=symbol)
    else:
        bars_list = list(bars)
        resolved_symbol = symbol or (bars_list[0].symbol if bars_list else "")
        if not bars_list:
            raise TrainingError("Передан пустой список баров")
    calendar = build_calendar(config)
    validation_report = validate_bars(bars_list, tf, calendar=calendar)
    if not validation_report.ok:
        raise TrainingError(
            "Бары не прошли валидацию:\n" + validation_report.summary_text(config.validation.max_reported_issues)
        )

    local_tz = local_timezone or config.data.source_timezone
    dataset = assemble_dataset(
        bars_list,
        timeframe=tf,
        horizon_bars=horizon,
        lookback_bars=lookback,
        label_cfg=label_cfg,
        features_cfg=config.features,
        symbol=resolved_symbol,
        local_timezone=local_tz,
    )

    wf = config.validation.walk_forward
    folds = purged_walk_forward(
        dataset.n_samples,
        n_splits=wf.n_splits,
        horizon_bars=horizon,
        embargo_bars=wf.embargo_bars,
        min_train_bars=wf.min_train_bars,
        min_val_bars=wf.min_val_bars,
    )
    for fold in folds:
        assert_no_label_overlap(fold)  # инвариант D6 — проверяется на каждом прогоне

    classification = dataset.is_classification
    objective = "binary" if classification else "regression"
    metric = "binary_logloss" if classification else "l2"
    threshold = config.evaluation.decision_threshold

    outcomes = [_train_fold(config, dataset, fold, threshold, objective, metric) for fold in folds]

    # ---- OOS-агрегация (объединение неперекрывающихся валидационных окон) ----
    oos_pred = np.concatenate([o.val_proba for o in outcomes])
    oos_y = np.concatenate([o.val_y for o in outcomes])
    oos_positions = np.concatenate([o.val_positions for o in outcomes])
    if classification:
        oos_metrics = classification_metrics(oos_y, oos_pred, threshold).as_dict()
    else:
        oos_metrics = regression_metrics(oos_y, oos_pred).__dict__

    oos_segments = [
        _simulate(dataset, fold.val_idx, o.val_positions, config) for fold, o in zip(folds, outcomes)
    ]
    oos_trading_stats = aggregate_trading(oos_segments)

    baselines_oos_metrics: dict[str, dict[str, Any]] = {}
    baselines_oos_trading: dict[str, dict[str, Any]] = {}
    baseline_names = [name for name in ("majority", "momentum", "logreg") if any(name in o.baseline_val_proba for o in outcomes)]
    for name in baseline_names:
        proba_parts = [o.baseline_val_proba[name] for o in outcomes if name in o.baseline_val_proba]
        pos_parts = [o.baseline_val_positions[name] for o in outcomes if name in o.baseline_val_positions]
        if not proba_parts:
            continue
        if len(proba_parts) != len(folds) or len(pos_parts) != len(folds):
            raise TrainingError(
                f"Базовый ориентир {name!r} присутствует не во всех фолдах — "
                "OOS-агрегация невозможна (внутренняя ошибка согласованности)"
            )
        proba_all = np.concatenate(proba_parts)
        if classification:
            baselines_oos_metrics[name] = classification_metrics(oos_y, proba_all, threshold).as_dict()
        else:
            baselines_oos_metrics[name] = regression_metrics(oos_y, proba_all).__dict__
        segments = [
            _simulate(dataset, fold.val_idx, pos, config)
            for fold, pos in zip(folds, pos_parts)
        ]
        baselines_oos_trading[name] = aggregate_trading(segments).as_dict()

    # ---- допущения годового пересчёта ----
    bars_per_year, bars_per_day_median, bpr_source = derive_bars_per_year(
        dataset.ts,
        trading_days_per_year=config.evaluation.trading_days_per_year,
        local_timezone=local_tz,
        config_override=config.evaluation.bars_per_year,
    )
    assumptions = TradingAssumptions(
        commission_bps=config.evaluation.commission_bps,
        slippage_bps=config.evaluation.slippage_bps,
        bars_per_year=bars_per_year,
        bars_per_day_median=bars_per_day_median,
        trading_days_per_year=config.evaluation.trading_days_per_year,
        risk_free_annual=config.evaluation.risk_free_annual,
        bars_per_year_source=bpr_source,
    )
    oos_annualized = {
        "cagr": annualized_cagr(oos_trading_stats.net_return, oos_trading_stats.n_bars, bars_per_year),
        "sharpe": sharpe_ratio(oos_trading_stats.per_bar_net, bars_per_year, config.evaluation.risk_free_annual),
        "bars_per_year": bars_per_year,
        "bars_per_year_source": bpr_source,
        "n_oos_bars": oos_trading_stats.n_bars,
    }

    # ---- аудит утечек ----
    audit = _audit_random_split(config, dataset, objective, metric, oos_metrics.get("roc_auc"))

    # ---- вердикт по заранее заданным порогам ----
    verdict = _build_verdict(config, oos_metrics, oos_trading_stats, baselines_oos_metrics, audit, classification)

    # ---- финальная модель и артефакты ----
    best_iterations = [o.best_iteration for o in outcomes if o.best_iteration]
    if best_iterations:
        final_n_estimators = max(10, int(round(float(np.median(best_iterations)))))
    else:
        final_n_estimators = config.model.params.n_estimators
    final_model, _ = _fit_final_model(config, dataset, objective, metric, final_n_estimators)

    fingerprint = dataset_fingerprint(dataset)
    period_first, period_last = dataset_period(dataset)
    artifact_name = f"{config.run.experiment_id}_{tf.value}_{fingerprint}"
    target_model_dir = Path(model_dir if model_dir is not None else config.run.model_dir) / artifact_name
    target_report_dir = Path(report_dir if report_dir is not None else config.run.report_dir)
    target_report_dir.mkdir(parents=True, exist_ok=True)

    oos_frame = pd.DataFrame(
        {
            "ts": np.concatenate([dataset.ts[fold.val_idx] for fold in folds]),
            "y": oos_y,
            "prediction": oos_pred,
            "position": oos_positions,
        }
    )
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config.run.experiment_id,
        "symbol": resolved_symbol,
        "timeframe": tf.value,
        "horizon_bars": horizon,
        "lookback_bars": lookback,
        "label": {"kind": label_cfg.kind, "neutral_band_bps": label_cfg.neutral_band_bps},
        "model": {
            "kind": config.model.kind,
            "objective": objective,
            "seed": config.model.seed,
            "params": config.model.params.model_dump(mode="json"),
            "final_n_estimators": final_n_estimators,
            "best_iterations_per_fold": best_iterations,
        },
        "training_period": {
            "first_ts": period_first.isoformat(),
            "last_ts": period_last.isoformat(),
            "n_rows": dataset.n_samples,
        },
        "dataset": {
            "fingerprint": fingerprint,
            "info": _dataset_info_dict(dataset.info),
            "feature_names": list(dataset.feature_names),
        },
        "walk_forward": {
            "n_splits": wf.n_splits,
            "embargo_bars": folds[0].embargo_bars if folds else None,
            "min_train_bars": wf.min_train_bars,
            "min_val_bars": wf.min_val_bars,
            "folds": [
                {
                    "fold_id": o.fold_id,
                    "n_train": o.n_train,
                    "n_val": o.n_val,
                    "train_range": list(o.train_range),
                    "val_range": list(o.val_range),
                    "model_metrics": o.model_metrics,
                    "model_trading": o.model_trading,
                    "baseline_metrics": o.baseline_metrics,
                    "best_iteration": o.best_iteration,
                }
                for o in outcomes
            ],
        },
        "results": {
            "oos_metrics": oos_metrics,
            "oos_trading": oos_trading_stats.as_dict(),
            "oos_annualized": oos_annualized,
            "baselines_oos_metrics": baselines_oos_metrics,
            "baselines_oos_trading": baselines_oos_trading,
            "audit_random_split": audit,
            "verdict": verdict,
        },
        "assumptions": assumptions.as_dict(),
        "versions": library_versions(),
        "config_hash": config_hash(config),
        "note": (
            "Финальная модель обучена на всех строках датасета для последующего "
            "использования; приведённые метрики получены ТОЛЬКО на валидационных "
            "окнах walk-forward и финальной модели не присвоены"
        ),
    }

    booster = final_model.booster_
    save_model_bundle(
        target_model_dir,
        booster,
        metadata,
        dataset.spec,
        oos_predictions_csv=oos_frame.to_csv(index=False),
    )

    result = TrainingRunResult(
        experiment_id=config.run.experiment_id,
        symbol=resolved_symbol,
        timeframe=tf.value,
        horizon_bars=horizon,
        lookback_bars=lookback,
        label={"kind": label_cfg.kind, "neutral_band_bps": label_cfg.neutral_band_bps},
        seed=config.model.seed,
        config_hash=config_hash(config),
        dataset_fingerprint=fingerprint,
        dataset_info=_dataset_info_dict(dataset.info),
        training_period=(period_first.isoformat(), period_last.isoformat()),
        folds=outcomes,
        oos_metrics=oos_metrics,
        oos_trading=oos_trading_stats.as_dict(),
        oos_trading_annualized=oos_annualized,
        baselines_oos_metrics=baselines_oos_metrics,
        baselines_oos_trading=baselines_oos_trading,
        assumptions=assumptions.as_dict(),
        audit=audit,
        verdict=verdict,
        versions=metadata["versions"],
        model_dir=str(target_model_dir),
        report_path=str(target_report_dir / f"train_report_{artifact_name}.md"),
        final_n_estimators=final_n_estimators,
    )

    from .report import render_markdown  # локальный импорт: report зависит от train-типов

    Path(result.report_path).write_text(render_markdown(result, dataset), encoding="utf-8")
    return result


def _fit_final_model(config: AppConfig, dataset: MlDataset, objective: str, metric: str, n_estimators: int):
    import lightgbm as lgb

    all_idx = np.arange(dataset.n_samples)
    params = _lightgbm_params(config, objective)
    if objective == "binary":
        model: Any = lgb.LGBMClassifier(n_estimators=n_estimators, metric=metric, **params)
    else:
        model = lgb.LGBMRegressor(n_estimators=n_estimators, metric=metric, **params)
    model.fit(dataset.X[all_idx], dataset.y[all_idx])
    return model, n_estimators


def _dataset_info_dict(info: DatasetInfo) -> dict[str, Any]:
    return {
        "n_bars": info.n_bars,
        "n_rows": info.n_rows,
        "warmup_dropped": info.warmup_dropped,
        "neutral_dropped": info.neutral_dropped,
        "label_tail_dropped": info.tail_dropped,
        "execution_tail_dropped": info.execution_tail_dropped,
        "target_info": {
            "kind": info.target_info.spec.kind if info.target_info else None,
            "horizon_bars": info.target_info.spec.horizon_bars if info.target_info else None,
            "neutral_band_bps": info.target_info.spec.neutral_band_bps if info.target_info else None,
            "class_balance": info.target_info.class_balance if info.target_info else {},
        },
    }


def _build_verdict(
    config: AppConfig,
    oos_metrics: dict[str, Any],
    oos_trading: TradingStats,
    baselines_oos_metrics: dict[str, dict[str, Any]],
    audit: dict[str, Any] | None,
    classification: bool,
) -> dict[str, Any]:
    thresholds = config.validation.thresholds
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool | None, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    auc = oos_metrics.get("roc_auc") if classification else None
    if thresholds.min_oos_auc is not None:
        if auc is None:
            add("min_oos_auc", None, "AUC не определён (регессия или один класс) — порог не проверялся")
        else:
            add("min_oos_auc", auc >= thresholds.min_oos_auc, f"AUC OOS {auc:.4f} vs порог {thresholds.min_oos_auc}")
    if thresholds.min_oos_trades is not None:
        add(
            "min_oos_trades",
            oos_trading.n_trades >= thresholds.min_oos_trades,
            f"сделок OOS {oos_trading.n_trades} vs порог {thresholds.min_oos_trades}",
        )
    if audit and audit.get("flagged"):
        add("leakage_audit", False, "аудит random-split показал подозрительно большой разрыв — см. раздел аудита")

    baseline_comparison: dict[str, Any] = {}
    if classification and auc is not None:
        for name in ("majority", "momentum", "logreg"):
            base_auc = baselines_oos_metrics.get(name, {}).get("roc_auc")
            baseline_comparison[name] = {
                "auc": base_auc,
                "model_beats": (auc > base_auc) if base_auc is not None else None,
            }

    explicit_checks = [c for c in checks if c["passed"] is not None]
    passed = all(c["passed"] for c in explicit_checks) if explicit_checks else None
    text = (
        "диагностические пороги не заданы — вердикт информационный"
        if passed is None
        else ("диагностические пороги пройдены" if passed else "диагностические пороги НЕ пройдены")
    )
    return {
        "checks": checks,
        "passed": passed,
        "baseline_comparison": baseline_comparison,
        "text": text + ". Вердикт НЕ является утверждением о прибыльности (decisions D16)",
    }
