"""ML-контур: датасет, сплиты, обучение, метрики, реестр моделей, отчёт."""

from .baselines import LogisticBaseline, MajorityBaseline, MomentumBaseline
from .dataset import DatasetInfo, MlDataset, assemble_dataset, dataset_fingerprint, dataset_period
from .errors import InsufficientHistoryError, ModelCompatibilityError, TrainingError
from .metrics import (
    TradingAssumptions,
    TradingStats,
    aggregate_trading,
    classification_metrics,
    regression_metrics,
    simulate_positions,
)
from .registry import ModelBundle, library_versions, load_model_bundle, save_model_bundle
from .splits import WalkForwardFold, assert_no_label_overlap, purged_walk_forward
from .train import FoldOutcome, TrainingRunResult, load_bars_for_timeframe, run_training

__all__ = [
    "DatasetInfo",
    "FoldOutcome",
    "InsufficientHistoryError",
    "LogisticBaseline",
    "MajorityBaseline",
    "MlDataset",
    "ModelBundle",
    "ModelCompatibilityError",
    "MomentumBaseline",
    "TradingAssumptions",
    "TradingStats",
    "TrainingError",
    "TrainingRunResult",
    "WalkForwardFold",
    "aggregate_trading",
    "assemble_dataset",
    "assert_no_label_overlap",
    "classification_metrics",
    "dataset_fingerprint",
    "dataset_period",
    "library_versions",
    "load_model_bundle",
    "load_bars_for_timeframe",
    "purged_walk_forward",
    "regression_metrics",
    "run_training",
    "save_model_bundle",
    "simulate_positions",
]
