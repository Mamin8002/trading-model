"""Тесты целевой переменной, walk-forward сплитов и сборки датасета."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.config.schema import FeaturesConfig, LabelConfig
from tradebot.data.synthetic import generate_synthetic_bars
from tradebot.labels.target import TargetSpec, build_target, compute_forward_return
from tradebot.ml.dataset import assemble_dataset, dataset_fingerprint, dataset_period
from tradebot.ml.errors import InsufficientHistoryError, TrainingError
from tradebot.ml.splits import assert_no_label_overlap, purged_walk_forward


class TestTarget:
    def test_forward_return_known_values(self):
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        fwd = compute_forward_return(close, 2)
        assert fwd.iloc[0] == pytest.approx(0.02)          # 102/100 - 1
        assert fwd.iloc[1] == pytest.approx(103 / 101 - 1)
        assert fwd.iloc[2] == pytest.approx(104 / 102 - 1)
        assert np.isnan(fwd.iloc[3]) and np.isnan(fwd.iloc[4])  # хвост H=2

    def test_classification_with_neutral_band(self):
        close = pd.Series([100.0, 100.0, 100.0, 100.0, 105.0, 99.0, 100.2])
        spec = TargetSpec(kind="fwd_return_class", horizon_bars=1, neutral_band_bps=100.0)  # band=1%
        y, info = build_target(close, spec)
        # fwd: [0.0, 0.0, 0.0, 0.05, -0.0576..., 0.012..., NaN]
        assert np.isnan(y.iloc[0])  # 0.0 внутри полосы [-1%, +1%] -> нейтраль
        assert y.iloc[3] == 1.0     # +5% > +1%
        assert y.iloc[4] == 0.0     # -5.76% < -1%
        assert y.iloc[5] == 1.0     # +1.2% > +1%
        assert np.isnan(y.iloc[6])  # хвост
        assert info.n_neutral_dropped == 3
        assert info.n_tail_dropped == 1
        assert info.n_valid == 3

    def test_regression_kind(self):
        close = pd.Series([100.0, 101.0, 102.0])
        spec = TargetSpec(kind="fwd_return_reg", horizon_bars=1, neutral_band_bps=0.0)
        y, info = build_target(close, spec)
        assert y.iloc[0] == pytest.approx(0.01)
        assert np.isnan(y.iloc[2])  # хвост H=1
        assert info.n_neutral_dropped == 0

    def test_triple_barrier_not_implemented(self):
        class _Cfg:
            kind = "triple_barrier"
            neutral_band_bps = 0.0

        with pytest.raises(NotImplementedError, match="B1"):
            TargetSpec.from_config(_Cfg(), horizon_bars=5)

    def test_zero_horizon_rejected(self):
        with pytest.raises(ValueError):
            TargetSpec(kind="fwd_return_reg", horizon_bars=0, neutral_band_bps=0.0)


class TestSplits:
    def test_fold_structure_and_purge(self):
        folds = purged_walk_forward(
            1000, n_splits=4, horizon_bars=6, min_train_bars=200, min_val_bars=50
        )
        assert len(folds) == 4
        prev_val_end = -1
        for fold in folds:
            assert_no_label_overlap(fold)
            # хронологичность: train строго до val
            assert fold.train_idx.max() < fold.val_idx.min()
            # embargo >= horizon
            assert fold.embargo_bars >= fold.horizon_bars
            # точный инвариант очистки: max(train) + H < min(val)
            assert fold.train_idx.max() + fold.horizon_bars < fold.val_idx.min()
            # расширяющееся train-окно
            assert fold.n_train >= 200
            # валидационные окна не пересекаются и идут по времени
            assert fold.val_idx.min() > prev_val_end
            prev_val_end = fold.val_idx.max()
            assert np.all(np.diff(fold.val_idx) == 1)  # непрерывность val-окна

    def test_last_fold_extends_to_end(self):
        folds = purged_walk_forward(
            1000, n_splits=4, horizon_bars=6, min_train_bars=200, min_val_bars=50
        )
        assert folds[-1].val_idx.max() == 999

    def test_explicit_embargo_larger_than_horizon(self):
        folds = purged_walk_forward(
            1000, n_splits=3, horizon_bars=4, embargo_bars=20, min_train_bars=100, min_val_bars=50
        )
        for fold in folds:
            assert fold.embargo_bars == 20
            assert fold.train_idx.max() + 20 < fold.val_idx.min()

    def test_insufficient_history_raises_with_explanation(self):
        with pytest.raises(InsufficientHistoryError) as exc:
            purged_walk_forward(
                300, n_splits=5, horizon_bars=6, min_train_bars=200, min_val_bars=50
            )
        assert exc.value.required_bars is not None
        assert "недостаточно" in str(exc.value).lower()
        # сообщение объясняет, что train-метрики не являются оценкой
        assert "обучающей" in str(exc.value)

    def test_no_shuffling_indices_are_ranges(self):
        folds = purged_walk_forward(
            800, n_splits=3, horizon_bars=5, min_train_bars=100, min_val_bars=30
        )
        for fold in folds:
            assert np.array_equal(fold.train_idx, np.arange(fold.train_idx[0], fold.train_idx[-1] + 1))
            assert fold.train_idx[0] == 0  # expanding anchor от начала


@pytest.fixture(scope="module")
def small_dataset():
    bars = generate_synthetic_bars(600, seed=13, timeframe="5m")
    return assemble_dataset(
        bars,
        timeframe="5m",
        horizon_bars=4,
        lookback_bars=240,
        label_cfg=LabelConfig(kind="fwd_return_class", neutral_band_bps=5.0),
        features_cfg=FeaturesConfig(),
        symbol="SYNTH@DEMO",
    )


class TestDataset:
    @pytest.fixture()
    def dataset(self, small_dataset):
        return small_dataset

    def test_shapes_and_no_nan(self, dataset):
        assert dataset.X.shape == (dataset.n_samples, len(dataset.feature_names))
        assert np.isfinite(dataset.X).all()
        assert dataset.y.shape == (dataset.n_samples,)
        assert set(np.unique(dataset.y)) <= {0.0, 1.0}

    def test_accounting_identity(self, dataset):
        info = dataset.info
        # строки датасета <= всех баров; счётчики исключений неотрицательны
        assert info.n_rows <= info.n_bars
        assert info.warmup_dropped > 0
        assert info.tail_dropped >= 4  # хвост H баров как минимум
        assert info.n_rows + info.warmup_dropped <= info.n_bars

    def test_execution_arrays_consistency(self, dataset):
        # open_t1 > 0, open_t2 > 0; close_t > 0; prev_ret конечен
        assert (dataset.open_t1 > 0).all() and (dataset.open_t2 > 0).all()
        assert (dataset.close_t > 0).all()
        assert np.isfinite(dataset.prev_ret).all()

    def test_ts_sorted(self, dataset):
        assert np.all(np.diff(dataset.ts.astype("datetime64[s]")).astype("int64") >= 0)

    def test_lookback_smaller_than_warmup_rejected(self):
        bars = generate_synthetic_bars(300, seed=14, timeframe="5m")
        with pytest.raises(TrainingError, match="lookback_bars"):
            assemble_dataset(
                bars,
                timeframe="5m",
                horizon_bars=4,
                lookback_bars=10,  # < warmup 61
                label_cfg=LabelConfig(),
                features_cfg=FeaturesConfig(),
            )

    def test_fingerprint_and_period(self, dataset):
        fp = dataset_fingerprint(dataset)
        assert len(fp) == 16
        first, last = dataset_period(dataset)
        assert first < last

    def test_labels_not_in_features(self, dataset):
        # контракт: y существует отдельно от X; в feature_names нет целевых колонок
        assert not any("fwd" in name or "target" in name or "label" in name for name in dataset.feature_names)
