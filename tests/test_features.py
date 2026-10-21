"""Тесты признаков: причинность (КЛЮЧЕВОЙ критерий приёмки), warmup, известные значения.

Причинность проверяется двумя способами:
1. префикс-инвариантность: признаки, построенные на серии, обрезанной в t,
   совпадают с признаками полной серии в строках <= t;
2. возмущение будущего: изменение ЛЮБЫХ баров после t не меняет признаки в t.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from tradebot.config.schema import FeatureParamsConfig, FeaturesConfig
from tradebot.data.bar import UTC, Bar
from tradebot.data.synthetic import generate_synthetic_bars
from tradebot.features.builder import FeatureBuilder
from tradebot.features.indicators import log_return, rsi_wilder
from tradebot.features.spec import FeatureSpec


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return FeatureSpec.from_config(FeaturesConfig())


@pytest.fixture(scope="module")
def bars() -> list[Bar]:
    return generate_synthetic_bars(400, seed=11, timeframe="5m")


@pytest.fixture(scope="module")
def frame(spec, bars) -> pd.DataFrame:
    return FeatureBuilder(spec, local_timezone="Europe/Moscow").build(bars)


class TestSpec:
    def test_feature_names_deterministic(self):
        s1 = FeatureSpec.from_config(FeaturesConfig())
        s2 = FeatureSpec.from_config(FeaturesConfig())
        assert s1.feature_names == s2.feature_names
        assert len(s1.feature_names) == 31  # v1: документированное число колонок

    def test_params_change_names(self):
        cfg = FeaturesConfig(params=FeatureParamsConfig(ema_periods=[9], return_lags=[2]))
        spec = FeatureSpec.from_config(cfg)
        assert "ema_ratio_9" in spec.feature_names
        assert "ret_2" in spec.feature_names
        assert "ret_1" not in spec.feature_names

    def test_warmup_positive_and_manifest_roundtrip(self, spec):
        assert spec.warmup_bars >= 2
        manifest = spec.to_manifest()
        restored = FeatureSpec.from_manifest(manifest)
        assert restored.feature_names == spec.feature_names
        assert restored.warmup_bars == spec.warmup_bars


class TestBuilder:
    def test_columns_match_spec_exactly(self, spec, frame):
        assert list(frame.columns) == list(spec.feature_names)

    def test_warmup_rows_are_nan_then_clean(self, spec, frame):
        assert frame.iloc[0].isna().any()
        tail = frame.iloc[spec.warmup_bars:]
        assert tail.notna().all().all(), "после warmup NaN быть не должно"

    def test_rejects_non_final_bars(self, spec, bars):
        mutated = list(bars)
        mutated[10] = replace(mutated[10], is_final=False)
        with pytest.raises(ValueError, match="финальные"):
            FeatureBuilder(spec).build(mutated)

    def test_rejects_unsorted(self, spec, bars):
        mutated = list(bars)
        mutated[5], mutated[6] = mutated[6], mutated[5]
        with pytest.raises(ValueError, match="упорядочены"):
            FeatureBuilder(spec).build(mutated)

    def test_no_inf_values(self, frame):
        values = frame.to_numpy(dtype="float64")
        assert not np.isinf(values[~np.isnan(values)]).any()

    def test_time_features_local_timezone(self, spec):
        # бар в 07:00 UTC = 10:00 Europe/Moscow, четверг (weekday=3)
        bar_ts = datetime(2024, 1, 4, 7, 0, tzinfo=UTC)
        bars = generate_synthetic_bars(80, seed=3, start=bar_ts, timeframe="5m")
        frame = FeatureBuilder(spec, local_timezone="Europe/Moscow").build(bars)
        first = frame.iloc[0]
        local = bars[0].ts_open.astimezone(ZoneInfo("Europe/Moscow"))
        assert first["minute_of_day"] == local.hour * 60 + local.minute
        assert first["day_of_week"] == float(local.weekday())


class TestCausality:
    """Критерий приёмки: признаки в момент t не используют данные позже t."""

    def test_prefix_invariance(self, spec, bars, frame):
        builder = FeatureBuilder(spec, local_timezone="Europe/Moscow")
        for cut in (spec.warmup_bars + 5, spec.warmup_bars + 60, len(bars) - 37):
            prefix_frame = builder.build(bars[:cut])
            common = frame.iloc[:cut]
            pd.testing.assert_frame_equal(
                prefix_frame,
                common,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_future_perturbation_does_not_change_past_features(self, spec, bars, frame):
        builder = FeatureBuilder(spec, local_timezone="Europe/Moscow")
        rng = np.random.RandomState(5)
        for t in (spec.warmup_bars + 2, 150, len(bars) - 30):
            perturbed = list(bars)
            for i in range(t + 1, len(perturbed)):
                factor = 1.0 + rng.uniform(-0.2, 0.2)
                b = perturbed[i]
                perturbed[i] = replace(
                    b,
                    open=b.open * factor,
                    high=max(b.high * factor, b.open * factor, b.close * factor),
                    low=min(b.low * factor, b.open * factor, b.close * factor),
                    close=b.close * factor,
                    volume=int(b.volume * rng.uniform(0.2, 5.0)) + 1,
                )
            perturbed_frame = builder.build(perturbed)
            left = frame.iloc[: t + 1].to_numpy(dtype="float64")
            right = perturbed_frame.iloc[: t + 1].to_numpy(dtype="float64")
            assert np.allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True), (
                f"признаки в t={t} изменились после возмущения будущего — утечка!"
            )

    def test_appending_future_bars_keeps_history(self, spec, bars, frame):
        builder = FeatureBuilder(spec, local_timezone="Europe/Moscow")
        extended = builder.build(list(bars) + generate_synthetic_bars(50, seed=99, start=bars[-1].ts_open + timedelta(minutes=5), timeframe="5m"))
        pd.testing.assert_frame_equal(
            extended.iloc[: len(bars)], frame, check_exact=False, rtol=1e-12, atol=1e-12
        )


class TestIndicatorValues:
    def test_log_return_known(self):
        close = pd.Series([100.0, 110.0, 99.0])
        result = log_return(close, 1)
        assert math.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(math.log(1.1))
        assert result.iloc[2] == pytest.approx(math.log(0.9))

    def test_rsi_monotonic_series_is_100(self):
        close = pd.Series(np.linspace(100.0, 200.0, 60))
        rsi = rsi_wilder(close, 14)
        assert rsi.iloc[-1] == pytest.approx(100.0)
        assert rsi.iloc[:14].isna().all()

    def test_rsi_bounds(self, frame, spec):
        rsi_col = f"rsi_{spec.params['rsi_period']}"
        values = frame[rsi_col].dropna().to_numpy()
        assert ((values >= 0.0) & (values <= 100.0)).all()

    def test_atr_ratio_positive(self, frame, spec):
        values = frame[f"atr_ratio_{spec.params['atr_period']}"].dropna().to_numpy()
        assert (values > 0.0).all()

    def test_close_position_bounds(self, frame):
        values = frame["close_position"].dropna().to_numpy()
        assert ((values >= 0.0) & (values <= 1.0)).all()
