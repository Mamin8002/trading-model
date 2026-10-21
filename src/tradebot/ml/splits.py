"""Хронологическое purged walk-forward разбиение (docs/decisions.md D6, docs/ml.md §4).

Схема (expanding window, строго по времени, без перемешивания):

    train_k = [0, val_start_k - embargo)          <- обучающее окно растёт
    val_k   = [val_start_k, val_end_k)            <- проверочное окно идёт вперёд

    embargo = max(embargo_bars or 0, horizon_bars)

Гарантия отсутствия пересечения целевых горизонтов: метка обучающего образца t
использует close[t + H]; при embargo >= H для любого t из train_k:
t + H <= (val_start_k - embargo - 1) + H < val_start_k — т.е. ни одна обучающая
метка не заходит в проверочное окно.

Горизонт H измеряется В БАРАХ серии (индексное пространство), поэтому разрывы
сессий/выходные не ломают логику очистки: метка всегда определена как shift(-H)
по позициям той же серии.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import InsufficientHistoryError

__all__ = ["WalkForwardFold", "purged_walk_forward"]


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_idx: np.ndarray  # позиции в датасете (строки), возрастают
    val_idx: np.ndarray
    embargo_bars: int
    horizon_bars: int

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)


def purged_walk_forward(
    n_samples: int,
    *,
    n_splits: int,
    horizon_bars: int,
    embargo_bars: int | None = None,
    min_train_bars: int = 500,
    min_val_bars: int = 20,
) -> list[WalkForwardFold]:
    """Строит фолды. Бросает InsufficientHistoryError, если истории не хватает.

    Последний фолд продлевается до конца данных (иначе хвост пропадает);
    остальные валидационные блоки равной длины.
    """
    if n_splits < 2:
        raise ValueError("n_splits >= 2 (иначе это не walk-forward)")
    if horizon_bars < 1:
        raise ValueError("horizon_bars >= 1")
    embargo = max(embargo_bars or 0, horizon_bars)

    # сколько строк доступно под валидацию после первого train-окна и embargo
    available = n_samples - min_train_bars - embargo
    val_block = available // n_splits if available > 0 else 0
    required = min_train_bars + embargo + n_splits * min_val_bars
    if val_block < min_val_bars:
        raise InsufficientHistoryError(
            f"Истории недостаточно для честной walk-forward валидации: доступно {n_samples} "
            f"строк датасета, требуется минимум {required} "
            f"(min_train_bars={min_train_bars} + embargo={embargo} + {n_splits}×min_val_bars={min_val_bars}). "
            "Увеличьте историю, уменьшите горизонт/embargo или число фолдов. "
            "Метрики на обучающей выборке НЕ являются оценкой качества и не приводятся.",
            required_bars=required,
            available_bars=n_samples,
        )

    folds: list[WalkForwardFold] = []
    for k in range(n_splits):
        val_start = min_train_bars + embargo + k * val_block
        val_end = val_start + val_block if k < n_splits - 1 else n_samples
        train_end = val_start - embargo  # исключительно
        train_idx = np.arange(0, train_end, dtype=np.int64)
        val_idx = np.arange(val_start, val_end, dtype=np.int64)
        if len(train_idx) < min_train_bars:  # защита от ошибки арифметики
            raise InsufficientHistoryError(
                f"Фолд {k}: обучающее окно {len(train_idx)} < min_train_bars={min_train_bars}",
                required_bars=min_train_bars,
                available_bars=len(train_idx),
            )
        folds.append(
            WalkForwardFold(
                fold_id=k,
                train_idx=train_idx,
                val_idx=val_idx,
                embargo_bars=embargo,
                horizon_bars=horizon_bars,
            )
        )
    return folds


def assert_no_label_overlap(fold: WalkForwardFold) -> None:
    """Проверочный инвариант (используется тестами и аудитом):

    max(train) + horizon < min(val)  =>  целевые окна не пересекаются.
    """
    if len(fold.train_idx) == 0 or len(fold.val_idx) == 0:
        raise AssertionError(f"Фолд {fold.fold_id}: пустое окно")
    max_train = int(fold.train_idx.max())
    min_val = int(fold.val_idx.min())
    if max_train + fold.horizon_bars >= min_val:
        raise AssertionError(
            f"Фолд {fold.fold_id}: пересечение целевых горизонтов "
            f"(max_train={max_train} + H={fold.horizon_bars} >= min_val={min_val})"
        )
