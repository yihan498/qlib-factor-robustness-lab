"""Leakage-aware factor stability utilities.

The functions in this module are original project code. They accept ordinary
pandas objects so the research method can be tested independently before it is
connected to a specific Qlib dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_panel(features: pd.DataFrame, labels: pd.Series) -> None:
    if not isinstance(features.index, pd.MultiIndex) or features.index.nlevels != 2:
        raise ValueError("features must use a two-level (datetime, instrument) index")
    if not isinstance(labels.index, pd.MultiIndex) or labels.index.nlevels != 2:
        raise ValueError("labels must use a two-level (datetime, instrument) index")
    if features.columns.empty:
        raise ValueError("features must contain at least one factor column")
    if features.columns.has_duplicates:
        raise ValueError("factor names must be unique")


def cross_sectional_ic(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    method: str = "spearman",
    min_assets: int = 5,
) -> pd.DataFrame:
    """Calculate daily cross-sectional information coefficients.

    Parameters
    ----------
    features:
        Factor values indexed by ``(datetime, instrument)``.
    labels:
        Forward return or another target on the same index.
    method:
        ``"spearman"`` for rank IC or ``"pearson"`` for linear IC.
    min_assets:
        Minimum number of valid instruments required for a daily estimate.
    """

    _validate_panel(features, labels)
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    if min_assets < 2:
        raise ValueError("min_assets must be at least 2")

    aligned = features.join(labels.rename("__label__"), how="inner")

    def daily_ic(group: pd.DataFrame) -> pd.Series:
        left = group[features.columns]
        right = group["__label__"]
        if method == "spearman":
            left = left.rank(method="average")
            right = right.rank(method="average")
        valid_count = left.notna().mul(right.notna(), axis=0).sum(axis=0)
        values = left.corrwith(right, axis=0)
        return values.where(valid_count >= min_assets)

    result = aligned.groupby(level=0, sort=True).apply(daily_ic)
    result.index.name = features.index.names[0] or "datetime"
    return result


def stability_adjusted_weights(
    ic: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: int = 20,
    information_lag: int = 1,
    uncertainty_penalty: float = 1.0,
    turnover: pd.DataFrame | None = None,
    turnover_penalty: float = 0.0,
) -> pd.DataFrame:
    """Build dynamic factor weights using only information available before t.

    The raw score subtracts an estimation-uncertainty penalty from the absolute
    trailing mean IC. An optional trailing turnover penalty discourages costly
    factors. Absolute weights are normalized to one on every date.

    ``information_lag`` must match the time at which the label used to compute
    IC becomes observable.  For example, Qlib Alpha158 predicts the T+1 to T+2
    return, so a close-of-T decision requires ``information_lag=2``.
    """

    if window < 2:
        raise ValueError("window must be at least 2")
    if not 2 <= min_periods <= window:
        raise ValueError("min_periods must be between 2 and window")
    if uncertainty_penalty < 0 or turnover_penalty < 0:
        raise ValueError("penalties must be non-negative")
    if not isinstance(information_lag, int) or information_lag < 1:
        raise ValueError("information_lag must be a positive integer")
    if ic.columns.empty or ic.columns.has_duplicates:
        raise ValueError("ic must contain unique factor columns")

    past_ic = ic.shift(information_lag)
    rolling = past_ic.rolling(window=window, min_periods=min_periods)
    mean_ic = rolling.mean()
    observations = rolling.count()
    standard_error = rolling.std(ddof=1) / np.sqrt(observations)

    confidence_adjusted = (mean_ic.abs() - uncertainty_penalty * standard_error).clip(lower=0.0)
    score = np.sign(mean_ic) * confidence_adjusted

    if turnover is not None:
        aligned_turnover = turnover.reindex(index=ic.index, columns=ic.columns)
        past_turnover = (
            aligned_turnover.shift(information_lag)
            .rolling(window=window, min_periods=min_periods)
            .mean()
        )
        score = score / (1.0 + turnover_penalty * past_turnover.clip(lower=0.0))

    denominator = score.abs().sum(axis=1).replace(0.0, np.nan)
    return score.div(denominator, axis=0).fillna(0.0)


def combine_factors(features: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Create a daily composite score from cross-sectionally standardized factors."""

    if not isinstance(features.index, pd.MultiIndex) or features.index.nlevels != 2:
        raise ValueError("features must use a two-level (datetime, instrument) index")
    missing = features.columns.difference(weights.columns)
    if not missing.empty:
        raise ValueError(f"weights are missing factors: {missing.tolist()}")

    dates = features.index.get_level_values(0)
    aligned_weights = weights.reindex(dates).set_axis(features.index)

    grouped = features.groupby(level=0)
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0.0, np.nan)
    standardized = (features - means) / stds

    composite = (standardized * aligned_weights[features.columns]).sum(axis=1, min_count=1)
    composite.name = "factor_score"
    return composite
