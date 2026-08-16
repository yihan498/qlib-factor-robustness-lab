"""Trading-calendar helpers for purged train/validation/test boundaries."""

from __future__ import annotations

import pandas as pd


def observable_ic_cutoff(
    dates: pd.Index, decision_start: str | pd.Timestamp, information_lag: int
) -> pd.Timestamp:
    """Return the latest IC date observable at the first decision timestamp.

    ``information_lag`` is measured in trading rows, not calendar days. With
    Alpha158's T+1-to-T+2 label, a decision on row T can use IC through T-2.
    """

    if not isinstance(information_lag, int) or information_lag < 1:
        raise ValueError("information_lag must be a positive integer")
    ordered = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
    if ordered.empty:
        raise ValueError("dates must not be empty")
    position = int(ordered.searchsorted(pd.Timestamp(decision_start), side="left"))
    cutoff_position = position - information_lag
    if cutoff_position < 0 or cutoff_position >= len(ordered):
        raise ValueError("insufficient history for the requested decision boundary")
    return pd.Timestamp(ordered[cutoff_position])
