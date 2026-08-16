import unittest

import numpy as np
import pandas as pd

from qlib_factor_lab import (
    combine_factors,
    cross_sectional_ic,
    stability_adjusted_weights,
)


class StabilityTests(unittest.TestCase):
    def test_cross_sectional_ic_recovers_rank_direction(self):
        dates = pd.date_range("2024-01-01", periods=3)
        instruments = list("ABCDE")
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        base = np.tile(np.arange(5, dtype=float), len(dates))
        features = pd.DataFrame({"positive": base, "negative": -base}, index=index)
        labels = pd.Series(base, index=index)

        ic = cross_sectional_ic(features, labels, min_assets=5)

        self.assertTrue(np.allclose(ic["positive"], 1.0))
        self.assertTrue(np.allclose(ic["negative"], -1.0))

    def test_weights_respect_label_availability_lag(self):
        dates = pd.date_range("2024-01-01", periods=8)
        ic = pd.DataFrame({"f1": [0.1] * 8, "f2": [0.05] * 8}, index=dates)
        baseline = stability_adjusted_weights(
            ic,
            window=4,
            min_periods=3,
            information_lag=2,
            uncertainty_penalty=0.0,
        )

        changed = ic.copy()
        changed.loc[dates[5] :, "f1"] = -100.0
        revised = stability_adjusted_weights(
            changed,
            window=4,
            min_periods=3,
            information_lag=2,
            uncertainty_penalty=0.0,
        )

        pd.testing.assert_series_equal(baseline.loc[dates[5]], revised.loc[dates[5]])
        pd.testing.assert_series_equal(baseline.loc[dates[6]], revised.loc[dates[6]])
        self.assertFalse(baseline.loc[dates[7]].equals(revised.loc[dates[7]]))

    def test_information_lag_must_be_positive_integer(self):
        dates = pd.date_range("2024-01-01", periods=5)
        ic = pd.DataFrame({"f1": 0.1}, index=dates)
        with self.assertRaises(ValueError):
            stability_adjusted_weights(ic, window=3, min_periods=2, information_lag=0)

    def test_stable_factor_receives_more_weight_than_noisy_factor(self):
        dates = pd.date_range("2024-01-01", periods=30)
        ic = pd.DataFrame(
            {
                "stable": [0.05] * 30,
                "noisy": [0.30, -0.20] * 15,
            },
            index=dates,
        )

        weights = stability_adjusted_weights(ic, window=20, min_periods=10, uncertainty_penalty=1.0)

        self.assertGreater(abs(weights.iloc[-1]["stable"]), abs(weights.iloc[-1]["noisy"]))
        self.assertAlmostEqual(weights.iloc[-1].abs().sum(), 1.0)

    def test_turnover_penalty_reduces_expensive_factor_weight(self):
        dates = pd.date_range("2024-01-01", periods=20)
        ic = pd.DataFrame({"cheap": 0.05, "expensive": 0.05}, index=dates)
        turnover = pd.DataFrame({"cheap": 0.1, "expensive": 2.0}, index=dates)

        weights = stability_adjusted_weights(
            ic,
            window=10,
            min_periods=5,
            uncertainty_penalty=0.0,
            turnover=turnover,
            turnover_penalty=1.0,
        )

        self.assertGreater(weights.iloc[-1]["cheap"], weights.iloc[-1]["expensive"])

    def test_combine_factors_aligns_date_weights(self):
        dates = pd.date_range("2024-01-01", periods=2)
        index = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C"]], names=["datetime", "instrument"]
        )
        features = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
                "f2": [3.0, 2.0, 1.0] * 2,
            },
            index=index,
        )
        weights = pd.DataFrame({"f1": [1.0, 0.0], "f2": [0.0, 1.0]}, index=dates)

        score = combine_factors(features, weights)

        self.assertGreater(score.loc[(dates[0], "C")], score.loc[(dates[0], "A")])
        self.assertGreater(score.loc[(dates[1], "A")], score.loc[(dates[1], "C")])


if __name__ == "__main__":
    unittest.main()
