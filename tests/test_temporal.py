import unittest

import pandas as pd

from qlib_factor_lab import observable_ic_cutoff


class TemporalBoundaryTests(unittest.TestCase):
    def test_alpha158_two_session_cutoff(self):
        dates = pd.to_datetime(["2016-12-28", "2016-12-29", "2016-12-30", "2017-01-03"])
        cutoff = observable_ic_cutoff(dates, "2017-01-03", information_lag=2)
        self.assertEqual(cutoff, pd.Timestamp("2016-12-29"))

    def test_cutoff_uses_trading_rows_not_calendar_days(self):
        dates = pd.to_datetime(["2024-02-07", "2024-02-08", "2024-02-19"])
        cutoff = observable_ic_cutoff(dates, "2024-02-19", information_lag=2)
        self.assertEqual(cutoff, pd.Timestamp("2024-02-07"))

    def test_insufficient_history_is_rejected(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        with self.assertRaises(ValueError):
            observable_ic_cutoff(dates, "2024-01-02", information_lag=2)


if __name__ == "__main__":
    unittest.main()
