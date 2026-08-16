"""INVALIDATED v1 experiment retained only for audit history.

Do not use its metrics. Alpha158 labels require a two-session availability
lag; the corrected executable pipeline is ``run_portfolio_v2.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP

from qlib_factor_lab import combine_factors, cross_sectional_ic, stability_adjusted_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/qlib/cn_data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/factor_stability"))
    parser.add_argument("--top-factors", type=int, default=30)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--min-periods", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    return parser.parse_args()


def score_metrics(score: pd.Series, label: pd.Series, cost_bps: float) -> dict[str, float]:
    frame = pd.concat([score.rename("score"), label.rename("label")], axis=1).dropna()
    daily_ic = frame.groupby(level="datetime").apply(
        lambda x: x["score"].rank().corr(x["label"].rank()), include_groups=False
    )

    def daily_portfolio(group: pd.DataFrame) -> pd.Series:
        count = max(1, int(len(group) * 0.1))
        ordered = group.sort_values("score")
        spread = ordered.tail(count)["label"].mean() - ordered.head(count)["label"].mean()
        top = tuple(
            sorted(ordered.tail(min(50, len(ordered))).index.get_level_values("instrument"))
        )
        return pd.Series({"spread": spread, "top": top})

    daily = frame.groupby(level="datetime").apply(daily_portfolio, include_groups=False)
    prior = None
    turnover = []
    for holdings in daily["top"]:
        current = set(holdings)
        turnover.append(
            np.nan if prior is None else 1.0 - len(current & prior) / max(1, len(current))
        )
        prior = current
    daily["turnover"] = turnover
    daily["net_spread"] = daily["spread"] - daily["turnover"].fillna(0.0) * cost_bps / 10_000.0
    ic_std = daily_ic.std(ddof=1)
    return {
        "rank_ic_mean": float(daily_ic.mean()),
        "rank_icir_annualized": float(daily_ic.mean() / ic_std * np.sqrt(252))
        if ic_std
        else float("nan"),
        "long_short_return_annualized": float(daily["spread"].mean() * 252),
        "cost_adjusted_long_short_annualized": float(daily["net_spread"].mean() * 252),
        "top50_turnover_mean": float(daily["turnover"].mean()),
        "observations": int(daily_ic.notna().sum()),
    }


def main() -> None:
    raise SystemExit("v1 is invalidated; run scripts/run_portfolio_v2.py")
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(args.data.resolve()), region="cn")
    handler = Alpha158(
        instruments="csi300",
        start_time="2014-01-01",
        end_time="2020-08-01",
        fit_start_time="2014-01-01",
        fit_end_time="2016-12-31",
    )
    # Features are unchanged by Alpha158's default learning processors, while
    # DK_L labels are cross-sectionally z-scored for model fitting.  Portfolio
    # return diagnostics must use the raw DK_I forward-return label.
    features = (
        handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_L).sort_index().astype("float32")
    )
    labels = (
        handler.fetch(col_set="label", data_key=DataHandlerLP.DK_I)
        .sort_index()
        .iloc[:, 0]
        .rename("label")
    )

    all_ic = cross_sectional_ic(features, labels, min_assets=30)
    train_ic = all_ic.loc[:"2016-12-31"]
    selected = train_ic.mean().abs().nlargest(args.top_factors).index.tolist()
    features = features[selected]
    ic = all_ic[selected]

    train_mean = train_ic[selected].mean()
    equal_signed_row = np.sign(train_mean) / max(1, len(selected))
    fixed_weights = pd.DataFrame(
        np.tile(equal_signed_row.to_numpy(), (len(ic), 1)), index=ic.index, columns=selected
    )
    rolling_weights = stability_adjusted_weights(
        ic, window=args.window, min_periods=args.min_periods, uncertainty_penalty=0.0
    )
    stability_weights = stability_adjusted_weights(
        ic, window=args.window, min_periods=args.min_periods, uncertainty_penalty=1.0
    )

    test_slice = slice("2017-01-01", "2020-08-01")
    test_features = features.loc[test_slice]
    test_labels = labels.loc[test_slice]
    methods = {
        "fixed_equal_signed": fixed_weights,
        "rolling_ic": rolling_weights,
        "stability_adjusted_ic": stability_weights,
    }
    results = {
        name: score_metrics(
            combine_factors(test_features, weights.loc[test_slice]), test_labels, args.cost_bps
        )
        for name, weights in methods.items()
    }
    payload = {
        "design": {
            "universe": "CSI300",
            "factor_library": "Qlib Alpha158",
            "factor_selection_period": ["2014-01-01", "2016-12-31"],
            "out_of_sample_period": ["2017-01-01", "2020-08-01"],
            "top_factors": args.top_factors,
            "rolling_window": args.window,
            "minimum_history": args.min_periods,
            "assumed_one_way_cost_bps": args.cost_bps,
            "leakage_control": "factor selection uses train only; dynamic weights shift IC by one trading day",
        },
        "selected_factors": selected,
        "results": results,
        "limitations": [
            "Qlib example data is sourced from Yahoo Finance and is not institutional-grade market data.",
            "Long-short returns use the forward-return label and a simplified turnover cost diagnostic, not Qlib execution simulation.",
            "This experiment tests one historical period and does not establish future profitability.",
        ],
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(results).T.to_csv(args.output / "method_comparison.csv", encoding="utf-8-sig")
    stability_weights.loc[test_slice].to_csv(
        args.output / "stability_weights.csv", encoding="utf-8-sig"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
