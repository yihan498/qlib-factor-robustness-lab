"""Leakage-controlled personal signal pipeline with Qlib execution backtest."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data.dataset.handler import DataHandlerLP

from qlib_factor_lab import (
    combine_factors,
    cross_sectional_ic,
    evaluate_research_gate,
    observable_ic_cutoff,
    stability_adjusted_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/personal_signal_v2.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/portfolio_v2"))
    return parser.parse_args()


def normalize(weights: pd.DataFrame) -> pd.DataFrame:
    denominator = weights.abs().sum(axis=1).replace(0.0, np.nan)
    return weights.div(denominator, axis=0).fillna(0.0)


def rank_ic_summary(score: pd.Series, labels: pd.Series, min_assets: int) -> dict[str, float | int]:
    daily = cross_sectional_ic(score.to_frame("score"), labels, min_assets=min_assets)[
        "score"
    ].dropna()
    std = daily.std(ddof=1)
    return {
        "rank_ic_mean": float(daily.mean()),
        "rank_icir_annualized": float(daily.mean() / std * np.sqrt(252)) if std else float("nan"),
        "trading_days": len(daily),
    }


def qlib_backtest(
    score: pd.Series, config: dict, benchmark: str, output_dir: Path, name: str
) -> dict[str, float | int]:
    backtest_config = config["backtest"]
    strategy = TopkDropoutStrategy(
        signal=score.dropna(),
        topk=int(backtest_config["topk"]),
        n_drop=int(backtest_config["n_drop"]),
    )
    dates = score.index.get_level_values("datetime")
    report, _ = backtest_daily(
        start_time=dates.min(),
        end_time=dates.max(),
        strategy=strategy,
        account=float(backtest_config["account"]),
        benchmark=benchmark,
        exchange_kwargs=backtest_config["exchange_kwargs"],
    )
    report.to_csv(output_dir / f"backtest_{name}.csv", encoding="utf-8-sig")
    excess_before_cost = report["return"] - report["bench"]
    excess_after_cost = excess_before_cost - report["cost"]
    before = risk_analysis(excess_before_cost, N=252, freq=None)["risk"]
    after = risk_analysis(excess_after_cost, N=252, freq=None)["risk"]
    return {
        "annualized_excess_return_before_cost": float(before["annualized_return"]),
        "annualized_excess_return_after_cost": float(after["annualized_return"]),
        "information_ratio_after_cost": float(after["information_ratio"]),
        "max_drawdown_after_cost": float(after["max_drawdown"]),
        "average_turnover": float(report["turnover"].mean()),
        "average_daily_cost": float(report["cost"].mean()),
        "backtest_days": len(report),
    }


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    args.output.mkdir(parents=True, exist_ok=True)

    periods = config["periods"]
    train_start, train_end = map(str, periods["train"])
    validation_start, validation_end = map(str, periods["validation"])
    test_start, test_end = map(str, periods["test"])
    lag = int(config["label_information_lag"])
    min_assets = int(config["minimum_assets_per_ic"])

    qlib.init(
        provider_uri=str(Path(config["data_path"]).resolve()),
        region="cn",
        kernels=int(config.get("runtime_kernels", 4)),
    )
    handler = Alpha158(
        instruments=config["universe"],
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
    )
    features = (
        handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).sort_index().astype("float32")
    )
    labels = (
        handler.fetch(col_set="label", data_key=DataHandlerLP.DK_I)
        .sort_index()
        .iloc[:, 0]
        .rename("label")
    )
    all_ic = cross_sectional_ic(features, labels, min_assets=min_assets)
    calendar = pd.DatetimeIndex(all_ic.index)
    train_cutoff = observable_ic_cutoff(calendar, validation_start, lag)
    validation_cutoff = observable_ic_cutoff(calendar, test_start, lag)

    train_ic = all_ic.loc[train_start:train_cutoff]
    selected = train_ic.mean().abs().nlargest(int(config["top_factors"])).index.tolist()
    features = features[selected]
    ic = all_ic[selected]

    train_mean = train_ic[selected].mean()
    fixed_row = np.sign(train_mean) / len(selected)
    fixed_weights = pd.DataFrame(
        np.tile(fixed_row.to_numpy(), (len(ic), 1)), index=ic.index, columns=selected
    )

    validation_slice = slice(validation_start, validation_cutoff)
    validation_features = features.loc[validation_slice]
    validation_labels = labels.loc[validation_slice]
    candidates: list[dict] = []
    weight_cache: dict[tuple[int, float], pd.DataFrame] = {}
    grid = config["candidate_grid"]
    for window, penalty in itertools.product(grid["windows"], grid["uncertainty_penalties"]):
        key = (int(window), float(penalty))
        dynamic = stability_adjusted_weights(
            ic,
            window=key[0],
            min_periods=min(20, key[0]),
            information_lag=lag,
            uncertainty_penalty=key[1],
        )
        weight_cache[key] = dynamic
        for shrinkage in grid["shrinkage_to_dynamic"]:
            blended = normalize(
                (1.0 - float(shrinkage)) * fixed_weights + float(shrinkage) * dynamic
            )
            score = combine_factors(validation_features, blended.loc[validation_slice])
            metrics = rank_ic_summary(score, validation_labels, min_assets)
            candidates.append(
                {
                    "window": key[0],
                    "uncertainty_penalty": key[1],
                    "shrinkage_to_dynamic": float(shrinkage),
                    **metrics,
                }
            )
    best = max(candidates, key=lambda item: item["rank_icir_annualized"])
    best_dynamic = weight_cache[(best["window"], best["uncertainty_penalty"])]
    adaptive_weights = normalize(
        (1.0 - best["shrinkage_to_dynamic"]) * fixed_weights
        + best["shrinkage_to_dynamic"] * best_dynamic
    )
    rolling_weights = stability_adjusted_weights(
        ic,
        window=60,
        min_periods=20,
        information_lag=lag,
        uncertainty_penalty=0.0,
    )

    test_slice = slice(test_start, test_end)
    test_features = features.loc[test_slice]
    test_labels = labels.loc[test_slice]
    methods = {
        "fixed_equal_signed": fixed_weights,
        "rolling_ic_60": rolling_weights,
        "adaptive_shrinkage": adaptive_weights,
    }
    results = {}
    for name, weights in methods.items():
        score = combine_factors(test_features, weights.loc[test_slice]).sort_index()
        score.to_frame("score").to_csv(args.output / f"signal_{name}.csv", encoding="utf-8-sig")
        results[name] = {
            "signal": rank_ic_summary(score, test_labels, min_assets),
            "portfolio": qlib_backtest(score, config, config["benchmark"], args.output, name),
        }

    promotion_gate = evaluate_research_gate(
        results["adaptive_shrinkage"],
        {
            "fixed_equal_signed": results["fixed_equal_signed"],
            "rolling_ic_60": results["rolling_ic_60"],
        },
        config["research_promotion_gate"],
    )

    payload = {
        "status": "valid_v2",
        "runtime": {
            "python": platform.python_version(),
            "pyqlib": qlib.__version__,
            "lightgbm": lightgbm.__version__,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "data_manifest": {
            "source": "Qlib-packaged downloader using the community SunsetWolf/qlib_dataset release; underlying example prices are described as Yahoo Finance data",
            "universe": config["universe"],
            "feature_rows": len(features),
            "instruments": int(features.index.get_level_values("instrument").nunique()),
            "first_date": str(features.index.get_level_values("datetime").min().date()),
            "last_date": str(features.index.get_level_values("datetime").max().date()),
        },
        "temporal_controls": {
            "label": "Ref($close, -2)/Ref($close, -1)-1 (T+1 to T+2 return)",
            "information_lag_trading_sessions": lag,
            "train_requested_end": train_end,
            "train_effective_ic_cutoff": str(train_cutoff.date()),
            "validation_requested_end": validation_end,
            "validation_effective_ic_cutoff": str(validation_cutoff.date()),
            "test_period": [test_start, test_end],
        },
        "selected_factors": selected,
        "validation_candidates": candidates,
        "selected_adaptive_parameters": best,
        "test_results": results,
        "research_promotion_gate": promotion_gate,
        "interpretation_rule": "The adaptive method is called an improvement only if it beats both practical baselines on after-cost Qlib backtest metrics; otherwise it remains an engineering experiment.",
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
