"""Machine-readable research promotion gate.

The gate answers a narrow question: whether an evaluated candidate has enough
out-of-sample evidence to proceed to a higher-fidelity research stage.  It is
not a claim of production readiness or expected investment return.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evaluate_research_gate(
    candidate: Mapping[str, Mapping[str, float | int]],
    controls: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    thresholds: Mapping[str, float | int | bool],
) -> dict[str, Any]:
    """Evaluate absolute and relative evidence requirements.

    Parameters use the same ``signal`` and ``portfolio`` schema emitted by the
    experiment runner.  Every check is returned with its observed value,
    threshold and pass/fail state so the final decision is auditable.
    """

    signal = candidate["signal"]
    portfolio = candidate["portfolio"]
    control_returns = [
        float(item["portfolio"]["annualized_excess_return_after_cost"])
        for item in controls.values()
    ]
    control_irs = [
        float(item["portfolio"]["information_ratio_after_cost"])
        for item in controls.values()
    ]
    best_control_return = max(control_returns)
    best_control_ir = max(control_irs)

    observed = {
        "test_days": int(signal["trading_days"]),
        "rank_ic": float(signal["rank_ic_mean"]),
        "rank_icir": float(signal["rank_icir_annualized"]),
        "after_cost_annualized_excess": float(
            portfolio["annualized_excess_return_after_cost"]
        ),
        "after_cost_information_ratio": float(portfolio["information_ratio_after_cost"]),
        "max_drawdown": float(portfolio["max_drawdown_after_cost"]),
        "return_advantage_vs_best_control": float(
            portfolio["annualized_excess_return_after_cost"]
        )
        - best_control_return,
        "ir_advantage_vs_best_control": float(portfolio["information_ratio_after_cost"])
        - best_control_ir,
    }
    limits = {
        "test_days": int(thresholds["minimum_test_days"]),
        "rank_ic": float(thresholds["minimum_rank_ic"]),
        "rank_icir": float(thresholds["minimum_rank_icir"]),
        "after_cost_annualized_excess": float(
            thresholds["minimum_after_cost_annualized_excess"]
        ),
        "after_cost_information_ratio": float(
            thresholds["minimum_after_cost_information_ratio"]
        ),
        "max_drawdown": float(thresholds["maximum_drawdown_floor"]),
        "return_advantage_vs_best_control": float(
            thresholds["minimum_return_advantage_vs_control"]
        ),
        "ir_advantage_vs_best_control": float(thresholds["minimum_ir_advantage_vs_control"]),
    }

    checks = {
        "test_days": observed["test_days"] >= limits["test_days"],
        "rank_ic": observed["rank_ic"] >= limits["rank_ic"],
        "rank_icir": observed["rank_icir"] >= limits["rank_icir"],
        "after_cost_annualized_excess": observed["after_cost_annualized_excess"]
        >= limits["after_cost_annualized_excess"],
        "after_cost_information_ratio": observed["after_cost_information_ratio"]
        >= limits["after_cost_information_ratio"],
        "max_drawdown": observed["max_drawdown"] >= limits["max_drawdown"],
        "return_advantage_vs_best_control": observed["return_advantage_vs_best_control"]
        >= limits["return_advantage_vs_best_control"],
        "ir_advantage_vs_best_control": observed["ir_advantage_vs_best_control"]
        >= limits["ir_advantage_vs_best_control"],
    }
    detail = {
        key: {"observed": observed[key], "threshold": limits[key], "passed": passed}
        for key, passed in checks.items()
    }
    return {
        "gate": "research_promotion_v1",
        "scope": "Promotion to a higher-fidelity research stage; not production approval",
        "decision": "PASS" if all(checks.values()) else "BLOCK",
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "checks": detail,
    }
