import unittest

from qlib_factor_lab import evaluate_research_gate


def result(rank_ic=0.02, icir=1.0, annualized=0.08, ir=0.8, drawdown=-0.15, days=700):
    return {
        "signal": {
            "rank_ic_mean": rank_ic,
            "rank_icir_annualized": icir,
            "trading_days": days,
        },
        "portfolio": {
            "annualized_excess_return_after_cost": annualized,
            "information_ratio_after_cost": ir,
            "max_drawdown_after_cost": drawdown,
        },
    }


THRESHOLDS = {
    "minimum_test_days": 500,
    "minimum_rank_ic": 0.0,
    "minimum_rank_icir": 0.5,
    "minimum_after_cost_annualized_excess": 0.0,
    "minimum_after_cost_information_ratio": 0.0,
    "maximum_drawdown_floor": -0.30,
    "minimum_return_advantage_vs_control": 0.0,
    "minimum_ir_advantage_vs_control": 0.0,
}


class ResearchGateTests(unittest.TestCase):
    def test_candidate_passes_when_all_absolute_and_relative_checks_pass(self):
        gate = evaluate_research_gate(
            result(),
            {"static": result(annualized=0.02, ir=0.2)},
            THRESHOLDS,
        )

        self.assertEqual(gate["decision"], "PASS")
        self.assertEqual(gate["passed_checks"], gate["total_checks"])

    def test_candidate_is_blocked_with_explicit_failure_reasons(self):
        candidate = result(annualized=-0.12, ir=-1.0, drawdown=-0.45)
        gate = evaluate_research_gate(
            candidate,
            {"static": result(annualized=-0.05, ir=-0.4)},
            THRESHOLDS,
        )

        self.assertEqual(gate["decision"], "BLOCK")
        self.assertFalse(gate["checks"]["after_cost_annualized_excess"]["passed"])
        self.assertFalse(gate["checks"]["return_advantage_vs_best_control"]["passed"])


if __name__ == "__main__":
    unittest.main()
