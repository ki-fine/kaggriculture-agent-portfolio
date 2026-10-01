"""Prevent incomplete grids, seat errors and health laundering in reports."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analyze_results


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "data/r7_cross24.json").read_text(encoding="utf-8"))

    def test_complete_grid_keeps_original_fail_and_both_summaries(self):
        report = analyze_results.analyze_cross(self.data)
        self.assertEqual(report["completed_games"], 24)
        self.assertEqual(report["unique_world_seeds"], 2)
        self.assertEqual(report["strict_full_panel_health"], "FAIL")
        self.assertEqual(report["strict_healthy_games"], 22)
        self.assertEqual(report["candidate_strict_healthy_games"], 12)
        self.assertEqual(report["raw_all_pairs_including_flagged_controls"]["mean_margin_delta"], 961)
        self.assertEqual(report["strict_clean_pairs_diagnostic_only"]["mean_margin_delta"], 1072.4)
        self.assertEqual(report["official_cash_checked_transitions"], 17256)
        self.assertEqual(len(report["retained_flagged_games"]), 2)

    def test_missing_pair_is_rejected(self):
        self.data["rows"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            analyze_results.analyze_cross(self.data)

    def test_duplicate_pair_cannot_replace_missing_pair(self):
        self.data["rows"][-1] = copy.deepcopy(self.data["rows"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze_results.analyze_cross(self.data)

    def test_money_is_aligned_by_seat(self):
        row = next(r for r in self.data["rows"] if r["seat"] == 1
                   and r["candidate"]["own_cash"] != r["candidate"]["rival_cash"])
        row["candidate"]["cash_audit"]["rewards_in_player_order"].reverse()
        with self.assertRaisesRegex(ValueError, "seat-aligned"):
            analyze_results.analyze_cross(self.data)

    def test_cash_residual_cannot_be_ignored_for_flagged_control(self):
        row = next(r for r in self.data["rows"] if not r["control"]["strict_healthy"])
        row["control"]["cash_audit"]["cash_residuals"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "Nonzero"):
            analyze_results.analyze_cross(self.data)

    def test_done_status_does_not_launder_layer_errors(self):
        row = next(r for r in self.data["rows"] if not r["control"]["strict_healthy"])
        row["control"]["strict_healthy"] = True
        with self.assertRaisesRegex(ValueError, "flagged game"):
            analyze_results.analyze_cross(self.data)

    def test_independent_margin_includes_opponent_cash(self):
        rows = [{"candidate": {"own_cash": 150, "rival_cash": 140},
                 "control": {"own_cash": 100, "rival_cash": 50}}]
        result = analyze_results.summarize(rows)
        self.assertEqual(result["mean_own_cash_delta"], 50)
        self.assertEqual(result["mean_rival_cash_delta"], 90)
        self.assertEqual(result["mean_margin_delta"], -40)

    def test_runtime_only_ablation_reproduces_action_cash_and_ratio(self):
        data = json.loads((ROOT / "data/r7_performance.json").read_text(encoding="utf-8"))
        report = analyze_results.analyze_performance(data)
        self.assertEqual(report["pairs"], 16)
        self.assertEqual(report["actions_exact_pairs"], 16)
        self.assertEqual(report["final_cash_exact_pairs"], 16)
        self.assertAlmostEqual(report["callback_total_reduction_percent"], 15.56497942052834)
        self.assertAlmostEqual(report["later_callback_reduction_percent"], 17.35615733381362)
        analyze_results.verify_extract_hashes(ROOT / "data")
        analyze_results.check_reproduction(self.data, data,
                                          analyze_results.analyze_cross(self.data), report)


if __name__ == "__main__":
    unittest.main()
