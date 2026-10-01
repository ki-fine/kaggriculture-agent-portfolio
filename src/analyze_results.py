"""Recompute paired results from portable, sanitized evidence (Python stdlib).

This reads retained evidence summaries; it does not run agents or independently
re-audit unavailable replay archives. A successful --check means reproduction
checks passed. The original full-panel strict health FAIL stays visible.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{name} must be a finite number")
    return value


def validate_grid(data):
    require(data.get("schema_version") == 1, "Unsupported evidence schema")
    expected = set(itertools.product(data["declared_opponents"],
                                     data["declared_seeds"], data["declared_seats"]))
    keys = [(r["opponent"], r["seed"], r["seat"]) for r in data["rows"]]
    require(len(keys) == len(set(keys)), "Duplicate opponent/seed/seat pair")
    require(set(keys) == expected, "Incomplete or unexpected frozen comparison grid")
    require(len(keys) == data["expected_pairs"], "Pair count differs from declaration")
    require(data["expected_games"] == 2 * len(keys), "Each pair must contain two games")
    require(set(data["declared_seats"]) == {0, 1}, "Both player seats are required")
    return data["rows"]


def cash(game):
    own = number(game["own_cash"], "own_cash")
    rival = number(game["rival_cash"], "rival_cash")
    return own, rival, own - rival


def summarize(rows):
    require(bool(rows), "Cannot summarize an empty comparison")
    deltas = []
    own_deltas, rival_deltas = [], []
    for row in rows:
        a, b = cash(row["candidate"]), cash(row["control"])
        own_deltas.append(a[0] - b[0])
        rival_deltas.append(a[1] - b[1])
        deltas.append(a[2] - b[2])
    return {
        "pairs": len(rows), "mean_margin_delta": mean(deltas),
        "min_margin_delta": min(deltas), "max_margin_delta": max(deltas),
        "better_tied_worse": [sum(d > 0 for d in deltas),
                              sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)],
        "mean_own_cash_delta": mean(own_deltas),
        "mean_rival_cash_delta": mean(rival_deltas),
    }


def analyze_cross(data):
    rows = validate_grid(data)
    audited_replays, healthy_games, transitions = set(), 0, 0
    flagged = []
    for row in rows:
        for role in ("candidate", "control"):
            game = row[role]
            require(type(game["strict_healthy"]) is bool, "Health must be an explicit bool")
            require(game["states"] == 720 and game["callbacks"] == [719, 719]
                    and game["status"] == ["DONE", "DONE"], "Incomplete retained game")
            require(type(game["retained_layer_errors"]) is dict, "Missing layer error record")
            require(not game["strict_healthy"] or not game["retained_layer_errors"],
                    "A flagged game cannot be declared strictly healthy")
            healthy_games += game["strict_healthy"]
            if not game["strict_healthy"]:
                flagged.append({"opponent": row["opponent"], "seed": row["seed"],
                                "seat": row["seat"], "role": role,
                                "retained_layer_errors": game["retained_layer_errors"]})
            audit = game["cash_audit"]
            replay = game["replay_sha256"]
            require(replay == audit["replay_sha256"], "Cash audit is for a different replay")
            require(replay not in audited_replays, "Duplicate replay in cash audit coverage")
            audited_replays.add(replay)
            require(audit["checked_transitions"] == 719, "Incomplete cash audit")
            require(audit["cash_residuals"] == [0, 0], "Nonzero official cash audit residual")
            rewards = audit["rewards_in_player_order"]
            require(len(rewards) == 2, "Both players' settlement rewards are required")
            own, rival, _ = cash(game)
            require(rewards[row["seat"]] == own and rewards[1 - row["seat"]] == rival,
                    "Audit rewards do not match seat-aligned final cash")
            transitions += audit["checked_transitions"]
    clean = [r for r in rows if r["candidate"]["strict_healthy"]
             and r["control"]["strict_healthy"]]
    all_healthy = healthy_games == data["expected_games"]
    original_pass = data["original_strict_gate"] == "PASS_NATIVE_ARTIFACTS"
    require(original_pass == all_healthy, "Original strict health gate was changed or contradicted")
    return {
        "panel": data["panel"], "completed_games": 2 * len(rows),
        "unique_world_seeds": len(data["declared_seeds"]),
        "seeds": data["declared_seeds"], "opponent_families": len(data["declared_opponents"]),
        "strict_full_panel_health": "PASS" if all_healthy else "FAIL",
        "strict_healthy_games": healthy_games,
        "candidate_strict_healthy_games": sum(r["candidate"]["strict_healthy"] for r in rows),
        "raw_all_pairs_including_flagged_controls": summarize(rows),
        "strict_clean_pairs_diagnostic_only": summarize(clean) if clean else None,
        "raw_by_opponent": {op: summarize([r for r in rows if r["opponent"] == op])
                            for op in data["declared_opponents"]},
        "official_cash_checked_transitions": transitions,
        "official_cash_residuals_all_zero": True,
        "retained_flagged_games": flagged,
        "evidence_boundary": data["sampling"],
        "audit_boundary": "Checks saved audit coverage and cash identity; does not re-run the engine or re-audit original replays.",
    }


def analyze_performance(data):
    rows = validate_grid(data)
    total_ratios, later_ratios = [], []
    exact_cash, exact_actions, exact_semantics, exact_optimizers = 0, 0, 0, 0
    for r in rows:
        a, b = r["candidate"], r["control"]
        for field, target in (("callback_total_seconds", total_ratios),
                              ("later_callback_seconds", later_ratios)):
            numerator, denominator = number(a[field], field), number(b[field], field)
            require(numerator > 0 and denominator > 0, "Callback durations must be positive")
            target.append(numerator / denominator)
        exact_cash += cash(a) == cash(b)
        for k in ("both_player_actions_exact", "all_callback_runtime_semantics_exact",
                  "all_callback_optimizer_nontiming_exact"):
            require(type(r[k]) is bool, "Equivalence fields must be explicit booleans")
        exact_actions += r["both_player_actions_exact"]
        exact_semantics += r["all_callback_runtime_semantics_exact"]
        exact_optimizers += r["all_callback_optimizer_nontiming_exact"]
    return {
        "panel": data["panel"], "pairs": len(rows), "games": 2 * len(rows),
        "unique_world_seeds": len(data["declared_seeds"]),
        "actions_exact_pairs": exact_actions, "final_cash_exact_pairs": exact_cash,
        "runtime_semantics_exact_pairs": exact_semantics,
        "optimizer_nontiming_exact_pairs": exact_optimizers,
        "mean_callback_total_ratio": mean(total_ratios),
        "mean_later_callback_ratio": mean(later_ratios),
        "callback_total_reduction_percent": 100 * (1 - mean(total_ratios)),
        "later_callback_reduction_percent": 100 * (1 - mean(later_ratios)),
        "evidence_boundary": data["scope"],
    }


def verify_extract_hashes(data_dir):
    provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    for item in provenance["published_extracts"]:
        name = item["basename"]
        require(Path(name).name == name, "Extract filename must be a basename")
        actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        require(actual == item["sha256"], f"Published extract hash mismatch: {name}")


def check_reproduction(cross_data, perf_data, cross, perf):
    expected = cross_data["expected_recomputed"]
    require(cross["strict_full_panel_health"] == "FAIL", "Preserve original whole-panel FAIL")
    require(cross["strict_healthy_games"] == expected["strict_healthy_games"], "Health count changed")
    require(cross["official_cash_checked_transitions"] == expected["official_cash_checked_transitions"],
            "Cash audit transition count changed")
    require(cross["raw_all_pairs_including_flagged_controls"]["mean_margin_delta"]
            == expected["raw_mean_margin_delta"], "Raw paired margin changed")
    require(cross["strict_clean_pairs_diagnostic_only"]["mean_margin_delta"]
            == expected["clean_diagnostic_mean_margin_delta"], "Clean diagnostic margin changed")
    for k, v in perf_data["expected_recomputed"].items():
        require(math.isclose(perf[k], v, rel_tol=1e-12), f"Performance reproduction changed: {k}")
    require(all(perf[k] == perf["pairs"] for k in ("actions_exact_pairs", "final_cash_exact_pairs",
                                                  "runtime_semantics_exact_pairs", "optimizer_nontiming_exact_pairs")),
            "Runtime-only ablation lost action/cash/semantic equivalence")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--check", action="store_true", help="Verify frozen extract hashes and known recomputed metrics")
    args = ap.parse_args(argv)
    try:
        cross_data = json.loads((args.data_dir / "r7_cross24.json").read_text(encoding="utf-8"))
        perf_data = json.loads((args.data_dir / "r7_performance.json").read_text(encoding="utf-8"))
        cross, perf = analyze_cross(cross_data), analyze_performance(perf_data)
        if args.check:
            verify_extract_hashes(args.data_dir)
            check_reproduction(cross_data, perf_data, cross, perf)
        print(json.dumps({"reproduction_check": "PASS" if args.check else "NOT_REQUESTED",
                          "cross24": cross, "runtime_only_normal32": perf}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        ap.exit(2, f"Evidence validation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
