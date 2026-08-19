"""Ordering contract for the leaderboard: file order is the ranking.

rank_order() is what print_table and write_json share, so this pins the
committed-artifact order: qualifiers (EOT fp within the 0.15 ceiling) by
EOT recall, then over-ceiling entries. Over-ceiling submissions are
displayed and ranked last, never dropped.
"""

import math

from turnbench.analysis.leaderboard import TEST_FP_BOUND, rank_order


def _row(recall: float, fp: float) -> dict:
    return {
        "EOT": (recall, fp, None),
        "INT": (math.nan, math.nan, None),
        "supported": {"EOT": True, "INT": False},
    }


def test_rank_order_sinks_over_ceiling_below_all_qualifiers():
    scored = {
        "spam": _row(1.0, 0.9),  # top recall, over the ceiling
        "good": _row(0.8, 0.05),
        "meh": _row(0.4, TEST_FP_BOUND),  # exactly at the ceiling qualifies
    }
    assert [m for m, _ in rank_order(scored)] == ["good", "meh", "spam"]


def test_rank_order_keeps_every_entry():
    scored = {"spam": _row(1.0, 0.9), "good": _row(0.8, 0.05)}
    assert len(rank_order(scored)) == len(scored)


def test_rank_order_sorts_qualifiers_by_recall():
    scored = {"low": _row(0.3, 0.01), "high": _row(0.9, 0.14)}
    assert [m for m, _ in rank_order(scored)] == ["high", "low"]
