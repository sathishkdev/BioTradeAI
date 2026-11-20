"""ToT arithmetic: bounded projection, residual collapse, EV over all outcomes."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from biotradeai.agents import tot
from biotradeai.guardrails.rules import compute_ev
from biotradeai.schemas import ToTNode

BASE = [0.500, 0.078, 0.422]


def _node(nid: str, p: float, r: float, base: float | None = None, *, pruned: bool = False,
          ) -> ToTNode:
    b = p if base is None else base
    return ToTNode(node_id=nid, parent_id=tot.parent_of(nid), depth=nid.count("/") + 1,
                   label=nid.rsplit("/", 1)[-1], base_rate=b, adjustment=p - b, probability=p,
                   step_return=r, justification="t", pruned=pruned,
                   prune_reason="test" if pruned else None)


@given(st.lists(st.floats(-1, 1, allow_nan=False), min_size=3, max_size=3),
       st.floats(0.01, 0.5))
@settings(max_examples=300)
def test_projection_sums_to_one_and_stays_in_band(adj: list[float], d: float) -> None:
    p = tot.bounded_projection(BASE, adj, d)
    assert math.isclose(math.fsum(p), 1.0, abs_tol=1e-12)
    for b, x in zip(BASE, p, strict=True):
        assert 0.0 <= x <= 1.0
        assert abs(x - b) <= d + 1e-12


def test_zero_adjustment_keeps_base_rates() -> None:
    assert tot.bounded_projection(BASE, [0, 0, 0], 0.15) == pytest.approx(BASE)


def test_symmetric_adjustment_is_kept_exactly() -> None:
    assert tot.bounded_projection(BASE, [0.15, 0, -0.15], 0.15) == pytest.approx(
        [0.65, 0.078, 0.272])


def test_oversized_adjustment_is_clipped_to_band() -> None:
    p = tot.bounded_projection(BASE, [0.9, 0, 0], 0.15)
    assert p[0] <= 0.65 + 1e-12


def test_projection_rejects_bad_base() -> None:
    with pytest.raises(tot.TreeError):
        tot.bounded_projection([0.5, 0.4], [0, 0], 0.1)


def _tree() -> list[ToTNode]:
    return [
        _node("success", 0.6, 1.0), _node("success/clean", 0.8, 0.0),
        _node("success/delay", 0.2, -0.1),
        _node("partial", 0.1, 0.0, pruned=True),
        _node("failure", 0.3, -0.6), _node("failure/stop", 1.0 - 0.4, -0.1),
        _node("failure/go", 0.4, 0.05),
    ]


def test_scenarios_include_failure_and_residual_and_sum_to_one() -> None:
    s = tot.build_scenarios(_tree(), residual_return=-0.7, tolerance=0.001)
    labels = [x.label for x in s.scenarios]
    assert "failure > stop" in labels and "failure > go" in labels
    res = [x for x in s.scenarios if x.is_residual]
    assert len(res) == 1 and res[0].probability == pytest.approx(0.1)
    assert res[0].expected_return == -0.7
    assert math.fsum(x.probability for x in s.scenarios) == pytest.approx(1.0)


def test_leaf_return_compounds_steps() -> None:
    assert tot.leaf_return(_tree(), "success/delay") == pytest.approx(2.0 * 0.9 - 1)
    assert tot.leaf_return(_tree(), "failure/stop") == pytest.approx(0.4 * 0.9 - 1)


def test_tree_ev_matches_risk_recomputation() -> None:
    nodes = _tree()
    s = tot.build_scenarios(nodes, residual_return=-0.7, tolerance=0.001)
    assert tot.tree_ev(nodes, -0.7) == pytest.approx(compute_ev(s), abs=1e-12)


@given(st.floats(0.05, 0.95), st.floats(0.05, 0.95), st.floats(-1, 3), st.floats(-1, 3))
def test_tree_ev_equals_scenario_ev_property(p: float, q: float, r1: float, r2: float) -> None:
    nodes = [_node("a", p, r1), _node("b", 1 - p, r2), _node("a/x", q, 0.1),
             _node("a/y", 1 - q, -0.2)]
    s = tot.build_scenarios(nodes, residual_return=-0.7, tolerance=0.001)
    assert tot.tree_ev(nodes, -0.7) == pytest.approx(compute_ev(s), abs=1e-9)


def test_siblings_not_summing_to_one_rejected() -> None:
    bad = [_node("a", 0.5, 0.0), _node("b", 0.4, 0.0)]
    with pytest.raises(tot.TreeError):
        tot.build_scenarios(bad, residual_return=-0.7, tolerance=0.001)


def test_inconsistent_parent_rejected() -> None:
    n = _node("a/x", 1.0, 0.0).model_copy(update={"parent_id": "b"})
    with pytest.raises(tot.TreeError):
        tot.by_id([_node("a", 1.0, 0.0), n])
