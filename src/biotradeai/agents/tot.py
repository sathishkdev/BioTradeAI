"""Tree-of-Thoughts arithmetic. Pure code, no LLM, no I/O (MUST #5).

The tree is a flat list of ``ToTNode``. ``node_id`` is the path of labels joined with
"/" (e.g. ``success/clean_path/no_raise``), so a node's parent is its id minus the last
segment. ``probability`` is conditional on the parent and siblings always sum to 1.

Pruned nodes are never expanded. Their joint probability is collapsed into ONE
residual scenario with ``tot.residual_scenario_return`` (plan correction #2), so the
scenario set still covers the whole outcome space and EV includes every branch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from biotradeai.schemas import Scenario, ScenarioSet, ToTNode

RESIDUAL_LABEL = "residual (pruned branches)"


class TreeError(ValueError):
    pass


def bounded_projection(base: Sequence[float], proposed: Sequence[float],
                       max_adjustment: float) -> list[float]:
    """Turn LLM-proposed adjustments into probabilities that (a) sum to 1 and (b) stay
    within ``max_adjustment`` of their base rate and inside [0, 1].

    ``base`` must already sum to 1. Each proposal is first clipped to the band; then one
    shift ``lam`` is applied to all outcomes, clipped to each box, found by bisection so the
    total is 1. A solution always exists because ``base`` itself is feasible.
    """
    if len(base) != len(proposed) or not base:
        raise TreeError("base and proposed must be the same, non-zero length")
    if abs(math.fsum(base) - 1.0) > 1e-9:
        raise TreeError("base rates must sum to 1")
    d = max_adjustment
    lo = [max(0.0, b - d) for b in base]
    hi = [min(1.0, b + d) for b in base]
    target = [min(max(b + max(-d, min(d, a)), lo_i), hi_i)
              for b, a, lo_i, hi_i in zip(base, proposed, lo, hi, strict=True)]

    def total(lam: float) -> float:
        return math.fsum(min(max(t + lam, lo_i), hi_i)
                         for t, lo_i, hi_i in zip(target, lo, hi, strict=True))

    a, b = -1.0, 1.0
    for _ in range(200):
        mid = (a + b) / 2
        if total(mid) < 1.0:
            a = mid
        else:
            b = mid
    p = [min(max(t + b, lo_i), hi_i) for t, lo_i, hi_i in zip(target, lo, hi, strict=True)]
    # Put the last ~1e-15 of rounding error on the outcome with the most room.
    err = 1.0 - math.fsum(p)
    i = max(range(len(p)), key=lambda k: (hi[k] - p[k]) if err > 0 else (p[k] - lo[k]))
    p[i] = min(max(p[i] + err, lo[i]), hi[i])
    return p


# --------------------------------------------------------------------------- tree queries


def parent_of(node_id: str) -> str | None:
    head, sep, _ = node_id.rpartition("/")
    return head if sep else None


def by_id(nodes: Sequence[ToTNode]) -> dict[str, ToTNode]:
    out: dict[str, ToTNode] = {}
    for n in nodes:
        if n.node_id in out:
            raise TreeError(f"duplicate node id {n.node_id}")
        if n.parent_id != parent_of(n.node_id):
            raise TreeError(f"node {n.node_id} has inconsistent parent {n.parent_id}")
        out[n.node_id] = n
    return out


def children(nodes: Sequence[ToTNode], parent_id: str | None) -> list[ToTNode]:
    return [n for n in nodes if n.parent_id == parent_id]


def joint_probability(nodes: Sequence[ToTNode], node_id: str) -> float:
    idx = by_id(nodes)
    p = 1.0
    cur: str | None = node_id
    while cur is not None:
        p *= idx[cur].probability
        cur = idx[cur].parent_id
    return p


def path(nodes: Sequence[ToTNode], node_id: str) -> list[ToTNode]:
    idx = by_id(nodes)
    out: list[ToTNode] = []
    cur: str | None = node_id
    while cur is not None:
        out.append(idx[cur])
        cur = idx[cur].parent_id
    return list(reversed(out))


def leaf_return(nodes: Sequence[ToTNode], node_id: str) -> float:
    growth = 1.0
    for n in path(nodes, node_id):
        growth *= 1.0 + n.step_return
    return max(-1.0, growth - 1.0)


def check_siblings(nodes: Sequence[ToTNode], tolerance: float) -> None:
    parents = {n.parent_id for n in nodes}
    for pid in parents:
        s = math.fsum(n.probability for n in children(nodes, pid))
        if abs(s - 1.0) > tolerance:
            raise TreeError(f"children of {pid or 'root'} sum to {s:.6f}, not 1")


def frontier(nodes: Sequence[ToTNode], depth: int) -> list[ToTNode]:
    """Nodes at ``depth`` that survived pruning (the ones to expand next)."""
    return [n for n in nodes if n.depth == depth and not n.pruned]


# --------------------------------------------------------------------------- outputs


def tree_ev(nodes: Sequence[ToTNode], residual_return: float) -> float:
    """Research Agent's EV: depth-first walk carrying joint probability and growth.

    Deliberately a different code path from ``guardrails.rules.compute_ev`` (which sums
    the flattened ScenarioSet), so the Risk Agent's recomputation is a real cross-check.
    """
    terms: list[float] = []

    def walk(node: ToTNode, joint: float, growth: float) -> None:
        joint *= node.probability
        if node.pruned:
            terms.append(joint * residual_return)
            return
        growth *= 1.0 + node.step_return
        kids = children(nodes, node.node_id)
        if not kids:
            terms.append(joint * max(-1.0, growth - 1.0))
            return
        for k in kids:
            walk(k, joint, growth)

    for root in children(nodes, None):
        walk(root, 1.0, 1.0)
    return math.fsum(terms)


def build_scenarios(nodes: Sequence[ToTNode], *, residual_return: float,
                    tolerance: float) -> ScenarioSet:
    """Flatten the tree: one Scenario per surviving leaf, plus the residual."""
    by_id(nodes)
    check_siblings(nodes, tolerance)
    scenarios: list[Scenario] = []
    residual = 0.0
    residual_evidence: list[str] = []
    for n in nodes:
        if n.pruned:
            # Only count the topmost pruned node of a branch (its subtree was never built).
            if not any(a.pruned for a in path(nodes, n.node_id)[:-1]):
                residual += joint_probability(nodes, n.node_id)
                residual_evidence += n.evidence_ids
            continue
        if children(nodes, n.node_id):
            continue
        evid = sorted({e for a in path(nodes, n.node_id) for e in a.evidence_ids})
        scenarios.append(Scenario(
            depth=n.depth, label=" > ".join(a.label for a in path(nodes, n.node_id)),
            probability=joint_probability(nodes, n.node_id),
            expected_return=leaf_return(nodes, n.node_id), evidence_ids=evid))
    if residual > 0.0:
        scenarios.append(Scenario(depth=1, label=RESIDUAL_LABEL, probability=residual,
                                  expected_return=residual_return,
                                  evidence_ids=sorted(set(residual_evidence)),
                                  is_residual=True))
    return ScenarioSet(scenarios=scenarios, tolerance=tolerance)
