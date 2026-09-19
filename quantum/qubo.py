"""
QUBO builder for Q-Signal.
Constructs the Q matrix encoding the multi-objective signal optimization problem.

Energy function:
  E(x) =
    - Σ_i [benefit(i,NS)×(1-x_i) + benefit(i,EW)×x_i]    (local demand)
    + λ_coord × Σ_(i,j) (x_i + x_j - 2×x_i×x_j)           (coordination)
    + λ_switch × Σ_i switch_penalty(i) × x_i                (switching)
    + λ_fair  × Σ_i starvation_penalty(i)                    (fairness)
    + λ_spill × Σ_i spillback_penalty(i)                     (spillback)

Variables: x_i = 0 → NS phase, x_i = 1 → EW phase
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from qsignal.models import TrafficState, PHASE_NS, PHASE_EW

# Default QUBO weights (overridden by config)
DEFAULT_LAMBDA_COORD = 0.5
DEFAULT_LAMBDA_SWITCH = 0.3
DEFAULT_LAMBDA_FAIR = 0.4
DEFAULT_LAMBDA_SPILL = 0.6
DEFAULT_WAITING_WEIGHT = 1.0


def build_qubo(
    state: TrafficState,
    graph=None,
    config: Optional[dict] = None,
) -> Tuple[np.ndarray, float]:
    """
    Build the QUBO matrix Q and constant offset for the given traffic state.

    Returns:
        Q: (n × n) upper-triangular QUBO matrix.
           Energy = x^T Q x  (with Q[i,j] for i≤j)
        offset: constant energy offset (not encoded in Q).

    Note: In QUBO form, diagonal Q[i,i] captures linear terms,
          off-diagonal Q[i,j] captures quadratic coupling.
    """
    cfg = config or {}
    lambda_coord = cfg.get("lambda_coord", DEFAULT_LAMBDA_COORD)
    lambda_switch = cfg.get("lambda_switch", DEFAULT_LAMBDA_SWITCH)
    lambda_fair = cfg.get("lambda_fair", DEFAULT_LAMBDA_FAIR)
    lambda_spill = cfg.get("lambda_spill", DEFAULT_LAMBDA_SPILL)
    waiting_weight = cfg.get("waiting_weight", DEFAULT_WAITING_WEIGHT)
    spill_threshold = cfg.get("spill_threshold", 10.0)  # queue length for spillback

    node_ids = list(state.intersections.keys())
    n = len(node_ids)
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Group approaches by intersection
    ns_queue: Dict[str, float] = {}
    ew_queue: Dict[str, float] = {}
    ns_starvation: Dict[str, int] = {}
    ew_starvation: Dict[str, int] = {}

    for approach in state.approaches.values():
        nid = approach.intersection_id
        if approach.movement == "NS":
            ns_queue[nid] = approach.queue
            ns_starvation[nid] = approach.starvation_timer
        else:
            ew_queue[nid] = approach.queue
            ew_starvation[nid] = approach.starvation_timer

    Q = np.zeros((n, n))
    offset = 0.0

    # ------------------------------------------------------------------
    # 1. Local demand benefit term
    #    benefit(i,NS) = D(i,NS) × waiting_weight
    #    benefit(i,EW) = D(i,EW) × waiting_weight
    #    Encoding: -benefit_NS × (1-x_i) - benefit_EW × x_i
    #    = -benefit_NS + benefit_NS×x_i - benefit_EW×x_i
    #    = -benefit_NS + (benefit_NS - benefit_EW)×x_i
    # ------------------------------------------------------------------
    for nid, i in node_idx.items():
        b_ns = ns_queue.get(nid, 0.0) * waiting_weight
        b_ew = ew_queue.get(nid, 0.0) * waiting_weight
        # Diagonal: (benefit_NS - benefit_EW)  [linear in x_i]
        Q[i, i] += (b_ns - b_ew)
        # Constant: -benefit_NS
        offset -= b_ns

    # ------------------------------------------------------------------
    # 2. Coordination term (penalise adjacent intersections with
    #    opposite phases on the same corridor direction)
    #    (x_i - x_j)^2 = x_i + x_j - 2x_i x_j
    # ------------------------------------------------------------------
    if graph is not None:
        from qsignal.network import get_adjacent_pairs, get_edge_direction
        pairs = get_adjacent_pairs(graph)
        for (u, v) in pairs:
            i, j = node_idx.get(u), node_idx.get(v)
            if i is None or j is None:
                continue
            direction = get_edge_direction(graph, u, v)
            # On a NS corridor, prefer both intersections to have the same phase
            # Penalise disagreement
            Q[i, i] += lambda_coord       # +x_i
            Q[j, j] += lambda_coord       # +x_j
            qi, qj = (i, j) if i < j else (j, i)
            Q[qi, qj] -= 2.0 * lambda_coord  # -2x_i x_j

    # ------------------------------------------------------------------
    # 3. Phase-switch penalty
    #    If intersection i is currently NS (phase=0) and we pick EW (x_i=1),
    #    incur a switch penalty proportional to how recently it last switched.
    # ------------------------------------------------------------------
    for nid, i in node_idx.items():
        inter = state.intersections.get(nid)
        if inter is None:
            continue
        current_is_ns = inter.phase == PHASE_NS
        if current_is_ns:
            # Penalty for choosing EW (x_i = 1)
            Q[i, i] += lambda_switch
        else:
            # Penalty for choosing NS (x_i = 0): linear in (1-x_i) → +constant - x_i
            Q[i, i] -= lambda_switch
            offset += lambda_switch

    # ------------------------------------------------------------------
    # 4. Fairness (starvation) penalty
    #    If NS is starved, penalise choosing EW (x_i=1)
    #    If EW is starved, penalise choosing NS (x_i=0)
    # ------------------------------------------------------------------
    for nid, i in node_idx.items():
        ns_starve = ns_starvation.get(nid, 0)
        ew_starve = ew_starvation.get(nid, 0)
        # Normalise to [0, 1] range
        ns_score = min(ns_starve / 60.0, 1.0)
        ew_score = min(ew_starve / 60.0, 1.0)
        # Penalty for choosing EW when NS is starved
        Q[i, i] += lambda_fair * ns_score
        # Penalty for choosing NS when EW is starved: +constant - x_i
        Q[i, i] -= lambda_fair * ew_score
        offset += lambda_fair * ew_score

    # ------------------------------------------------------------------
    # 5. Spillback penalty
    #    If a queue exceeds the spill threshold, add extra urgency
    # ------------------------------------------------------------------
    for nid, i in node_idx.items():
        ns_q = ns_queue.get(nid, 0.0)
        ew_q = ew_queue.get(nid, 0.0)
        if ns_q > spill_threshold:
            spill = lambda_spill * (ns_q - spill_threshold) / spill_threshold
            Q[i, i] += spill        # penalise choosing EW
        if ew_q > spill_threshold:
            spill = lambda_spill * (ew_q - spill_threshold) / spill_threshold
            Q[i, i] -= spill
            offset += spill

    return Q, offset


def qubo_energy(Q: np.ndarray, bits: List[int]) -> float:
    """Compute the QUBO energy x^T Q x for a given bitstring."""
    x = np.array(bits, dtype=float)
    return float(x @ Q @ x)
