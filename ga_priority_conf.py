r"""
ga_priority_conf.py
-------------------
Genetic-algorithm conformance search over the policy grammar for a single-line
manufacturing process governed by a *priority look-ahead* decision policy.

Process
-------

Both waiting places are priority queues: highest `priority` first (see
_sort_by_priority).  That ordering is what decides *which* case `process`
consumes, since the solver takes the first available binding.

Reference policy (pi_prio)
--------------------------
    IF MAX(waiting_1, priority) <= MAX(waiting_2, priority)
        THEN FIRE process
        ELSE POSTPONE

"Do not start on what is ready if something better is still upstream."

Seed policies (three complementary imperfect starts)
----------------------------------------------------
    A: right shape, wrong aggregate  (MIN instead of MAX)
    B: availability only             (ignores priority entirely)
    C: no policy at all              (always fire)
"""

import numpy as np
from sortedcontainers import SortedList

import gympn_path  # noqa: F401  -- makes gympn importable; see its docstring

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

from policy_grammar import (
    IfThenElse, Compare,
    Count, Min, Max,
    Fire, Postpone,
)
import symbolic_regression as sr
import ga_shared as shared

# ─── scenario parameters ──────────────────────────────────────────────────────

# Mean time between case announcements.
ARRIVAL_SCALE    = 1.2
LEAD_TIME        = 2.0
PROCESS_TIME     = 1.0    # machine occupied per case

# Inclusive priority range carried by each case.
PRIORITY_LO      = 1
PRIORITY_HI      = 9

# ─── domain ───────────────────────────────────────────────────────────────────

PLACES      = ['arrival', 'waiting_1', 'waiting_2', 'completed', 'machine']
TRANSITIONS = ['process']

ATTR_FEATURES = [
    ('waiting_1', 'priority'),
    ('waiting_2', 'priority'),
    ('completed', 'priority'),
]

# ─── GA hyper-parameters (shared -- see ga_shared) ────────────────────────────

POP_SIZE      = shared.POP_SIZE
N_GENERATIONS = shared.N_GENERATIONS
TOURNAMENT_K  = shared.TOURNAMENT_K
P_CROSSOVER   = shared.P_CROSSOVER
P_MUTATION    = shared.P_MUTATION
MAX_DEPTH     = shared.MAX_DEPTH
ELITE_K       = shared.ELITE_K
COMPARATORS   = shared.COMPARATORS
# 6 x 60 yields 489 decision points.
N_ROLLOUTS    = 6
HORIZON       = shared.HORIZON

COSMETIC_CONFIG = shared.COSMETIC_CONFIG

SCALE_CONFIG = shared.SCALE_CONFIG

# ─── priority-queue helper ────────────────────────────────────────────────────

def _sort_by_priority(var):
    """Turn a place into a priority queue: highest `priority` attribute first.

    simpn's SimVar keeps its marking in a SortedList whose key decides the
    order tokens are offered to transitions (lower key first, defaulting to
    token time).  gympn's GymProblem.add_var does not expose that key — and it
    forwards its own `initial` argument into simpn's `priority` slot
    positionally — so the SortedList is replaced here instead, right after
    creation while the marking is still empty.

    Negating the attribute turns "lowest key first" into "highest priority
    first".  `_sorted_by_time` must be cleared too: simpn checks it to decide
    whether marking[0] can be trusted as the earliest token, and falls back to
    walking the whole marking when it cannot.
    """
    var.marking = SortedList(key=lambda token: -token.value["priority"])
    var._sorted_by_time = False
    return var


def _entry_priority(entry):
    """Highest `priority` value appearing anywhere in one candidate binding."""
    best = None
    for item in entry:
        candidate = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else item
        value = getattr(candidate, 'value', candidate)
        if isinstance(value, dict) and 'priority' in value:
            p = value['priority']
            if best is None or p > best:
                best = p
    return best


def _select_binding(entries):
    """Choose the candidate binding carrying the highest-priority case.

    Sorting the marking is necessary but *not sufficient* to give waiting_2 a
    priority discipline.  gympn sorts the candidate bindings by binding time
    before passing them to the solver (GymProblem.bindings), so the marking's
    order survives only as a tie-break among equally-timed bindings — taking
    entries[0] processes the earliest-available case, not the best one.

    Selecting here restores the intended discipline.  This is part of the
    *environment*, not of any policy: it decides which case is processed once a
    policy has decided to process, and it is applied identically to every
    candidate tree, so conformance comparisons stay fair.
    """
    best_i, best_p = 0, None
    for i, entry in enumerate(entries):
        p = _entry_priority(entry)
        if p is not None and (best_p is None or p > best_p):
            best_i, best_p = i, p
    return entries[best_i]

# ─── manufacturing-line factory ───────────────────────────────────────────────

def make_priority_system(seed: int = 42) -> GymProblem:
    """Return a fully initialised single-line manufacturing GymProblem."""
    pn  = GymProblem(allow_postpone=True, causal_rl=False)
    rng = np.random.default_rng(seed)
    pn.rng = rng

    # Priority is the only case data in this process; cases carry no identifier.
    arrival   = pn.add_var("arrival",   var_attributes=[])
    waiting_1 = pn.add_var("waiting_1", var_attributes=["priority"])
    waiting_2 = pn.add_var("waiting_2", var_attributes=["priority"])
    completed = pn.add_var("completed", var_attributes=["priority"])
    machine   = pn.add_var("machine",   var_attributes=["machine_id"])

    # Both waiting places are priority queues, as required.
    _sort_by_priority(waiting_1)
    _sort_by_priority(waiting_2)

    # ── arrive (evolution) ────────────────────────────────────────────────────
    def arrive(tok):
        inter    = float(rng.exponential(scale=ARRIVAL_SCALE))
        priority = int(rng.integers(PRIORITY_LO, PRIORITY_HI + 1))
        # The generator advances by `inter`; the case itself only becomes
        # available LEAD_TIME later, so it is visible-but-pending in waiting_1.
        return [SimToken({}, delay=inter),
                SimToken({"priority": priority}, delay=inter + LEAD_TIME)]

    pn.add_event([arrival], [arrival, waiting_1], behavior=arrive, name="arrive")

    # ── pre_process (evolution) ───────────────────────────────────────────────
    def pre_process(case):
        return [SimToken(case, delay=0)]

    pn.add_event([waiting_1], [waiting_2], behavior=pre_process, name="pre_process")

    # ── process (ACTION) ──────────────────────────────────────────────────────
    def process(case, mach):
        return [SimToken(case, delay=0),
                SimToken(mach, delay=PROCESS_TIME)]

    pn.add_action([waiting_2, machine], [completed, machine],
                  behavior=process,
                  reward_function=lambda case, mach: case["priority"],
                  name="process")

    # ── initial state ─────────────────────────────────────────────────────────
    arrival.put({})
    machine.put({"machine_id": 1})

    return pn


# ─── reference policy (pi_prio) ───────────────────────────────────────────────

REFERENCE_POLICY = IfThenElse(
    condition=Compare('<=',
                      Max('waiting_1', 'priority'),
                      Max('waiting_2', 'priority')),
    then=Fire('process'),
    else_=Postpone(),
)

# ─── fitness ──────────────────────────────────────────────────────────────────

# Scoring is target-driven: the reference is simulated once, its decision
# points are recorded, and candidates are scored by replay.  `_select_binding`
# is the binding picker for that recording, so the priority-queue discipline
# shapes the recorded trajectory -- see its docstring for why sorting the
# marking alone is not enough.
_TRACES = sr.TraceCache(
    make_system=lambda s: make_priority_system(seed=s),
    seed_base=3000,
    pick_binding=_select_binding,
)


def conformance_counts(tree, reference=None) -> tuple[int, int] | None:
    """Divergences and decision points vs *reference* (default REFERENCE_POLICY).

    The decision points are *reference*'s: recorded once from a simulation it
    drives, then replayed for every candidate.  N_ROLLOUTS / HORIZON are read
    here rather than baked into `_TRACES` so `evaluation.py` can override them.
    """
    return _TRACES.counts(
        tree, reference if reference is not None else REFERENCE_POLICY,
        n_rollouts=N_ROLLOUTS, horizon=HORIZON,
    )


def evaluate_primary(tree) -> float | None:
    """Negative divergence rate (see sr.primary_rate)."""
    return sr.primary_rate(conformance_counts(tree))


evaluate = sr.make_evaluator(evaluate_primary, COSMETIC_CONFIG)

# ─── expression vocabulary ────────────────────────────────────────────────────

VOCAB = shared.make_vocab(PLACES, ATTR_FEATURES)

CONFIG = shared.make_config(VOCAB, TRANSITIONS)

# ─── seed policies ────────────────────────────────────────────────────────────

# Seed A — right comparison shape, wrong aggregate.  A single aggregate swap
#          (MIN -> MAX) turns it into the reference.
SEED_A = IfThenElse(
    condition=Compare('<=',
                      Min('waiting_1', 'priority'),
                      Min('waiting_2', 'priority')),
    then=Fire('process'),
    else_=Postpone(),
)

# Seed B — right structure (compare the two queues), wrong feature: queue
#          length instead of best priority.  Note a `COUNT(waiting_2) > 0`
#          seed would be useless here: process cannot bind unless waiting_2 is
#          non-empty, so that condition is true at every decision point and the
#          seed collapses onto Seed C.
SEED_B = IfThenElse(
    condition=Compare('<=', Count('waiting_1'), Count('waiting_2')),
    then=Fire('process'),
    else_=Postpone(),
)

# Seed C — no policy at all.
SEED_C = Fire('process')

SEED_POLICIES = [SEED_A, SEED_B, SEED_C]
SEED_POLICY   = SEED_A

# ─── experiment record ──────────────────────────────────────

EXPERIMENT = sr.Experiment(
    name="priority look-ahead conformance search",
    reference_label="pi_prio",
    config=CONFIG,
    evaluate=evaluate,
    evaluate_primary=evaluate_primary,
    cosmetic_config=COSMETIC_CONFIG,
    seed_policies=SEED_POLICIES,
    seed_policy=SEED_POLICY,
    pop_size=POP_SIZE,
    n_generations=N_GENERATIONS,
    tournament_k=TOURNAMENT_K,
    p_crossover=P_CROSSOVER,
    elite_k=ELITE_K,
    reference_policy=REFERENCE_POLICY,
    conformance_counts=conformance_counts,
    scale_config=SCALE_CONFIG,
)


def run_ga(seed: int = 42):
    return sr.run_ga(EXPERIMENT, seed=seed)


if __name__ == "__main__":
    run_ga(seed=42)
