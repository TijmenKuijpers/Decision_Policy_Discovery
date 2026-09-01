"""
ga_batching_conf.py
-------------------
Genetic-algorithm conformance search over the policy grammar for a
transportation process with a *batching* decision policy.

Process
-------

The single truck token carries a `status` attribute.  While the status is
TRANSPORTING the `loading` guard fails, so no new cases can be staged: the
truck must first drain `ready`.

A transport therefore has two cases, and this is what makes the process batch:

  * cases remain in `ready` — the truck is still at the dock being filled, so
    it comes back after DISPATCH_TIME (zero) and stays TRANSPORTING.  Loading
    is blocked and the policy is consulted again immediately, so the rest of
    the batch is dispatched at the same clock;
  * `ready` is now empty — that was the last case, so the truck departs,
    delivers, and is unavailable for RETURN_TIME before returning in the
    LOADING status.

The round trip is thus paid once per trip rather than once per case, which is
the cost a batching policy exists to amortise: a larger batch spreads
RETURN_TIME over more cases, while waiting for one delays every case already
staged.

Truck status is encoded numerically (LOADING=0, TRANSPORTING=1) 

Reference policy (pi_batch)
--------------------------
    IF COUNT(ready) > BATCH_SIZE OR MAX(truck, status) = TRANSPORTING
        THEN FIRE transporting
        ELSE POSTPONE

Seed policies
-------------
    A: right shape, wrong batch size   (no status-continuation branch)
    B: status-continuation branch only (degenerate alone — never starts a batch)
    C: no batching at all              (transport whenever a case is ready)

SEED_POLICY (= A) is the single pi_0 the evaluation starts its
some-pre-knowledge arm from; B and C are available to experiments that want
several seeds.
"""

import numpy as np

import gympn_path  # noqa: F401  -- makes gympn importable; see its docstring

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

from policy_grammar import (
    IfThenElse, Or, Compare,
    Count, Max, Number,
    Fire, Postpone,
)
import symbolic_regression as sr
import ga_shared as shared

# ─── scenario parameters ──────────────────────────────────────────────────────

# Truck status encoding.  Numeric so <feature> terminals can read it.
STATUS_LOADING      = 0
STATUS_TRANSPORTING = 1

# Mean inter-arrival time.
ARRIVAL_SCALE  = 3.0

LOADING_TIME   = 0.5    # truck occupied per case staged into `ready`
DISPATCH_TIME  = 0.0    # per case, while the truck is still at the dock
RETURN_TIME    = 6.0    # once per trip, after the batch is delivered

# Batch trigger of the reference policy: transport once `ready` exceeds this.
BATCH_SIZE     = 3

# ─── domain ───────────────────────────────────────────────────────────────────

PLACES      = ['arrival', 'waiting', 'ready', 'delivered', 'truck']
TRANSITIONS = ['transporting']
COMPARATORS = shared.COMPARATORS
ATTR_FEATURES = [('truck',     'status'),]

# ─── GA hyper-parameters ──────────────────────────────────────────────────────

POP_SIZE      = shared.POP_SIZE
N_GENERATIONS = shared.N_GENERATIONS
TOURNAMENT_K  = shared.TOURNAMENT_K
P_CROSSOVER   = shared.P_CROSSOVER
P_MUTATION    = shared.P_MUTATION
MAX_DEPTH     = shared.MAX_DEPTH
ELITE_K       = shared.ELITE_K
N_ROLLOUTS    = 18
HORIZON       = shared.HORIZON
COSMETIC_CONFIG = shared.COSMETIC_CONFIG
SCALE_CONFIG = shared.SCALE_CONFIG

# ─── transportation-system factory ────────────────────────────────────────────

def make_batching_system(seed: int = 42) -> GymProblem:
    """Return a fully initialised transportation GymProblem.

    The net is rebuilt per rollout rather than deep-copied from a prototype.
    The `transporting` behaviour closes over the `ready` place to decide the
    truck's next status, and `copy.deepcopy` does not copy function objects —
    a deep-copied net would keep behaviours pointing at the *original* places
    and silently read a stale marking.  Rebuilding is cheap next to simulating.

    *seed* drives the arrival stream, so different rollouts see different
    workloads instead of replaying one identical trace.
    """
    pn  = GymProblem(allow_postpone=True, causal_rl=False)
    rng = np.random.default_rng(seed)
    pn.rng = rng

    # Cases are indistinguishable units of work and carry no attributes; only
    # the truck holds data.
    arrival   = pn.add_var("arrival",   var_attributes=[])
    waiting   = pn.add_var("waiting",   var_attributes=[])
    ready     = pn.add_var("ready",     var_attributes=[])
    delivered = pn.add_var("delivered", var_attributes=[])
    truck     = pn.add_var("truck",     var_attributes=["truck_id", "status"])

    # ── arrive (evolution) ────────────────────────────────────────────────────
    def arrive(tok):
        delay = float(rng.exponential(scale=ARRIVAL_SCALE))
        # Next generator token and the new case both land at t + delay.
        return [SimToken({}, delay=delay), SimToken({}, delay=delay)]

    pn.add_event([arrival], [arrival, waiting], behavior=arrive, name="arrive")

    # ── loading (evolution) ───────────────────────────────────────────────────
    def loading(case, truck_tok):
        # Case is staged immediately; the truck is what stays busy.
        return [SimToken({}, delay=0),
                SimToken(truck_tok, delay=LOADING_TIME)]

    pn.add_event(
        [waiting, truck], [ready, truck],
        behavior=loading,
        guard=lambda case, truck_tok: truck_tok["status"] == STATUS_LOADING,
        name="loading",
    )

    # ── transporting (ACTION) ─────────────────────────────────────────────────
    def transporting(case, truck_tok):
        # simpn removes the bound token from the marking before running the
        # behaviour, so what is left in `ready` here is what remains *after*
        # this case has been loaded onto the truck.
        if ready.marking:
            # Mid-batch: the truck is being filled and has not left yet, so it
            # is available again at once and stays TRANSPORTING, which
            # guard-blocks `loading` until the batch is complete.
            status, delay = STATUS_TRANSPORTING, DISPATCH_TIME
        else:
            # That was the last case: the truck departs, delivers, and drives
            # back before it can load again.
            status, delay = STATUS_LOADING, RETURN_TIME
        return [SimToken({}, delay=0),
                SimToken({"truck_id": truck_tok["truck_id"], "status": status},
                         delay=delay)]

    pn.add_action(
        [ready, truck], [delivered, truck],
        behavior=transporting,
        reward_function=lambda case, truck_tok: 1,
        name="transporting",
    )

    # ── initial state ─────────────────────────────────────────────────────────
    arrival.put({})
    truck.put({"truck_id": 1, "status": STATUS_LOADING})

    return pn


# ─── reference policy (pi_batch) ───────────────────────────────────────────────

REFERENCE_POLICY = IfThenElse(
    condition=Or(
        Compare('>', Count('ready'), Number(BATCH_SIZE)),
        Compare('=', Max('truck', 'status'), Number(STATUS_TRANSPORTING)),
    ),
    then=Fire('transporting'),
    else_=Postpone(),
)

# ─── fitness ──────────────────────────────────────────────────────────────────

# Scoring is target-driven: the reference is simulated once, its decision
# points are recorded, and candidates are scored by replaying them.
_TRACES = sr.TraceCache(
    make_system=lambda s: make_batching_system(seed=s),
    seed_base=1000,
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
    """Negative divergence rate over the reference's recorded decision points.

    The denominator is the fixed trace length (see sr.primary_rate), so a rate
    is a plain rescale of the divergence count.  It is kept as the primary
    because a score in [-1, 0] is comparable across processes with different
    trace lengths, which the shared cosmetic weights depend on.
    """
    return sr.primary_rate(conformance_counts(tree))


evaluate = sr.make_evaluator(evaluate_primary, COSMETIC_CONFIG)

# ─── expression vocabulary ────────────────────────────────────────────────────

VOCAB = shared.make_vocab(PLACES, ATTR_FEATURES)

CONFIG = shared.make_config(VOCAB, TRANSITIONS)

# ─── seed policies ────────────────────────────────────────────────────────────

# Seed A — right shape, wrong threshold, and no status-continuation branch:
#          once a batch starts it stops early, diverging mid-batch.
SEED_A = IfThenElse(
    condition=Compare('>', Count('ready'), Number(1)),
    then=Fire('transporting'),
    else_=Postpone(),
)

# Seed B — the status-continuation branch on its own.  Degenerate in isolation
#          (nothing ever sets the status to TRANSPORTING, so it always
#          postpones), but it carries the building block Seed A is missing.
SEED_B = IfThenElse(
    condition=Compare('=', Max('truck', 'status'), Number(STATUS_TRANSPORTING)),
    then=Fire('transporting'),
    else_=Postpone(),
)

# Seed C — no batching: transport as soon as anything is ready.
SEED_C = IfThenElse(
    condition=Compare('>', Count('ready'), Number(0)),
    then=Fire('transporting'),
    else_=Postpone(),
)

SEED_POLICIES = [SEED_A, SEED_B, SEED_C]
SEED_POLICY   = SEED_A   # used for the "Seed policy fitness" summary line

# ─── experiment record ──────────────────────────────────────

EXPERIMENT = sr.Experiment(
    name="batching conformance policy search",
    reference_label="pi_batch",
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
