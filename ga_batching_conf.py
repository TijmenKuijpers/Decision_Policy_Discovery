"""
ga_batching_conf.py
-------------------
Genetic-algorithm conformance search over the policy grammar for a
transportation process with a *batching* decision policy.

Process
-------
Five places and three transitions:

    arrival --[arrive]--> waiting --[loading]--> ready --[transporting]--> delivered
                                       ^                     |
                                       |                     |
                                     truck  <----------------+

  * arrive       (evolution) — cases arrive and queue in `waiting`.
  * loading      (evolution) — moves one case from `waiting` to `ready`;
                               consumes and returns the truck, which is then
                               busy for LOADING_TIME.  Guarded on the truck
                               being in the LOADING status.
  * transporting (ACTION)    — moves one case from `ready` to `delivered`;
                               consumes and returns the truck, busy for
                               TRANSPORT_TIME.

The single truck token carries a `status` attribute.  While the status is
TRANSPORTING the `loading` guard fails, so no new cases can be staged: the
truck must first drain `ready`.  The status flips back to LOADING on the
transport that empties `ready`, which is what makes the process batch.

Truck status is encoded numerically (LOADING=0, TRANSPORTING=1) because every
<feature> terminal in the grammar returns a scalar — a string status could not
be read by SUM/MIN/MAX/MEAN.

When the policy is consulted (important)
----------------------------------------
gympn alternates the net between an *evolution* tag and an *action* tag, and
switches to the action tag only when no evolution binding is enabled at the
current clock (see GymProblem.bindings).  A permanently-enabled evolution
therefore starves the policy completely: the solver is never called and every
decision is made implicitly by the simulator.

That constrains the timing here.  `loading` competes with `transporting` for
the truck, so if `waiting` is never empty, `loading` is enabled at every clock
at which the truck is free and the policy never gets asked anything.  Keeping
ARRIVAL_SCALE > LOADING_TIME makes `waiting` drain between arrivals, which is
what creates the decision points:

  * status LOADING, `waiting` momentarily empty — the policy chooses between
    departing with the batch it has (FIRE) and waiting for more (POSTPONE).
    This is where COUNT(ready) > BATCH_SIZE does the work.
  * status TRANSPORTING — `loading` is guard-blocked outright, so the policy is
    consulted for every case in the batch until `ready` drains.  This is where
    the MAX(truck, status) branch does the work.

If you retune the timing, re-run `--check` and confirm the decision count is
still non-zero and that both branches are exercised.

Reference policy (pi_batch)
--------------------------
    IF COUNT(ready) > BATCH_SIZE OR MAX(truck, status) = TRANSPORTING
        THEN FIRE transporting
        ELSE POSTPONE

The disjunction has an equivalent ELIF spelling in this grammar,

    IF COUNT(ready) > BATCH_SIZE       THEN FIRE transporting
    ELSE IF MAX(truck, status) = 1     THEN FIRE transporting
    ELSE POSTPONE

so the GA can reach zero divergence through either form.  Since exactly one
truck token exists, MAX/MIN/MEAN/SUM over (truck, status) all coincide — the
reference condition has several equally-correct feature spellings, and any of
them scores identically.

Seed policies (three complementary imperfect starts)
----------------------------------------------------
    A: right shape, wrong batch size   (no status-continuation branch)
    B: status-continuation branch only (degenerate alone — never starts a batch)
    C: no batching at all              (transport whenever a case is ready)

Crossover between A and B assembles the full reference without seeding it
directly; ELITE_K=3 keeps all three seeds alive past generation 0.

Expected runtime: ~10-20 minutes for the default 60 x 50 configuration.
"""

import io
import sys
import contextlib

import numpy as np

sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")

from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import HeuristicSolver

from policy_grammar import (
    IfThenElse, Or, Compare,
    Count, Sum, Min, Max, Mean, Number,
    Fire, Postpone,
)
from ga_fitness import CosmeticConfig
import symbolic_regression as sr

# ─── scenario parameters ──────────────────────────────────────────────────────

# Truck status encoding.  Numeric so <feature> terminals can read it.
STATUS_LOADING      = 0
STATUS_TRANSPORTING = 1

# Mean inter-arrival time.  MUST stay above LOADING_TIME — see the note on
# decision points below.  With loading faster than arrivals, `waiting` drains
# between arrivals, which is what hands control to the policy.
ARRIVAL_SCALE  = 1.5

LOADING_TIME   = 0.5    # truck occupied per case staged into `ready`
TRANSPORT_TIME = 0.3    # truck occupied per case moved to `delivered`

# Batch trigger of the reference policy: transport once `ready` exceeds this.
BATCH_SIZE     = 3

# ─── domain ───────────────────────────────────────────────────────────────────

PLACES      = ['arrival', 'waiting', 'ready', 'delivered', 'truck']
TRANSITIONS = ['transporting']
COMPARATORS = ['>', '>=', '<', '<=', '=', '!=']

# (place, attribute) pairs the aggregate features may be built over.
# ('truck', 'status') is the one the reference policy needs; the case_id pairs
# are distractors, so the search has to find the informative feature rather
# than being handed it.  Extend this list to make discovery harder.
ATTR_FEATURES = [
    ('truck',     'status'),
    ('ready',     'case_id'),
    ('waiting',   'case_id'),
]

# Aggregate <feature> node types over an attribute.
AGGREGATES = [Sum, Min, Max, Mean]

# Integer literals; BATCH_SIZE and the status codes are all small.
NUM_POOL = list(range(9))

# ─── GA hyper-parameters ──────────────────────────────────────────────────────

POP_SIZE      = 60
N_GENERATIONS = 50
TOURNAMENT_K  = 3
P_CROSSOVER   = 0.70
P_MUTATION    = 0.35
MAX_DEPTH     = 3
ELITE_K       = 3
N_ROLLOUTS    = 3
HORIZON       = 50

# The shared DIVERGENCE_COSMETIC preset is tuned for a primary score measured
# in absolute divergence counts (tens), where a penalty capped at 3.0 is a
# tie-breaker.  This driver's primary is a rate in [-1, 0], so those weights
# would dominate conformance outright and the GA would just minimise tree size.
# Scaled down by ~1/50 to restore the intended tie-breaker role.
COSMETIC_CONFIG = CosmeticConfig(
    enabled=True,
    w_ite_depth=0.002,
    w_non_numeric=0.002,
    w_non_terminal=0.001,
    w_expr_size=0.001,
    max_cosmetic_penalty=0.05,
)

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

    arrival   = pn.add_var("arrival",   var_attributes=["case_id"])
    waiting   = pn.add_var("waiting",   var_attributes=["case_id"])
    ready     = pn.add_var("ready",     var_attributes=["case_id"])
    delivered = pn.add_var("delivered", var_attributes=["case_id"])
    truck     = pn.add_var("truck",     var_attributes=["truck_id", "status"])

    # ── arrive (evolution) ────────────────────────────────────────────────────
    def arrive(tok):
        cid   = tok["case_id"] + 1
        delay = float(rng.exponential(scale=ARRIVAL_SCALE))
        case  = {"case_id": cid}
        # Next generator token and the new case both land at t + delay.
        return [SimToken(case, delay=delay), SimToken(case, delay=delay)]

    pn.add_event([arrival], [arrival, waiting], behavior=arrive, name="arrive")

    # ── loading (evolution) ───────────────────────────────────────────────────
    def loading(case, truck_tok):
        # Case is staged immediately; the truck is what stays busy.
        return [SimToken({"case_id": case["case_id"]}, delay=0),
                SimToken(truck_tok, delay=LOADING_TIME)]

    pn.add_event(
        [waiting, truck], [ready, truck],
        behavior=loading,
        guard=lambda case, truck_tok: truck_tok["status"] == STATUS_LOADING,
        name="loading",
    )

    # ── transporting (ACTION) ─────────────────────────────────────────────────
    def transporting(case, truck_tok):
        # Filter by case_id rather than counting, so this is correct whether or
        # not the bound token has already been removed from the marking.
        remaining = [t for t in ready.marking
                     if t.value["case_id"] != case["case_id"]]
        status = STATUS_TRANSPORTING if remaining else STATUS_LOADING
        return [SimToken({"case_id": case["case_id"]}, delay=0),
                SimToken({"truck_id": truck_tok["truck_id"], "status": status},
                         delay=TRANSPORT_TIME)]

    pn.add_action(
        [ready, truck], [delivered, truck],
        behavior=transporting,
        reward_function=lambda case, truck_tok: 1,
        name="transporting",
    )

    # ── initial state ─────────────────────────────────────────────────────────
    arrival.put({"case_id": 0})
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

def conformance_counts(tree) -> tuple[int, int] | None:
    """Divergences and decision points vs REFERENCE_POLICY, over all rollouts."""
    return sr.conformance_counts(
        tree, REFERENCE_POLICY,
        make_system=lambda s: make_batching_system(seed=s),
        n_rollouts=N_ROLLOUTS, horizon=HORIZON, seed_base=1000,
    )


def evaluate_primary(tree) -> float | None:
    """Negative divergence rate (see sr.primary_rate for why a rate, not a count).

    Concretely here: a policy that postpones aggressively keeps the truck idle
    and reaches far fewer decisions, so under absolute counts bare `postpone`
    outranks every seed.  Normalising removes that.
    """
    return sr.primary_rate(conformance_counts(tree))


evaluate = sr.make_evaluator(evaluate_primary, COSMETIC_CONFIG)

# ─── expression vocabulary ────────────────────────────────────────────────────

VOCAB = sr.ExprVocab(
    places=PLACES,
    attr_features=ATTR_FEATURES,          # truck status + case_id distractors
    aggregates=AGGREGATES,
    int_pool=NUM_POOL,                    # int-only domain: BATCH_SIZE and 0/1 status
    float_nudges=(),                      # ... so the float mutation path is unused
    w_count=0.45,                         # token counts (the batch trigger needs one)
    w_number=0.20,                        # literal threshold
    w_aggregate=0.35,                     # attribute aggregate (truck status lives here)
    p_agg_swap=0.40,
    p_feature_swap=0.80,
    int_lo=0, int_hi=8,
)

CONFIG = sr.GrammarConfig.from_vocab(
    VOCAB,
    comparators=COMPARATORS,
    transitions=TRANSITIONS,
    max_depth=MAX_DEPTH,
    p_and=0.6,             # And-vs-Or split when building a random condition
    p_fire=0.6,            # Fire-vs-Postpone split when building a random action
    p_mutate_fire=0.5,     # Fire-vs-Postpone split when mutating an action
    p_mutation=P_MUTATION,
)

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

# ─── model smoke test ─────────────────────────────────────────────────────────

def smoke_test(seed: int = 1000, horizon: int = 25) -> None:
    """Run the reference policy once and print the batching cycle it produces.

    Use this to sanity-check the process model before spending GA time on it:
    `ready` should sawtooth up to BATCH_SIZE+1 and drain, and the truck status
    should stay TRANSPORTING for the whole drain.
    """
    trace = []

    def heuristic(pn, actions_dict):
        n_ready = Count('ready').evaluate(pn)
        status  = Max('truck', 'status').evaluate(pn)
        action  = REFERENCE_POLICY.evaluate(pn)
        trace.append((pn.clock, n_ready,
                      Count('waiting').evaluate(pn),
                      Count('delivered').evaluate(pn),
                      status, action))
        if action == 'postpone':
            return 'postpone'
        if action in actions_dict and actions_dict[action]:
            return {action: actions_dict[action][0]}
        return 'postpone'

    pn = make_batching_system(seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        pn.testing_run(solver=HeuristicSolver(heuristic_function=heuristic),
                       length=horizon)

    print("=" * 74)
    print(f"Reference policy trace  (BATCH_SIZE={BATCH_SIZE}, horizon={horizon})")
    print("=" * 74)
    print(f"{'clock':>7} {'ready':>6} {'waiting':>8} {'delivered':>10} "
          f"{'truck':>13}  decision")
    print("-" * 74)
    for clock, n_ready, n_wait, n_deliv, status, action in trace:
        status_s = 'TRANSPORTING' if status == STATUS_TRANSPORTING else 'LOADING'
        print(f"{clock:>7.2f} {n_ready:>6} {n_wait:>8} {n_deliv:>10} "
              f"{status_s:>13}  {action}")
    print("-" * 74)
    print(f"decision points: {len(trace)}   "
          f"fired: {sum(1 for t in trace if t[5] != 'postpone')}   "
          f"postponed: {sum(1 for t in trace if t[5] == 'postpone')}")

    delivered = Count('delivered').evaluate(pn)
    print(f"delivered at end: {delivered}")

    print()
    print("Seed / reference fitness:")
    # 'postpone' is included as a degeneracy check: with a rate-based primary it
    # must NOT outrank the real seeds (with absolute counts, it does).
    sr.seed_baseline_table(
        (('reference', REFERENCE_POLICY), ('SEED_A', SEED_A),
         ('SEED_B', SEED_B), ('SEED_C', SEED_C), ('postpone', Postpone())),
        conformance_counts, evaluate,
    )

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
)


def run_ga(seed: int = 42):
    return sr.run_ga(EXPERIMENT, seed=seed)


if __name__ == "__main__":
    if "--check" in sys.argv:
        smoke_test()
    else:
        run_ga(seed=42)
