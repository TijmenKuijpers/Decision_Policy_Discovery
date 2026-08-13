r"""
ga_ratelimit_conf.py
--------------------
Genetic-algorithm conformance search over the policy grammar for a two-line
manufacturing process governed by a *rate-limiting* decision policy.

Process
-------
Seven places and four transitions.  Two independent job lines share one
machine:

    arrival_a --[arrive_a]--> waiting_a --[process_a]--> completed_a
    arrival_b --[arrive_b]--> waiting_b --[process_b]--> completed_b
                                   \                /
                                    \--- machine --/

  * arrive_a / arrive_b (evolutions) — jobs arrive and queue in `waiting_x`.
  * process_a (ACTION)  — waiting_a + machine -> completed_a + machine
  * process_b (ACTION)  — waiting_b + machine -> completed_b + machine

The single machine token carries `nr_exec_a` and `nr_exec_b`, the running
count of executions on each line.  Both actions consume and return it, so the
counters are the machine's own state and the lines compete for it.

Reference policy (pi_rate)
--------------------------
    IF   SUM(machine, nr_exec_a) / CLOCK < MAX_PER_TIME_A  THEN FIRE process_a
    ELIF SUM(machine, nr_exec_b) / CLOCK < MAX_PER_TIME_B  THEN FIRE process_b
    ELSE POSTPONE

`SUM(machine, nr_exec_a) / CLOCK` is executions per time unit so far, so each
branch fires only while its line is under its cap.  A is checked first, which
is what gives it priority.  Since exactly one machine token exists,
SUM/MIN/MAX/MEAN over (machine, nr_exec_a) all coincide — several feature
spellings score identically.

Two deviations from the brief, both deliberate
----------------------------------------------
1. CLOCK is an *extension* to the published grammar (see policy_grammar).
   Every terminal in Algorithm 1 is a token aggregate, so elapsed time is not
   expressible and neither is any rate policy.  This experiment cannot be run
   against the language exactly as published.

2. The brief wrote the guards as `clock / nr_exec_X < max_per_time_X`.  That
   is time-per-execution, whose value *falls* as executions accumulate, so the
   condition gets easier to satisfy the more the line has already run — the
   opposite of a cap.  It is also degenerate from the start: nr_exec is 0, so
   Div returns its zero-denominator default of 1.0 and the comparison is
   decided by that constant rather than by the process.  The reference below
   uses the reciprocal, executions-per-time, which throttles as intended.
   The literal reading is kept as REFERENCE_POLICY_AS_BRIEFED so the two can
   be compared directly — run `--check` to see both.

Seed policies (three complementary imperfect starts)
----------------------------------------------------
    A: line-A rate branch only  (never runs B)
    B: line-B rate branch only  (never runs A)
    C: static priority, no rate limiting at all

Crossover between A and B assembles the full two-branch reference without it
ever being seeded directly; ELITE_K=3 keeps all three alive past generation 0.

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
    IfThenElse, Compare,
    Count, Sum, Min, Max, Mean, Clock, Number, Div,
    Fire, Postpone, ENABLED,
)
from ga_fitness import CosmeticConfig
import symbolic_regression as sr

# ─── scenario parameters ──────────────────────────────────────────────────────

# Mean inter-arrival times.  Both arrival rates (1/1.5 = 0.67, 1/2.0 = 0.50)
# sit above the corresponding caps, so the queues stay populated and the *cap*
# is what limits throughput rather than job availability.  If arrivals were the
# binding constraint the policy would have nothing to decide.
ARRIVAL_SCALE_A = 1.5
ARRIVAL_SCALE_B = 2.0

PROCESS_TIME_A  = 1.0   # machine occupied per line-A execution
PROCESS_TIME_B  = 1.0   # machine occupied per line-B execution

# Caps in executions per time unit.  Combined demand on the machine is
# (0.40 + 0.25) * 1.0 = 0.65 utilisation, so both caps are simultaneously
# satisfiable and the policy is not fighting a saturated machine.
MAX_PER_TIME_A  = 0.40
MAX_PER_TIME_B  = 0.25

# ─── domain ───────────────────────────────────────────────────────────────────

PLACES = [
    'arrival_a', 'waiting_a', 'completed_a',
    'arrival_b', 'waiting_b', 'completed_b',
    'machine',
]
TRANSITIONS = ['process_a', 'process_b']
COMPARATORS = ['>', '>=', '<', '<=', '=', '!=']

# (place, attribute) pairs the aggregate features may be built over.
ATTR_FEATURES = [
    ('machine', 'nr_exec_a'),
    ('machine', 'nr_exec_b'),
]

AGGREGATES = [Sum, Min, Max, Mean]

# Integer literals for token counts; floats for the per-time-unit caps.
NUM_INT_POOL   = list(range(9))
NUM_FLOAT_POOL = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]

# ─── GA hyper-parameters ──────────────────────────────────────────────────────

POP_SIZE      = 60
N_GENERATIONS = 50
TOURNAMENT_K  = 3
P_CROSSOVER   = 0.70
P_MUTATION    = 0.35
MAX_DEPTH     = 3
ELITE_K       = 3
N_ROLLOUTS    = 3
HORIZON       = 60

# Primary score is a divergence *rate* in [-1, 0] (see evaluate_primary), so
# the shared DIVERGENCE_COSMETIC preset — tuned for absolute counts and capped
# at 3.0 — would dominate it outright.  Scaled to stay a tie-breaker.
COSMETIC_CONFIG = CosmeticConfig(
    enabled=True,
    w_ite_depth=0.002,
    w_non_numeric=0.002,
    w_non_terminal=0.001,
    w_expr_size=0.001,
    max_cosmetic_penalty=0.05,
)

# ─── manufacturing-system factory ─────────────────────────────────────────────

def make_ratelimit_system(seed: int = 42) -> GymProblem:
    """Return a fully initialised two-line manufacturing GymProblem.

    Rebuilt per rollout rather than deep-copied, matching the other drivers:
    *seed* drives both arrival streams, so rollouts see different workloads
    instead of replaying one identical trace.
    """
    pn  = GymProblem(allow_postpone=True, causal_rl=False)
    rng = np.random.default_rng(seed)
    pn.rng = rng

    arrival_a   = pn.add_var("arrival_a",   var_attributes=["job_id"])
    waiting_a   = pn.add_var("waiting_a",   var_attributes=["job_id"])
    completed_a = pn.add_var("completed_a", var_attributes=["job_id"])
    arrival_b   = pn.add_var("arrival_b",   var_attributes=["job_id"])
    waiting_b   = pn.add_var("waiting_b",   var_attributes=["job_id"])
    completed_b = pn.add_var("completed_b", var_attributes=["job_id"])
    machine     = pn.add_var("machine",
                             var_attributes=["machine_id", "nr_exec_a", "nr_exec_b"])

    # ── arrivals (evolutions) ─────────────────────────────────────────────────
    def _arrival(scale):
        def behavior(tok):
            jid   = tok["job_id"] + 1
            delay = float(rng.exponential(scale=scale))
            job   = {"job_id": jid}
            return [SimToken(job, delay=delay), SimToken(job, delay=delay)]
        return behavior

    pn.add_event([arrival_a], [arrival_a, waiting_a],
                 behavior=_arrival(ARRIVAL_SCALE_A), name="arrive_a")
    pn.add_event([arrival_b], [arrival_b, waiting_b],
                 behavior=_arrival(ARRIVAL_SCALE_B), name="arrive_b")

    # ── processing (actions) ──────────────────────────────────────────────────
    def process_a(job, mach):
        new_mach = {"machine_id": mach["machine_id"],
                    "nr_exec_a":  mach["nr_exec_a"] + 1,
                    "nr_exec_b":  mach["nr_exec_b"]}
        return [SimToken({"job_id": job["job_id"]}, delay=0),
                SimToken(new_mach, delay=PROCESS_TIME_A)]

    def process_b(job, mach):
        new_mach = {"machine_id": mach["machine_id"],
                    "nr_exec_a":  mach["nr_exec_a"],
                    "nr_exec_b":  mach["nr_exec_b"] + 1}
        return [SimToken({"job_id": job["job_id"]}, delay=0),
                SimToken(new_mach, delay=PROCESS_TIME_B)]

    pn.add_action([waiting_a, machine], [completed_a, machine],
                  behavior=process_a,
                  reward_function=lambda job, mach: 1,
                  name="process_a")
    pn.add_action([waiting_b, machine], [completed_b, machine],
                  behavior=process_b,
                  reward_function=lambda job, mach: 1,
                  name="process_b")

    # ── initial state ─────────────────────────────────────────────────────────
    arrival_a.put({"job_id": 0})
    arrival_b.put({"job_id": 0})
    machine.put({"machine_id": 1, "nr_exec_a": 0, "nr_exec_b": 0})

    return pn


# ─── reference policies ───────────────────────────────────────────────────────

def _rate(attribute: str):
    """Executions per time unit so far on one line: SUM(machine, x) / CLOCK."""
    return Div(Sum('machine', attribute), Clock())


def _inverse_rate(attribute: str):
    """The brief's literal spelling: CLOCK / SUM(machine, x)."""
    return Div(Clock(), Sum('machine', attribute))


# Corrected form — throttles as a cap should.  This is the discovery target.
REFERENCE_POLICY = IfThenElse(
    condition=Compare('<', _rate('nr_exec_a'), Number(MAX_PER_TIME_A)),
    then=Fire('process_a'),
    else_=IfThenElse(
        condition=Compare('<', _rate('nr_exec_b'), Number(MAX_PER_TIME_B)),
        then=Fire('process_b'),
        else_=Postpone(),
    ),
)

# Literal reading of the brief, kept for comparison only — see module docstring.
REFERENCE_POLICY_AS_BRIEFED = IfThenElse(
    condition=Compare('<', _inverse_rate('nr_exec_a'), Number(MAX_PER_TIME_A)),
    then=Fire('process_a'),
    else_=IfThenElse(
        condition=Compare('<', _inverse_rate('nr_exec_b'), Number(MAX_PER_TIME_B)),
        then=Fire('process_b'),
        else_=Postpone(),
    ),
)

# ─── fitness ──────────────────────────────────────────────────────────────────

def conformance_counts(tree, reference=None) -> tuple[int, int] | None:
    """Divergences and decision points vs *reference* (default REFERENCE_POLICY)."""
    return sr.conformance_counts(
        tree, reference if reference is not None else REFERENCE_POLICY,
        make_system=lambda s: make_ratelimit_system(seed=s),
        n_rollouts=N_ROLLOUTS, horizon=HORIZON, seed_base=2000,
    )


def evaluate_primary(tree) -> float | None:
    """Negative divergence rate (see sr.primary_rate)."""
    return sr.primary_rate(conformance_counts(tree))


evaluate = sr.make_evaluator(evaluate_primary, COSMETIC_CONFIG)

# ─── random tree constructors ─────────────────────────────────────────────────

VOCAB = sr.ExprVocab(
    places=PLACES,
    attr_features=ATTR_FEATURES,          # the per-line execution counters
    aggregates=AGGREGATES,
    with_clock=True,                      # bare elapsed time
    ratio=sr.RatioSpec(sr.CLOCK_DENOM),   # AGG(...) / CLOCK -- the rate form
                                          # the reference policy is built from
    int_pool=NUM_INT_POOL,
    float_pool=NUM_FLOAT_POOL,
    p_float=0.5,
    w_count=0.30,                         # queue lengths
    w_number=0.15,                        # literal (int count or float cap)
    w_clock=0.10,
    w_aggregate=0.20,                     # raw execution counters
    w_ratio=0.25,                         # the rate expression
    p_agg_swap=0.40,
    p_feature_swap=0.80,
    p_ratio_keep=0.70,                    # keep the rate shape rather than decay
                                          # into its numerator
    int_lo=0, int_hi=8,
    float_lo=0.0, float_hi=2.0,
)

CONFIG = sr.GrammarConfig.from_vocab(
    VOCAB,
    comparators=COMPARATORS,
    transitions=TRANSITIONS,
    max_depth=MAX_DEPTH,
    p_and=0.6,             # And-vs-Or split when building a random condition
    p_fire=0.7,            # Fire-vs-Postpone split when building a random action
    p_mutate_fire=0.6,     # Fire-vs-Postpone split when mutating an action
    p_mutation=P_MUTATION,
)

# ─── seed policies ────────────────────────────────────────────────────────────

# Seed A — line-A rate branch only.  Correct on every A decision, but never
#          runs B, so it diverges wherever the reference falls through.
SEED_A = IfThenElse(
    condition=Compare('<', _rate('nr_exec_a'), Number(MAX_PER_TIME_A)),
    then=Fire('process_a'),
    else_=Postpone(),
)

# Seed B — line-B rate branch only: the block Seed A is missing, and vice versa.
SEED_B = IfThenElse(
    condition=Compare('<', _rate('nr_exec_b'), Number(MAX_PER_TIME_B)),
    then=Fire('process_b'),
    else_=Postpone(),
)

# Seed C — static priority with no rate limiting: always run A when a job is
#          queued, otherwise B.  A strong baseline on the A-priority ordering
#          but blind to both caps.
SEED_C = IfThenElse(
    condition=Compare('>', Count('waiting_a'), Number(0)),
    then=Fire('process_a'),
    else_=Fire('process_b'),
)

SEED_POLICIES = [SEED_A, SEED_B, SEED_C]
SEED_POLICY   = SEED_A

# ─── model smoke test ─────────────────────────────────────────────────────────

def smoke_test(seed: int = 2000, horizon: int = 40) -> None:
    """Run the reference policy once and print the rate-limiting behaviour.

    Both realised rates should sit at or just under their caps, and the
    decision count must be non-zero — if it is zero the policy is never being
    consulted and nothing downstream is meaningful (see ga_batching_conf for
    why that can happen).
    """
    trace = []

    def heuristic(pn, actions_dict):
        action = REFERENCE_POLICY.evaluate(pn)
        trace.append((
            pn.clock,
            Count('waiting_a', ENABLED).evaluate(pn),
            Count('waiting_b', ENABLED).evaluate(pn),
            Sum('machine', 'nr_exec_a').evaluate(pn),
            Sum('machine', 'nr_exec_b').evaluate(pn),
            _rate('nr_exec_a').evaluate(pn),
            _rate('nr_exec_b').evaluate(pn),
            action,
        ))
        if action == 'postpone':
            return 'postpone'
        if action in actions_dict and actions_dict[action]:
            return {action: actions_dict[action][0]}
        return 'postpone'

    pn = make_ratelimit_system(seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        pn.testing_run(solver=HeuristicSolver(heuristic_function=heuristic),
                       length=horizon)

    print("=" * 86)
    print(f"Reference policy trace  (cap_A={MAX_PER_TIME_A}, "
          f"cap_B={MAX_PER_TIME_B}, horizon={horizon})")
    print("=" * 86)
    print(f"{'clock':>7} {'wait_a':>7} {'wait_b':>7} {'n_a':>5} {'n_b':>5} "
          f"{'rate_a':>8} {'rate_b':>8}  decision")
    print("-" * 86)
    for row in trace[:28]:
        clock, wa, wb, na, nb, ra, rb, action = row
        print(f"{clock:>7.2f} {wa:>7} {wb:>7} {na:>5} {nb:>5} "
              f"{ra:>8.3f} {rb:>8.3f}  {action}")
    if len(trace) > 28:
        print(f"  ... {len(trace) - 28} more decisions")
    print("-" * 86)

    fired_a = sum(1 for t in trace if t[7] == 'process_a')
    fired_b = sum(1 for t in trace if t[7] == 'process_b')
    postponed = sum(1 for t in trace if t[7] == 'postpone')
    print(f"decision points: {len(trace)}   process_a: {fired_a}   "
          f"process_b: {fired_b}   postpone: {postponed}")

    done_a = Count('completed_a').evaluate(pn)
    done_b = Count('completed_b').evaluate(pn)
    print(f"completed_a={done_a}  completed_b={done_b}  clock={pn.clock:.2f}")
    if pn.clock > 0:
        print(f"realised rate A = {done_a / pn.clock:.3f}  (cap {MAX_PER_TIME_A})")
        print(f"realised rate B = {done_b / pn.clock:.3f}  (cap {MAX_PER_TIME_B})")

    # ── the briefed vs corrected guard ───────────────────────────────────────
    print()
    print("Briefed guard (CLOCK / nr_exec) vs corrected (nr_exec / CLOCK):")
    # NOTE: gympn calls heuristic_function(net, tokens, bindings) first and only
    # falls back to the 2-argument form on TypeError.  A heuristic that declares
    # extra *defaulted* parameters therefore accepts the 3-argument call and
    # silently receives `bindings` in that slot instead of its default.  Close
    # over what you need and absorb the rest with *_.
    def _make_tracer(ref, acts):
        def h(pn_, actions_dict, *_):
            a = ref.evaluate(pn_)
            acts.append(a)
            if a == 'postpone':
                return 'postpone'
            if a in actions_dict and actions_dict[a]:
                return {a: actions_dict[a][0]}
            return 'postpone'
        return h

    for name, ref in (('corrected', REFERENCE_POLICY),
                      ('as-briefed', REFERENCE_POLICY_AS_BRIEFED)):
        pn2 = make_ratelimit_system(seed=seed)
        acts = []
        h = _make_tracer(ref, acts)

        with contextlib.redirect_stdout(io.StringIO()):
            pn2.testing_run(solver=HeuristicSolver(heuristic_function=h),
                            length=horizon)
        print(f"  {name:<11} decisions={len(acts):>4}  "
              f"A={acts.count('process_a'):>4}  B={acts.count('process_b'):>4}  "
              f"postpone={acts.count('postpone'):>4}  "
              f"completed_a={Count('completed_a').evaluate(pn2):>3}  "
              f"completed_b={Count('completed_b').evaluate(pn2):>3}")

    # ── seed / baseline fitness ──────────────────────────────────────────────
    print()
    print("Seed / reference fitness:")
    sr.seed_baseline_table(
        (('reference', REFERENCE_POLICY), ('SEED_A', SEED_A),
         ('SEED_B', SEED_B), ('SEED_C', SEED_C), ('postpone', Postpone())),
        conformance_counts, evaluate,
    )

# ─── experiment record ──────────────────────────────────────

EXPERIMENT = sr.Experiment(
    name="rate-limit conformance policy search",
    reference_label="pi_rate",
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
