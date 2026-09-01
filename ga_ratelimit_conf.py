r"""
ga_ratelimit_conf.py
--------------------
Genetic-algorithm conformance search over the policy grammar for a two-line
manufacturing process governed by a *rate-limiting* decision policy.

Process
-------

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


Seed policies (three complementary imperfect starts)
----------------------------------------------------
    A: line-A rate branch only  (never runs B)
    B: line-B rate branch only  (never runs A)
    C: static priority, no rate limiting at all

Crossover between A and B assembles the full two-branch reference without it
ever being seeded directly; ELITE_K=3 keeps all three alive past generation 0.
"""

import numpy as np

import gympn_path  # noqa: F401  -- makes gympn importable; see its docstring

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

from policy_grammar import (
    IfThenElse, Compare,
    Count, Sum, Clock, Number, Div,
    Fire, Postpone,
)
import symbolic_regression as sr
import ga_shared as shared

# ─── scenario parameters ──────────────────────────────────────────────────────
ARRIVAL_SCALE_A = 1.5
ARRIVAL_SCALE_B = 2.0
PROCESS_TIME_A  = 1.0   # machine occupied per line-A execution
PROCESS_TIME_B  = 1.0   # machine occupied per line-B execution

# Caps in executions per time unit.  
MAX_PER_TIME_A  = 0.40
MAX_PER_TIME_B  = 0.25

# ─── domain ───────────────────────────────────────────────────────────────────
PLACES = [
    'arrival_a', 'waiting_a', 'completed_a',
    'arrival_b', 'waiting_b', 'completed_b',
    'machine',
]
TRANSITIONS = ['process_a', 'process_b']
COMPARATORS = shared.COMPARATORS

# (place, attribute) pairs the aggregate features may be built over.
ATTR_FEATURES = [
    ('machine', 'nr_exec_a'),
    ('machine', 'nr_exec_b'),
]

# ─── GA hyper-parameters ──────────────────────────────────────────────────────

POP_SIZE      = shared.POP_SIZE
N_GENERATIONS = shared.N_GENERATIONS
TOURNAMENT_K  = shared.TOURNAMENT_K
P_CROSSOVER   = shared.P_CROSSOVER
P_MUTATION    = shared.P_MUTATION
MAX_DEPTH     = shared.MAX_DEPTH
ELITE_K       = shared.ELITE_K
# 8 x 60 yields 486 decision points.
N_ROLLOUTS    = 8
HORIZON       = shared.HORIZON

COSMETIC_CONFIG = shared.COSMETIC_CONFIG

SCALE_CONFIG = shared.SCALE_CONFIG

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

# Scoring is target-driven: the reference is simulated once, its decision
# points are recorded, and candidates are scored by replaying them.
_TRACES = sr.TraceCache(
    make_system=lambda s: make_ratelimit_system(seed=s),
    seed_base=2000,
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

# ─── random tree constructors ─────────────────────────────────────────────────

VOCAB = shared.make_vocab(PLACES, ATTR_FEATURES)

CONFIG = shared.make_config(VOCAB, TRANSITIONS)

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
    scale_config=SCALE_CONFIG,
)


def run_ga(seed: int = 42):
    return sr.run_ga(EXPERIMENT, seed=seed)


if __name__ == "__main__":
    run_ga(seed=42)
