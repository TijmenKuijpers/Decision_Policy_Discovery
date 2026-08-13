"""
ga_priority_conf.py
-------------------
Genetic-algorithm conformance search over the policy grammar for a single-line
manufacturing process governed by a *priority look-ahead* decision policy.

Process
-------
Five places and three transitions:

    arrival --[arrive]--> waiting_1 --[pre_process]--> waiting_2 --[process]--> completed
                                                            \                /
                                                             \--- machine --/

  * arrive      (evolution) — announces a case, with a priority attribute.
  * pre_process (evolution) — moves a case from waiting_1 to waiting_2 as soon
                              as it becomes available.  No resource needed.
  * process     (ACTION)    — waiting_2 + machine -> completed + machine.

Both waiting places are priority queues: highest `priority` first (see
_sort_by_priority).  That ordering is what decides *which* case `process`
consumes, since the solver takes the first available binding.

Reference policy (pi_prio)
--------------------------
    IF MAX(waiting_1, priority) <= MAX(waiting_2, priority)
        THEN FIRE process
        ELSE POSTPONE

"Do not start on what is ready if something better is still upstream."

Why arrivals are announced ahead of time
----------------------------------------
gympn only consults the policy once no evolution is enabled at the current
clock (see GymProblem.bindings, and the note in ga_batching_conf).  waiting_1
is the input place of an evolution, so by the time any decision is taken
pre_process has already drained every *enabled* token out of it — MAX over
waiting_1's enabled tokens is always the empty-set value at exactly the moments
the policy runs, and the reference condition would degenerate to "always fire".

So `arrive` announces each case LEAD_TIME before it becomes available.  The
announced cases sit in waiting_1 as future-dated tokens: visible to the ALL
selector, not yet consumable by pre_process.  waiting_1 therefore means "known
but not yet arrived" and waiting_2 means "ready to run", which is what makes
the look-ahead comparison meaningful.  This also makes the ALL/ENABLED
distinction load-bearing — under ENABLED the reference policy is degenerate.

Postponing is always productive here: the clock advances, the better upstream
case becomes available, pre_process moves it into waiting_2 and the condition
flips to true.  (A capacity-limited pre_process would deadlock instead — with
waiting_2 full, postponing could never let the better case through.)

Empty-set note: MAX over an empty selection is -inf (see policy_grammar).  At
any decision point waiting_2 is non-empty, since `process` could not otherwise
bind, so MAX(waiting_2) is finite.  When waiting_1 is empty MAX is -inf and the
condition holds — nothing upstream, so run what is ready.  Both edges behave.

Seed policies (three complementary imperfect starts)
----------------------------------------------------
    A: right shape, wrong aggregate  (MIN instead of MAX)
    B: availability only             (ignores priority entirely)
    C: no policy at all              (always fire)

Expected runtime: ~10-20 minutes for the default 60 x 50 configuration.
"""

import io
import sys
import contextlib

import numpy as np
from sortedcontainers import SortedList

sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")

from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import HeuristicSolver

from policy_grammar import (
    IfThenElse, Compare,
    Count, Sum, Min, Max, Mean,
    Fire, Postpone, ENABLED,
)
from ga_fitness import CosmeticConfig
import symbolic_regression as sr

# ─── scenario parameters ──────────────────────────────────────────────────────

# Mean time between case announcements.
ARRIVAL_SCALE    = 1.2

# How far ahead a case is announced.  Must exceed ARRIVAL_SCALE for waiting_1
# to hold more than one pending case — that is what gives MAX(waiting_1) and
# the priority sort on waiting_1 something to range over.
#
# It also controls how often the reference policy fires: a longer lead means
# more pending cases, a higher MAX(waiting_1), and so more postponing.  Swept
# at horizon 60: lead 1.0 -> 38% fire, 2.0 -> 35%, 3.0 -> 28%, 4.0 -> 21%.
# 2.0 keeps waiting_1 genuinely multi-token while leaving the policy a healthy
# mix of both decisions.
LEAD_TIME        = 2.0

PROCESS_TIME     = 1.0    # machine occupied per case

# Inclusive priority range carried by each case.
PRIORITY_LO      = 1
PRIORITY_HI      = 9

# ─── domain ───────────────────────────────────────────────────────────────────

PLACES      = ['arrival', 'waiting_1', 'waiting_2', 'completed', 'machine']
TRANSITIONS = ['process']
COMPARATORS = ['>', '>=', '<', '<=', '=', '!=']

ATTR_FEATURES = [
    ('waiting_1', 'priority'),
    ('waiting_2', 'priority'),
    ('completed', 'priority'),
]

AGGREGATES = [Sum, Min, Max, Mean]

NUM_INT_POOL   = list(range(11))
NUM_FLOAT_POOL = [0.5, 1.5, 2.5, 5.0]

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

# Primary score is a divergence rate in [-1, 0]; scaled so the cosmetic term
# stays a tie-breaker rather than the dominant objective.
COSMETIC_CONFIG = CosmeticConfig(
    enabled=True,
    w_ite_depth=0.002,
    w_non_numeric=0.002,
    w_non_terminal=0.001,
    w_expr_size=0.001,
    max_cosmetic_penalty=0.05,
)

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

    arrival   = pn.add_var("arrival",   var_attributes=["case_id"])
    waiting_1 = pn.add_var("waiting_1", var_attributes=["case_id", "priority"])
    waiting_2 = pn.add_var("waiting_2", var_attributes=["case_id", "priority"])
    completed = pn.add_var("completed", var_attributes=["case_id", "priority"])
    machine   = pn.add_var("machine",   var_attributes=["machine_id"])

    # Both waiting places are priority queues, as required.
    _sort_by_priority(waiting_1)
    _sort_by_priority(waiting_2)

    # ── arrive (evolution) ────────────────────────────────────────────────────
    def arrive(tok):
        cid      = tok["case_id"] + 1
        inter    = float(rng.exponential(scale=ARRIVAL_SCALE))
        priority = int(rng.integers(PRIORITY_LO, PRIORITY_HI + 1))
        case     = {"case_id": cid, "priority": priority}
        # The generator advances by `inter`; the case itself only becomes
        # available LEAD_TIME later, so it is visible-but-pending in waiting_1.
        return [SimToken({"case_id": cid}, delay=inter),
                SimToken(case, delay=inter + LEAD_TIME)]

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
    arrival.put({"case_id": 0})
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

def conformance_counts(tree, reference=None) -> tuple[int, int] | None:
    """Divergences and decision points vs *reference* (default REFERENCE_POLICY).

    `_select_binding` is passed as the binding picker so the priority-queue
    discipline applies during measurement -- see its docstring for why sorting
    the marking alone is not enough.
    """
    return sr.conformance_counts(
        tree, reference if reference is not None else REFERENCE_POLICY,
        make_system=lambda s: make_priority_system(seed=s),
        n_rollouts=N_ROLLOUTS, horizon=HORIZON, seed_base=3000,
        pick_binding=_select_binding,
    )


def evaluate_primary(tree) -> float | None:
    """Negative divergence rate (see sr.primary_rate)."""
    return sr.primary_rate(conformance_counts(tree))


evaluate = sr.make_evaluator(evaluate_primary, COSMETIC_CONFIG)

# ─── expression vocabulary ────────────────────────────────────────────────────

VOCAB = sr.ExprVocab(
    places=PLACES,
    attr_features=ATTR_FEATURES,          # priority on each queue + completed
    aggregates=AGGREGATES,
    int_pool=NUM_INT_POOL,
    float_pool=NUM_FLOAT_POOL,
    p_float=0.25,
    w_count=0.25,                         # queue lengths
    w_number=0.15,                        # literal threshold
    w_aggregate=0.60,                     # the priority aggregates the reference needs

    # p_family_escape lets a Count-based tree reach the priority aggregates
    # without discarding the comparison structure that made it good.  The
    # reference here needs exactly that jump (COUNT -> MAX over priority), and a
    # GA seeded with a strong count-based policy otherwise stalls on it.
    p_family_escape=0.15,
    p_agg_swap=0.45,
    p_feature_swap=0.85,

    int_lo=0, int_hi=10,
    float_nudges=(-0.5, 0.5),
    float_lo=0.0,
    float_hi=None,                        # NOTE: no upper clamp -- a float literal
                                          # can random-walk upward without bound.
                                          # Preserved deliberately; see the plan's
                                          # "defects preserved" list.
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

# ─── model smoke test ─────────────────────────────────────────────────────────

def smoke_test(seed: int = 3000, horizon: int = 30) -> None:
    """Run the reference policy once and check the model behaves as designed.

    Checks, in order:
      * decision points exist at all (see the module note on why they might not)
      * waiting_1 and waiting_2 both hold several tokens, so MAX ranges over
        something and the priority sort has work to do
      * waiting_2 really is a priority queue - the case `process` consumes is
        the highest-priority one available
      * the policy both fires and postpones
    """
    trace   = []
    picks   = []   # (priority chosen, priorities available) per firing

    def heuristic(pn, actions_dict):
        action = REFERENCE_POLICY.evaluate(pn)
        w1 = [t.value["priority"] for p in pn.places if p._id == 'waiting_1'
              for t in p.marking]
        w2 = [t.value["priority"] for p in pn.places if p._id == 'waiting_2'
              for t in p.marking if t.time <= pn.clock]
        trace.append((pn.clock, list(w1), list(w2),
                      Max('waiting_1', 'priority').evaluate(pn),
                      Max('waiting_2', 'priority').evaluate(pn),
                      action))
        if action == 'postpone':
            return 'postpone'
        if action in actions_dict and actions_dict[action]:
            entries = actions_dict[action]
            chosen  = _select_binding(entries)
            # Compare what was taken against everything that was on offer.
            taken     = _entry_priority(chosen)
            offered   = [p for p in (_entry_priority(e) for e in entries)
                         if p is not None]
            naive     = _entry_priority(entries[0])   # what entries[0] would give
            if taken is not None and offered:
                picks.append((taken, sorted(offered, reverse=True), naive))
            return {action: chosen}
        return 'postpone'

    pn = make_priority_system(seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        pn.testing_run(solver=HeuristicSolver(heuristic_function=heuristic),
                       length=horizon)

    print("=" * 88)
    print(f"Reference policy trace  (lead={LEAD_TIME}, horizon={horizon})")
    print("=" * 88)
    print(f"{'clock':>7}  {'waiting_1 (pending)':<26} {'waiting_2 (ready)':<22} "
          f"{'maxW1':>6} {'maxW2':>6}  decision")
    print("-" * 88)
    def _fmt(queue, width):
        s = ",".join(str(p) for p in queue)
        return s if len(s) <= width else s[:width - 3] + "..."

    for clock, w1, w2, m1, m2, action in trace[:24]:
        m1s = '-inf' if m1 == -float('inf') else f"{m1:g}"
        m2s = '-inf' if m2 == -float('inf') else f"{m2:g}"
        print(f"{clock:>7.2f}  {_fmt(w1, 26):<26} {_fmt(w2, 22):<22} "
              f"{m1s:>6} {m2s:>6}  {action}")
    if len(trace) > 24:
        print(f"  ... {len(trace) - 24} more decisions")
    print("-" * 88)

    fired     = sum(1 for t in trace if t[5] != 'postpone')
    postponed = sum(1 for t in trace if t[5] == 'postpone')
    print(f"decision points: {len(trace)}   fired: {fired}   postponed: {postponed}")

    # ── model sanity checks ──────────────────────────────────────────────────
    print()
    print("Model checks:")
    ok = True

    if len(trace) == 0:
        print("  [FAIL] no decision points - policy never consulted")
        ok = False
    else:
        print(f"  [ OK ] {len(trace)} decision points")

    if fired == 0 or postponed == 0:
        print(f"  [FAIL] policy is one-sided (fired={fired}, postponed={postponed})")
        ok = False
    else:
        print(f"  [ OK ] policy both fires and postpones")

    max_w1 = max((len(t[1]) for t in trace), default=0)
    max_w2 = max((len(t[2]) for t in trace), default=0)
    if max_w1 < 2 or max_w2 < 2:
        print(f"  [FAIL] queues too shallow for MAX/sorting to matter "
              f"(max |waiting_1|={max_w1}, max |waiting_2|={max_w2})")
        ok = False
    else:
        print(f"  [ OK ] queues deep enough (max |waiting_1|={max_w1}, "
              f"max |waiting_2|={max_w2})")

    bad_picks = [(p, avail) for p, avail, _ in picks if avail and p != avail[0]]
    if not picks:
        print("  [WARN] no firings recorded, priority ordering unverified")
        ok = False
    elif bad_picks:
        print(f"  [FAIL] priority discipline broken in {len(bad_picks)}/{len(picks)} "
              f"firings, e.g. took {bad_picks[0][0]} from {bad_picks[0][1]}")
        ok = False
    else:
        print(f"  [ OK ] all {len(picks)} firings took the highest available "
              f"priority (waiting_2 discipline works)")

    # How often gympn's time-ordered bindings[0] would have picked differently.
    would_differ = [(t, n, av) for t, av, n in picks if n is not None and n != t]
    if would_differ:
        t, n, av = would_differ[0]
        print(f"  [note] taking bindings[0] instead would have differed in "
              f"{len(would_differ)}/{len(picks)} firings "
              f"(e.g. {n} instead of {t}, offered {av}) - "
              f"the marking sort alone does not control binding order")
    else:
        print("  [note] bindings[0] would have agreed on every firing here")

    print(f"\n  => {'MODEL OK' if ok else 'MODEL NEEDS ATTENTION'}")

    # ── ALL vs ENABLED degeneracy demonstration ──────────────────────────────
    enabled_ref = IfThenElse(
        condition=Compare('<=',
                          Max('waiting_1', 'priority', ENABLED),
                          Max('waiting_2', 'priority', ENABLED)),
        then=Fire('process'),
        else_=Postpone(),
    )
    counts_all = conformance_counts(REFERENCE_POLICY)
    counts_en  = conformance_counts(enabled_ref)
    print()
    print("ALL vs ENABLED on waiting_1 (why the reference uses ALL):")
    print(f"  ALL     : {counts_all}")
    print(f"  ENABLED : {counts_en}  <- degenerate, waiting_1 is empty of "
          f"enabled tokens at every decision")

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
)


def run_ga(seed: int = 42):
    return sr.run_ga(EXPERIMENT, seed=seed)


if __name__ == "__main__":
    if "--check" in sys.argv:
        smoke_test()
    else:
        run_ga(seed=42)
