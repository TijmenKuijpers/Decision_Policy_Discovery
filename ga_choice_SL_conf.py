"""
ga_choice_SL_conf.py
--------------------
Genetic-algorithm conformance search over the policy grammar for the
two-product assembly system with service-level (SLA) tracking.

The fitness is *negative divergence* from the reference SLA policy
(π_sla): the GA searches for alternative grammar trees that make the
same action choice as π_sla as often as possible.

Reference policy (π_sla)
------------------------
    IF   SLA(phone_delivered) < 0.95  →  FIRE phone_production
    ELIF SLA(game_delivered)  < 0.95  →  FIRE game_production
    ELIF game_demand > 0              →  FIRE game_production
    ELSE                              →  FIRE phone_production

where SLA(p) = SUM(p, 'on_time', ALL) / COUNT(p, ALL).

Seed policies (three complementary imperfect starts)
----------------------------------------------------
    A:  phone-SLA rescue only  (missing game-SLA branch)
    B:  game-SLA rescue only   (missing phone-SLA branch)
    C:  trivial demand-driven  (missing both SLA branches)

Crossover between A and B can assemble the full reference policy without
seeding it directly.  ELITE_K=3 preserves all three seeds past generation 0.

Expected runtime: ~10–20 minutes for the default 80 × 100 configuration.
"""

import copy
import sys
import random

import numpy as np

sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

from policy_grammar import (
    IfThenElse, And, Compare,
    Count, Sum, Div, Number,
    Fire,
)
from ga_fitness import DIVERGENCE_COSMETIC
import symbolic_regression as sr

# ─── scenario parameters (mirror choice_gym_SL.py) ───────────────────────────

CHIP_ARRIVAL_SCALE       = 2.0
PHONE_CASE_ARRIVAL_SCALE = 3.5
GAME_PRODUCTION_DELAY    = 1
PHONE_PRODUCTION_DELAY   = 2
GAME_DEMAND_INTERVAL     = 3
PHONE_DEMAND_INTERVAL    = 5
GAME_DEADLINE_LO,  GAME_DEADLINE_HI  = 3, 6
PHONE_DEADLINE_LO, PHONE_DEADLINE_HI = 2, 5

# ─── domain ───────────────────────────────────────────────────────────────────

PLACES = [
    'stock_chip', 'stock_phone_case',
    'game_resource', 'phone_resource',
    'phone_demand', 'game_demand',
    'phone_delivered', 'game_delivered',
]
DELIVERED_PLACES = ['phone_delivered', 'game_delivered']
TRANSITIONS      = ['game_production', 'phone_production']
COMPARATORS      = ['>', '>=', '<', '<=', '=', '!=']

# Number literals: integers 0-8 for token counts; floats for SLA thresholds.
NUM_INT_POOL   = list(range(9))
NUM_FLOAT_POOL = [0.5, 0.75, 0.90, 0.95, 0.99]
NUM_POOL       = NUM_INT_POOL + NUM_FLOAT_POOL

# ─── GA hyper-parameters ──────────────────────────────────────────────────────

POP_SIZE      = 80
N_GENERATIONS = 100
TOURNAMENT_K  = 3
P_CROSSOVER   = 0.70
P_MUTATION    = 0.35
MAX_DEPTH     = 3
N_ROLLOUTS    = 5
HORIZON       = 50
ELITE_K       = 3   # number of best individuals carried forward unchanged

COSMETIC_CONFIG = DIVERGENCE_COSMETIC

# ─── assembly-system factory ──────────────────────────────────────────────────

def make_sl_assembly_system() -> GymProblem:
    """Return a fully initialised SL assembly-system GymProblem.

    Behavior functions are defined as inner closures that capture *pn*
    so that pn.clock is always the clock of the correct instance.
    """
    pn = GymProblem(allow_postpone=True, causal_rl=False)
    pn.rng = np.random.default_rng(42)

    a_chip = pn.add_var("chip supply",         var_attributes=["chip_id"])
    a_pc   = pn.add_var("phone case supply",   var_attributes=["phone_case_id"])
    gda    = pn.add_var("game_demand_arrival",  var_attributes=["request_date"])
    pda    = pn.add_var("phone_demand_arrival", var_attributes=["request_date"])
    p_dem  = pn.add_var("phone_demand",         var_attributes=["request_date"])
    g_dem  = pn.add_var("game_demand",          var_attributes=["request_date"])
    s_chip = pn.add_var("stock_chip",           var_attributes=["chip_id"])
    s_pc   = pn.add_var("stock_phone_case",     var_attributes=["phone_case_id"])
    r_ph   = pn.add_var("phone_resource",       var_attributes=["phone_id"])
    r_gm   = pn.add_var("game_resource",        var_attributes=["game_id"])
    g_del  = pn.add_var("game_delivered",       var_attributes=["request_date", "on_time"])
    p_del  = pn.add_var("phone_delivered",      var_attributes=["request_date", "on_time"])

    pn.set_unobservable(token_attrs={
        'stock_chip':       ['chip_id'],
        'stock_phone_case': ['phone_case_id'],
        'phone_resource':   ['phone_id'],
        'game_resource':    ['game_id'],
    })

    def chip_arrival(tok):
        cid   = tok["chip_id"] + 1
        delay = np.random.default_rng(42 + cid).exponential(scale=CHIP_ARRIVAL_SCALE)
        t     = {"chip_id": cid}
        return [SimToken(t, delay=delay), SimToken(t, delay=delay)]

    def pc_arrival(tok):
        pid   = tok["phone_case_id"] + 1
        delay = np.random.default_rng(42 + pid).exponential(scale=PHONE_CASE_ARRIVAL_SCALE)
        t     = {"phone_case_id": pid}
        return [SimToken(t, delay=delay), SimToken(t, delay=delay)]

    def game_demand_arrive(demand):
        new_d = {"request_date": pn.clock + random.randint(GAME_DEADLINE_LO, GAME_DEADLINE_HI)}
        return [SimToken(demand, delay=GAME_DEMAND_INTERVAL), SimToken(new_d, delay=0)]

    def phone_demand_arrive(demand):
        new_d = {"request_date": pn.clock + random.randint(PHONE_DEADLINE_LO, PHONE_DEADLINE_HI)}
        return [SimToken(demand, delay=PHONE_DEMAND_INTERVAL), SimToken(new_d, delay=0)]

    def game_production_behavior(stock_chip, game_resource, game_demand):
        on_time   = pn.clock <= game_demand["request_date"]
        delivered = {"request_date": game_demand["request_date"], "on_time": on_time}
        return [SimToken(game_resource, delay=GAME_PRODUCTION_DELAY), SimToken(delivered, delay=0)]

    def phone_production_behavior(stock_chip, stock_phone_case, phone_resource, phone_demand):
        on_time   = pn.clock <= phone_demand["request_date"]
        delivered = {"request_date": phone_demand["request_date"], "on_time": on_time}
        return [SimToken(phone_resource, delay=PHONE_PRODUCTION_DELAY), SimToken(delivered, delay=0)]

    pn.add_event([a_chip], [a_chip, s_chip], behavior=chip_arrival,        name="chip_arrival")
    pn.add_event([a_pc],   [a_pc,   s_pc],  behavior=pc_arrival,           name="phone_case_arrival")
    pn.add_event([gda],    [gda,    g_dem], behavior=game_demand_arrive,   name="game_demand_arrive")
    pn.add_event([pda],    [pda,    p_dem], behavior=phone_demand_arrive,  name="phone_demand_arrive")

    pn.add_action(
        [s_chip, r_gm, g_dem], [r_gm, g_del],
        behavior=game_production_behavior,
        reward_function=lambda sc, gr, gd: 1 if pn.clock <= gd["request_date"] else 0,
        name="game_production",
    )
    pn.add_action(
        [s_chip, s_pc, r_ph, p_dem], [r_ph, p_del],
        behavior=phone_production_behavior,
        reward_function=lambda sc, sp, pr, pd: 3 if pn.clock <= pd["request_date"] else 0,
        name="phone_production",
    )

    a_chip.put({"chip_id": 0})
    a_pc.put({"phone_case_id": 0})
    gda.put({"request_date": 0})
    pda.put({"request_date": 0})
    r_gm.put({"game_id": 1})
    r_ph.put({"phone_id": 1})

    return pn


# Build once; deepcopy per fitness evaluation keeps simulation state isolated.
_BASE_PN: GymProblem = make_sl_assembly_system()

# ─── reference policy (π_sla) ─────────────────────────────────────────────────

def _sla_below(delivered_place: str, threshold: float) -> And:
    """SLA(place) < threshold, expressed without division-by-zero risk."""
    return And(
        Compare('>', Count(delivered_place), Number(0)),
        Compare('<',
            Div(Sum(delivered_place, 'on_time'), Count(delivered_place)),
            Number(threshold),
        ),
    )


REFERENCE_POLICY = IfThenElse(
    condition=_sla_below('phone_delivered', 0.95),
    then=Fire('phone_production'),
    else_=IfThenElse(
        condition=_sla_below('game_delivered', 0.95),
        then=Fire('game_production'),
        else_=IfThenElse(
            condition=Compare('>', Count('game_demand'), Number(0)),
            then=Fire('game_production'),
            else_=Fire('phone_production'),
        ),
    ),
)

# ─── fitness ──────────────────────────────────────────────────────────────────

def conformance_counts(tree, reference=None) -> tuple[int, int] | None:
    """Divergences and decision points vs REFERENCE_POLICY.

    Unlike the other drivers this one deep-copies a prototype net per rollout
    rather than rebuilding it, so the `make_system` callback ignores the seed.
    """
    return sr.conformance_counts(
        tree, reference if reference is not None else REFERENCE_POLICY,
        make_system=lambda _s: copy.deepcopy(_BASE_PN),
        n_rollouts=N_ROLLOUTS, horizon=HORIZON,
    )


def evaluate_primary(tree) -> float | None:
    """Negative MEAN ABSOLUTE divergence per rollout -- not a rate.

    This experiment predates the switch to a rate-based primary and its
    COSMETIC_CONFIG (the unscaled DIVERGENCE_COSMETIC, capped at 3.0) is tuned
    against a primary measured in tens.  Swapping in sr.primary_rate here would
    put the primary in [-1, 0] and let the cosmetic term dominate conformance
    outright, so the two must be changed together or not at all.
    """
    return sr.primary_count(conformance_counts(tree), N_ROLLOUTS)


evaluate = sr.make_evaluator(evaluate_primary, COSMETIC_CONFIG)

# ─── random tree constructors ─────────────────────────────────────────────────

VOCAB = sr.ExprVocab(
    places=PLACES,

    # This net's only attribute-bearing feature is the boolean `on_time` flag on
    # the two delivered places, and SUM is the only aggregate that says anything
    # useful about it: MIN is 1 iff every delivery was on time, MAX is 1 iff any
    # was, and MEAN duplicates the ratio branch below.  Declaring aggregates
    # narrowly is a vocabulary fact, not a compatibility shim.
    attr_features=[(p, 'on_time') for p in DELIVERED_PLACES],
    aggregates=(Sum,),

    # SLA ratio: SUM(delivered, on_time) / COUNT(delivered) -- the same place on
    # both sides, which is what makes it a share rather than a rate.
    ratio=sr.RatioSpec(sr.COUNT_SAME_PLACE,
                       places=DELIVERED_PLACES, attribute='on_time'),

    # No future-dated tokens here, so ALL and ENABLED coincide.
    use_selectors=False,

    int_pool=NUM_INT_POOL,
    float_pool=NUM_FLOAT_POOL,
    # The old flat NUM_POOL drew from ints and floats together, giving
    # P(float) = 5/14; kept explicitly so stratifying does not shift it.
    p_float=5 / 14,

    w_count=0.50,                         # simple token count
    w_number=0.15,                        # literal (int 0-8 or SLA threshold)
    w_aggregate=0.15,                     # on-time delivery count
    w_ratio=0.20,                         # SLA ratio (needed for conformance)

    p_count_place=1.0,                    # no selector to swap, so always the place
    p_ratio_keep=0.5,

    int_lo=0, int_hi=8,
    float_lo=0.0, float_hi=1.0,           # an SLA is a fraction
)

CONFIG = sr.GrammarConfig.from_vocab(
    VOCAB,
    comparators=COMPARATORS,
    transitions=TRANSITIONS,
    max_depth=MAX_DEPTH,
    p_and=0.7,             # And-vs-Or split when building a random condition
    p_fire=0.6,            # Fire-vs-Postpone split when building a random action
    p_mutate_fire=0.5,     # Fire-vs-Postpone split when mutating an action
    p_mutation=P_MUTATION,
)

# ─── seed policy ──────────────────────────────────────────────────────────────

# Three complementary imperfect seeds.  Each captures one structural building
# block of the reference policy.  Crossover between them can assemble the
# full three-branch reference without ever seeding it directly.
#
# Seed A — phone-SLA rescue branch only (matches reference in the phone cases,
#           but misses the game-SLA rescue → fitness ~ -0.85).
SEED_A = IfThenElse(
    condition=_sla_below('phone_delivered', 0.95),
    then=Fire('phone_production'),
    else_=IfThenElse(
        condition=Compare('>', Count('game_demand'), Number(0)),
        then=Fire('game_production'),
        else_=Fire('phone_production'),
    ),
)

# Seed B — game-SLA rescue branch only (handles the game-SLA case that makes
#           the reference diverge from the trivial tree → fitness ~ -0.25).
SEED_B = IfThenElse(
    condition=_sla_below('game_delivered', 0.95),
    then=Fire('game_production'),
    else_=IfThenElse(
        condition=Compare('>', Count('game_demand'), Number(0)),
        then=Fire('game_production'),
        else_=Fire('phone_production'),
    ),
)

# Seed C — trivial demand-driven policy (strong baseline; missing both SLA
#           checks → fitness ~ -0.25, but correct on the dominant case).
SEED_C = IfThenElse(
    condition=Compare('>', Count('game_demand'), Number(0)),
    then=Fire('game_production'),
    else_=Fire('phone_production'),
)

SEED_POLICIES = [SEED_A, SEED_B, SEED_C]
SEED_POLICY   = SEED_A   # used for the "Seed policy fitness" summary line

# ─── experiment record ──────────────────────────────────────

EXPERIMENT = sr.Experiment(
    name="SLA conformance policy search",
    reference_label="pi_sla",
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
    run_ga(seed=42)
