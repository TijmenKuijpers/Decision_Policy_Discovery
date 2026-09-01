"""
ga_shared.py
------------
Every hyper-parameter that is shared by the four evaluation drivers.

A driver imports these values instead of restating them, so the four processes
cannot drift apart: policy discovery does not depend on a tuned per-process
configuration.  What stays in a driver is only a fact about that process -- its
places, token attributes, action transitions, normative policy, and the rollout
count that reaches the shared log size.  Any value a driver does override is
passed as an explicit keyword argument at the call site.

These are the settings of Section 6.3.
"""

from __future__ import annotations

from ga_fitness import CosmeticConfig, NUMERIC_PENALTY
import symbolic_regression as sr


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────
HORIZON          = 60
TARGET_DECISIONS = 500

# ─────────────────────────────────────────────────────────────────────────────
# Genetic algorithm
# ─────────────────────────────────────────────────────────────────────────────

POP_SIZE      = 60      # N
N_GENERATIONS = 300     # G
TOURNAMENT_K  = 3       # k
ELITE_K       = 3       # e
P_CROSSOVER   = 0.70    # p_c
P_MUTATION    = 0.35    # p_m

# Tree shape.  MAX_DEPTH bounds the if-then-else chain; the condition arity
# (at most two comparisons) and the expression arity (at most two terms) are
# fixed in symbolic_regression and are not per-process either.
MAX_DEPTH   = 3
COMPARATORS = ['>', '>=', '<', '<=', '=', '!=']

# Sub-operator probabilities.  These are the draws made *inside* an operator
# once it has been selected, and there is no principled reason for them to
# differ between processes.
P_AND         = 0.6     # And vs Or when building a random condition
P_FIRE        = 0.6     # Fire vs Postpone when building a random action
P_MUTATE_FIRE = 0.5     # Fire vs Postpone when mutating an action

# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────
# One number pool for every process.  
INT_POOL   = tuple(range(11))                     # 0-10
FLOAT_POOL = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
              0.5, 0.75, 0.9, 0.95, 0.99)
P_FLOAT    = 0.35

# Numeric mutation: how a literal moves once it exists.
INT_LO, INT_HI = 0, 10
INT_STEP       = (-2, 2)
FLOAT_NUDGES   = (-0.1, -0.05, 0.05, 0.1)
FLOAT_LO       = 0.0
FLOAT_HI       = 1.0
ROUND_TO       = 2

# Branch weights: the probability mass each <Factor> form gets when an
# expression is drawn.  All five branches are enabled for all four processes,
# so one set of weights applies everywhere and they sum to 1.0.
W_COUNT     = 0.25
W_NUMBER    = 0.20
W_AGGREGATE = 0.25
W_CLOCK     = 0.05
W_BINARY    = 0.25      # two features joined by + - * / (a share, a rate, ...)

# Expression mutation.
P_FAMILY_ESCAPE = 0.15  # abandon the node's feature family entirely
P_COUNT_PLACE   = 0.75  # a Count mutation swaps the place, else the selector
P_AGG_SWAP      = 0.40  # swap the aggregate function
P_FEATURE_SWAP  = 0.80  # cumulative: swap the (place, attribute) pair

# Every evaluation net carries future-dated tokens, so ALL and ENABLED are a
# real distinction everywhere and the selector is sampled for all four.
USE_SELECTORS = True
P_ALL         = 0.80

# CLOCK is available to every process.  
WITH_CLOCK = True

AGGREGATES = sr.AGGREGATES_DEFAULT      # (Sum, Min, Max, Mean)

# ─────────────────────────────────────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────────────────────────────────────

# The interpretability penalty: a weighted sum over the five
# structural properties in Xi, capped at P_max.

COSMETIC_CONFIG = CosmeticConfig(
    enabled=True,
    numeric_mode=NUMERIC_PENALTY,
    w_ite_depth=0.002,           # iota  -- if-then-else depth
    w_non_numeric=-0.002,        # eta   -- numeric-literal count (reward, see above)
    w_non_terminal=0.001,        # nu    -- non-terminal count
    w_expr_size=0.001,           # eps   -- expression size
    w_duplicate_condition=0.005, # delta -- duplicate conditions
    max_cosmetic_penalty=0.05,   # P_max
)

# None disables the per-generation rescaling of the penalty, so FITNESS is
# exactly `conformance - penalty` as Eq. (3) states it.  The raw penalty is
# capped at 0.05 against a primary spanning 1.0, so it is already a tie-breaker.
SCALE_CONFIG = None


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def make_vocab(places, attr_features, **overrides) -> sr.ExprVocab:
    """An ExprVocab carrying the shared settings.

    *places* and *attr_features* are the domain facts; everything else comes
    from this module.
    """
    settings = dict(
        places=places,
        attr_features=attr_features,
        aggregates=AGGREGATES,
        with_clock=WITH_CLOCK,
        use_selectors=USE_SELECTORS,
        p_all=P_ALL,
        int_pool=INT_POOL,
        float_pool=FLOAT_POOL,
        p_float=P_FLOAT,
        w_count=W_COUNT,
        w_number=W_NUMBER,
        w_aggregate=W_AGGREGATE,
        w_clock=W_CLOCK,
        w_binary=W_BINARY,
        p_family_escape=P_FAMILY_ESCAPE,
        p_count_place=P_COUNT_PLACE,
        p_agg_swap=P_AGG_SWAP,
        p_feature_swap=P_FEATURE_SWAP,
        int_lo=INT_LO,
        int_hi=INT_HI,
        int_step=INT_STEP,
        float_nudges=FLOAT_NUDGES,
        float_lo=FLOAT_LO,
        float_hi=FLOAT_HI,
        round_to=ROUND_TO,
    )
    settings.update(overrides)
    return sr.ExprVocab(**settings)


def make_config(vocab: sr.ExprVocab, transitions, **overrides) -> sr.GrammarConfig:
    """A GrammarConfig carrying the shared settings.  *transitions* is T_A."""
    settings = dict(
        comparators=COMPARATORS,
        transitions=transitions,
        max_depth=MAX_DEPTH,
        p_and=P_AND,
        p_fire=P_FIRE,
        p_mutate_fire=P_MUTATE_FIRE,
        p_mutation=P_MUTATION,
    )
    settings.update(overrides)
    return sr.GrammarConfig.from_vocab(vocab, **settings)
