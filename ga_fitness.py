"""
ga_fitness.py
-------------
Shared cosmetic (interpretability) penalties for GA policy-tree fitness.

Fitness is combined as:

    total = primary_score - scale * cosmetic_penalty(tree)

The GA maximises *total*.  `scale` is normally 1.0 (the configured weights are
used as-is); `symbolic_regression.run_ga` can instead recompute it once per
generation from the population's primary-fitness spread, so the cosmetic term
stays a tie-breaker no matter how the primary objective is scaled.  See
`ScaleConfig` and `cosmetic_scale`.

Penalty terms
-------------
    ite_depth              length of the IF/ELSE chain
    numeric_count          numeric literals -- neutral by default, see NUMERIC_MODES
    non_terminal_count     internal (non-leaf) nodes
    expr_size_penalty      summed size of every Compare subtree
    duplicate_conditions   Compare subtrees that repeat elsewhere in the tree

The last term exists because the size-based terms count *nodes*, not *distinct
logic*: a tree that re-tests the same condition in two branches pays only for
the extra nodes, which is far less than the interpretability cost of a reader
having to work out that the second test can never be true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from policy_grammar import (
    IfThenElse, And, Or, Not, Compare,
    Number, Fire, Postpone,
    Add, Sub, Mul, Div,
    FEATURES,
)

# Leaf node types.  FEATURES is imported rather than re-listed so that adding a
# <feature> to the grammar cannot silently leave it counted as a non-terminal.
TERMINALS = (Number, Fire, Postpone) + FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# How a <number> literal is priced.
#
#   NEUTRAL  numeric literals cost nothing either way (default).
#   PENALTY  every literal costs, i.e. magic numbers are actively discouraged.
#
# The sign of `w_non_numeric` decides the direction under PENALTY: a negative
# weight promotes numeric thresholds instead of penalising them.  See
# `ga_shared.COSMETIC_CONFIG`, which is what the four drivers use.
NUMERIC_NEUTRAL = 'neutral'
NUMERIC_PENALTY = 'penalty'
NUMERIC_MODES   = (NUMERIC_NEUTRAL, NUMERIC_PENALTY)


@dataclass
class CosmeticConfig:
    """Weights for interpretability penalties.  Tune per primary objective."""

    enabled: bool = True
    w_ite_depth: float = 0.05
    w_non_numeric: float = 0.01          # applied per NUMERIC_MODES; see numeric_mode
    w_non_terminal: float = 0.01
    w_expr_size: float = 0.02
    w_duplicate_condition: float = 0.10  # per repeated Compare subtree
    numeric_mode: str = NUMERIC_NEUTRAL
    max_cosmetic_penalty: float | None = 1.0

    def __post_init__(self):
        if self.numeric_mode not in NUMERIC_MODES:
            raise ValueError(
                f"Invalid numeric_mode '{self.numeric_mode}'. "
                f"Must be one of {NUMERIC_MODES}"
            )


@dataclass
class ScaleConfig:
    """How the cosmetic weights are rescaled against the primary objective.

    `CosmeticConfig` weights are absolute, so whether the cosmetic term is a
    tie-breaker or the dominant objective depends on how the primary score is
    scaled.  This derives the scale from the population instead: each
    generation, shrink the weights so the worst cosmetic penalty is at most
    `budget_frac` of the primary spread.

    The spread is not `max(primary) - min(primary)`: early populations contain
    individuals that diverge on nearly every decision, so the full range is
    dominated by policies selection has already discarded.  It is measured over
    the selection-relevant band -- from the `spread_quantile` quantile up to
    the best individual -- falling back to the full range when that band is
    degenerate.

    The four drivers leave this off (`ga_shared.SCALE_CONFIG is None`), so
    fitness is exactly `conformance - penalty` as in Eq. (3).
    """

    enabled: bool = True
    budget_frac: float = 0.10      # cosmetic <= 10% of the primary spread
    spread_quantile: float = 0.75  # spread is measured over the top quartile
    tie_break_scale: float = 1e-6  # used when the primary spread is exactly 0
    max_scale: float = 1.0         # never *amplify* the configured weights



# ─────────────────────────────────────────────────────────────────────────────
# Tree walking
# ─────────────────────────────────────────────────────────────────────────────

def iter_nodes(tree):
    """Yield every node in a policy grammar tree."""
    yield tree
    if isinstance(tree, IfThenElse):
        yield from iter_nodes(tree.condition)
        yield from iter_nodes(tree.then)
        yield from iter_nodes(tree.else_)
    elif isinstance(tree, (And, Or)):
        yield from iter_nodes(tree.left)
        yield from iter_nodes(tree.right)
    elif isinstance(tree, Not):
        yield from iter_nodes(tree.operand)
    elif isinstance(tree, Compare):
        yield from iter_nodes(tree.left)
        yield from iter_nodes(tree.right)
    elif isinstance(tree, (Add, Sub, Mul, Div)):
        yield from iter_nodes(tree.left)
        yield from iter_nodes(tree.right)


def ite_depth(tree) -> int:
    """Number of IF-THEN-ELSE levels along the else_ chain."""
    return 1 + ite_depth(tree.else_) if isinstance(tree, IfThenElse) else 0


def numeric_count(tree) -> int:
    """Count numeric literals in the tree."""
    return sum(1 for node in iter_nodes(tree) if isinstance(node, Number))

def non_terminal_count(tree) -> int:
    """Count internal (non-terminal) nodes in the tree."""
    return sum(1 for node in iter_nodes(tree) if not isinstance(node, TERMINALS))


def expr_size(expr) -> int:
    """Node count of a numeric expression subtree (1 for a terminal)."""
    if isinstance(expr, TERMINALS):
        return 1
    if isinstance(expr, (Add, Sub, Mul, Div)):
        return 1 + expr_size(expr.left) + expr_size(expr.right)
    return 1


def condition_size(cond) -> int:
    """Node count of a condition subtree."""
    if isinstance(cond, Compare):
        return 1 + expr_size(cond.left) + expr_size(cond.right)
    if isinstance(cond, (And, Or)):
        return 1 + condition_size(cond.left) + condition_size(cond.right)
    if isinstance(cond, Not):
        return 1 + condition_size(cond.operand)
    return 1


def expr_size_penalty(tree) -> float:
    """Sum of expression/condition sizes over all Compare nodes."""
    return float(sum(condition_size(node) for node in iter_nodes(tree)
                 if isinstance(node, Compare)))


# ─────────────────────────────────────────────────────────────────────────────
# Structural redundancy
# ─────────────────────────────────────────────────────────────────────────────

# `a > b` and `b < a` are the same test written two ways; canonicalising the
# operand order means one has to be rewritten to the other before hashing.
_FLIPPED_OP = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}
# `=` and `!=` are symmetric, so their operands can simply be sorted.
_SYMMETRIC_OPS = frozenset({'=', '!='})
# Commutative arithmetic: SUM(p,x) + CLOCK and CLOCK + SUM(p,x) are one expr.
_COMMUTATIVE_EXPR = (Add, Mul)


def expr_key(expr):
    """Hashable canonical form of an <expr> subtree.

    Leaves are keyed by their repr, which the grammar already renders
    canonically (`COUNT(p, s)`, `SUM(p, x, s)`, `CLOCK`).  Numbers are keyed by
    value so `Number(2)` and `Number(2.0)` collapse to one key.  Commutative
    operators sort their operands; the others keep them ordered.
    """
    if isinstance(expr, Number):
        return ('num', float(expr.value))
    if isinstance(expr, _COMMUTATIVE_EXPR):
        operands = sorted((expr_key(expr.left), expr_key(expr.right)), key=repr)
        return (type(expr).__name__, *operands)
    if isinstance(expr, (Sub, Div)):
        return (type(expr).__name__, expr_key(expr.left), expr_key(expr.right))
    return ('leaf', repr(expr))


def compare_key(cond: Compare):
    """Hashable canonical form of a Compare node.

    Two Compares get the same key exactly when they test the same thing:
    operands of a symmetric comparator are sorted, and an asymmetric one is
    flipped so the operands come out in a fixed order.  `MAX(w2) >= MAX(w1)`
    and `MAX(w1) <= MAX(w2)` therefore collide, as they should.
    """
    left, right = expr_key(cond.left), expr_key(cond.right)
    op = cond.op
    if op in _SYMMETRIC_OPS:
        left, right = sorted((left, right), key=repr)
    elif repr(right) < repr(left):
        left, right, op = right, left, _FLIPPED_OP[op]
    return (op, left, right)


def duplicate_condition_count(tree) -> int:
    """How many Compare subtrees repeat a test already made in the same tree.

    A tree containing one condition twice scores 1, three times scores 2, and
    so on -- i.e. the count of occurrences beyond the first of each distinct
    test.  Repeats across branches (dead code: the second test is unreachable
    or already decided) and repeats within one condition (`A AND A`) both
    count, because both make a reader check something the tree already knows.
    """
    keys = [compare_key(node) for node in iter_nodes(tree)
            if isinstance(node, Compare)]
    return len(keys) - len(set(keys))


# ─────────────────────────────────────────────────────────────────────────────
# Penalty
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_term(tree, cfg: CosmeticConfig) -> float:
    """Contribution of numeric literals -- see NUMERIC_MODES."""
    if cfg.numeric_mode == NUMERIC_NEUTRAL:
        return 0.0
    return cfg.w_non_numeric * numeric_count(tree)


def cosmetic_penalty(tree, config: CosmeticConfig | None = None) -> float:
    """Return the total cosmetic penalty for *tree*."""
    cfg = config or CosmeticConfig()
    if not cfg.enabled:
        return 0.0

    penalty = (
        cfg.w_ite_depth    * ite_depth(tree)
        + _numeric_term(tree, cfg)
        + cfg.w_non_terminal * non_terminal_count(tree)
        + cfg.w_expr_size    * expr_size_penalty(tree)
        + cfg.w_duplicate_condition * duplicate_condition_count(tree)
    )
    if cfg.max_cosmetic_penalty is not None:
        penalty = min(penalty, cfg.max_cosmetic_penalty)
    return penalty


def cosmetic_breakdown(tree, config: CosmeticConfig | None = None) -> dict:
    """Per-term view of `cosmetic_penalty`, for reporting and diagnosis."""
    cfg = config or CosmeticConfig()
    terms = {
        'ite_depth':       cfg.w_ite_depth * ite_depth(tree),
        'numeric':         _numeric_term(tree, cfg),
        'non_terminal':    cfg.w_non_terminal * non_terminal_count(tree),
        'expr_size':       cfg.w_expr_size * expr_size_penalty(tree),
        'duplicate_condition': (cfg.w_duplicate_condition
                                * duplicate_condition_count(tree)),
    }
    terms['raw_total'] = sum(terms.values()) if cfg.enabled else 0.0
    terms['total'] = cosmetic_penalty(tree, cfg)
    terms['counts'] = {
        'ite_depth':          ite_depth(tree),
        'numeric':            numeric_count(tree),
        'non_terminal':       non_terminal_count(tree),
        'expr_size':          expr_size_penalty(tree),
        'duplicate_condition': duplicate_condition_count(tree),
    }
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# Population-relative scaling
# ─────────────────────────────────────────────────────────────────────────────

def _finite(primaries):
    return [p for p in primaries if p is not None and math.isfinite(p)]


def _quantile(values, q: float) -> float:
    """Linear-interpolation quantile of a non-empty list (numpy-free)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def smallest_primary_gap(primaries) -> float:
    """The smallest nonzero difference between two distinct primary scores.

    One "unit of discrimination": the least the primary objective can say.  For
    a divergence-rate primary that is one divergence out of the decisions the
    run reached; for a count primary, one divergence over the rollouts.  0.0
    when every individual scores alike.
    """
    distinct = sorted(set(_finite(primaries)))
    gaps = [b - a for a, b in zip(distinct, distinct[1:]) if b > a]
    return min(gaps) if gaps else 0.0


def primary_spread(primaries, config: ScaleConfig | None = None) -> float:
    """Spread of the selection-relevant band of primary scores.

    `max - quantile(spread_quantile)` -- the band selection actually
    discriminates over.

    When that band is flat (the whole top quartile scoring alike, i.e. the run
    has converged on primary) the fallback is *not* the full range.  Falling
    back to `max - min` is what a first cut did, and it is exactly backwards:
    the full range is set by individuals selection has already discarded, so on
    a converged choice_sl population it read 20.0, put the budget at 2.0, and
    saturated the scale at 1.0 -- restoring the very domination this is meant
    to remove.  The right reference once the band is flat is one unit of
    primary discrimination: the smallest gap between distinct primary scores
    still present in the population.  Cosmetics then order the tied top band
    while staying provably below what a single divergence is worth.

    Returns 0.0 only when every individual shares one primary score, which is
    the signal that nothing but the cosmetic term can order the population.
    """
    cfg = config or ScaleConfig()
    finite = _finite(primaries)
    if len(finite) < 2:
        return 0.0
    spread = max(finite) - _quantile(finite, cfg.spread_quantile)
    if spread <= 0.0:
        spread = smallest_primary_gap(finite)
    return max(spread, 0.0)


def cosmetic_scale(primaries, cosmetics, config: ScaleConfig | None = None) -> float:
    """Factor to multiply cosmetic penalties by this generation.

    Chosen so the largest cosmetic penalty in the population costs at most
    `budget_frac` of the primary spread, and never more than the configured
    weights already say (`max_scale`).  When the primary spread is exactly 0
    every comparison in the population is between equal primaries, so the
    factor only has to be positive for the cosmetic ordering to survive --
    hence the tiny `tie_break_scale`.
    """
    cfg = config or ScaleConfig()
    if not cfg.enabled:
        return cfg.max_scale

    worst = max((c for c, p in zip(cosmetics, primaries)
                 if p is not None and math.isfinite(p)), default=0.0)
    if worst <= 0.0:
        return cfg.max_scale

    budget = cfg.budget_frac * primary_spread(primaries, cfg)
    if budget <= 0.0:
        return min(cfg.max_scale, cfg.tie_break_scale)
    return min(cfg.max_scale, budget / worst)


# ─────────────────────────────────────────────────────────────────────────────
# Combination
# ─────────────────────────────────────────────────────────────────────────────

def combined_fitness(primary: float, tree, config: CosmeticConfig | None = None,
                     scale: float = 1.0) -> float:
    """Combine a primary score with cosmetic penalties (GA maximises this)."""
    return primary - scale * cosmetic_penalty(tree, config)


def fitness_breakdown(primary: float, tree, config: CosmeticConfig | None = None,
                      scale: float = 1.0) -> dict:
    """Return primary, cosmetic penalty, and combined fitness."""
    cosmetic = cosmetic_penalty(tree, config)
    return {
        "primary": primary,
        "cosmetic": cosmetic,
        "scale": scale,
        "total": primary - scale * cosmetic,
    }
