"""
ga_fitness.py
-------------
Shared cosmetic (interpretability) penalties for GA policy-tree fitness.

Fitness is combined as:

    total = primary_score - cosmetic_penalty(tree)

The GA maximises *total*, so cosmetic terms act as tie-breakers / soft
regularisers when weights are kept small relative to the primary objective.
"""

from __future__ import annotations

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


@dataclass
class CosmeticConfig:
    """Weights for interpretability penalties.  Tune per primary objective."""

    enabled: bool = True
    w_ite_depth: float = 0.05
    w_non_numeric: float = 0.01
    w_non_terminal: float = 0.01
    w_expr_size: float = 0.02
    max_cosmetic_penalty: float | None = 1.0


# Presets tuned so primary objective dominates.
DIVERGENCE_COSMETIC = CosmeticConfig(
    enabled=True,
    w_ite_depth=0.10,
    w_non_numeric=0.10,
    w_non_terminal=0.05,
    w_expr_size=0.05,
    max_cosmetic_penalty=3.0,
)

REWARD_COSMETIC = CosmeticConfig(
    enabled=True,
    w_ite_depth=0.05,
    w_non_numeric=0.01,
    w_non_terminal=0.01,
    w_expr_size=0.02,
    max_cosmetic_penalty=5.0,
)


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
    """Count internal (numeric) nodes in the tree."""
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


def cosmetic_penalty(tree, config: CosmeticConfig | None = None) -> float:
    """Return the total cosmetic penalty for *tree*."""
    cfg = config or CosmeticConfig()
    if not cfg.enabled:
        return 0.0

    penalty = (
        cfg.w_ite_depth    * ite_depth(tree)
        - cfg.w_non_numeric * numeric_count(tree)
        + cfg.w_non_terminal * non_terminal_count(tree)
        + cfg.w_expr_size    * expr_size_penalty(tree)
    )
    if cfg.max_cosmetic_penalty is not None:
        penalty = min(penalty, cfg.max_cosmetic_penalty)
    return penalty


def combined_fitness(primary: float, tree, config: CosmeticConfig | None = None) -> float:
    """Combine a primary score with cosmetic penalties (GA maximises this)."""
    return primary - cosmetic_penalty(tree, config)


def fitness_breakdown(primary: float, tree, config: CosmeticConfig | None = None) -> dict:
    """Return primary, cosmetic penalty, and combined fitness."""
    cosmetic = cosmetic_penalty(tree, config)
    return {
        "primary": primary,
        "cosmetic": cosmetic,
        "total": primary - cosmetic,
    }
