"""
symbolic_regression.py
-----------------------
Process- and policy-agnostic genetic-algorithm machinery for symbolic policy
discovery over the Petri-net policy grammar in `policy_grammar.py`.

An experiment supplies three things and nothing else:

  1. a domain model         -- a `make_system(seed)` factory returning a GymProblem
  2. an `ExprVocab`         -- WHICH features exist and how often to sample them
  3. an `Experiment` record -- the reference policy, seeds, and hyper-parameters

Everything else lives here: expression generation, expression mutation, tree
construction, the mutation operator, crossover, tournament selection, conformance
measurement, and the generational loop.

Layering
--------
    ExprVocab      <feature> / <number> vocabulary  -- the only domain-aware part
    GrammarConfig  <condition> / <action> shape, comparators, transitions
    Experiment     fitness, seeds, hyper-parameters, banner
    run_ga         the generational loop

`ExprVocab.rnd_expr` / `.mutate_expr` are bound methods matching the callable
signatures `GrammarConfig` expects, so a vocabulary drops straight in:

    VOCAB  = ExprVocab(places=PLACES, attr_features=ATTR_FEATURES, ...)
    CONFIG = GrammarConfig(..., rnd_expr=VOCAB.rnd_expr,
                                mutate_expr=VOCAB.mutate_expr)

A domain that needs a novel expression form can also pass its own callables
into `GrammarConfig`; the declarative path is the default, not the only one.
"""

from __future__ import annotations

import contextlib
import copy
import io
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from policy_grammar import (
    IfThenElse, And, Or, Not, Compare, Fire, Postpone,
    Count, Sum, Min, Max, Mean, Clock, Number, Add, Sub, Mul, Div,
    ALL, ENABLED,
)
from ga_fitness import (
    ite_depth, combined_fitness, cosmetic_penalty, cosmetic_scale, ScaleConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Expression vocabulary
# ─────────────────────────────────────────────────────────────────────────────

AGGREGATES_DEFAULT = (Sum, Min, Max, Mean)

# The operators `_rnd_binary` combines two <Feature> terminals with.
#
# This is the whole of the <Expression>/<Term>/<Factor> restriction: an <expr>
# is either a single term, or exactly two terms joined by one operator.  One
# rule serves every process -- a share (AGG/COUNT) and a rate (AGG/CLOCK) are
# just two of the Div expressions this general form already reaches.
BINARY_OPS = (Add, Sub, Mul, Div)


@dataclass
class ExprVocab:
    """Declarative description of a domain's <expr> vocabulary.

    The `w_*` weights are the probability mass given to each branch and must sum
    to 1.0 over the branches that actually have vocabulary behind them.  A weight
    set on a branch with no vocabulary is an error rather than silently ignored
    (that mistake is otherwise invisible and quietly reshapes the search space).
    """

    # ── vocabulary (domain facts) ────────────────────────────────────────────
    places:        Sequence[str] = ()
    attr_features: Sequence[tuple[str, str]] = ()     # (place, attribute)
    aggregates:    Sequence[type] = AGGREGATES_DEFAULT
    with_clock:    bool = False

    # False when the net has no meaningful ALL/ENABLED distinction, i.e. no
    # future-dated tokens.  Purely a vocabulary fact: Count(p) and Count(p, ALL)
    # are equal and print identically, so this only decides whether a selector
    # is sampled at all.
    use_selectors: bool = True
    p_all:         float = 0.80        # P(ALL) when selectors are sampled

    # ── number literals ──────────────────────────────────────────────────────
    int_pool:   Sequence[int]   = tuple(range(9))
    float_pool: Sequence[float] = ()
    p_float:    float = 0.0            # ignored when float_pool is empty

    # ── generation weights ───────────────────────────────────────────────────
    w_count:     float = 0.40
    w_number:    float = 0.20
    w_aggregate: float = 0.40
    w_clock:     float = 0.0
    # Two <Feature> terminals joined by one of BINARY_OPS.  This is the branch
    # that reaches a share (AGG/COUNT) or a rate (AGG/CLOCK).
    w_binary:    float = 0.0

    # ── mutation ─────────────────────────────────────────────────────────────
    # Probability an expression mutation abandons its node family entirely.
    # Without it mutation is family-closed -- a Count only ever becomes another
    # Count, an aggregate only another aggregate -- so a tree built on one
    # feature family can only reach another by discarding its whole condition
    # and losing whatever structure made it good.  Set it when the target policy
    # needs a cross-family jump (e.g. COUNT -> MAX over an attribute).
    p_family_escape: float = 0.0
    p_count_place:   float = 0.75      # else swap the selector
    p_agg_swap:      float = 0.40      # swap the aggregate function
    p_feature_swap:  float = 0.80      # cumulative: swap the (place, attribute)

    int_lo:   int = 0
    int_hi:   int = 8
    int_step: tuple[int, int] = (-2, 2)
    float_nudges: Sequence[float] = (-0.1, -0.05, 0.05, 0.1)
    float_lo: float | None = 0.0
    float_hi: float | None = 1.0       # None disables the upper clamp
    round_to: int = 2

    # ── derived ──────────────────────────────────────────────────────────────
    _branches:         list = field(default_factory=list, init=False, repr=False)
    _feature_branches: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # The <Feature>-only weights, computed first because `binary`'s own
        # vocabulary check depends on them: a binary expression needs two
        # feature terminals, so it has none to draw from unless at least one
        # of count/aggregate/clock is both present and given nonzero weight.
        feature_weights = {
            'count':     self.w_count     if self.places else 0.0,
            'aggregate': self.w_aggregate if (self.attr_features and self.aggregates) else 0.0,
            'clock':     self.w_clock     if self.with_clock else 0.0,
        }
        total_feat = sum(feature_weights.values())

        enabled = {
            'count':     bool(self.places),
            'number':    bool(self.int_pool or self.float_pool),
            'aggregate': bool(self.attr_features and self.aggregates),
            'clock':     bool(self.with_clock),
            'binary':    total_feat > 0,
        }
        weights = {
            'count':     self.w_count,
            'number':    self.w_number,
            'aggregate': self.w_aggregate,
            'clock':     self.w_clock,
            'binary':    self.w_binary,
        }
        for key, w in weights.items():
            if w and not enabled[key]:
                raise ValueError(
                    f"w_{key}={w} but the '{key}' branch has no vocabulary. "
                    f"Populate the matching field, or set w_{key}=0.0."
                )
        total = sum(w for key, w in weights.items() if enabled[key])
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"enabled branch weights sum to {total!r}, expected 1.0 "
                f"(enabled: {[k for k, v in enabled.items() if v]})"
            )

        order = [
            ('count',     self._rnd_count),
            ('number',    self.rnd_number),
            ('aggregate', self._rnd_aggregate),
            ('clock',     self._rnd_clock),
            ('binary',    self._rnd_binary),
        ]
        cum = 0.0
        branches = []
        for key, build in order:
            if not enabled[key] or not weights[key]:
                continue
            cum += weights[key]
            branches.append((cum, build))
        self._branches = branches

        # Operands of a binary expression are drawn UNIFORMLY over the feature
        # families the vocabulary has, not by the generation weights above.
        #
        # The two answer different questions.  A generation weight says how
        # often a bare feature of that kind should stand as a whole expression,
        # and CLOCK earns a small one there because a policy comparing raw
        # elapsed time to a constant is rarely what is wanted.  As an *operand*
        # CLOCK is structurally central -- it is the denominator that turns a
        # counter into a rate -- so inheriting its small generation weight made
        # AGG/CLOCK far rarer than the shape deserves, and the rate-limit target
        # became unreachable in practice.  A family that has no vocabulary is
        # still excluded, so this cannot invent operands a process lacks.
        feature_order = [
            ('count',     self._rnd_count),
            ('aggregate', self._rnd_aggregate),
            ('clock',     self._rnd_clock),
        ]
        available = [build for key, build in feature_order if feature_weights[key]]
        self._feature_branches = [((i + 1) / len(available), build)
                                  for i, build in enumerate(available)]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _selector(self, rng):
        return rnd_selector(rng, self.p_all) if self.use_selectors else ALL

    def _agg_node(self, agg, place, attribute, rng):
        if self.use_selectors:
            return agg(place, attribute, rnd_selector(rng, self.p_all))
        return agg(place, attribute)

    # ── branch builders ──────────────────────────────────────────────────────

    def _rnd_count(self, rng):
        place = rng.choice(list(self.places))
        if self.use_selectors:
            return Count(place, rnd_selector(rng, self.p_all))
        return Count(place)

    def rnd_number(self, rng):
        if self.float_pool and rng.random() < self.p_float:
            return Number(rng.choice(list(self.float_pool)))
        return Number(rng.choice(list(self.int_pool)))

    def _rnd_aggregate(self, rng):
        agg = rng.choice(list(self.aggregates))
        place, attribute = rng.choice(list(self.attr_features))
        return self._agg_node(agg, place, attribute, rng)

    def _rnd_clock(self, rng):
        return Clock()

    def _rnd_feature_leaf(self, rng):
        """One <Feature> terminal -- COUNT, an aggregate, or CLOCK, never a
        Number.  Used only by `_rnd_binary`, so both operands of a binary
        expression are features rather than one feature and a bare constant."""
        r = rng.random()
        for threshold, build in self._feature_branches:
            if r < threshold:
                return build(rng)
        return self._feature_branches[-1][1](rng)

    def _rnd_binary(self, rng):
        """<expr> ::= <expr> (+|-|*) <term> over two <Feature> terminals.

        Operands are drawn independently and may repeat (two counts on
        different places, or by chance the same feature twice).  No
        distinctness check: the interpretability penalty already prices the
        extra nodes, and a self-cancelling `X - X` is a milder version of the
        redundancy `duplicate_condition_count` already flags at the
        comparison level, not a new failure mode to guard against here.
        """
        left  = self._rnd_feature_leaf(rng)
        right = self._rnd_feature_leaf(rng)
        return rng.choice(BINARY_OPS)(left, right)

    # ── public API (plugs into GrammarConfig) ────────────────────────────────

    def rnd_expr(self, rng):
        """Sample one <expr>: a single draw, then cumulative-threshold dispatch."""
        r = rng.random()
        for threshold, build in self._branches:
            if r < threshold:
                return build(rng)
        return self._branches[-1][1](rng)

    def mutate_number(self, n: Number, rng):
        v = n.value
        if self.float_nudges and isinstance(v, float):
            v = v + rng.choice(list(self.float_nudges))
            if self.float_lo is not None:
                v = max(self.float_lo, v)
            if self.float_hi is not None:
                v = min(self.float_hi, v)
            return Number(round(v, self.round_to))
        return Number(max(self.int_lo,
                          min(self.int_hi, int(v) + rng.randint(*self.int_step))))

    def mutate_expr(self, expr, rng):
        """Mutate within the node's own family, preserving surrounding structure."""
        if self.p_family_escape and rng.random() < self.p_family_escape:
            return self.rnd_expr(rng)

        if isinstance(expr, Count):
            if rng.random() < self.p_count_place:
                return Count(rng.choice(list(self.places)), expr.selector)
            return Count(expr.place, self._selector(rng))

        if isinstance(expr, BINARY_OPS):
            if not self._feature_branches:
                # No vocabulary to draw a feature operand from -- possible if
                # a hand-authored seed uses Add/Sub/Mul in a vocabulary that
                # never samples them (w_binary=0).  Escape rather than index
                # into an empty branch list.
                return self.rnd_expr(rng)
            r = rng.random()
            if r < 1 / 3:
                other_ops = [op for op in BINARY_OPS if op is not type(expr)]
                return rng.choice(other_ops)(expr.left, expr.right)
            if r < 2 / 3:
                return type(expr)(self._rnd_feature_leaf(rng), expr.right)
            return type(expr)(expr.left, self._rnd_feature_leaf(rng))

        if self.aggregates and isinstance(expr, tuple(self.aggregates)):
            r = rng.random()
            if r < self.p_agg_swap:
                agg = rng.choice(list(self.aggregates))
                return agg(expr.place, expr.attribute, expr.selector)
            if r < self.p_feature_swap:
                place, attribute = rng.choice(list(self.attr_features))
                return type(expr)(place, attribute, expr.selector)
            return type(expr)(expr.place, expr.attribute, self._selector(rng))

        if isinstance(expr, Clock):
            return self.rnd_expr(rng)

        if isinstance(expr, Number):
            return self.mutate_number(expr, rng)

        return self.rnd_expr(rng)


# ─────────────────────────────────────────────────────────────────────────────
# Grammar configuration (tree shape)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GrammarConfig:
    """Domain parameters the generic tree operators need.

    `rnd_expr` / `mutate_expr` are normally `ExprVocab` bound methods, but any
    callable with the same signature works.
    """
    comparators: list[str]
    transitions: list[str]
    max_depth: int
    rnd_expr: Callable[[random.Random], object]
    mutate_expr: Callable[[object, random.Random], object]
    p_and: float = 0.6            # P(And vs Or) when building a random condition
    p_fire: float = 0.6           # P(Fire vs Postpone) when building a random action
    p_mutate_fire: float = 0.5    # P(Fire vs Postpone) when mutating an action
    p_mutation: float = 0.35      # P(an offspring is mutated at all); one edit if so
    # Clause-arity edits.  Without these `mutate_condition` can only ever
    # rewrite a Compare *in place*, so a single-comparison condition can never
    # grow the second clause that <Condition> ::= <Condition> OR <Conjunction>
    # allows -- and neither can crossover, which exchanges else-branch suffixes
    # rather than conditions.  Two-clause targets (pi_batch's
    # `COUNT(ready) > 3 OR MAX(truck, status) = 1`, pi_sla's guarded SLA
    # ratios) were therefore reachable only by being *born* with the right
    # shape.  Grow is weighted above shrink because the cosmetic penalty
    # already pushes the other way.
    p_condition_grow: float = 0.25    # P(Compare -> And/Or(Compare, new Compare))
    p_condition_shrink: float = 0.15  # P(And/Or -> one of its operands)
    # P(a bare-action policy gains a rule instead of just swapping its action).
    # A bare <Action> is still a <Policy>, so the grammar can wrap it via
    # <Policy> ::= IF <Condition> THEN <Action> ELSE <Policy> -- the paper's
    # "prepend a rule" edit.  Without this a bare action is an ABSORBING state:
    # `_mutate_once` finds no ITE node and can only swap the action, and
    # `crossover` returns both parents untouched when either has no ITE node.
    # Since bare actions carry the smallest cosmetic penalty, a population that
    # drifts onto one can never re-grow a rule and the run dies there.
    p_action_prepend: float = 0.5

    @classmethod
    def from_vocab(cls, vocab: ExprVocab, **kwargs) -> "GrammarConfig":
        """Build a config from a vocabulary without naming the two callables."""
        return cls(rnd_expr=vocab.rnd_expr, mutate_expr=vocab.mutate_expr, **kwargs)


def rnd_selector(rng: random.Random, p_all: float = 0.80) -> str:
    """ALL dominates; ENABLED is sampled as a minority alternative."""
    return ALL if rng.random() < p_all else ENABLED


# ─────────────────────────────────────────────────────────────────────────────
# Random tree constructors
# ─────────────────────────────────────────────────────────────────────────────

def rnd_compare(rng: random.Random, cfg: GrammarConfig) -> Compare:
    return Compare(rng.choice(cfg.comparators), cfg.rnd_expr(rng), cfg.rnd_expr(rng))


def rnd_condition(rng: random.Random, cfg: GrammarConfig, depth: int = 0):
    """Random condition, at most 2 levels deep."""
    if depth >= 1 or rng.random() < 0.5:
        return rnd_compare(rng, cfg)
    cls = And if rng.random() < cfg.p_and else Or
    return cls(rnd_condition(rng, cfg, depth + 1), rnd_condition(rng, cfg, depth + 1))


def rnd_action(rng: random.Random, cfg: GrammarConfig):
    return Fire(rng.choice(cfg.transitions)) if rng.random() < cfg.p_fire else Postpone()


def rnd_policy(rng: random.Random, cfg: GrammarConfig, depth: int = 0):
    """Random policy tree with IF-THEN-ELSE depth limited to cfg.max_depth."""
    if depth >= cfg.max_depth or rng.random() < 0.25:
        return rnd_action(rng, cfg)
    return IfThenElse(
        condition=rnd_condition(rng, cfg),
        then=rnd_action(rng, cfg),
        else_=rnd_policy(rng, cfg, depth + 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tree helpers
# ─────────────────────────────────────────────────────────────────────────────

def ite_nodes(tree):
    """All IfThenElse nodes reachable through the else_ chain."""
    if not isinstance(tree, IfThenElse):
        return []
    return [tree] + ite_nodes(tree.else_)


# ─────────────────────────────────────────────────────────────────────────────
# Mutation
# ─────────────────────────────────────────────────────────────────────────────

def mutate_compare(c: Compare, rng: random.Random, cfg: GrammarConfig) -> Compare:
    r = rng.random()
    if r < 0.33:
        return Compare(rng.choice(cfg.comparators), copy.deepcopy(c.left),      copy.deepcopy(c.right))
    if r < 0.66:
        return Compare(c.op,                        cfg.mutate_expr(c.left, rng), copy.deepcopy(c.right))
    return     Compare(c.op,                        copy.deepcopy(c.left),        cfg.mutate_expr(c.right, rng))


def mutate_condition(cond, rng: random.Random, cfg: GrammarConfig, depth: int = 0):
    """Mutate a condition, keeping it inside the search space's shape.

    `depth` counts enclosing And/Or nodes.  The clause-grow edit only fires at
    depth 0, so a condition can reach `And/Or(Compare, Compare)` and no
    further: that is the "at most one level of conjunction/disjunction over
    comparisons" the search space allows.  Without the bound, growing a Compare
    nested inside an And/Or compounds without limit and conditions run away --
    deep enough, on a long run, to overflow the stack in `copy.deepcopy`.
    """
    if isinstance(cond, Compare):
        if depth == 0 and rng.random() < cfg.p_condition_grow:   # gain a clause
            cls   = And if rng.random() < cfg.p_and else Or
            fresh = rnd_compare(rng, cfg)
            kept  = copy.deepcopy(cond)
            # Either side, so the new clause is not always the right operand.
            return cls(kept, fresh) if rng.random() < 0.5 else cls(fresh, kept)
        return mutate_compare(cond, rng, cfg)
    if isinstance(cond, (And, Or)):
        inner = depth + 1
        r = rng.random()
        if r < cfg.p_condition_shrink:                 # lose a clause
            # The inverse of the grow edit above: without it conditions only
            # ever ratchet up in arity.
            return copy.deepcopy(cond.left if rng.random() < 0.5 else cond.right)
        r = rng.random()
        if r < 0.10:                                   # flip And <-> Or
            cls = Or if isinstance(cond, And) else And
            return cls(mutate_condition(cond.left, rng, cfg, inner),
                       mutate_condition(cond.right, rng, cfg, inner))
        if r < 0.55:
            return type(cond)(mutate_condition(cond.left, rng, cfg, inner),
                              copy.deepcopy(cond.right))
        return     type(cond)(copy.deepcopy(cond.left),
                              mutate_condition(cond.right, rng, cfg, inner))
    if isinstance(cond, Not):
        return Not(mutate_condition(cond.operand, rng, cfg, depth))
    return rnd_condition(rng, cfg)


def mutate_action(action, rng: random.Random, cfg: GrammarConfig):
    return Fire(rng.choice(cfg.transitions)) if rng.random() < cfg.p_mutate_fire else Postpone()


def _mutate_once(root, rng: random.Random, cfg: GrammarConfig):
    """Apply exactly one edit to *root*, in place where possible.

    The caller owns *root* (it is already a private copy), so nodes are edited
    or re-spliced directly rather than rebuilt.  Returns the new root, which
    differs from the one passed in only when the edit landed on the first rule.
    """
    nodes = ite_nodes(root)
    if not nodes:                       # a bare action node
        # Prepend a rule above it, or fall back to swapping the action itself.
        # See GrammarConfig.p_action_prepend for why the first branch has to
        # exist at all.
        if cfg.max_depth >= 1 and rng.random() < cfg.p_action_prepend:
            return IfThenElse(condition=rnd_condition(rng, cfg),
                              then=rnd_action(rng, cfg),
                              else_=root)
        return mutate_action(root, rng, cfg)

    i = rng.randrange(len(nodes))       # chain position == AST depth of nodes[i]
    n = nodes[i]

    def splice(replacement):
        """Put *replacement* where nodes[i] sits; return the (new) root."""
        if i == 0:
            return replacement
        nodes[i - 1].else_ = replacement
        return root

    # The five edits are drawn uniformly (0.2 each).  The paper names the edit
    # distribution `q` but fixes no values; a uniform `q` is the choice that
    # needs no per-process justification, which is the point -- discovery should
    # not rest on a tuned mutation mix.
    edit = rng.randrange(5)
    if edit == 0:
        n.condition = mutate_condition(n.condition, rng, cfg)
        return root
    if edit == 1:
        n.then = mutate_action(n.then, rng, cfg)
        return root
    if edit == 2:
        # Prepending adds a level, so it needs room under cfg.max_depth; when
        # blocked it falls through to the prune branch, as it always has.
        if ite_depth(root) < cfg.max_depth:
            return splice(IfThenElse(condition=rnd_condition(rng, cfg),
                                     then=rnd_action(rng, cfg),
                                     else_=n))
        return splice(n.else_)
    if edit == 3:
        return splice(n.else_)
    return splice(rnd_policy(rng, cfg, depth=i))


def mutate(tree, rng: random.Random, cfg: GrammarConfig, n_edits: int = 1):
    """Return a deep copy of *tree* with exactly *n_edits* single edits applied.

    One edit picks ONE rule uniformly from the else_ chain and changes it; the
    number of edits per call therefore does not grow with the length of the
    policy.  Whether an offspring is mutated at all is the caller's decision
    (`cfg.p_mutation` in `run_ga`), not this operator's.

    Given `nodes = ite_nodes(tree)` and a uniformly drawn position `i`, the
    single edit is:

        op < 0.35   mutate condition     nodes[i].condition = mutate_condition(...)
        op < 0.55   mutate action        nodes[i].then      = mutate_action(...)
        op < 0.70   prepend a rule       IfThenElse(rnd_condition, rnd_action, nodes[i])
        op < 0.85   prune this rule      nodes[i] -> nodes[i].else_
        otherwise   replace the suffix   nodes[i] -> rnd_policy(depth=i)

    A tree with no IfThenElse node at all (a bare action) is mutated with
    `mutate_action`; that counts as the edit.

    The depth bound uses the *absolute* chain position `i`, so both the prepend
    guard and the `rnd_policy` depth argument bound the depth of the whole
    policy by `cfg.max_depth`.  `n_edits` edits are drawn successively, each on
    the result of the previous.  *tree* itself is never modified.
    """
    result = copy.deepcopy(tree)
    for _ in range(n_edits):
        result = _mutate_once(result, rng, cfg)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Crossover / selection
# ─────────────────────────────────────────────────────────────────────────────

def terminal_action(tree):
    """The <Action> a policy falls through to when no rule fires."""
    while isinstance(tree, IfThenElse):
        tree = tree.else_
    return tree


def truncate_depth(tree, max_depth: int):
    """Cut the if-then-else chain back to *max_depth* rules.

    The discarded suffix is replaced by the policy's own fall-through action,
    so truncation drops rules the policy can no longer reach rather than
    silently changing what it does when none of the kept rules fire.
    """
    if ite_depth(tree) <= max_depth:
        return tree
    if max_depth == 0:
        return copy.deepcopy(terminal_action(tree))
    node = tree
    for _ in range(max_depth - 1):
        node = node.else_
    node.else_ = copy.deepcopy(terminal_action(node.else_))
    return tree


def crossover(p1, p2, rng: random.Random, cfg: GrammarConfig):
    """Subtree crossover: swap else_ branches at randomly chosen ITE nodes.

    Both parents are deep-copied first so the originals are never modified.

    The swap can lengthen a chain -- a rule low in one parent can receive a
    long suffix from the other -- so each child is truncated back to
    `cfg.max_depth` afterwards.  Without that, `max_depth` binds on mutation
    (whose prepend edit checks it) but not on crossover, and the chain
    compounds across generations until it is deep enough to overflow the stack
    in `copy.deepcopy`.
    """
    c1, c2  = copy.deepcopy(p1), copy.deepcopy(p2)
    nodes1  = ite_nodes(c1)
    nodes2  = ite_nodes(c2)
    if not nodes1 or not nodes2:
        return c1, c2
    n1, n2   = rng.choice(nodes1), rng.choice(nodes2)
    tmp      = copy.deepcopy(n1.else_)
    n1.else_ = copy.deepcopy(n2.else_)
    n2.else_ = tmp
    return (truncate_depth(c1, cfg.max_depth),
            truncate_depth(c2, cfg.max_depth))


def tournament(population, fitnesses, k: int, rng: random.Random):
    idx = rng.sample(range(len(population)), k)
    return population[max(idx, key=lambda i: fitnesses[i])]


# ─────────────────────────────────────────────────────────────────────────────
# Conformance measurement
# ─────────────────────────────────────────────────────────────────────────────
#
# Scoring is *target-driven* (off-policy): the reference policy is simulated
# once, every decision point it reaches is recorded together with the action it
# took there, and a candidate is scored by replaying that fixed list of states
# and asking what it would have done at each.  The candidate never drives the
# simulation, so it cannot influence which states it is judged on, and every
# candidate in a run faces exactly the same decision points.  The state
# distribution and the denominator are both fixed by the target.
#
# What is recorded is a snapshot, not the live net.  `policy_grammar` reads a
# state through a deliberately narrow interface -- `.clock`, and `.places` with
# `._id` / `.marking` of tokens carrying `.time` and `.value` -- so a snapshot
# only has to reproduce that much.  Copying is not optional: gympn hands the
# solver an observable view whose markings alias the live SimVars, and the
# simulator mutates those on the next step.


def first_binding(entries):
    """Default binding picker: whatever gympn offered first."""
    return entries[0]


# ─── recorded state ──────────────────────────────────────────────────────────

class _RecordedToken:
    """A token frozen at record time: `.time` and `.value`, nothing else."""

    __slots__ = ('value', 'time')

    def __init__(self, token):
        value = getattr(token, 'value', None)
        # Attribute dicts are copied because the simulator reuses token objects
        # across steps.  Non-dict values (gympn allows bare scalars) are
        # immutable in practice and kept as-is.
        self.value = dict(value) if isinstance(value, dict) else value
        self.time  = token.time


class _RecordedPlace:
    """A place frozen at record time: `._id` and a copied `.marking`."""

    __slots__ = ('_id', 'marking')

    def __init__(self, place):
        self._id     = place._id
        self.marking = [_RecordedToken(t) for t in place.marking]


class RecordedState:
    """Frozen snapshot of the observable net at one decision point.

    Implements the duck-typed `State` interface `policy_grammar` documents --
    `.clock` and `.places` -- so a policy tree cannot tell it from the live net,
    and cannot reach past it into the simulator.

    It also answers `selected(place, selector)` directly.  Because the snapshot
    never changes, which tokens are ENABLED is fixed the moment it is taken, so
    both selections are computed once here instead of being re-derived for
    every <feature> node of every candidate at every decision point.  That is
    the same answer the generic scan gives, just not recomputed.
    """

    __slots__ = ('clock', 'places', '_all', '_enabled')

    def __init__(self, pn):
        self.clock  = pn.clock
        self.places = [_RecordedPlace(p) for p in pn.places]
        self._all     = {p._id: tuple(p.marking) for p in self.places}
        self._enabled = {p._id: tuple(t for t in p.marking if t.time <= self.clock)
                         for p in self.places}

    def selected(self, place, selector):
        source = self._enabled if selector == ENABLED else self._all
        return source.get(place, ())

    def __repr__(self):
        sizes = {p._id: len(p.marking) for p in self.places}
        return f"RecordedState(clock={self.clock:g}, {sizes})"


# ─── the trace ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TargetTrace:
    """The decision points of one simulation of a target policy.

    `states[i]` is the net at the i-th decision point and `actions[i]` is what
    the target did there -- its raw `evaluate` output, taken before the
    enabled-action fallback, so a candidate naming an action the net cannot
    currently fire counts as a divergence instead of silently collapsing onto
    `postpone`.  That is the same comparison the candidate-driven version made;
    only the states it is made in have changed.
    """

    states: tuple
    actions: tuple

    def __len__(self) -> int:
        return len(self.states)

    def counts(self, tree) -> tuple[int, int] | None:
        """Return (divergences, decision_points) for *tree* over this trace.

        `decision_points` is `len(self)` for every candidate: the denominator
        is a property of the trace, not of the tree being scored.  Returns None
        if the tree cannot be evaluated on some recorded state, so a tree that
        raises can never outrank one that runs.
        """
        try:
            divergences = sum(tree.evaluate(state) != action
                              for state, action in zip(self.states, self.actions))
        except Exception:
            return None
        return divergences, len(self.states)


def record_trace(reference, make_system, n_rollouts, horizon,
                 seed_base=0, pick_binding=None) -> TargetTrace:
    """Simulate *reference* and record every decision point it reaches.

    `pick_binding` decides WHICH binding is taken once the policy has chosen to
    fire.  It is part of the environment rather than of any policy; since only
    the reference is ever simulated, it shapes the recorded trajectory alone,
    and every candidate is then scored on that one trajectory.  Defaults to
    gympn's first offer.

    The reference is evaluated on the snapshot rather than on the live net, so
    the action stored for a state is the action the reference takes on exactly
    the input every candidate will later be given.

    Raises RuntimeError if the reference crashes the model or reaches no
    decision at all.  Either is a broken experiment setup, and an empty trace
    would hand every candidate a perfect score instead of failing.
    """
    from gympn.solvers import HeuristicSolver

    pick    = pick_binding or first_binding
    states  = []
    actions = []

    def heuristic(pn, actions_dict):
        state  = RecordedState(pn)
        action = reference.evaluate(state)
        states.append(state)
        actions.append(action)
        if action == 'postpone':
            return 'postpone'
        if action in actions_dict and actions_dict[action]:
            return {action: pick(actions_dict[action])}
        return 'postpone'

    solver = HeuristicSolver(heuristic_function=heuristic)
    try:
        for r in range(n_rollouts):
            pn = make_system(seed_base + r)
            with contextlib.redirect_stdout(io.StringIO()):
                pn.testing_run(solver=solver, length=horizon)
    except Exception as exc:
        raise RuntimeError(
            f"reference policy {reference!r} failed to drive the model: {exc}"
        ) from exc
    if not states:
        raise RuntimeError(
            f"reference policy {reference!r} reached no decision point in "
            f"{n_rollouts} rollouts of length {horizon}; there is nothing to "
            f"score candidates on"
        )
    return TargetTrace(tuple(states), tuple(actions))


class TraceCache:
    """Records target traces on demand and keeps them for the rest of the run.

    A trace depends only on the target policy and the environment, never on the
    candidate being scored, so it is recorded once and replayed for every
    fitness evaluation.  That is where the speed-up over candidate-driven
    scoring comes from: one simulation per target, instead of `n_rollouts`
    simulations per individual per generation.

    `n_rollouts` and `horizon` are per-call arguments rather than constructor
    state because the drivers read them as module globals at call time and
    `evaluation.py` rewrites those globals between runs (`_apply_overrides`);
    taking them per call, and keying the cache on them, keeps an override from
    being served a trace recorded under the old settings.
    """

    def __init__(self, make_system, seed_base=0, pick_binding=None):
        self.make_system  = make_system
        self.seed_base    = seed_base
        self.pick_binding = pick_binding
        self._traces      = {}

    def trace(self, reference, n_rollouts, horizon) -> TargetTrace:
        # `repr` is the key because the grammar renders canonically and policy
        # trees are unhashable (mutable dataclasses): two trees with the same
        # repr are the same policy and must share a trace.
        key   = (repr(reference), n_rollouts, horizon)
        trace = self._traces.get(key)
        if trace is None:
            trace = record_trace(
                reference, self.make_system, n_rollouts, horizon,
                seed_base=self.seed_base, pick_binding=self.pick_binding,
            )
            self._traces[key] = trace
        return trace

    def counts(self, tree, reference, n_rollouts, horizon):
        """(divergences, decision_points) for *tree* against *reference*."""
        return self.trace(reference, n_rollouts, horizon).counts(tree)


def primary_rate(counts):
    """Negative divergence RATE -- divergences per decision point, in [-1, 0].

    The denominator is the length of the target's recorded trace, so it is the
    same for every candidate in a run and the rate is a plain rescale of the
    divergence count.  It is kept as the primary because a score in [-1, 0] is
    comparable across experiments with different trace lengths, and because the
    cosmetic weights in each driver are tuned against that range.

    Under candidate-driven scoring this normalisation was load-bearing rather
    than cosmetic: a tree controlled how many decisions it reached, so absolute
    counts rewarded policies that simply did less.  Target-driven scoring
    closes that route at the source.
    """
    if counts is None:
        return None
    divergences, decisions = counts
    if decisions == 0:
        return None
    return -divergences / decisions


def make_evaluator(evaluate_primary, cosmetic_config, scale: float = 1.0):
    """Wrap a primary score with the cosmetic penalty; -inf on simulation error.

    `scale` multiplies the cosmetic penalty.  It defaults to 1.0 -- the
    configured weights taken at face value -- which is what a single tree in
    isolation (a seed, a reference policy, a baseline row) has to be scored
    with.  Inside `run_ga` the scale is instead recomputed per generation from
    the population; see `Experiment.scale_config`.
    """
    def evaluate(tree) -> float:
        primary = evaluate_primary(tree)
        if primary is None:
            return -float('inf')
        return combined_fitness(primary, tree, cosmetic_config, scale)
    return evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Experiment driver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Experiment:
    """Everything `run_ga` needs that is not the grammar itself."""
    name: str                          # banner text
    config: GrammarConfig
    evaluate: Callable
    evaluate_primary: Callable
    cosmetic_config: object
    seed_policies: Sequence
    seed_policy: object                # the one summarised in the header line
    pop_size: int
    n_generations: int
    tournament_k: int
    p_crossover: float
    elite_k: int
    reference_policy: object = None
    reference_label: str = 'reference'
    conformance_counts: Callable | None = None   # for the final divergence-rate line
    p_seed_mutation: float = 0.35       # share of the initial population from seeds
    seed_mutation_edits: int = 2        # single edits applied to each mutated seed
    # None (or a disabled config) keeps `evaluate` exactly as the driver built
    # it -- fixed weights, scale 1.0.  A ScaleConfig makes run_ga recompute the
    # cosmetic scale each generation from the population's primary spread.
    scale_config: ScaleConfig | None = None


def _population_scores(exp: "Experiment", trees):
    """Score a whole population at once.

    Returns (fitnesses, parts, scale), where `parts` is the per-individual
    (primary, cosmetic) pair -- None when scaling is off, since the fixed-scale
    path goes through `exp.evaluate` and never splits the two apart.
    """
    cfg = exp.scale_config
    if cfg is None or not cfg.enabled:
        return [exp.evaluate(t) for t in trees], None, 1.0

    parts = [(exp.evaluate_primary(t), cosmetic_penalty(t, exp.cosmetic_config))
             for t in trees]
    scale = cosmetic_scale([p for p, _ in parts], [c for _, c in parts], cfg)
    fitnesses = [-float('inf') if p is None else p - scale * c for p, c in parts]
    return fitnesses, parts, scale


def log_gen(gen, fitnesses, gen_best_fit, gen_best_tree,
            evaluate_primary, cosmetic_config, scale: float = 1.0):
    import numpy as np
    avg      = float(np.mean(fitnesses))
    primary  = evaluate_primary(gen_best_tree)
    cosmetic = cosmetic_penalty(gen_best_tree, cosmetic_config)
    primary_str = f"{primary:.4f}" if primary is not None else "err"
    scale_str   = "" if scale == 1.0 else f" x{scale:.3g}"
    print(
        f"Gen {gen:2d} | best={gen_best_fit:.4f}  avg={avg:.4f}  "
        f"(primary={primary_str}, cosmetic=-{cosmetic:.4f}{scale_str})  "
        f"| {repr(gen_best_tree)}"
    )


def run_ga(exp: Experiment, seed: int = 42, verbose: bool = True,
           on_generation: Callable | None = None):
    """Run the genetic algorithm.

    Parameters
    ----------
    verbose : print the banner, the per-generation log and the closing summary.
    on_generation : called once per generation (including generation 0) with a
        dict of {gen, mean_fit, gen_best_fit, gen_best_tree, best_fit,
        best_tree, cosmetic_scale}.  `gen_best_*` is this generation's best,
        `best_*` the best seen so far -- a batch runner needs both and neither
        survives the printed log.  `cosmetic_scale` is the factor the cosmetic
        penalty was multiplied by this generation (always 1.0 when
        `exp.scale_config` is off); the fitness numbers are only comparable to
        each other at equal scale, so a caller reporting them needs it.

    Returns
    -------
    best_tree : grammar policy tree with the highest observed fitness
    best_fit  : corresponding combined fitness score
    """
    import numpy as np
    import time

    rng = random.Random(seed)
    np.random.seed(seed)
    cfg = exp.config

    def say(*args):
        if verbose:
            print(*args)

    def report(gen, fitnesses, gen_best_fit, gen_best_tree, best_fit, best_tree,
               scale):
        if verbose:
            log_gen(gen, fitnesses, gen_best_fit, gen_best_tree,
                    exp.evaluate_primary, exp.cosmetic_config, scale)
        if on_generation is not None:
            on_generation({
                'gen': gen,
                'mean_fit': float(np.mean(fitnesses)),
                'gen_best_fit': gen_best_fit,
                'gen_best_tree': gen_best_tree,
                'best_fit': best_fit,
                'best_tree': best_tree,
                'cosmetic_scale': scale,
            })

    say("-" * 70)
    say(f"Genetic Algorithm - {exp.name}")
    say(f"  population={exp.pop_size}  generations={exp.n_generations}")
    if exp.reference_policy is not None:
        say(f"\nReference policy ({exp.reference_label}):\n  {repr(exp.reference_policy)}")
    say("-" * 70)

    say("\nEvaluating seed policies ...")
    seed_fit = exp.evaluate(exp.seed_policy)
    for i, s in enumerate(exp.seed_policies):
        say(f"Seed {chr(ord('A') + i)} fitness: {exp.evaluate(s):.4f}  {repr(s)}")
    say()

    # Initial population: one copy of each seed, then a mix of mutated seeds and
    # fresh random trees.  Keeping every seed alive lets crossover combine the
    # building blocks they each carry.
    population = [copy.deepcopy(s) for s in exp.seed_policies]
    while len(population) < exp.pop_size:
        if rng.random() < exp.p_seed_mutation:
            src = rng.choice(list(exp.seed_policies))
            population.append(mutate(src, rng, cfg, n_edits=exp.seed_mutation_edits))
        else:
            population.append(rnd_policy(rng, cfg))

    say("Evaluating initial population ...")
    t0        = time.time()
    fitnesses, parts, scale = _population_scores(exp, population)
    say(f"Done in {time.time() - t0:.1f} s\n")

    best_idx   = int(np.argmax(fitnesses))
    best_fit   = fitnesses[best_idx]
    best_tree  = copy.deepcopy(population[best_idx])
    # (primary, cosmetic) of the incumbent, so it can be re-scored under a
    # later generation's scale without re-simulating it.
    best_parts = None if parts is None else parts[best_idx]

    history = [(0, float(np.mean(fitnesses)), best_fit, best_tree)]
    report(0, fitnesses, best_fit, best_tree, best_fit, best_tree, scale)

    for gen in range(1, exp.n_generations + 1):
        # elitism: carry the top ELITE_K individuals forward unchanged
        elite_idx = sorted(range(len(fitnesses)),
                           key=lambda i: fitnesses[i], reverse=True)[:exp.elite_k]
        new_pop = [copy.deepcopy(population[i]) for i in elite_idx]
        history.append((gen, float(np.mean(fitnesses)), best_fit, best_tree))

        while len(new_pop) < exp.pop_size:
            if rng.random() < exp.p_crossover:
                c1, c2 = crossover(
                    tournament(population, fitnesses, exp.tournament_k, rng),
                    tournament(population, fitnesses, exp.tournament_k, rng),
                    rng, cfg,
                )
                new_pop.append(mutate(c1, rng, cfg)
                               if rng.random() < cfg.p_mutation else c1)
                if len(new_pop) < exp.pop_size:
                    new_pop.append(mutate(c2, rng, cfg)
                                   if rng.random() < cfg.p_mutation else c2)
            else:
                parent = tournament(population, fitnesses, exp.tournament_k, rng)
                new_pop.append(mutate(parent, rng, cfg)
                               if rng.random() < cfg.p_mutation else copy.deepcopy(parent))

        population = new_pop
        fitnesses, parts, scale = _population_scores(exp, population)

        # Under adaptive scaling the scale moves between generations, so the
        # incumbent's stored fitness is on a different footing than this
        # generation's.  Re-score it at the current scale before comparing.
        if best_parts is not None:
            b_primary, b_cosmetic = best_parts
            best_fit = (-float('inf') if b_primary is None
                        else b_primary - scale * b_cosmetic)

        gen_best_i   = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_i]
        if gen_best_fit > best_fit:
            best_fit   = gen_best_fit
            best_tree  = copy.deepcopy(population[gen_best_i])
            best_parts = None if parts is None else parts[gen_best_i]

        report(gen, fitnesses, gen_best_fit, population[gen_best_i],
               best_fit, best_tree, scale)

    if verbose:
        print("\n" + "=" * 70)
        print("GA complete.")
        if scale != 1.0:
            print(f"  Cosmetic scale      : {scale:.4g}  "
                  f"(final generation; seed fitness below is at scale 1.0)")
        print(f"  Seed policy fitness : {seed_fit:.4f}")
        print(f"  Best policy fitness : {best_fit:.4f}  (delta = {best_fit - seed_fit:+.4f})")
        if exp.conformance_counts is not None:
            counts = exp.conformance_counts(best_tree)
            if counts is not None and counts[1]:
                print(f"  Best divergence rate: {counts[0] / counts[1]:.4f}  "
                      f"({counts[0]} of {counts[1]} decisions)")
        if exp.reference_policy is not None:
            print(f"\nReference policy:\n  {repr(exp.reference_policy)}")
        print(f"\nBest policy:\n  {repr(best_tree)}")
        print("=" * 70)

        print("\nHistory:")
        for gen, avg_fitness, best_fitness, tree in history:
            print(f"Gen {gen:2d} | best={best_fitness:.4f}  avg={avg_fitness:.4f}")
            print(f"  {repr(tree)}")
            print()

    return best_tree, best_fit
