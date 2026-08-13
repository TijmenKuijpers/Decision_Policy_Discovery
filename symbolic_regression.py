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
construction, the mutation walk, crossover, tournament selection, conformance
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

A domain that needs a genuinely novel expression form can still pass its own
callables into `GrammarConfig` -- the declarative path is the default, not a
cage.

Note on reproducibility: this module replaced an earlier design in which each
experiment hand-wrote its own expression sampler.  The rewrite changes the
random-number call sequence, so runs seeded before it will not reproduce.  The
*distributions* are preserved; the specific draws are not.
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
    Count, Sum, Min, Max, Mean, Clock, Number, Div,
    ALL, ENABLED,
)
from ga_fitness import ite_depth, combined_fitness, cosmetic_penalty


# ─────────────────────────────────────────────────────────────────────────────
# Expression vocabulary
# ─────────────────────────────────────────────────────────────────────────────

AGGREGATES_DEFAULT = (Sum, Min, Max, Mean)

# <ratio> denominators
COUNT_SAME_PLACE = 'count_same_place'   # Div(AGG(p, x), COUNT(p))  -- a share
CLOCK_DENOM      = 'clock'              # Div(AGG(p, x), CLOCK)     -- a rate


@dataclass(frozen=True)
class RatioSpec:
    """A composite Div(...) expression form.

    Two denominators are supported because they mean different things:

      COUNT_SAME_PLACE -- a *share*: the same place appears in numerator and
                          denominator, e.g. SUM(delivered, on_time) / COUNT(delivered).
      CLOCK_DENOM      -- a *rate*: per unit of elapsed time, e.g.
                          SUM(machine, nr_exec) / CLOCK.

    `places` / `attribute` narrow the ratio to a subset of the vocabulary; both
    default to the vocabulary's own `attr_features`.
    """
    denominator: str
    places:      Sequence[str] = ()
    attribute:   str | None    = None


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
    ratio:         RatioSpec | None = None
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
    w_ratio:     float = 0.0

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
    p_ratio_keep:    float = 0.70      # keep the ratio shape, else escape

    int_lo:   int = 0
    int_hi:   int = 8
    int_step: tuple[int, int] = (-2, 2)
    float_nudges: Sequence[float] = (-0.1, -0.05, 0.05, 0.1)
    float_lo: float | None = 0.0
    float_hi: float | None = 1.0       # None disables the upper clamp
    round_to: int = 2

    # ── derived ──────────────────────────────────────────────────────────────
    _branches: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        enabled = {
            'count':     bool(self.places),
            'number':    bool(self.int_pool or self.float_pool),
            'aggregate': bool(self.attr_features and self.aggregates),
            'clock':     bool(self.with_clock),
            'ratio':     self.ratio is not None,
        }
        weights = {
            'count':     self.w_count,
            'number':    self.w_number,
            'aggregate': self.w_aggregate,
            'clock':     self.w_clock,
            'ratio':     self.w_ratio,
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
            ('ratio',     self._rnd_ratio),
        ]
        cum = 0.0
        branches = []
        for key, build in order:
            if not enabled[key] or not weights[key]:
                continue
            cum += weights[key]
            branches.append((cum, build))
        self._branches = branches

    # ── helpers ──────────────────────────────────────────────────────────────

    def _selector(self, rng):
        return rnd_selector(rng, self.p_all) if self.use_selectors else ALL

    def _agg_node(self, agg, place, attribute, rng):
        if self.use_selectors:
            return agg(place, attribute, rnd_selector(rng, self.p_all))
        return agg(place, attribute)

    def _ratio_place_attr(self, rng):
        spec = self.ratio
        places = spec.places or [p for p, _ in self.attr_features]
        place = rng.choice(list(places))
        if spec.attribute is not None:
            return place, spec.attribute
        # fall back to the attribute paired with this place in the vocabulary
        for p, a in self.attr_features:
            if p == place:
                return place, a
        return place, self.attr_features[0][1]

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

    def _rnd_ratio(self, rng):
        agg = rng.choice(list(self.aggregates))
        if self.ratio.denominator == CLOCK_DENOM:
            place, attribute = rng.choice(list(self.attr_features))
            return Div(self._agg_node(agg, place, attribute, rng), Clock())
        place, attribute = self._ratio_place_attr(rng)
        # The denominator normalises by the whole place, so it takes no selector.
        return Div(self._agg_node(agg, place, attribute, rng), Count(place))

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
        """Mutate within the node's own family, preserving surrounding structure.

        Div is dispatched before the aggregates so a ratio keeps its shape rather
        than decaying into its numerator.
        """
        if self.p_family_escape and rng.random() < self.p_family_escape:
            return self.rnd_expr(rng)

        if isinstance(expr, Count):
            if rng.random() < self.p_count_place:
                return Count(rng.choice(list(self.places)), expr.selector)
            return Count(expr.place, self._selector(rng))

        if isinstance(expr, Div):
            if self.ratio is not None and rng.random() < self.p_ratio_keep:
                return self._rnd_ratio(rng)
            return self.rnd_expr(rng)

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
    p_mutation: float = 0.35      # default per-node mutation probability

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


def mutate_condition(cond, rng: random.Random, cfg: GrammarConfig):
    if isinstance(cond, Compare):
        return mutate_compare(cond, rng, cfg)
    if isinstance(cond, (And, Or)):
        r = rng.random()
        if r < 0.10:                                   # flip And <-> Or
            cls = Or if isinstance(cond, And) else And
            return cls(mutate_condition(cond.left, rng, cfg), mutate_condition(cond.right, rng, cfg))
        if r < 0.55:
            return type(cond)(mutate_condition(cond.left, rng, cfg), copy.deepcopy(cond.right))
        return     type(cond)(copy.deepcopy(cond.left),             mutate_condition(cond.right, rng, cfg))
    if isinstance(cond, Not):
        return Not(mutate_condition(cond.operand, rng, cfg))
    return rnd_condition(rng, cfg)


def mutate_action(action, rng: random.Random, cfg: GrammarConfig):
    return Fire(rng.choice(cfg.transitions)) if rng.random() < cfg.p_mutate_fire else Postpone()


def mutate(tree, rng: random.Random, cfg: GrammarConfig, p: float | None = None):
    """Return a (possibly mutated) deep copy of *tree*.

    Each IfThenElse node is visited in turn.  With probability *p* the node
    is altered; otherwise only its else_-branch is recursed into.
    Operations:
        • mutate condition         (35 %)
        • mutate then-action       (20 %)
        • prepend a new rule       (15 %, only if depth < cfg.max_depth)
        • prune this rule          (15 %)
        • replace subtree randomly (15 %)
    """
    if p is None:
        p = cfg.p_mutation

    if not isinstance(tree, IfThenElse):
        return mutate_action(tree, rng, cfg) if rng.random() < p else copy.deepcopy(tree)

    if rng.random() < p:
        op = rng.random()
        if op < 0.35:
            return IfThenElse(
                condition=mutate_condition(tree.condition, rng, cfg),
                then=copy.deepcopy(tree.then),
                else_=mutate(tree.else_, rng, cfg, p),
            )
        if op < 0.55:
            return IfThenElse(
                condition=copy.deepcopy(tree.condition),
                then=mutate_action(tree.then, rng, cfg),
                else_=mutate(tree.else_, rng, cfg, p),
            )
        if op < 0.70 and ite_depth(tree) < cfg.max_depth:
            return IfThenElse(
                condition=rnd_condition(rng, cfg),
                then=rnd_action(rng, cfg),
                else_=copy.deepcopy(tree),
            )
        if op < 0.85:
            return copy.deepcopy(tree.else_)
        return rnd_policy(rng, cfg, depth=ite_depth(tree))

    return IfThenElse(
        condition=copy.deepcopy(tree.condition),
        then=copy.deepcopy(tree.then),
        else_=mutate(tree.else_, rng, cfg, p),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Crossover / selection
# ─────────────────────────────────────────────────────────────────────────────

def crossover(p1, p2, rng: random.Random):
    """Subtree crossover: swap else_ branches at randomly chosen ITE nodes.

    Both parents are deep-copied first so the originals are never modified.
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
    return c1, c2


def tournament(population, fitnesses, k: int, rng: random.Random):
    idx = rng.sample(range(len(population)), k)
    return population[max(idx, key=lambda i: fitnesses[i])]


# ─────────────────────────────────────────────────────────────────────────────
# Conformance measurement
# ─────────────────────────────────────────────────────────────────────────────

def first_binding(entries):
    """Default binding picker: whatever gympn offered first."""
    return entries[0]


def conformance_counts(tree, reference, make_system, n_rollouts, horizon,
                       seed_base=0, pick_binding=None):
    """Return (divergences, decision_points) summed over all rollouts.

    Divergence is measured *on-policy*: the candidate drives the simulation and
    is compared, at each decision point it reaches, against what *reference*
    would have chosen in that same state.  The two therefore see the same state
    only for as long as they agree.

    `pick_binding` decides WHICH binding is taken once a policy has chosen to
    fire -- part of the environment, applied identically to every candidate, so
    conformance comparisons stay fair.  Defaults to gympn's first offer.

    Returns None on simulation error, so a tree that crashes the model can never
    outrank one that runs.
    """
    from gympn.solvers import HeuristicSolver

    pick = pick_binding or first_binding
    counters = {'div': 0, 'decisions': 0}

    def heuristic(pn, actions_dict):
        action_tree = tree.evaluate(pn)
        action_ref  = reference.evaluate(pn)
        counters['decisions'] += 1
        if action_tree != action_ref:
            counters['div'] += 1
        if action_tree == 'postpone':
            return 'postpone'
        if action_tree in actions_dict and actions_dict[action_tree]:
            return {action_tree: pick(actions_dict[action_tree])}
        return 'postpone'

    solver = HeuristicSolver(heuristic_function=heuristic)
    try:
        for r in range(n_rollouts):
            pn = make_system(seed_base + r)
            with contextlib.redirect_stdout(io.StringIO()):
                pn.testing_run(solver=solver, length=horizon)
    except Exception:
        return None
    return counters['div'], counters['decisions']


def primary_rate(counts):
    """Negative divergence RATE -- divergences per decision point, in [-1, 0].

    A rate rather than an absolute count: because divergence is measured
    on-policy, a tree also controls how many decision points it reaches, so
    absolute counts reward policies that simply do less.  A tree reaching no
    decision at all scores None (-> -inf), closing the other degenerate route.
    """
    if counts is None:
        return None
    divergences, decisions = counts
    if decisions == 0:
        return None
    return -divergences / decisions


def primary_count(counts, n_rollouts):
    """Negative MEAN ABSOLUTE divergences per rollout.

    Kept for experiments whose cosmetic weights are tuned against a
    count-scaled primary; pair it with an unscaled CosmeticConfig.
    """
    if counts is None:
        return None
    return -counts[0] / n_rollouts


def make_evaluator(evaluate_primary, cosmetic_config):
    """Wrap a primary score with the cosmetic penalty; -inf on simulation error."""
    def evaluate(tree) -> float:
        primary = evaluate_primary(tree)
        if primary is None:
            return -float('inf')
        return combined_fitness(primary, tree, cosmetic_config)
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
    seed_mutation_strength: float = 0.50


def log_gen(gen, fitnesses, gen_best_fit, gen_best_tree,
            evaluate_primary, cosmetic_config):
    import numpy as np
    avg      = float(np.mean(fitnesses))
    primary  = evaluate_primary(gen_best_tree)
    cosmetic = cosmetic_penalty(gen_best_tree, cosmetic_config)
    primary_str = f"{primary:.4f}" if primary is not None else "err"
    print(
        f"Gen {gen:2d} | best={gen_best_fit:.4f}  avg={avg:.4f}  "
        f"(primary={primary_str}, cosmetic=-{cosmetic:.4f})  | {repr(gen_best_tree)}"
    )


def seed_baseline_table(baselines, conformance_counts_fn, evaluate_fn):
    """Print the reference/seed/degenerate fitness table used by every --check.

    `baselines` is a sequence of (label, tree).  Including a bare Postpone() is
    worth doing: with a rate-based primary it must NOT outrank the real seeds,
    and with an absolute-count primary it does -- so the row is a live check on
    the fitness definition, not decoration.
    """
    print(f"  {'policy':<12} {'diverg.':>8} {'decisions':>10} {'rate':>7} {'fitness':>9}")
    print("  " + "-" * 50)
    for name, tree in baselines:
        counts = conformance_counts_fn(tree)
        if counts is None:
            print(f"  {name:<12} {'err':>8}")
            continue
        div, dec = counts
        rate = div / dec if dec else float('nan')
        print(f"  {name:<12} {div:>8} {dec:>10} {rate:>7.3f} {evaluate_fn(tree):>9.4f}")


def run_ga(exp: Experiment, seed: int = 42, verbose: bool = True,
           on_generation: Callable | None = None):
    """Run the genetic algorithm.

    Parameters
    ----------
    verbose : print the banner, the per-generation log and the closing summary.
    on_generation : called once per generation (including generation 0) with a
        dict of {gen, mean_fit, gen_best_fit, gen_best_tree, best_fit,
        best_tree}.  `gen_best_*` is this generation's best, `best_*` the best
        seen so far -- a batch runner needs both and neither survives the
        printed log.

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

    def report(gen, fitnesses, gen_best_fit, gen_best_tree, best_fit, best_tree):
        if verbose:
            log_gen(gen, fitnesses, gen_best_fit, gen_best_tree,
                    exp.evaluate_primary, exp.cosmetic_config)
        if on_generation is not None:
            on_generation({
                'gen': gen,
                'mean_fit': float(np.mean(fitnesses)),
                'gen_best_fit': gen_best_fit,
                'gen_best_tree': gen_best_tree,
                'best_fit': best_fit,
                'best_tree': best_tree,
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
            population.append(mutate(src, rng, cfg, p=exp.seed_mutation_strength))
        else:
            population.append(rnd_policy(rng, cfg))

    say("Evaluating initial population ...")
    t0        = time.time()
    fitnesses = [exp.evaluate(ind) for ind in population]
    say(f"Done in {time.time() - t0:.1f} s\n")

    best_idx  = int(np.argmax(fitnesses))
    best_fit  = fitnesses[best_idx]
    best_tree = copy.deepcopy(population[best_idx])

    history = [(0, float(np.mean(fitnesses)), best_fit, best_tree)]
    report(0, fitnesses, best_fit, best_tree, best_fit, best_tree)

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
                    rng,
                )
                new_pop.append(mutate(c1, rng, cfg))
                if len(new_pop) < exp.pop_size:
                    new_pop.append(mutate(c2, rng, cfg))
            else:
                parent = tournament(population, fitnesses, exp.tournament_k, rng)
                new_pop.append(mutate(parent, rng, cfg))

        population = new_pop
        fitnesses  = [exp.evaluate(ind) for ind in population]

        gen_best_i   = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_i]
        if gen_best_fit > best_fit:
            best_fit  = gen_best_fit
            best_tree = copy.deepcopy(population[gen_best_i])

        report(gen, fitnesses, gen_best_fit, population[gen_best_i],
               best_fit, best_tree)

    if verbose:
        print("\n" + "=" * 70)
        print("GA complete.")
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
