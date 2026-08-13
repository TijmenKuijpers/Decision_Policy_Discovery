"""
policy_grammar.py
-----------------
Python AST node classes for rule-based decision policies in Petri nets,
implementing the policy grammar G_pi (Algorithm 1 of the accompanying paper,
"Policy Discovery in Process Models with Symbolic Regression").

Grammar recap (G_pi)
--------------------
<policy>      ::= IF <condition> THEN <action> ELSE <policy> | <action>
<condition>   ::= <condition> OR <conjunction> | <conjunction>
<conjunction> ::= <conjunction> AND <literal>  | <literal>
<literal>     ::= NOT <literal> | ( <condition> ) | <expr> <compop> <expr>
<compop>      ::= < | > | = | != | <= | >=
<expr>        ::= <expr> + <term> | <expr> - <term> | <term>
<term>        ::= <term> * <factor> | <term> / <factor> | <factor>
<factor>      ::= <feature> | <number> | ( <expr> )
<feature>     ::= COUNT(p, s) | SUM(p, x, s) | MIN(p, x, s)
                | MAX(p, x, s) | MEAN(p, x, s) | CLOCK
                  where p in P, x in C(p), s in {ALL, ENABLED}
<number>      ::= c, c in R
<action>      ::= FIRE(t) | POSTPONE,  t in T_A

Extensions beyond the published G_pi
------------------------------------
CLOCK is an addition, not part of Algorithm 1 in the paper.  The published
feature set is entirely token aggregates, so the simulation time tau enters a
policy only through the ENABLED selector and can never be used as a quantity.
That makes rate-based policies — executions per time unit, throughput targets,
deadline slack — inexpressible.  Any policy using CLOCK is outside the
language as published; see the Clock class for detail.

The classes below are the *AST* form of this grammar, not its parse tree: the
precedence-only non-terminals <conjunction>, <literal>, <term> and <factor>
collapse into the nodes they wrap, so e.g. `COUNT(chips, ALL) > 0` is a single
Compare node.  Parentheses likewise carry no node of their own — nesting is
expressed by the tree structure itself.

Empty-selection conventions
---------------------------
Aggregates over an empty token selection would otherwise be undefined, so each
node is made total by returning its identity element:

    SUM  -> 0          (additive identity)
    MIN  -> +inf       (identity for min; MIN(...) < c is then False)
    MAX  -> -inf       (identity for max; MAX(...) > c is then False)
    MEAN -> 0.0        (no identity exists — chosen by convention)
    Div  -> 1.0        when the denominator is 0 (see Div)

These conventions are an implementation decision; the grammar itself does not
fix them.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union


# ── Type aliases ──────────────────────────────────────────────────────────────

# State is the *full observable Petri net* at the current decision point.
# It is duck-typed: any object that exposes
#   .clock   : float            — current simulation time
#   .places  : iterable         — each item has
#                  ._id     : str
#                  .marking : iterable of tokens, each with .time : float
# Feature nodes (e.g. Count) are responsible for extracting their own
# numeric values from this raw state; no pre-processing happens outside them.
State  = Any
Action = str   # name of a transition, or 'postpone'


# ─────────────────────────────────────────────────────────────────────────────
# Selectors  (s in {ALL, ENABLED})
# ─────────────────────────────────────────────────────────────────────────────

ALL       = 'ALL'
ENABLED   = 'ENABLED'
SELECTORS = {ALL, ENABLED}


def _selected_tokens(state: State, place: str, selector: str) -> list:
    """Return M^s(p): the tokens of *place* retained by *selector*.

        M^ALL(p)     = M(p)
        M^ENABLED(p) = {(v, t') in M(p) | t' <= tau}     (tau = state.clock)

    A place that does not exist in the net yields the empty selection rather
    than raising, so a policy naming an unknown place stays evaluable.
    """
    for p in state.places:
        if p._id == place:
            if selector == ENABLED:
                return [t for t in p.marking if t.time <= state.clock]
            return list(p.marking)
    return []


class _FeatureNode:
    """Mixin validating the ALL/ENABLED selector shared by all <feature> nodes."""

    def __post_init__(self):
        if self.selector not in SELECTORS:
            raise ValueError(
                f"Invalid selector '{self.selector}'. Must be one of {SELECTORS}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Expressions  (<expr>, <term>, <factor>, <feature>)
# Each node implements evaluate(state) -> numeric value
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Count(_FeatureNode):
    """<feature> ::= COUNT(p, s)

        COUNT(p, s) = |M^s(p)|

    Example:
        Count('chips').evaluate(pn)             -> all tokens in 'chips'
        Count('chips', ENABLED).evaluate(pn)    -> only those with time <= clock
    """
    place: str
    selector: str = ALL

    def evaluate(self, state: State) -> int:
        return len(_selected_tokens(state, self.place, self.selector))

    def __repr__(self) -> str:
        return f"COUNT({self.place}, {self.selector})"


@dataclass
class Sum(_FeatureNode):
    """<feature> ::= SUM(p, x, s)

    Sum of attribute *x* over M^s(p).  An empty selection sums to 0.

    Example:
        Sum('delivered', 'on_time').evaluate(pn)
            -> number of on-time deliveries (Python sums booleans as 0/1)
    """
    place: str
    attribute: str
    selector: str = ALL

    def evaluate(self, state: State) -> float:
        return sum(t.value[self.attribute]
                   for t in _selected_tokens(state, self.place, self.selector))

    def __repr__(self) -> str:
        return f"SUM({self.place}, {self.attribute}, {self.selector})"


@dataclass
class Min(_FeatureNode):
    """<feature> ::= MIN(p, x, s)

    Minimum of attribute *x* over M^s(p).  An empty selection returns +inf,
    the identity for min, so `MIN(...) < c` is False when nothing qualifies.
    """
    place: str
    attribute: str
    selector: str = ALL

    def evaluate(self, state: State) -> float:
        values = [t.value[self.attribute]
                  for t in _selected_tokens(state, self.place, self.selector)]
        return min(values) if values else float('inf')

    def __repr__(self) -> str:
        return f"MIN({self.place}, {self.attribute}, {self.selector})"


@dataclass
class Max(_FeatureNode):
    """<feature> ::= MAX(p, x, s)

    Maximum of attribute *x* over M^s(p).  An empty selection returns -inf,
    the identity for max, so `MAX(...) > c` is False when nothing qualifies.
    """
    place: str
    attribute: str
    selector: str = ALL

    def evaluate(self, state: State) -> float:
        values = [t.value[self.attribute]
                  for t in _selected_tokens(state, self.place, self.selector)]
        return max(values) if values else -float('inf')

    def __repr__(self) -> str:
        return f"MAX({self.place}, {self.attribute}, {self.selector})"


@dataclass
class Mean(_FeatureNode):
    """<feature> ::= MEAN(p, x, s)

    Mean of attribute *x* over M^s(p).  An empty selection returns 0.0.
    Unlike MIN/MAX the mean has no identity element, so 0.0 is a convention
    rather than a derivation — see the module docstring.
    """
    place: str
    attribute: str
    selector: str = ALL

    def evaluate(self, state: State) -> float:
        values = [t.value[self.attribute]
                  for t in _selected_tokens(state, self.place, self.selector)]
        return sum(values) / len(values) if values else 0.0

    def __repr__(self) -> str:
        return f"MEAN({self.place}, {self.attribute}, {self.selector})"


@dataclass
class Clock:
    """<feature> ::= CLOCK        [EXTENSION — not in the published G_pi]

    The current simulation time tau of state <M, tau>.

    Every other <feature> is an aggregate over tokens, so tau is reachable only
    indirectly, through the ENABLED selector's tau comparison.  That is enough
    to ask "is this token available yet" but not to use elapsed time as a
    quantity, which rules out any rate policy — executions per time unit,
    throughput targets, deadline slack.  CLOCK closes that gap.

    Takes no place, attribute or selector: tau is a property of the state as a
    whole, not of any one place's marking.

    Example:
        Div(Sum('machine', 'nr_exec'), Clock()).evaluate(pn)
            -> executions per time unit so far
    """

    def evaluate(self, state: State) -> float:
        return state.clock

    def __repr__(self) -> str:
        return "CLOCK"


@dataclass
class Number:
    """<factor> ::= <number>

    A numeric literal.

    Example:
        Number(5).evaluate({}) -> 5
    """
    value: int | float

    def evaluate(self, state: State) -> int | float:
        return self.value

    def __repr__(self) -> str:
        return str(self.value)


# All <feature> node types.  Kept in one place so downstream consumers
# (e.g. ga_fitness.TERMINALS) cannot drift out of sync when a feature is added.
FEATURES = (Count, Sum, Min, Max, Mean, Clock)

# Expr is anything that evaluates to a number
Expr = Union[Count, Sum, Min, Max, Mean, Clock, Number,
             "Add", "Sub", "Mul", "Div"]


@dataclass
class Add:
    """<expr> ::= <expr> + <term>"""
    left: Expr
    right: Expr

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) + self.right.evaluate(state)

    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class Sub:
    """<expr> ::= <expr> - <term>"""
    left: Expr
    right: Expr

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) - self.right.evaluate(state)

    def __repr__(self) -> str:
        return f"({self.left} - {self.right})"


@dataclass
class Mul:
    """<term> ::= <term> * <factor>"""
    left: Expr
    right: Expr

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) * self.right.evaluate(state)

    def __repr__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class Div:
    """<term> ::= <term> / <factor>

    Returns 1.0 when the denominator is zero.  For SLA ratios this is the
    correct default (empty delivered place → perfect SLA = 1.0 → no violation).
    """
    left: Expr
    right: Expr

    def evaluate(self, state: State) -> float:
        r = self.right.evaluate(state)
        if r == 0:
            return 1.0
        return self.left.evaluate(state) / r

    def __repr__(self) -> str:
        return f"({self.left} / {self.right})"


# ─────────────────────────────────────────────────────────────────────────────
# Conditions  (<condition>, <conjunction>, <literal>)
# Each node implements evaluate(state) -> bool
# ─────────────────────────────────────────────────────────────────────────────

# Condition is anything that evaluates to a bool
Condition = Union["Compare", "And", "Or", "Not"]

COMPARATORS = {'>', '>=', '<', '<=', '=', '!='}


@dataclass
class Compare:
    """<literal> ::= <expr> <comparator> <expr>

    Compares two expressions. Valid comparators: >, >=, <, <=, =, !=

    Example:
        Compare('>', Count('material_c', ENABLED), Number(0)).evaluate(pn) -> True
            when pn has >= 1 enabled token in place 'material_c'.
    """
    op: str
    left: Expr
    right: Expr

    def __post_init__(self):
        if self.op not in COMPARATORS:
            raise ValueError(
                f"Invalid comparator '{self.op}'. Must be one of {COMPARATORS}"
            )

    def evaluate(self, state: State) -> bool:
        l = self.left.evaluate(state)
        r = self.right.evaluate(state)
        return {
            '>' : l > r,
            '>=': l >= r,
            '<' : l < r,
            '<=': l <= r,
            '=' : l == r,
            '!=': l != r,
        }[self.op]

    def __repr__(self) -> str:
        return f"{self.left} {self.op} {self.right}"


@dataclass
class And:
    """<conjunction> ::= <conjunction> AND <literal>

    Short-circuits: right is not evaluated if left is False.

    Example:
        And(Compare('>', Count('mc', ENABLED), Number(0)),
            Compare('>', Count('md', ENABLED), Number(1)))
            .evaluate(pn) -> True  when mc > 0 and md > 1 in pn.
    """
    left: Condition
    right: Condition

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) and self.right.evaluate(state)

    def __repr__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass
class Or:
    """<condition> ::= <condition> OR <conjunction>

    Short-circuits: right is not evaluated if left is True.
    """
    left: Condition
    right: Condition

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) or self.right.evaluate(state)

    def __repr__(self) -> str:
        return f"({self.left} OR {self.right})"


@dataclass
class Not:
    """<literal> ::= NOT <literal>"""
    operand: Condition

    def evaluate(self, state: State) -> bool:
        return not self.operand.evaluate(state)

    def __repr__(self) -> str:
        return f"NOT {self.operand}"


# ─────────────────────────────────────────────────────────────────────────────
# Policy  (<policy>, <action>)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fire:
    """<action> ::= FIRE <transition>

    Selects a transition to fire.

    Example:
        Fire('produce_A').evaluate({}) -> 'produce_A'
    """
    transition: str

    def evaluate(self, state: State) -> Action:
        return self.transition

    def __repr__(self) -> str:
        return f"FIRE {self.transition}"


@dataclass
class Postpone:
    """<action> ::= postpone

    Fires no action transition at this decision point.

    Example:
        Postpone().evaluate({}) -> 'postpone'
    """
    def evaluate(self, state: State) -> Action:
        return 'postpone'

    def __repr__(self) -> str:
        return 'postpone'


# Action is anything that evaluates to an action string
ActionNode = Union[Fire, Postpone]


@dataclass
class IfThenElse:
    """<policy> ::= IF <condition> THEN <action> ELSE <policy>

    If the condition holds, evaluates the then-branch; otherwise recurses
    into the else-branch, which is itself a <policy> (another IfThenElse
    or a base-case ActionNode).

    Example:
        IfThenElse(
            condition = Compare('>', Count('material_c', ENABLED), Number(0)),
            then      = Fire('produce_A'),
            else_     = Postpone()
        ).evaluate(pn) -> 'produce_A'  when material_c has >= 1 enabled token.
    """
    condition: Condition
    then: ActionNode
    else_: Union["IfThenElse", ActionNode]

    def evaluate(self, state: State) -> Action:
        if self.condition.evaluate(state):
            return self.then.evaluate(state)
        return self.else_.evaluate(state)

    def __repr__(self) -> str:
        return f"IF {self.condition} THEN {self.then} ELSE ({self.else_})"


# ─────────────────────────────────────────────────────────────────────────────
# GymPN adapter
# ─────────────────────────────────────────────────────────────────────────────

def make_heuristic(policy_tree: Union[IfThenElse, ActionNode]):
    """Wrap a grammar policy tree in the gympn HeuristicSolver interface.

    The returned function has the signature expected by HeuristicSolver:
        heuristic(pn, actions_dict) -> 'postpone' | {action_name: binding}

    The observable Petri net (*pn*) is passed directly as the state to the
    policy tree.  Feature nodes such as Count are responsible for querying
    whatever they need from *pn* themselves — no pre-processing happens here.

    Args:
        policy_tree: root node of a grammar policy tree (IfThenElse or
                     ActionNode).

    Returns:
        A callable compatible with gympn's HeuristicSolver.
    """
    def heuristic(pn, actions_dict):
        action = policy_tree.evaluate(pn)

        if action == 'postpone':
            return 'postpone'
        if action in actions_dict and actions_dict[action]:
            return {action: actions_dict[action][0]}
        return 'postpone'

    return heuristic


# ─────────────────────────────────────────────────────────────────────────────
# Example: π₁ from the production process running example
# ─────────────────────────────────────────────────────────────────────────────

pi1 = IfThenElse(
    condition=And(
        Compare('>', Count('phone_resource', ENABLED),   Number(0)),
        And(
            Compare('>', Count('stock_chip', ENABLED),       Number(0)),
            Compare('>', Count('stock_phone_case', ENABLED), Number(0))
        )
    ),
    then=Fire('phone_production'),
    else_=IfThenElse(
        condition=And(
            Compare('>', Count('game_resource', ENABLED), Number(0)),
            Compare('>', Count('stock_chip', ENABLED),    Number(0))
        ),
        then=Fire('game_production'),
        else_=Postpone()
    )
)


if __name__ == '__main__':
    # ── minimal mock Petri net for testing ────────────────────────────────────

    class _MockToken:
        def __init__(self, value=None, time=0.0):
            self.value = value if value is not None else {}
            self.time  = time

    class _MockPlace:
        def __init__(self, id_: str, tokens: list):
            self._id     = id_
            self.marking = tokens

    class _MockPN:
        def __init__(self, places: dict, clock: float = 0.0):
            self.clock  = clock
            self.places = [_MockPlace(k, v) for k, v in places.items()]

        def __repr__(self):
            return str({p._id: len(p.marking) for p in self.places})

    def _pn(**counts):
        """Petri net with *n* always-enabled, attribute-less tokens per place."""
        return _MockPN({k: [_MockToken() for _ in range(n)] for k, n in counts.items()})

    failures = 0

    # ── sanity checks for π₁ ─────────────────────────────────────────────────

    policy_cases = [
        # both components + phone resource -> phone
        (_pn(phone_resource=1, game_resource=1, stock_chip=2, stock_phone_case=1),
         'phone_production'),
        # no phone cases, but chips + game resource -> game
        (_pn(phone_resource=1, game_resource=1, stock_chip=2, stock_phone_case=0),
         'game_production'),
        # no chips -> neither branch fires
        (_pn(phone_resource=1, game_resource=1, stock_chip=0, stock_phone_case=1),
         'postpone'),
        # empty system
        (_pn(phone_resource=0, game_resource=0, stock_chip=0, stock_phone_case=0),
         'postpone'),
    ]

    print(f"{'State':<62} {'Expected':<18} {'Got':<18} {'OK'}")
    print('-' * 104)
    for pn, expected in policy_cases:
        result = pi1.evaluate(pn)
        ok = '/' if result == expected else 'X'
        failures += result != expected
        print(f"{str(pn):<62} {expected:<18} {result:<18} {ok}")

    # ── <feature> semantics ──────────────────────────────────────────────────
    # 'orders' holds three tokens; at clock=5.0 only the first two are enabled.

    feature_pn = _MockPN({
        'orders': [_MockToken({'value': 10.0}, 1.0),
                   _MockToken({'value':  4.0}, 5.0),
                   _MockToken({'value':  7.0}, 9.0)],
        'empty':  [],
    }, clock=5.0)

    feature_cases = [
        (Count('orders'),                   3),
        (Count('orders', ENABLED),           2),
        (Sum('orders', 'value'),             21.0),
        (Sum('orders', 'value', ENABLED),    14.0),
        (Min('orders', 'value'),             4.0),
        (Min('orders', 'value', ENABLED),    4.0),
        (Max('orders', 'value'),             10.0),
        (Max('orders', 'value', ENABLED),    10.0),
        (Mean('orders', 'value'),            21.0 / 3),
        (Mean('orders', 'value', ENABLED),   7.0),
        # empty-selection conventions
        (Count('empty'),                     0),
        (Sum('empty', 'value'),              0),
        (Min('empty', 'value'),              float('inf')),
        (Max('empty', 'value'),             -float('inf')),
        (Mean('empty', 'value'),             0.0),
        # Div guards a zero denominator
        (Div(Sum('empty', 'value'), Count('empty')), 1.0),
        # CLOCK reads tau straight off the state
        (Clock(),                            5.0),
        (Div(Count('orders'), Clock()),      3 / 5.0),
    ]

    print()
    print(f"{'Feature':<46} {'Expected':<12} {'Got':<12} {'OK'}")
    print('-' * 84)
    for node, expected in feature_cases:
        result = node.evaluate(feature_pn)
        ok = '/' if result == expected else 'X'
        failures += result != expected
        print(f"{repr(node):<46} {str(expected):<12} {str(result):<12} {ok}")

    # ── selector validation ──────────────────────────────────────────────────

    try:
        Count('orders', 'SOMETIMES')
    except ValueError:
        print("\nSelector validation: rejects unknown selector  /")
    else:
        failures += 1
        print("\nSelector validation: accepted an unknown selector  X")

    print()
    print("pi1 repr:")
    print(repr(pi1))

    print()
    print("FAILURES:", failures)
