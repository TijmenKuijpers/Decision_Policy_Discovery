"""
tests/test_mutation.py
----------------------
Unit tests for the one-edit-per-call mutation operator in
`symbolic_regression.mutate`.

The property the operator is supposed to have is that mutation *strength* is
fixed by the caller, not by the length of the policy: one call applies exactly
one edit at one uniformly chosen rule, however long the else_ chain is.

Note on "one edit" and depth.  Four of the five edits are local -- they change
at most one rule and move `ite_depth` by at most one level.  The fifth
(`replace the suffix`, 20 % of draws) discards everything from the chosen rule
downwards and regrows it, so it can change depth by more than one level; on a
3-rule policy an edit at position 0 can return a bare action.  That is the
operator as specified, so the tests below bound the *local* edits tightly and
bound the replacement branch by frequency instead.

Run with `python -m pytest tests` from the repository root, or directly with
`python tests/test_mutation.py`.  Nothing here starts a simulation.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ga_fitness import ite_depth
from policy_grammar import IfThenElse, Compare, Count, Number, Fire, Postpone
from symbolic_regression import ExprVocab, GrammarConfig, ite_nodes, mutate

VOCAB = ExprVocab(
    places=('ready', 'busy', 'done'),
    use_selectors=False,
    w_count=0.60,
    w_number=0.40,
    w_aggregate=0.0,
)

CFG = GrammarConfig.from_vocab(
    VOCAB,
    comparators=['<', '<=', '>', '>=', '=', '!='],
    transitions=['process', 'reject'],
    max_depth=3,
)

N_SEEDS = 500        # draws per statistical test; enough to pin the weights down


def three_rules():
    """A fixed 3-rule policy -- the longest chain CFG.max_depth allows."""
    return IfThenElse(
        condition=Compare('>', Count('ready'), Number(0)),
        then=Fire('process'),
        else_=IfThenElse(
            condition=Compare('<', Count('busy'), Number(2)),
            then=Fire('reject'),
            else_=IfThenElse(
                condition=Compare('>=', Count('done'), Number(5)),
                then=Fire('process'),
                else_=Postpone(),
            ),
        ),
    )


def one_rule():
    """A 1-rule policy, so prepending has room and is actually exercised."""
    return IfThenElse(
        condition=Compare('>', Count('ready'), Number(0)),
        then=Fire('process'),
        else_=Postpone(),
    )


def rule_reprs(tree):
    """Per-rule repr of each node on the else_ chain, in chain order.

    Each entry excludes the else_ branch, so a rule compares equal to itself
    however the chain below it was re-spliced.
    """
    return [f"{n.condition} -> {n.then}" for n in ite_nodes(tree)]


def edit_shape(base, got):
    """Which single edit turns rule list *base* into *got*.

    Returns 'in_place' (one rule rewritten, or none), 'prepend', 'prune', or
    'replace' -- the catch-all for the suffix-replacement branch, which is the
    only one that may leave more than one rule changed.
    """
    if len(got) == len(base) and sum(1 for a, b in zip(base, got) if a != b) <= 1:
        return 'in_place'
    if len(got) == len(base) + 1:
        if any(got[:j] + got[j + 1:] == base for j in range(len(got))):
            return 'prepend'
    if len(got) == len(base) - 1:
        if any(base[:j] + base[j + 1:] == got for j in range(len(base))):
            return 'prune'
    return 'replace'


def shapes(tree, n=N_SEEDS, **kw):
    """(shape, depth shift) for one mutation of *tree* per seed."""
    base = rule_reprs(tree)
    out = []
    for seed in range(n):
        result = mutate(tree, random.Random(seed), CFG, **kw)
        out.append((edit_shape(base, rule_reprs(result)),
                    ite_depth(result) - ite_depth(tree)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The input is never touched
# ─────────────────────────────────────────────────────────────────────────────

def test_input_tree_is_not_modified():
    t = three_rules()
    before = repr(t)
    for seed in range(N_SEEDS):
        mutate(t, random.Random(seed), CFG)
        assert repr(t) == before


def test_result_is_a_fresh_tree():
    t = three_rules()
    out = mutate(t, random.Random(0), CFG)
    assert out is not t
    assert all(a is not b for a in ite_nodes(out) for b in ite_nodes(t))


def test_result_usually_differs_from_the_input():
    """Every branch is a real edit.  A redrawn action or expression can land on
    the value it already had, so a few fixed points are expected -- but the
    operator must not be a no-op in the common case."""
    t = three_rules()
    unchanged = sum(1 for s in range(N_SEEDS)
                    if repr(mutate(t, random.Random(s), CFG)) == repr(t))
    assert unchanged < N_SEEDS // 5


# ─────────────────────────────────────────────────────────────────────────────
# Exactly one edit lands
# ─────────────────────────────────────────────────────────────────────────────

def test_local_edits_change_one_rule_and_at_most_one_level():
    """The four local edits: one rule rewritten, inserted or dropped, and depth
    moved by at most one level."""
    for tree in (one_rule(), three_rules()):
        for shape, shift in shapes(tree):
            if shape == 'replace':
                continue
            assert abs(shift) <= 1, (shape, shift)
            assert (shift == 0) == (shape == 'in_place')


def test_the_suffix_replacement_is_the_only_wider_edit():
    """Anything that moves depth by more than one level must be the
    replacement branch, and it must stay a minority of draws."""
    for tree in (one_rule(), three_rules()):
        drawn = shapes(tree)
        assert all(shape == 'replace' for shape, shift in drawn if abs(shift) > 1)
        wide = sum(1 for shape, _ in drawn if shape == 'replace')
        assert wide < 0.30 * len(drawn)       # nominal weight is 0.20


def test_max_depth_is_never_exceeded():
    for tree in (one_rule(), three_rules()):
        for seed in range(N_SEEDS):
            assert ite_depth(mutate(tree, random.Random(seed), CFG)) <= CFG.max_depth


def test_every_edit_kind_is_reachable():
    """A sanity check on the weights: a policy with room to grow must produce
    longer, equal-length and shorter results."""
    drawn = shapes(one_rule())
    assert {'in_place', 'prepend', 'prune', 'replace'} <= {s for s, _ in drawn}


def test_prepend_is_blocked_at_max_depth_and_falls_back_to_prune():
    """A policy already at cfg.max_depth can never grow; the blocked prepend
    prunes instead, which shows up as a prune rate near 0.20 + 0.20."""
    drawn = shapes(three_rules())
    assert all(shift <= 0 for _, shift in drawn)
    assert not any(shape == 'prepend' for shape, _ in drawn)
    pruned = sum(1 for shape, _ in drawn if shape == 'prune') / len(drawn)
    assert 0.32 < pruned < 0.48               # nominal 0.40


def test_the_edit_point_is_uniform_over_the_chain():
    """Every rule on the chain gets edited, not just the head: an in-place edit
    must be observed at each of the three positions."""
    t = three_rules()
    base = rule_reprs(t)
    touched = set()
    for seed in range(N_SEEDS):
        got = rule_reprs(mutate(t, random.Random(seed), CFG))
        if len(got) != len(base):
            continue
        touched.update(j for j in range(len(base)) if base[j] != got[j])
    assert touched == {0, 1, 2}


# ─────────────────────────────────────────────────────────────────────────────
# Degenerate input: a policy with no rules at all
# ─────────────────────────────────────────────────────────────────────────────

def test_bare_action_either_swaps_or_gains_one_rule():
    """A bare action is a <Policy>, so mutation may wrap it in a rule.

    It used to be an absorbing state -- only ever swapped for another bare
    action, and left untouched by crossover -- so a population that drifted
    onto the cheapest tree in the space could never re-grow a rule.  The two
    legal outcomes are now: another bare action, or exactly one rule whose
    else-branch is the action it started from.
    """
    t = Postpone()
    saw_action = saw_rule = False
    for seed in range(N_SEEDS):
        out = mutate(t, random.Random(seed), CFG)
        if isinstance(out, (Fire, Postpone)):
            assert ite_depth(out) == 0
            saw_action = True
        else:
            assert ite_depth(out) == 1
            assert isinstance(out.else_, (Fire, Postpone))
            saw_rule = True
    assert saw_action and saw_rule
    assert isinstance(t, Postpone)          # the input is never mutated in place


def test_bare_action_is_redrawn():
    """`d == 0` spends its edit on the action or on prepending a rule, so over
    many seeds a Postpone() must sometimes come back as something else."""
    outs = {repr(mutate(Postpone(), random.Random(s), CFG)) for s in range(N_SEEDS)}
    assert len(outs) > 1


# ─────────────────────────────────────────────────────────────────────────────
# n_edits
# ─────────────────────────────────────────────────────────────────────────────

def test_n_edits_applies_that_many_edits():
    """Three edits equal three successive one-edit calls on the same stream."""
    t = three_rules()

    three_at_once = mutate(t, random.Random(7), CFG, n_edits=3)

    rng = random.Random(7)
    stepwise = t
    for _ in range(3):
        stepwise = mutate(stepwise, rng, CFG)

    assert repr(three_at_once) == repr(stepwise)


def test_n_edits_compounds():
    """The counterpart of the one-edit bound: three edits may change more than
    one rule, and sometimes do."""
    t = three_rules()
    local = sum(1 for shape, _ in shapes(t, n_edits=3) if shape == 'in_place')
    assert local < sum(1 for shape, _ in shapes(t) if shape == 'in_place')


def test_zero_edits_returns_an_untouched_copy():
    t = three_rules()
    out = mutate(t, random.Random(1), CFG, n_edits=0)
    assert repr(out) == repr(t)
    assert out is not t


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_same_seed_same_result():
    t = three_rules()
    assert (repr(mutate(t, random.Random(42), CFG))
            == repr(mutate(t, random.Random(42), CFG)))


def test_same_seed_same_result_for_a_sequence_of_calls():
    t = three_rules()

    def run():
        rng = random.Random(42)
        return [repr(mutate(t, rng, CFG)) for _ in range(20)]

    assert run() == run()


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


# ─────────────────────────────────────────────────────────────────────────────
# Search-space shape: conditions stay within one level of AND/OR
# ─────────────────────────────────────────────────────────────────────────────

def _condition_depth(cond):
    """Nesting depth of And/Or in a condition; 0 for a bare Compare."""
    from policy_grammar import And, Or, Not
    if isinstance(cond, (And, Or)):
        return 1 + max(_condition_depth(cond.left), _condition_depth(cond.right))
    if isinstance(cond, Not):
        return _condition_depth(cond.operand)
    return 0


def test_conditions_never_exceed_one_level_of_and_or():
    """The search space allows one level of conjunction/disjunction, no more.

    Repeated mutation used to compound: the clause-grow edit fired on Compares
    nested inside an And/Or as well as on top-level ones, so conditions grew
    without bound and eventually overflowed the stack in `copy.deepcopy`.
    """
    from symbolic_regression import mutate, ite_nodes
    rng = random.Random(0)
    tree = IfThenElse(condition=Compare('>', Count('ready'), Number(0)),
                      then=Fire('transporting'), else_=Postpone())
    worst = 0
    for _ in range(3000):                      # far longer than any GA lineage
        tree = mutate(tree, rng, CFG)
        for node in ite_nodes(tree):
            worst = max(worst, _condition_depth(node.condition))
    assert worst <= 1, f"condition nested {worst} levels of AND/OR deep"
