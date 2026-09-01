"""
tests/test_ga_fitness.py
------------------------
Unit tests for the three cosmetic-fitness fixes.

    A  duplicate-condition penalty      (test_duplicate_*)
    B  numeric literals made neutral    (test_numeric_*)
    C  scale-aware cosmetic weighting   (test_scale_*, test_choice_sl_*)

Run with `python -m pytest tests` from the repository root, or directly with
`python tests/test_ga_fitness.py`.  Nothing here starts a simulation, so the
whole file runs in well under a second.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ga_fitness import (
    CosmeticConfig, ScaleConfig,
    NUMERIC_NEUTRAL, NUMERIC_PENALTY,
    compare_key, cosmetic_penalty, cosmetic_scale, duplicate_condition_count,
    primary_spread,
)
from policy_grammar import (
    IfThenElse, And, Compare, Count, Sum, Max, Mean, Clock, Number, Div,
    Fire, Postpone, ALL,
)

# The weights the three rate-primary drivers share.
RATE_COSMETIC = CosmeticConfig(
    enabled=True,
    w_ite_depth=0.002,
    w_non_numeric=0.002,
    w_non_terminal=0.001,
    w_expr_size=0.001,
    w_duplicate_condition=0.005,
    max_cosmetic_penalty=0.05,
)

# Weights two orders of magnitude heavier than RATE_COSMETIC -- what a driver
# scoring an absolute divergence *count* needed, and what `ga_fitness` used to
# ship as the DIVERGENCE_COSMETIC preset.  The scale tests below exist because
# these weights were large enough to outvote conformance outright, so they are
# reproduced here as a fixture rather than dropped with the preset.
BULKY_COSMETIC = CosmeticConfig(
    enabled=True,
    w_ite_depth=0.10,
    w_non_numeric=0.10,
    w_non_terminal=0.05,
    w_expr_size=0.05,
    w_duplicate_condition=0.25,
    max_cosmetic_penalty=3.0,
)


def rate(attribute: str) -> Div:
    """SUM(machine, x) / CLOCK -- the ratelimit driver's rate expression."""
    return Div(Sum('machine', attribute), Clock())


# ─────────────────────────────────────────────────────────────────────────────
# A -- duplicate condition subtrees
# ─────────────────────────────────────────────────────────────────────────────

def _ratelimit_t1_discovered():
    """The T1/S1 tree from evaluation.xlsx: `rate_a < 0.4` in two nested else
    branches, the inner one unreachable."""
    return IfThenElse(
        condition=And(
            Compare('<=', Sum('machine', 'nr_exec_b'), Count('waiting_b', 'ENABLED')),
            Compare('>', Count('completed_a'), Count('machine')),
        ),
        then=Fire('process_b'),
        else_=IfThenElse(
            condition=Compare('<', rate('nr_exec_a'), Number(0.4)),
            then=Fire('process_a'),
            else_=IfThenElse(
                condition=Compare('<', rate('nr_exec_a'), Number(0.4)),
                then=Fire('process_a'),
                else_=Postpone(),
            ),
        ),
    )


def _ratelimit_t1_deduplicated():
    """The same tree with the duplicated branch collapsed to a Postpone()."""
    return IfThenElse(
        condition=And(
            Compare('<=', Sum('machine', 'nr_exec_b'), Count('waiting_b', 'ENABLED')),
            Compare('>', Count('completed_a'), Count('machine')),
        ),
        then=Fire('process_b'),
        else_=IfThenElse(
            condition=Compare('<', rate('nr_exec_a'), Number(0.4)),
            then=Fire('process_a'),
            else_=Postpone(),
        ),
    )


def test_duplicate_condition_counted_once_per_repeat():
    assert duplicate_condition_count(_ratelimit_t1_discovered()) == 1
    assert duplicate_condition_count(_ratelimit_t1_deduplicated()) == 0


def test_duplicate_penalty_exceeds_deduplicated_tree():
    """The whole point of the term: the redundant tree must cost strictly more,
    and by more than the node-count terms alone charged for it."""
    dup   = _ratelimit_t1_discovered()
    clean = _ratelimit_t1_deduplicated()

    no_dup_term = replace(RATE_COSMETIC, w_duplicate_condition=0.0)
    gap_size    = cosmetic_penalty(dup, no_dup_term) - cosmetic_penalty(clean, no_dup_term)
    gap_total   = cosmetic_penalty(dup, RATE_COSMETIC) - cosmetic_penalty(clean, RATE_COSMETIC)

    assert gap_total > gap_size
    assert gap_total == pytest.approx(gap_size + RATE_COSMETIC.w_duplicate_condition)


def test_duplicate_costs_more_than_a_distinct_condition_of_equal_size():
    """The failure mode itself: the size terms price a branch by its node
    count, so repeating a test and introducing a genuinely new one of the same
    shape cost exactly the same.  Only the duplicate term separates them."""
    dup      = _ratelimit_t1_discovered()
    distinct = IfThenElse(                      # same shape, inner cap retuned
        condition=dup.condition,
        then=dup.then,
        else_=IfThenElse(
            condition=Compare('<', rate('nr_exec_a'), Number(0.4)),
            then=Fire('process_a'),
            else_=IfThenElse(
                condition=Compare('<', rate('nr_exec_b'), Number(0.25)),
                then=Fire('process_b'),
                else_=Postpone(),
            ),
        ),
    )

    no_dup_term = replace(RATE_COSMETIC, w_duplicate_condition=0.0)
    assert (cosmetic_penalty(dup, no_dup_term)
            == pytest.approx(cosmetic_penalty(distinct, no_dup_term)))
    assert (cosmetic_penalty(dup, RATE_COSMETIC)
            > cosmetic_penalty(distinct, RATE_COSMETIC))


def test_duplicate_penalty_is_off_when_weight_is_zero():
    cfg = CosmeticConfig(w_duplicate_condition=0.0)
    assert (cosmetic_penalty(_ratelimit_t1_discovered(), cfg)
            > cosmetic_penalty(_ratelimit_t1_deduplicated(), cfg))  # node count only


def test_repeated_condition_inside_one_conjunction_counts():
    """`A AND A` is redundant too, not only repeats across branches."""
    a = Compare('>', Count('ready'), Number(3))
    tree = IfThenElse(And(a, Compare('>', Count('ready'), Number(3))),
                      Fire('t'), Postpone())
    assert duplicate_condition_count(tree) == 1


def test_compare_key_canonicalises_operand_order():
    """priority T1 discovered `MAX(w2) >= MAX(w1)`; the target writes the same
    test as `MAX(w1) <= MAX(w2)`.  One key, not two."""
    a = Compare('<=', Max('waiting_1', 'priority'), Max('waiting_2', 'priority'))
    b = Compare('>=', Max('waiting_2', 'priority'), Max('waiting_1', 'priority'))
    assert compare_key(a) == compare_key(b)


def test_compare_key_separates_genuinely_different_tests():
    a = Compare('<', rate('nr_exec_a'), Number(0.4))
    assert compare_key(a) != compare_key(Compare('<', rate('nr_exec_b'), Number(0.4)))
    assert compare_key(a) != compare_key(Compare('<', rate('nr_exec_a'), Number(0.25)))
    assert compare_key(a) != compare_key(Compare('>', rate('nr_exec_a'), Number(0.4)))


def test_compare_key_ignores_int_float_spelling():
    assert (compare_key(Compare('>', Count('ready'), Number(3)))
            == compare_key(Compare('>', Count('ready'), Number(3.0))))


def test_compare_key_respects_selector():
    assert (compare_key(Compare('>', Count('ready', ALL), Number(0)))
            != compare_key(Compare('>', Count('ready', 'ENABLED'), Number(0))))


# ─────────────────────────────────────────────────────────────────────────────
# B -- numeric literals
# ─────────────────────────────────────────────────────────────────────────────

def _priority_magic_number():
    """priority T2's discovered policy: a bare threshold on one feature."""
    return IfThenElse(Compare('<=', Number(2), Mean('waiting_2', 'priority')),
                      Fire('process'), Postpone())


def _priority_two_features():
    """priority T2's target: the same shape, comparing two features."""
    return IfThenElse(Compare('<=', Mean('waiting_1', 'priority'),
                              Max('waiting_2', 'priority')),
                      Fire('process'), Postpone())


def test_neutral_mode_prices_the_two_forms_alike():
    assert (cosmetic_penalty(_priority_magic_number(), RATE_COSMETIC)
            == pytest.approx(cosmetic_penalty(_priority_two_features(), RATE_COSMETIC)))


def test_penalty_mode_makes_magic_numbers_dearer():
    cfg = CosmeticConfig(numeric_mode=NUMERIC_PENALTY, w_non_numeric=0.01)
    assert (cosmetic_penalty(_priority_magic_number(), cfg)
            > cosmetic_penalty(_priority_two_features(), cfg))


def test_neutral_is_the_default():
    assert CosmeticConfig().numeric_mode == NUMERIC_NEUTRAL


def test_unknown_numeric_mode_is_rejected():
    with pytest.raises(ValueError):
        CosmeticConfig(numeric_mode='subsidise')


# ─────────────────────────────────────────────────────────────────────────────
# C -- scale awareness
# ─────────────────────────────────────────────────────────────────────────────

def test_scale_bounds_cosmetic_to_a_fraction_of_the_spread():
    primaries = [0.0, -0.1, -0.2, -0.4, -0.8]
    cosmetics = [1.0, 0.5, 0.2, 0.1, 0.05]
    cfg = ScaleConfig(budget_frac=0.10)

    scale = cosmetic_scale(primaries, cosmetics, cfg)
    worst = max(c * scale for c in cosmetics)
    assert worst <= cfg.budget_frac * primary_spread(primaries, cfg) + 1e-12


def test_scale_never_amplifies_the_configured_weights():
    """A population with a huge primary spread must not blow the weights up."""
    scale = cosmetic_scale([0.0, -100.0], [0.1, 0.1], ScaleConfig())
    assert scale <= 1.0


def test_scale_ignores_failed_individuals():
    """A tree that crashes the simulation scores None; it must not define the
    spread (and must not silently become the worst cosmetic)."""
    with_error = cosmetic_scale([0.0, -0.2, None], [0.5, 0.1, 9.9], ScaleConfig())
    without    = cosmetic_scale([0.0, -0.2],       [0.5, 0.1],      ScaleConfig())
    assert with_error == pytest.approx(without)


def test_scale_falls_back_to_a_tie_break_when_primary_is_flat():
    """Every primary equal -> the cosmetic term is the only ordering left, so
    any positive scale preserves it; it just must not be zero."""
    scale = cosmetic_scale([0.0, 0.0, 0.0], [1.35, 0.35, 0.10], ScaleConfig())
    assert 0.0 < scale < 1.0
    assert 0.0 - scale * 1.35 < 0.0 - scale * 0.35      # ordering survives


def test_spread_uses_the_selection_relevant_band():
    """The full range is set by individuals selection has already discarded;
    the top-quartile spread is what the run actually discriminates over."""
    primaries = [0.0, -0.05, -0.1, -0.15, -20.0]
    assert primary_spread(primaries, ScaleConfig()) < 1.0
    assert max(primaries) - min(primaries) == 20.0


def test_spread_falls_back_to_one_unit_of_discrimination_when_band_is_flat():
    """A converged top band must NOT fall back to the full range -- that is
    dominated by discarded individuals and would restore full-weight cosmetics
    exactly when they should be a tie-break.  One divergence (0.2 here) is the
    reference instead."""
    primaries = [0.0, 0.0, 0.0, 0.0, -0.2, -0.4, -20.0]
    assert primary_spread(primaries, ScaleConfig()) == pytest.approx(0.2)


def test_converged_choice_sl_population_keeps_cosmetics_a_tie_break():
    """The regression that a first cut of this got wrong: on a population whose
    top quartile has converged on zero divergences, the scale must not
    saturate."""
    target    = _choice_sl_t2_target()
    spurious  = _choice_sl_t2_discovered()
    trees     = [target, spurious] * 6 + [spurious] * 4
    primaries = [0.0] * 12 + [-0.2, -0.4, -1.0, -20.0]
    cosmetics = [cosmetic_penalty(t, BULKY_COSMETIC) for t in trees]

    scale = cosmetic_scale(primaries, cosmetics, ScaleConfig())
    assert scale < 1.0
    # The target's penalty stays under what one divergence is worth (0.2).
    assert scale * cosmetic_penalty(target, BULKY_COSMETIC) < 0.2


# ── the choice_sl T2 regression ──────────────────────────────────────────────

def _sla_below(place, threshold):
    return And(Compare('>', Count(place), Number(0)),
               Compare('<', Div(Sum(place, 'on_time'), Count(place)),
                       Number(threshold)))


def _choice_sl_t2_target():
    """The T2 target: game rescued first at SLA 0.9, phone fall-through."""
    return IfThenElse(
        condition=_sla_below('game_delivered', 0.90),
        then=Fire('game_production'),
        else_=IfThenElse(
            condition=_sla_below('phone_delivered', 0.90),
            then=Fire('phone_production'),
            else_=IfThenElse(
                condition=Compare('>', Count('phone_demand'), Number(0)),
                then=Fire('phone_production'),
                else_=Fire('game_production'),
            ),
        ),
    )


def _choice_sl_t2_discovered():
    """The one-line spurious rule the GA returned for T2."""
    return IfThenElse(
        condition=Compare('!=', Count('game_resource'), Count('phone_demand')),
        then=Fire('game_production'),
        else_=Fire('phone_production'),
    )


def test_choice_sl_t2_target_is_no_longer_dominated_after_rescaling():
    """With the population-relative scale the target's cosmetic cost is a
    fraction of one divergence, so anything that actually diverges loses."""
    target   = _choice_sl_t2_target()
    spurious = _choice_sl_t2_discovered()

    # A plausible late-run population: the target and the spurious rule both at
    # zero divergences, plus individuals that get some decisions wrong.
    trees      = [target, spurious, spurious, target]
    primaries  = [0.0, 0.0, -0.2, -1.0]
    cosmetics  = [cosmetic_penalty(t, BULKY_COSMETIC) for t in trees]
    scale      = cosmetic_scale(primaries, cosmetics, ScaleConfig())

    target_cost = cosmetic_penalty(target, BULKY_COSMETIC) * scale
    # Less than one divergence's worth of primary (0.2 in this fixture) --
    # i.e. a tie-breaker, not an objective.
    assert target_cost < 0.2

    fitness = [p - scale * c for p, c in zip(primaries, cosmetics)]
    assert fitness[0] > fitness[2]     # conforming target beats a diverging rule
    assert fitness[0] > fitness[3]

    # And the residual preference for the simpler tree is now small enough to
    # be a tie-break rather than a verdict.
    assert 0 < fitness[1] - fitness[0] < 0.2


def test_choice_sl_t2_target_fitness_is_now_essentially_its_primary():
    """The headline number: -1.35 was almost entirely cosmetic; it should not
    be any more.  The target is the bulkiest tree in the population, so its
    penalty is exactly the budget -- and the budget is a tenth of the spread."""
    target    = _choice_sl_t2_target()
    primaries = [0.0, 0.0, -0.2, -1.0]
    cosmetics = [cosmetic_penalty(t, BULKY_COSMETIC)
                 for t in (target, _choice_sl_t2_discovered(),
                           _choice_sl_t2_discovered(), target)]
    cfg   = ScaleConfig()
    scale = cosmetic_scale(primaries, cosmetics, cfg)

    target_fitness = 0.0 - scale * cosmetic_penalty(target, BULKY_COSMETIC)
    budget = cfg.budget_frac * primary_spread(primaries, cfg)
    assert abs(target_fitness) <= budget + 1e-12
    # One divergence is worth 0.2 in this fixture.
    assert abs(target_fitness) < 0.2


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
