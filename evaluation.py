"""
evaluation.py
-------------
Batch experiment runner over the `ga_*_conf.py` drivers.

For every driver it runs the GA once per (target policy, seed set) pair and
collects, per run:

    target policy | seed policy | fitness | fitness primary | fitness cosmetic
                  | discovered policy

Those six columns are written to one Excel sheet per driver, and the
best-fitness-per-generation curve of every run is drawn into one plot per
driver.

    fitness = fitness primary - fitness cosmetic

(the cosmetic column is the *penalty*, always >= 0, exactly as
`ga_fitness.fitness_breakdown` reports it).

Everything a re-run would want to change lives in the RUN SETTINGS block below,
and the (target, seed) variants live in the `_spec_*` builders further down --
one builder per driver, each importing only its own module.  Nothing else in
the file needs editing to change an experiment.

Usage
-----
    python evaluation.py                 # every driver
    python evaluation.py batching        # a subset, by key
    python evaluation.py --list          # keys and the rows they would run
    python evaluation.py --plot-only     # re-plot from a previous curves.csv

Outputs land in OUT_DIR:
    evaluation.xlsx         one sheet per driver + settings + diagnostics
    curves.csv              long-format fitness curves (re-plottable)
    <driver>.png            best fitness over generations, one line per row
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from policy_grammar import (
    IfThenElse, And, Or, Compare,
    Count, Sum, Max, Mean, Clock, Number, Div,
    Fire, Postpone,
)
from ga_fitness import cosmetic_penalty
import symbolic_regression as sr


# ═════════════════════════════════════════════════════════════════════════════
# RUN SETTINGS -- change these, then re-run
# ═════════════════════════════════════════════════════════════════════════════

# Random seed handed to `sr.run_ga`.  Each row is seeded identically, so rows
# differ only in their target/seed policies.  Put several values here to repeat
# every row under different GA seeds (the table then carries one row per repeat).
GA_SEEDS = (42,)

# None = use the value the driver module declares.  Set an int to override it
# for every driver -- handy for a quick smoke run (e.g. POP_SIZE = 20,
# N_GENERATIONS = 10) before committing to the full sweep.
POP_SIZE      = None
N_GENERATIONS = None
N_ROLLOUTS    = None     # rollouts per fitness evaluation
HORIZON       = None     # simulation length per rollout

# How targets and seed sets are combined into rows:
#   'zip'   -> target[i] with seed_set[i]        (2 rows per driver)
#   'cross' -> every target with every seed set  (4 rows per driver)
PAIRING = 'zip'

# Which drivers to run.  Keys are defined in SPEC_BUILDERS at the bottom.
EXPERIMENTS = ('batching', 'ratelimit', 'priority', 'choice_sl')

# Print each driver's own per-generation GA log (very verbose).
VERBOSE_GA = False

OUT_DIR   = Path(__file__).resolve().parent / "evaluation_results"
XLSX_NAME = "evaluation.xlsx"

# ── plot styling (data-viz reference palette, light surface) ─────────────────
SERIES_COLORS  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE        = "#fcfcfb"
TEXT_PRIMARY   = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR     = "#e4e3df"
FIG_SIZE       = (9.0, 5.6)
FIG_DPI        = 200


def _style_axes(ax, title: str) -> None:
    """Shared chart chrome: recessive grid and axes, integer generations."""
    from matplotlib.ticker import MaxNLocator

    ax.set_title(f"{title}\nbest fitness over generations",
                 color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("generation", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("best fitness so far", color=TEXT_SECONDARY, fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.margins(x=0.06)


def _legend_below(ax, handles=None, labels=None) -> None:
    """Legend under the axes -- the curves rise into every in-axes corner."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc="upper left",
                       bbox_to_anchor=(0.0, -0.14), fontsize=8, frameon=False,
                       borderaxespad=0.0, handlelength=2.4, labelspacing=0.5)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)


# ═════════════════════════════════════════════════════════════════════════════
# Experiment description
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Variant:
    """One named policy tree -- either a target or a single seed."""
    short: str          # plot label, e.g. 'T1'
    label: str          # human description
    tree: object


@dataclass(frozen=True)
class SeedSet:
    """The seed policies a GA run starts from.

    A *set* rather than a single tree because these experiments are built
    around crossover assembling a reference from complementary seeds -- see the
    driver docstrings.  All of them are reported in the 'seed policy' column.
    """
    short: str
    label: str
    trees: Sequence[object]


@dataclass
class ExperimentSpec:
    key: str
    module: object
    title: str                      # plot title / sheet caption
    targets: Sequence[Variant]
    seed_sets: Sequence[SeedSet]
    # counts(module, tree, reference) -> (divergences, decisions) | None
    counts: Callable
    # primary(module, counts) -> float | None
    primary: Callable

    @property
    def sheet(self) -> str:
        return self.module.__name__[:31]


@dataclass
class RowResult:
    key: str
    target: Variant
    seed_set: SeedSet
    ga_seed: int
    fitness: float
    primary: float
    cosmetic: float
    discovered: object
    curve: list[float] = field(default_factory=list)   # best-so-far per generation
    target_fitness: float = float('nan')
    seed_fitness: float = float('nan')
    divergences: int | None = None
    decisions: int | None = None
    seconds: float = 0.0

    @property
    def row_id(self) -> str:
        base = f"{self.target.short}/{self.seed_set.short}"
        return base if len(GA_SEEDS) == 1 else f"{base}#{self.ga_seed}"


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def _apply_overrides(module) -> None:
    """Push the RUN SETTINGS overrides onto a driver module.

    The drivers read N_ROLLOUTS / HORIZON as module globals at call time, so
    setting them here changes what `module.conformance_counts` measures.
    """
    for name, value in (("N_ROLLOUTS", N_ROLLOUTS), ("HORIZON", HORIZON)):
        if value is not None:
            setattr(module, name, value)


def _rows(spec: ExperimentSpec) -> list[tuple[Variant, SeedSet]]:
    if PAIRING == 'cross':
        return [(t, s) for t in spec.targets for s in spec.seed_sets]
    if len(spec.targets) != len(spec.seed_sets):
        raise ValueError(
            f"{spec.key}: PAIRING='zip' needs as many targets as seed sets "
            f"({len(spec.targets)} vs {len(spec.seed_sets)}); use PAIRING='cross'."
        )
    return list(zip(spec.targets, spec.seed_sets))


def run_row(spec: ExperimentSpec, target: Variant, seed_set: SeedSet,
            ga_seed: int) -> RowResult:
    """Run one GA against *target*, started from *seed_set*."""
    mod = spec.module

    def counts_fn(tree):
        return spec.counts(mod, tree, target.tree)

    def evaluate_primary(tree):
        return spec.primary(mod, counts_fn(tree))

    evaluate = sr.make_evaluator(evaluate_primary, mod.COSMETIC_CONFIG)

    experiment = sr.Experiment(
        name=f"{spec.title} | {target.short} {target.label} | "
             f"{seed_set.short} {seed_set.label}",
        reference_label=target.label,
        config=mod.CONFIG,
        evaluate=evaluate,
        evaluate_primary=evaluate_primary,
        cosmetic_config=mod.COSMETIC_CONFIG,
        seed_policies=list(seed_set.trees),
        seed_policy=seed_set.trees[0],
        pop_size=POP_SIZE or mod.POP_SIZE,
        n_generations=N_GENERATIONS or mod.N_GENERATIONS,
        tournament_k=mod.TOURNAMENT_K,
        p_crossover=mod.P_CROSSOVER,
        elite_k=mod.ELITE_K,
        reference_policy=target.tree,
        conformance_counts=counts_fn,
    )

    curve: list[float] = []
    t0 = time.time()
    best_tree, best_fit = sr.run_ga(
        experiment, seed=ga_seed, verbose=VERBOSE_GA,
        on_generation=lambda rec: curve.append(rec['best_fit']),
    )
    elapsed = time.time() - t0

    counts   = counts_fn(best_tree)
    primary  = evaluate_primary(best_tree)
    cosmetic = cosmetic_penalty(best_tree, mod.COSMETIC_CONFIG)

    return RowResult(
        key=spec.key, target=target, seed_set=seed_set, ga_seed=ga_seed,
        fitness=best_fit,
        primary=float('nan') if primary is None else primary,
        cosmetic=cosmetic,
        discovered=best_tree,
        curve=curve,
        target_fitness=evaluate(target.tree),
        seed_fitness=max(evaluate(s) for s in seed_set.trees),
        divergences=None if counts is None else counts[0],
        decisions=None if counts is None else counts[1],
        seconds=elapsed,
    )


def run_experiment(spec: ExperimentSpec) -> list[RowResult]:
    _apply_overrides(spec.module)
    results = []
    for target, seed_set in _rows(spec):
        for ga_seed in GA_SEEDS:
            print(f"  [{spec.key}] {target.short} {target.label}  <-  "
                  f"{seed_set.short} {seed_set.label}  (ga_seed={ga_seed}) ...",
                  flush=True)
            res = run_row(spec, target, seed_set, ga_seed)
            rate = ("n/a" if not res.decisions
                    else f"{res.divergences / res.decisions:.3f}")
            print(f"      fitness={res.fitness:.4f}  "
                  f"(primary={res.primary:.4f}, cosmetic=-{res.cosmetic:.4f})  "
                  f"divergence rate={rate}  [{res.seconds:.0f} s]", flush=True)
            print(f"      {repr(res.discovered)}", flush=True)
            results.append(res)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════════════════════

def _cell(short: str, label: str, trees) -> str:
    """A policy cell: id + description, then the policy itself, one per line."""
    if not isinstance(trees, (list, tuple)):
        trees = [trees]
    body = "\n".join(repr(t) for t in trees)
    return f"{short} - {label}\n{body}"


def table_for(results: Sequence[RowResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "target policy":    _cell(r.target.short, r.target.label, r.target.tree),
        "seed policy":      _cell(r.seed_set.short, r.seed_set.label,
                                  list(r.seed_set.trees)),
        "fitness":          r.fitness,
        "fitness primary":  r.primary,
        "fitness cosmetic": r.cosmetic,
        "discovered policy": repr(r.discovered),
    } for r in results])


def diagnostics_for(results: Sequence[RowResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "experiment":        r.key,
        "row":               r.row_id,
        "ga seed":           r.ga_seed,
        "target fitness":    r.target_fitness,
        "best seed fitness": r.seed_fitness,
        "best fitness":      r.fitness,
        "divergences":       r.divergences,
        "decisions":         r.decisions,
        "divergence rate":   (None if not r.decisions
                              else r.divergences / r.decisions),
        "generations":       max(len(r.curve) - 1, 0),
        "runtime (s)":       round(r.seconds, 1),
    } for r in results])


def curves_frame(all_results: dict[str, list[RowResult]]) -> pd.DataFrame:
    rows = []
    for key, results in all_results.items():
        for r in results:
            for gen, fit in enumerate(r.curve):
                rows.append({
                    "experiment": key,
                    "row": r.row_id,
                    "target": r.target.label,
                    "seeds": r.seed_set.label,
                    "generation": gen,
                    "best fitness": fit,
                })
    return pd.DataFrame(rows)


def settings_frame() -> pd.DataFrame:
    return pd.DataFrame([
        ("run at",        datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("ga seeds",      ", ".join(str(s) for s in GA_SEEDS)),
        ("pairing",       PAIRING),
        ("experiments",   ", ".join(EXPERIMENTS)),
        ("pop size",      "driver default" if POP_SIZE is None else POP_SIZE),
        ("generations",   "driver default" if N_GENERATIONS is None else N_GENERATIONS),
        ("rollouts",      "driver default" if N_ROLLOUTS is None else N_ROLLOUTS),
        ("horizon",       "driver default" if HORIZON is None else HORIZON),
        ("fitness",       "fitness = fitness primary - fitness cosmetic"),
    ], columns=["setting", "value"])


def write_excel(all_results: dict[str, list[RowResult]],
                specs: dict[str, ExperimentSpec], path: Path) -> None:
    from openpyxl.styles import Alignment, Font

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        settings_frame().to_excel(writer, sheet_name="settings", index=False)
        for key, results in all_results.items():
            table_for(results).to_excel(writer, sheet_name=specs[key].sheet,
                                        index=False)
        diagnostics = pd.concat([diagnostics_for(r) for r in all_results.values()],
                                ignore_index=True)
        diagnostics.to_excel(writer, sheet_name="diagnostics", index=False)

        widths = {"target policy": 60, "seed policy": 60, "discovered policy": 70,
                  "fitness": 12, "fitness primary": 15, "fitness cosmetic": 16,
                  "setting": 16, "value": 46}
        for sheet in writer.book.worksheets:
            headers = [c.value for c in sheet[1]]
            for cell in sheet[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(vertical="top")
            for idx, header in enumerate(headers, start=1):
                letter = sheet.cell(row=1, column=idx).column_letter
                sheet.column_dimensions[letter].width = widths.get(header, 18)
                for row in range(2, sheet.max_row + 1):
                    c = sheet.cell(row=row, column=idx)
                    c.alignment = Alignment(vertical="top", wrap_text=True)
                    if isinstance(c.value, float):
                        c.number_format = "0.0000"
            sheet.freeze_panes = "A2"


def plot_experiment(key: str, results: Sequence[RowResult], title: str,
                    path: Path) -> None:
    """Best fitness over generations, one line per row of the table."""
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, r in enumerate(results):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        gens  = range(len(r.curve))
        ax.plot(gens, r.curve, color=color, linewidth=2.0, solid_capstyle="round",
                label=f"{r.row_id}  target: {r.target.label}  |  "
                      f"seeds: {r.seed_set.label}")
        # What the target policy itself scores: zero divergences, minus its own
        # cosmetic penalty.  Not strictly a ceiling -- a smaller tree that is
        # behaviourally equivalent scores *above* it, which is the whole point
        # of the cosmetic term.
        ax.axhline(r.target_fitness, color=color, linewidth=1.0,
                   linestyle=(0, (2, 3)), alpha=0.5, zorder=1)
        if r.curve:
            ax.annotate(r.row_id, xy=(len(r.curve) - 1, r.curve[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        color=TEXT_SECONDARY, fontsize=8,
                        va="center", ha="left")

    _style_axes(ax, title)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=TEXT_SECONDARY, linewidth=1.0,
                              linestyle=(0, (2, 3)), alpha=0.6))
    labels.append("fitness of the target policy itself (zero divergences)")
    _legend_below(ax, handles, labels)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_from_curves(curves: pd.DataFrame, specs: dict[str, ExperimentSpec],
                     out_dir: Path) -> None:
    """Re-draw the plots from a saved curves.csv (no GA run)."""
    for key, group in curves.groupby("experiment", sort=False):
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        for i, (row_id, run) in enumerate(group.groupby("row", sort=False)):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            ax.plot(run["generation"], run["best fitness"], color=color,
                    linewidth=2.0,
                    label=f"{row_id}  target: {run['target'].iloc[0]}  |  "
                          f"seeds: {run['seeds'].iloc[0]}")
        _style_axes(ax, specs[key].title if key in specs else key)
        _legend_below(ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", facecolor=SURFACE,
                    bbox_inches="tight")
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Experiment definitions -- one builder per driver
#
# Each builder names TWO target policies (T1 is always the driver's own
# REFERENCE_POLICY) and TWO seed sets (S1 is always the driver's own
# SEED_POLICIES).  Under PAIRING='zip' that is the two rows T1/S1 and T2/S2.
# ═════════════════════════════════════════════════════════════════════════════

def _spec_batching() -> ExperimentSpec:
    import ga_batching_conf as mod

    # Same shape as the driver's pi_batch, a larger batch trigger: the truck
    # waits for 6 staged cases instead of 4 before departing.
    ALT_BATCH_SIZE = 5
    target_alt = IfThenElse(
        condition=Or(
            Compare('>', Count('ready'), Number(ALT_BATCH_SIZE)),
            Compare('=', Max('truck', 'status'),
                    Number(mod.STATUS_TRANSPORTING)),
        ),
        then=Fire('transporting'),
        else_=Postpone(),
    )

    # Deliberately uninformative seeds: neither carries the status-continuation
    # branch, and one watches the wrong place entirely.
    seed_wrong_place = IfThenElse(
        condition=Compare('>', Count('waiting'), Number(2)),
        then=Fire('transporting'),
        else_=Postpone(),
    )

    return ExperimentSpec(
        key="batching",
        module=mod,
        title="Batching (pi_batch)",
        targets=[
            Variant("T1", f"original pi_batch (batch > {mod.BATCH_SIZE})",
                    mod.REFERENCE_POLICY),
            Variant("T2", f"larger batch (batch > {ALT_BATCH_SIZE})", target_alt),
        ],
        seed_sets=[
            SeedSet("S1", "original A+B+C", mod.SEED_POLICIES),
            SeedSet("S2", "wrong-place + no-batching",
                    [seed_wrong_place, mod.SEED_C]),
        ],
        # The driver hard-codes seed_base=1000 inside its own conformance_counts
        # and takes no reference argument, so the call is spelled out here.
        counts=lambda m, tree, ref: sr.conformance_counts(
            tree, ref,
            make_system=lambda s: m.make_batching_system(seed=s),
            n_rollouts=m.N_ROLLOUTS, horizon=m.HORIZON, seed_base=1000,
        ),
        primary=lambda m, counts: sr.primary_rate(counts),
    )


def _spec_ratelimit() -> ExperimentSpec:
    import ga_ratelimit_conf as mod

    def rate(attribute):
        """SUM(machine, x) / CLOCK -- executions per time unit so far."""
        return Div(Sum('machine', attribute), Clock())

    # Priority reversed and both caps retuned: line B is checked first, so the
    # discovered policy has to get the branch *order* right as well as the caps.
    ALT_CAP_A, ALT_CAP_B = 0.3, 0.4
    target_alt = IfThenElse(
        condition=Compare('<', rate('nr_exec_b'), Number(ALT_CAP_B)),
        then=Fire('process_b'),
        else_=IfThenElse(
            condition=Compare('<', rate('nr_exec_a'), Number(ALT_CAP_A)),
            then=Fire('process_a'),
            else_=Postpone(),
        ),
    )

    # Right shape on line B, wrong cap -- and no line-A branch at all.
    seed_b_wrong_cap = IfThenElse(
        condition=Compare('<', rate('nr_exec_b'), Number(0.75)),
        then=Fire('process_b'),
        else_=Postpone(),
    )

    return ExperimentSpec(
        key="ratelimit",
        module=mod,
        title="Rate limiting (pi_rate)",
        targets=[
            Variant("T1", f"original pi_rate (A first, caps "
                          f"{mod.MAX_PER_TIME_A}/{mod.MAX_PER_TIME_B})",
                    mod.REFERENCE_POLICY),
            Variant("T2", f"B first, caps {ALT_CAP_B}/{ALT_CAP_A}", target_alt),
        ],
        seed_sets=[
            SeedSet("S1", "original A+B+C", mod.SEED_POLICIES),
            SeedSet("S2", "B-branch wrong cap + static priority",
                    [seed_b_wrong_cap, mod.SEED_C]),
        ],
        counts=lambda m, tree, ref: m.conformance_counts(tree, ref),
        primary=lambda m, counts: sr.primary_rate(counts),
    )


def _spec_priority() -> ExperimentSpec:
    import ga_priority_conf as mod

    # Look-ahead on the *average* pending priority rather than the best one:
    # a softer rule that fires more often than pi_prio.
    target_alt = IfThenElse(
        condition=Compare('<=',
                          Mean('waiting_1', 'priority'),
                          Max('waiting_2', 'priority')),
        then=Fire('process'),
        else_=Postpone(),
    )

    return ExperimentSpec(
        key="priority",
        module=mod,
        title="Priority look-ahead (pi_prio)",
        targets=[
            Variant("T1", "original pi_prio (MAX vs MAX)", mod.REFERENCE_POLICY),
            Variant("T2", "MEAN upstream vs MAX ready", target_alt),
        ],
        seed_sets=[
            SeedSet("S1", "original A+B+C", mod.SEED_POLICIES),
            # No priority aggregate anywhere in the seeds: the GA has to make
            # the COUNT -> aggregate jump on its own (see p_family_escape).
            SeedSet("S2", "queue lengths + always fire",
                    [mod.SEED_B, mod.SEED_C]),
        ],
        counts=lambda m, tree, ref: m.conformance_counts(tree, ref),
        primary=lambda m, counts: sr.primary_rate(counts),
    )


def _spec_choice_sl() -> ExperimentSpec:
    import ga_choice_SL_conf as mod

    def sla_below(place, threshold):
        """SLA(place) < threshold, guarded against an empty place."""
        return And(
            Compare('>', Count(place), Number(0)),
            Compare('<', Div(Sum(place, 'on_time'), Count(place)),
                    Number(threshold)),
        )

    # Game rescued first, at a looser SLA target, and the fall-through favours
    # phones -- every branch of the original is inverted somewhere.
    ALT_SLA = 0.90
    target_alt = IfThenElse(
        condition=sla_below('game_delivered', ALT_SLA),
        then=Fire('game_production'),
        else_=IfThenElse(
            condition=sla_below('phone_delivered', ALT_SLA),
            then=Fire('phone_production'),
            else_=IfThenElse(
                condition=Compare('>', Count('phone_demand'), Number(0)),
                then=Fire('phone_production'),
                else_=Fire('game_production'),
            ),
        ),
    )

    # Seeds for T2.  They deliberately differ from it in the *fall-through*
    # rather than in an SLA branch: under this workload no delivery is ever
    # late, so SLA(...) is pinned at 1.0 and every SLA branch is dead code --
    # a seed that differed only there would already score zero divergences and
    # leave the GA nothing to search for.  (Root cause: ga_choice_SL_conf
    # deep-copies one prototype net per rollout, but the behaviour closures
    # still read the *prototype's* clock, which never advances -- so
    # `on_time = clock <= request_date` is always True.  Row T1/S1 is left
    # as-is precisely so that shows up in the results.)
    seed_always_phone = Fire('phone_production')

    return ExperimentSpec(
        key="choice_sl",
        module=mod,
        title="SLA choice (pi_sla)",
        targets=[
            Variant("T1", "original pi_sla (phone first, SLA 0.95)",
                    mod.REFERENCE_POLICY),
            Variant("T2", f"game first, SLA {ALT_SLA}, phone fall-through",
                    target_alt),
        ],
        seed_sets=[
            SeedSet("S1", "original A+B+C", mod.SEED_POLICIES),
            SeedSet("S2", "always phone + trivial demand-driven",
                    [seed_always_phone, mod.SEED_C]),
        ],
        counts=lambda m, tree, ref: m.conformance_counts(tree, ref),
        # This driver's cosmetic weights are tuned against an absolute-count
        # primary -- see evaluate_primary in ga_choice_SL_conf.  Do not swap in
        # primary_rate here without rescaling COSMETIC_CONFIG with it.
        primary=lambda m, counts: sr.primary_count(counts, m.N_ROLLOUTS),
    )


SPEC_BUILDERS = {
    "batching":  _spec_batching,
    "ratelimit": _spec_ratelimit,
    "priority":  _spec_priority,
    "choice_sl": _spec_choice_sl,
}


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main(argv: Sequence[str]) -> int:
    keys = [a for a in argv if not a.startswith("-")] or list(EXPERIMENTS)
    unknown = [k for k in keys if k not in SPEC_BUILDERS]
    if unknown:
        print(f"unknown experiment(s): {', '.join(unknown)}\n"
              f"available: {', '.join(SPEC_BUILDERS)}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "--list" in argv:
        for key in keys:
            spec = SPEC_BUILDERS[key]()
            print(f"\n{key}  ({spec.module.__name__})")
            for target, seed_set in _rows(spec):
                print(f"  {target.short}/{seed_set.short}  "
                      f"target: {target.label}  |  seeds: {seed_set.label}")
        return 0

    specs = {key: SPEC_BUILDERS[key]() for key in keys}

    if "--plot-only" in argv:
        curves = pd.read_csv(OUT_DIR / "curves.csv")
        plot_from_curves(curves[curves["experiment"].isin(keys)], specs, OUT_DIR)
        print(f"plots rewritten in {OUT_DIR}")
        return 0

    started = time.time()
    all_results: dict[str, list[RowResult]] = {}
    for key in keys:
        spec = specs[key]
        print(f"\n=== {spec.title}  ({spec.module.__name__}) ===", flush=True)
        all_results[key] = run_experiment(spec)
        plot_experiment(key, all_results[key], spec.title, OUT_DIR / f"{key}.png")

    xlsx = OUT_DIR / XLSX_NAME
    write_excel(all_results, specs, xlsx)
    curves_frame(all_results).to_csv(OUT_DIR / "curves.csv", index=False)

    print(f"\nDone in {(time.time() - started) / 60:.1f} min.")
    print(f"  {xlsx}")
    print(f"  {OUT_DIR / 'curves.csv'}")
    for key in keys:
        print(f"  {OUT_DIR / (key + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
