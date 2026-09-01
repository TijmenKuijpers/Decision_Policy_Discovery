# Decision Policy Discovery in Process Models with Symbolic Regression

Code and data for the paper

> Tijmen Kuijpers, Karolin Winter and Remco Dijkman.
> *Decision Policy Discovery in Process Models with Symbolic Regression.*

**DPDGen** discovers *decision policies* — rules that decide which activity a
process executes next based on the state of **every** running instance, not
just the one at hand — from state-enriched event logs. A decision policy is
expressed in a context-free grammar over Petri-net token aggregates,
represented as an abstract syntax tree, and searched with a genetic algorithm.

This repository holds the four evaluation processes, the state-action datasets
they produce, the grammar, the search, and the results of the reported run.

---

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q          # 51 tests, ~12 s
```

`gympn` is not on PyPI. `requirements.txt` installs it from git; alternatively
clone [bpogroup/gympn](https://github.com/bpogroup/gympn) and point
`GYMPN_PATH` at the clone (see [`gympn_path.py`](gympn_path.py)).

Reproduce the published results:

```bash
python export_datasets.py     # regenerate data/          (~1 min)
python run_n20.py             # 160 GA runs               (~1-2 h)
python report.py              # convergence figure
```

`evaluation_results_n20/` is committed, so `report.py` works without re-running
the search.

---

## The four experiments

Each process is a timed-arc coloured Petri net with a normative decision policy
inspired by an instance-spanning constraint category (Winter et al., 2020).
DPDGen is run against each twice: from a seed policy drawn at random from the
grammar ("no pre-knowledge") and from a seed carrying part of the target
("some pre-knowledge").

| Driver | Cat. | Process | Normative policy | Paper |
|---|---|---|---|---|
| [`ga_batching_conf.py`](ga_batching_conf.py) | I | Transportation, simultaneous execution | `pi_batch` — transport once the batch exceeds 3, or while the truck is already transporting | Fig. 4, Policy 2 |
| [`ga_choice_SL_conf.py`](ga_choice_SL_conf.py) | II-a | Assembly, time-based SLA | `pi_sla` — rescue whichever product's on-time rate has fallen below 0.95 | Fig. 5, Policy 3 |
| [`ga_ratelimit_conf.py`](ga_ratelimit_conf.py) | II-b | Dual manufacturing, number of executions | `pi_rate` — run each line while its realised executions per time unit stay under its cap | Fig. 6, Policy 4 |
| [`ga_priority_conf.py`](ga_priority_conf.py) | II-c | Manufacturing, priority on data | `pi_prio` — do not start on what is ready if something higher-priority is still upstream | Fig. 7, Policy 5 |

Each driver defines only facts about its process — its places, token
attributes, action transitions `T_A`, normative policy, seed policies, and the
rollout count that brings its log to ~500 decision points. Every tuning knob
is shared, in [`ga_shared.py`](ga_shared.py), so the four processes cannot
drift apart. Run one on its own with `python ga_priority_conf.py`.

## Methodology

| File | What it is |
|---|---|
| [`policy_grammar.py`](policy_grammar.py) | The decision policy language `G_pi` (Listing 1) as AST node classes, and their evaluation against a Petri-net state. The `<feature>` terminals of Table 2 — `COUNT`, `SUM`, `MIN`, `MAX`, `MEAN`, `CLOCK` — over `ALL` or `ENABLED` tokens. |
| [`symbolic_regression.py`](symbolic_regression.py) | The genetic algorithm: expression sampling, the five mutation types (COND, ACT, PREPEND, PRUNE, REPLACE), crossover, tournament selection, elitism, the generational loop (Alg. 1–3), and target-driven conformance measurement. Process-agnostic. |
| [`ga_fitness.py`](ga_fitness.py) | The interpretability penalty (Eq. 5–6): the five structural properties in `Xi` — if-then-else depth, numeric-literal count, non-terminal count, expression size, duplicate conditions — weighted and capped at `P_max`. |
| [`ga_shared.py`](ga_shared.py) | Every hyper-parameter shared by the four drivers: `N`, `G`, `p_c`, `p_m`, tournament and elitism sizes, `D_max`, the number pools, and the penalty weights. The values in Section 6.3. (`p_seed` is `Experiment.p_seed_mutation` in `symbolic_regression.py`.) |
| [`policy_listing.py`](policy_listing.py) | Renders a policy tree as an `IF / ELSE IF / ELSE` listing. Used to record the seed, target and discovered policies in the results. |

## Data

[`data/`](data/) holds the four state-action datasets and its own
[README](data/README.md): the decision points each normative policy reaches,
with the action it takes there, as the full Petri-net marking. Regenerate with
`python export_datasets.py`.

Decision counts, matching Section 6.2: 491 (`pi_batch`), 484 (`pi_sla`),
486 (`pi_rate`), 489 (`pi_prio`).

## Running the evaluation

| File | What it does |
|---|---|
| [`evaluation.py`](evaluation.py) | Runs every driver once per (target, seed set) pair, over `GA_SEEDS`. Writes an Excel workbook, `results_<key>.json`, the fitness curves, and one convergence plot per process, into `evaluation_results/`. |
| [`run_n20.py`](run_n20.py) | The reported run: the same pipeline at 20 GA seeds per cell, into `evaluation_results_n20/`. |
| [`report.py`](report.py) | The 2x2 convergence figure (Figure 8), from a results directory. Defaults to `evaluation_results_n20/`; pass another directory as its first argument. |

`evaluation_results_n20/` holds the committed results: `results.json` and one
`results_<key>.json` per process (fitness, conformance, penalty, divergences,
decisions, the discovered policy, and the per-generation curve of every run),
`curves.csv`, `evaluation.xlsx`, and the figures.

### Reading the fitness sign

Internally the primary fitness is the **negated** divergence rate `-div/dec`,
so a policy that never disagrees with the target scores `0` and every fitness
is `<= 0`. The paper states conformance as `1 - div/dec` (Eq. 4), which puts a
perfect policy at `1`. The two differ by exactly 1: a run reported at `0.992`
in the paper appears as `-0.008` in `results.json`.

### Which replications the paper's Table 4 reports

Its fitness, conformance and penalty columns are the best over the first five
GA seeds — `(42, 7, 13, 101, 2024)`, the order `run_n20.py` runs them in —
while its remine-consistency column counts all twenty. Taking the best over
all twenty instead raises two cells (Cat I with pre-knowledge 0.990 → 0.992,
Cat II-b with pre-knowledge 0.777 → 0.805) and leaves the other six unchanged.

## Tests

```bash
python -m pytest tests -q
```

- [`tests/test_ga_fitness.py`](tests/test_ga_fitness.py) — the interpretability penalty terms.
- [`tests/test_mutation.py`](tests/test_mutation.py) — the mutation operator and its five edit types.
- [`tests/test_datasets.py`](tests/test_datasets.py) — the published datasets reproduce the recorded actions under the normative policies, and the decision counts match the paper.

## Notes on the implementation

**`CLOCK` extends the published grammar.** Every `<feature>` terminal in
Listing 1 is a token aggregate, so elapsed time enters a policy only through
the `ENABLED` selector and can never be used as a quantity — which makes rate
policies inexpressible. `pi_rate` (Cat II-b) needs one, so `policy_grammar.py`
adds `CLOCK`. Any policy using it is outside the language exactly as
published; see the `Clock` class for the detail.

**Scoring is target-driven.** The normative policy is simulated once and every
decision point it reaches is recorded; candidates are replayed against that
fixed list and never drive the simulation themselves. So every candidate in a
run faces the same decision points and the same denominator — neither of which
it can influence.

**Reproducibility.** The GA is fully seeded (`random.Random(seed)` plus
`np.random.seed(seed)` in `run_ga`, and each rollout's own
`np.random.default_rng(seed)` in the driver's `make_system`), so re-running
reproduces the committed results exactly rather than approximately. Note that
`run_n20.py` writes into `evaluation_results_n20/`, overwriting what is
committed there — `git diff` after a re-run is the reproduction check.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
