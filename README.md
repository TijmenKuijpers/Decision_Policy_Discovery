# Decision Policy Discovery in Process Models with Symbolic Regression

Code and data for the paper

> Tijmen Kuijpers, Karolin Winter and Remco Dijkman.
> *Decision Policy Discovery in Process Models with Symbolic Regression.*

**DPDGen** discovers *decision policies*, rules that decide which activity a
process executes next based on the state of **every** running instance from state-enriched event logs. 
A decision policy is expressed in a context-free grammar over Petri-net token aggregates,
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

## Tests

```bash
python -m pytest tests -q
```

- [`tests/test_ga_fitness.py`](tests/test_ga_fitness.py) — the interpretability penalty terms.
- [`tests/test_mutation.py`](tests/test_mutation.py) — the mutation operator and its five edit types.
- [`tests/test_datasets.py`](tests/test_datasets.py) — the published datasets reproduce the recorded actions under the normative policies, and the decision counts match the paper.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
