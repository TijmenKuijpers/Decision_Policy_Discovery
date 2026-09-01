"""
run_n20.py
----------

Usage:  python run_n20.py
"""

import time
from pathlib import Path

import evaluation as ev

ORIGINAL_SEEDS = (42, 7, 13, 101, 2024)
NEW_SEEDS      = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                  1100, 1200, 1300, 1400, 1500)
ALL_SEEDS      = ORIGINAL_SEEDS + NEW_SEEDS   # 20 total

OUT_DIR = Path(__file__).resolve().parent / "evaluation_results_n20"


def main():
    assert len(set(ALL_SEEDS)) == 20, "seed collision"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ev.GA_SEEDS = ALL_SEEDS      # run_experiment() reads this as a module global
    ev.VERBOSE_GA = False

    started = time.time()
    specs = {key: ev.SPEC_BUILDERS[key]() for key in ev.EXPERIMENTS}
    all_results = {}
    for key in ev.EXPERIMENTS:
        spec = specs[key]
        print(f"\n=== {spec.title}  ({spec.module.__name__})  "
              f"[{len(ALL_SEEDS)} seeds] ===", flush=True)
        all_results[key] = ev.run_experiment(spec)
        ev.plot_experiment(key, all_results[key], spec.title,
                           OUT_DIR / f"{key}.png")
        print(f"  -> {ev.write_results_json(key, all_results[key], spec, OUT_DIR)}",
              flush=True)

    ev.write_excel(all_results, specs, OUT_DIR / "evaluation.xlsx")
    ev.curves_frame(all_results).to_csv(OUT_DIR / "curves.csv", index=False)

    print(f"\nDone in {(time.time() - started) / 60:.1f} min. "
          f"({len(ALL_SEEDS)} seeds x 8 rows = {len(ALL_SEEDS) * 8} runs)")
    print(f"Output: {OUT_DIR}")
    print(f"Original 5-seed run is untouched: "
          f"{Path(__file__).resolve().parent / 'evaluation_results'}")


if __name__ == "__main__":
    main()
