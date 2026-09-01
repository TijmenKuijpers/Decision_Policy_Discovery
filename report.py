"""
report.py
---------
Draw the convergence figure: best fitness so far against generation, one panel
per process, one line per replication.

Reads `results_<key>.json` from a results directory (written by evaluation.py
or run_n20.py) and writes `convergence.png` into that same directory.  The
default directory is the 20-replication run.

Usage:  python report.py [results-dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = "evaluation_results_n20"

OUT_DIR = Path(__file__).resolve().parent / DEFAULT_RESULTS_DIR

SURFACE        = "#fcfcfb"
TEXT_PRIMARY   = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR     = "#e4e3df"
C_NONE         = "#eb6834"   # no pre-knowledge
C_SOME         = "#2a78d6"   # some pre-knowledge
C_TARGET       = "#52514e"

PANEL_TITLE = {
    "batching":  "(a) Cat I - transportation, simultaneous execution",
    "choice_sl": "(b) Cat II-a - assembly, time-based SLA",
    "ratelimit": "(c) Cat II-b - dual manufacturing, execution rate",
    "priority":  "(d) Cat II-c - manufacturing, priority on data",
}
ORDER = ["batching", "choice_sl", "ratelimit", "priority"]


def load() -> dict:
    """Merge every `results_<key>.json` in the output directory."""
    merged: dict = {"runs": {}}
    paths = sorted(OUT_DIR.glob("results_*.json"))
    whole = OUT_DIR / "results.json"
    if whole.exists():
        paths.append(whole)
    if not paths:
        raise SystemExit(f"no results_*.json in {OUT_DIR}; run evaluation.py first")
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            merged["runs"].update(json.load(fh)["runs"])
    return merged


def convergence(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    for ax, key in zip(axes.flat, ORDER):
        ax.set_facecolor(SURFACE)
        rows = data["runs"].get(key, [])
        for r in rows:
            color = C_NONE if r["seed_short"] == "S0" else C_SOME
            ax.plot(range(len(r["curve"])), r["curve"], color=color,
                    linewidth=2.0, solid_capstyle="round", label=r["seed_label"])
        if rows:
            # What the target policy itself scores: zero divergences, less its
            # own interpretability penalty.  Not a hard ceiling -- a smaller
            # behaviourally-equivalent tree scores above it.
            ax.axhline(rows[0]["target_fitness"], color=C_TARGET, linewidth=1.0,
                       linestyle=(0, (2, 3)), alpha=0.7)

        ax.set_title(PANEL_TITLE[key], color=TEXT_PRIMARY, fontsize=10,
                     loc="left", pad=8)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.set_xlabel("generation", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel("best fitness so far", color=TEXT_SECONDARY, fontsize=9)

    handles = [
        plt.Line2D([], [], color=C_NONE, linewidth=2.0),
        plt.Line2D([], [], color=C_SOME, linewidth=2.0),
        plt.Line2D([], [], color=C_TARGET, linewidth=1.0, linestyle=(0, (2, 3))),
    ]
    labels = ["seed with no pre-knowledge (random)",
              "seed with some pre-knowledge",
              "fitness of the target policy itself"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv=()) -> int:
    global OUT_DIR
    if argv:
        OUT_DIR = Path(argv[0]).resolve()
        if not OUT_DIR.is_dir():
            print(f"no such results directory: {OUT_DIR}")
            return 2

    convergence(load(), OUT_DIR / "convergence.png")
    print(f"wrote {OUT_DIR / 'convergence.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
