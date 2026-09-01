"""
export_datasets.py
------------------
Write the state-action datasets DPDGen is evaluated on to `data/`.

Usage
-----
    python export_datasets.py            # all four processes
    python export_datasets.py priority   # a subset, by key
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from policy_listing import listing

OUT_DIR = Path(__file__).resolve().parent / "data"

PROCESSES = {
    "batching":  ("Cat I",    "Transportation, simultaneous execution", "pi_batch"),
    "choice_sl": ("Cat II-a", "Assembly, time-based SLA",               "pi_sla"),
    "ratelimit": ("Cat II-b", "Dual manufacturing, number of executions", "pi_rate"),
    "priority":  ("Cat II-c", "Manufacturing, priority on data",        "pi_prio"),
}

MODULES = {
    "batching":  "ga_batching_conf",
    "choice_sl": "ga_choice_SL_conf",
    "ratelimit": "ga_ratelimit_conf",
    "priority":  "ga_priority_conf",
}


def load(key: str):
    """The driver module for *key*."""
    import importlib
    return importlib.import_module(MODULES[key])


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────

def jsonable(value):
    """Token attribute values as JSON, with anything exotic kept as its repr."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return repr(value)


def state_json(state):
    """One RecordedState as a plain dict: clock plus every place's marking."""
    return {
        "clock": state.clock,
        "places": {
            place._id: [{"time": token.time, "value": jsonable(token.value)}
                        for token in place.marking]
            for place in state.places
        },
    }


def rollout_index(states):
    """Which simulation replication each decision point came from.

    `record_trace` concatenates R independent rollouts into one trace and does
    not tag them.  Within a rollout the clock never runs backwards and each
    rollout restarts it, so a drop in the clock is a rollout boundary.
    """
    indices = []
    current = 0
    previous = None
    for state in states:
        if previous is not None and state.clock < previous:
            current += 1
        indices.append(current)
        previous = state.clock
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Reading back
# ─────────────────────────────────────────────────────────────────────────────

class LoadedToken:
    __slots__ = ("time", "value")

    def __init__(self, record):
        self.time = record["time"]
        self.value = record["value"]


class LoadedPlace:
    __slots__ = ("_id", "marking")

    def __init__(self, place_id, tokens):
        self._id = place_id
        self.marking = [LoadedToken(t) for t in tokens]


class LoadedState:
    """A published decision point, back in the shape a policy tree evaluates.

    `policy_grammar` reads a state through `.clock` and `.places`, with each
    place exposing `._id` and a `.marking` of tokens carrying `.time` and
    `.value` -- and nothing else.  Reproducing that much is enough to replay
    any policy on the published data without the simulator.
    """

    __slots__ = ("clock", "places")

    def __init__(self, record):
        self.clock = record["clock"]
        self.places = [LoadedPlace(pid, tokens)
                       for pid, tokens in record["places"].items()]


def load_state_action(path):
    """Read a published `<key>_state_action.jsonl` into (states, actions).

    The states are evaluatable: `tree.evaluate(state)` returns what that policy
    would have decided at that decision point, which is how a reader reproduces
    a conformance score from the data alone.
    """
    states, actions = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            states.append(LoadedState(record))
            actions.append(record["action"])
    return states, actions


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export(key: str) -> dict:
    module = load(key)
    category, description, pi = PROCESSES[key]

    reference = module.REFERENCE_POLICY
    n_rollouts, horizon = module.N_ROLLOUTS, module.HORIZON

    trace = module._TRACES.trace(reference, n_rollouts, horizon)
    states, actions = trace.states, trace.actions
    rollouts = rollout_index(states)

    observed_rollouts = rollouts[-1] + 1
    if observed_rollouts != n_rollouts:
        print(f"  note: {observed_rollouts} of {n_rollouts} rollouts reached a "
              f"decision point", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{key}_state_action.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for i, (state, action) in enumerate(zip(states, actions)):
            record = {"decision": i, "rollout": rollouts[i]}
            record.update(state_json(state))
            record["action"] = action
            fh.write(json.dumps(record, sort_keys=False) + "\n")

    distribution = {}
    for action in actions:
        distribution[action] = distribution.get(action, 0) + 1

    print(f"  {key:<10} {len(states):>4} decision points  ->  {path.name}")

    return {
        "category": category,
        "process": description,
        "normative_policy": pi,
        "normative_policy_listing": listing(reference),
        "n_rollouts": n_rollouts,
        "rollouts_with_decisions": observed_rollouts,
        "horizon": horizon,
        "trace_seed_base": module._TRACES.seed_base,
        "decision_points": len(states),
        "action_transitions": list(module.TRANSITIONS),
        "action_distribution": distribution,
        "file": path.name,
    }


def main(argv) -> int:
    keys = [a for a in argv if not a.startswith("-")] or list(PROCESSES)
    unknown = [k for k in keys if k not in PROCESSES]
    if unknown:
        print(f"unknown process(es): {', '.join(unknown)}\n"
              f"available: {', '.join(PROCESSES)}")
        return 2

    print(f"writing state-action datasets to {OUT_DIR}")
    manifest = {key: export(key) for key in keys}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "datasets.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"  manifest   -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
