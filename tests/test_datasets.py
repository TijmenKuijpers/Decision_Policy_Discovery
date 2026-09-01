"""
tests/test_datasets.py
----------------------
The published `data/` is what makes the results reproducible, so these tests
check that replaying each process's normative policy over the published states
reproduces the published actions exactly, using only `policy_grammar` and the
files -- no simulator.  That is the conformance denominator of Section 6.2
recomputed from the artefact, so a reader can score any candidate policy
against the same decision points the GA saw.

They also pin the decision counts reported in the paper (491 / 484 / 486 /
489), which is what would break first if a driver's process parameters or
rollout count changed without the datasets being regenerated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from export_datasets import PROCESSES, load, load_state_action  # noqa: E402

DATA = ROOT / "data"

# dec(pi) as reported in Section 6.2 of the paper.
REPORTED_DECISIONS = {
    "batching":  491,
    "choice_sl": 484,
    "ratelimit": 486,
    "priority":  489,
}

KEYS = list(PROCESSES)


@pytest.fixture(scope="module")
def manifest():
    path = DATA / "datasets.json"
    if not path.exists():
        pytest.skip("data/ not exported -- run `python export_datasets.py`")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read(key):
    path = DATA / f"{key}_state_action.jsonl"
    if not path.exists():
        pytest.skip("data/ not exported -- run `python export_datasets.py`")
    return load_state_action(path)


@pytest.mark.parametrize("key", KEYS)
def test_decision_count_matches_paper(key, manifest):
    states, actions = read(key)
    assert len(states) == len(actions) == REPORTED_DECISIONS[key]
    assert manifest[key]["decision_points"] == REPORTED_DECISIONS[key]


@pytest.mark.parametrize("key", KEYS)
def test_normative_policy_reproduces_recorded_actions(key):
    """Every published action is what the normative policy does on that state.

    This is the dataset's reason to exist: divergence is measured against these
    actions, so a state on which the target itself would now decide differently
    would silently shift every conformance score computed from the file.
    """
    module = load(key)
    states, actions = read(key)

    replayed = [module.REFERENCE_POLICY.evaluate(state) for state in states]
    mismatches = [i for i, (a, b) in enumerate(zip(replayed, actions)) if a != b]

    assert not mismatches, (
        f"{key}: normative policy disagrees with the recorded action at "
        f"{len(mismatches)} of {len(actions)} decision points "
        f"(first: index {mismatches[0]})"
    )


@pytest.mark.parametrize("key", KEYS)
def test_actions_are_declared_transitions_or_postpone(key, manifest):
    """No action outside T_A u {postpone} was recorded."""
    allowed = set(manifest[key]["action_transitions"]) | {"postpone"}
    _, actions = read(key)
    assert set(actions) <= allowed
    assert sum(manifest[key]["action_distribution"].values()) == len(actions)
