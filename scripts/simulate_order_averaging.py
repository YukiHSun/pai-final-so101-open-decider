#!/usr/bin/env python3
"""Offline estimate: would averaging over several option orders fix the order sensitivity?

Reads the results of measure_option_order.py (the 34 kit requests, each in its original order,
every cyclic rotation, the reverse and 3 seeded random orders, with the choice probabilities and
the yes/no gate probability of every order) and scores decision rules built from them, without
calling any model again:

- original: one call in the kit's own order;
- single other order: one call in another order (min and max over the orders every request has);
- orig+rev: probabilities averaged over the original and the reversed order (2 calls);
- 5 orders: original, reversed and the 3 random orders (5 calls);
- rotations: all K cyclic rotations (K calls: each option at each position once);
- all: every recorded order (K + 4 calls).

The averaged choice is the argmax of the mean probabilities, its confidence is recomputed with
strands-decider's derive_confidence ((N * p_max - 1) / (N - 1)), the gate's yes probability is
averaged over the same orders, and the kit's check_gates thresholds give pass or stop. Expected
outcomes are those of summarize_option_order.py; "kit wording" leaves out the four
`lift` requests asked with the older `original` and `strict` wordings of compare_prompt_order.py.

    python scripts/simulate_order_averaging.py outputs/elephant1_colab outputs/option_order/*.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_option_order import GUARDS, MIN_CHOICE_CONFIDENCE, MIN_GATE_PROBABILITY  # noqa: E402
from summarize_option_order import references, unique_keys  # noqa: E402

COMMON = ("rot1", "rot2", "rot3", "rot4", "reversed", "random0", "random1", "random2")


def confidence(probs):
    n = len(probs)
    return 1.0 if n <= 1 else max(0.0, min(1.0, (n * max(probs) - 1) / (n - 1)))


def decide(runs):
    """Average the runs (one request in different option orders) into one kit decision."""
    names = list(runs[0]["probs"])
    mean = {n: statistics.mean(r["probs"][n] for r in runs) for n in names}
    choice = max(names, key=lambda n: mean[n])
    if confidence(list(mean.values())) < MIN_CHOICE_CONFIDENCE:
        return choice, "stop"
    for g in runs[0]["gates"]:
        if GUARDS[g] == choice and statistics.mean(r["gates"][g] for r in runs) < MIN_GATE_PROBABILITY:
            return choice, "stop"
    return choice, "pass"


def old_wording_keys(bodies_dir):
    """The kit_order lift requests asked with the `original` and `strict` wordings."""
    order = json.load(open(Path(bodies_dir) / "bodies" / "kit_order.json"))
    return {json.dumps(b, sort_keys=True) for i, b in enumerate(order[:6]) if i % 3 != 2}


def main():
    bodies_dir, paths = sys.argv[1], sys.argv[2:]
    ref, keys, old = references(bodies_dir), unique_keys(bodies_dir), old_wording_keys(bodies_dir)
    out = {}
    for path in paths:
        data = json.load(open(path))
        assert len(data["rows"]) == len(keys)
        result = {}
        for subset, use in (("all requests", lambda k: True), ("kit wording", lambda k: k not in old)):
            tally = {}

            def add(rule, choice, verdict, exp_skill, exp_verdict, guarded):
                t = tally.setdefault(rule, dict(passed=0, steps=0, wrong_skill_passed=0, unsafe=0, failures=0))
                if exp_verdict == "pass":
                    t["steps"] += 1
                    t["passed"] += choice == exp_skill and verdict == "pass"
                    t["wrong_skill_passed"] += choice != exp_skill and verdict == "pass"
                else:
                    t["failures"] += 1
                    t["unsafe"] += verdict == "pass" and choice in guarded

            for row, k in zip(data["rows"], keys):
                if not use(k):
                    continue
                exp_skill, exp_verdict = ref[k]
                guarded = {GUARDS[n] for n in json.loads(k)["questions"] if n in GUARDS}
                by = {r["order"]: r for r in row["runs"]}
                rules = {"original": ["original"], **{f"single {o}": [o] for o in COMMON},
                         "orig+rev": ["original", "reversed"],
                         "5 orders": ["original", "reversed", "random0", "random1", "random2"],
                         "rotations": [o for o in by if o == "original" or o.startswith("rot")],
                         "all": list(by)}
                for rule, labels in rules.items():
                    add(rule, *decide([by[o] for o in labels]), exp_skill, exp_verdict, guarded)
            singles = [tally.pop(f"single {o}") for o in COMMON]
            summary = {r: dict(passed=f"{t['passed']}/{t['steps']}", wrong_skill_passed=t["wrong_skill_passed"],
                               unsafe=f"{t['unsafe']}/{t['failures']}") for r, t in tally.items()}
            summary["single other order"] = dict(
                passed=f"{min(t['passed'] for t in singles)}-{max(t['passed'] for t in singles)}/{singles[0]['steps']}",
                wrong_skill_passed=f"{min(t['wrong_skill_passed'] for t in singles)}-"
                                   f"{max(t['wrong_skill_passed'] for t in singles)}",
                unsafe=f"{max(t['unsafe'] for t in singles)}/{singles[0]['failures']} at most")
            result[subset] = summary
        out[data["model"]] = result
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
