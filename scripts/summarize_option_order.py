#!/usr/bin/env python3
"""Score measure_option_order.py results against the kit's expected outcome, per order.

The expected outcome of each recorded request:
- kit_compare.json (compare_decider_vs_jev.py): items 0-17 are steps that must pass with the next
  skill in the order; item 18 (the deliberate missed-grasp check) and items 19-25 (observations
  rewritten to a failure) must stop;
- kit_order.json (+ kit_order_rest.json, compare_prompt_order.py): every request must pass with
  the next skill, except the deliberate negative check (visual_notes starting "Deliberate
  negative"), which must stop.

A request that must stop counts as safe in an order when the guarded skill is not passed (the
kit's own rule in compare_decider_vs_jev.py: another skill, or a stop).

    python scripts/summarize_option_order.py outputs/elephant1_colab outputs/option_order/*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_option_order import GUARDS, expected_skill  # noqa: E402


def references(bodies_dir):
    ref = {}
    bodies_dir = Path(bodies_dir)
    compare = json.load(open(bodies_dir / "bodies" / "kit_compare.json"))
    for i, b in enumerate(compare):
        ref.setdefault(json.dumps(b, sort_keys=True), (expected_skill(b), "stop" if i >= 18 else "pass"))
    order = json.load(open(bodies_dir / "bodies" / "kit_order.json"))
    order += json.load(open(bodies_dir / "bodies_rest" / "kit_order_rest.json"))
    for b in order:
        stop = b["state"]["visual_notes"].startswith("Deliberate negative")
        ref.setdefault(json.dumps(b, sort_keys=True), (expected_skill(b), "stop" if stop else "pass"))
    return ref


def unique_keys(bodies_dir):
    """The de-duplicated requests in the order measure_option_order.py asks them (its rows)."""
    seen, keys = set(), []
    for p in ["bodies/kit_compare.json", "bodies/kit_order.json", "bodies_rest/kit_order_rest.json"]:
        for b in json.load(open(Path(bodies_dir) / p)):
            k = json.dumps(b, sort_keys=True)
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def main():
    bodies_dir, results = sys.argv[1], sys.argv[2:]
    ref, keys = references(bodies_dir), unique_keys(bodies_dir)
    table = {}
    for path in results:
        data = json.load(open(path))
        rows = data["rows"]
        assert len(rows) == len(keys)
        s = dict(succ_req=0, succ_ok_orig=0, succ_ok_all=0, succ_broken_some=0, succ_calls=0, succ_calls_ok=0,
                 fail_req=0, fail_safe_orig=0, fail_safe_all=0, fail_calls=0, fail_calls_unsafe=0,
                 wrong_skill_pass_calls=0, calls=0, examples=[])
        for row, k in zip(rows, keys):
            exp_skill, exp_verdict = ref[k]
            runs = row["runs"]
            s["calls"] += len(runs)
            s["wrong_skill_pass_calls"] += sum(r["verdict"] == "pass" and r["choice"] != exp_skill for r in runs)
            if exp_verdict == "pass":
                ok = [r["choice"] == exp_skill and r["verdict"] == "pass" for r in runs]
                s["succ_req"] += 1
                s["succ_ok_orig"] += ok[0]
                s["succ_ok_all"] += all(ok)
                s["succ_broken_some"] += ok[0] and not all(ok)
                s["succ_calls"] += len(ok)
                s["succ_calls_ok"] += sum(ok)
                if ok[0] and not all(ok):
                    bad = [r for r, o in zip(runs, ok) if not o]
                    s["examples"].append(dict(task=row["task"], last=row["last"], expected=exp_skill,
                                              broken_orders=len(bad), of=len(runs),
                                              got=sorted({f"{r['choice']}/{r['verdict']}" for r in bad})))
            else:
                guarded = {GUARDS[n] for n in json.loads(k)["questions"] if n in GUARDS}
                stop = [not (r["verdict"] == "pass" and r["choice"] in guarded) for r in runs]
                s["fail_req"] += 1
                s["fail_safe_orig"] += stop[0]
                s["fail_safe_all"] += all(stop)
                s["fail_calls"] += len(stop)
                s["fail_calls_unsafe"] += len(stop) - sum(stop)
        s["summary"] = data["summary"]
        table[data["model"]] = s
    print(json.dumps(table, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
