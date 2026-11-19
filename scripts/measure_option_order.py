#!/usr/bin/env python3
"""Measure how much a decision server's skill choice depends on the order of the options.

strands-labs/strands-decider issue #151 points out that a pointer head read under causal
attention may see option k only after options 1..k-1, so the listed order could leak into the
answer. This replays the kit's recorded `next_skill` requests (the same state, instructions and
yes/no gate) with only the order of the choice criteria changed, and reports the metrics the
issue proposes plus the kit's own pass/stop verdict:

- orders: the original, every cyclic rotation (each option sits at each position exactly once),
  the reverse, and a few seeded random shuffles;
- flip rate: share of requests whose chosen skill differs from the original order's in any order;
- TVD: total variation distance to the original order's probabilities, aligned by option name;
- position mass: mean probability at each list position over the rotations. An order-invariant
  model gives exactly 1/K at every position;
- verdict changes: share of requests whose kit verdict (chosen skill, pass or stop under
  check_gates) differs from the original order's in any order.

    B=outputs/elephant1_colab
    DECIDER_URL=http://127.0.0.1:8741 .venv/bin/python scripts/measure_option_order.py \\
        $B/bodies/kit_compare.json $B/bodies/kit_order.json $B/bodies_rest/kit_order_rest.json \\
        --tag h2o-lightning-4b --out outputs/option_order/h2o-lightning-4b.json

With --dump the reordered bodies are only written out (to be answered elsewhere, e.g. on a
GPU), and with --answers the recorded responses are read back instead of calling a server.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strands_so101_sim.claude_jev_controller import (  # noqa: E402
    MIN_CHOICE_CONFIDENCE, MIN_GATE_PROBABILITY, TASKS,
)

GUARDS = {gate.name: gate.guards for spec in TASKS.values() for gate in spec.gates.values()}
N_RANDOM = 3
SEED = 151


def post(url, body, timeout=300):
    req = request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                          headers={"content-type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def expected_skill(body):
    """The skill after last_completed_skill in the order the instructions spell out."""
    text = body["questions"]["next_skill"]["instructions"]
    order = re.search(r"ready(?: -> \w+)+", text).group(0).split(" -> ")
    last = body["state"]["last_completed_skill"]
    return order[order.index(last) + 1] if last in order and order.index(last) + 1 < len(order) else None


def orders(names):
    k = len(names)
    out = [("original", list(names))]
    out += [(f"rot{i}", names[i:] + names[:i]) for i in range(1, k)]
    out.append(("reversed", names[::-1]))
    rng = random.Random(SEED)
    for i in range(N_RANDOM):
        perm = list(names)
        rng.shuffle(perm)
        out.append((f"random{i}", perm))
    return out


def verdict(body, answers):
    choice = answers["next_skill"]
    action, conf = choice["choice"], float(choice.get("confidence") or 0.0)
    if conf < MIN_CHOICE_CONFIDENCE:
        return action, "stop"
    for name, ans in answers.items():
        if name in GUARDS and GUARDS[name] == action and float(ans["noul"]) < MIN_GATE_PROBABILITY:
            return action, "stop"
    return action, "pass"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bodies", nargs="+")
    ap.add_argument("--url", default=os.getenv("DECIDER_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--tag")
    ap.add_argument("--out")
    ap.add_argument("--dump", metavar="BODIES_JSON",
                    help="write every reordered request body to this file and exit (to answer elsewhere)")
    ap.add_argument("--answers", metavar="ANSWERS_JSON",
                    help="read the responses from this file ({rows: [{body, response, ms}]}) instead of a server; "
                         "bodies are matched with their key order kept, since only the order differs")
    args = ap.parse_args()
    if not args.dump and not (args.tag and args.out):
        ap.error("--tag and --out are needed unless --dump is given")

    recorded = None
    if args.answers:
        data = json.load(open(args.answers))
        recorded = {json.dumps(r["body"], ensure_ascii=False): (r["response"], r.get("ms")) for r in data["rows"]}
        args.url = f"recorded: {data.get('model')} on {data.get('device')} ({Path(args.answers).name})"
    dumped = []

    def answer(body):
        if args.dump:
            dumped.append(body)
            return None, 0
        if recorded is not None:
            return recorded[json.dumps(body, ensure_ascii=False)]
        t = time.time()
        resp = post(args.url, body)
        return resp, round((time.time() - t) * 1000)

    bodies, seen = [], set()
    for path in args.bodies:
        for b in json.load(open(path)):
            key = json.dumps(b, sort_keys=True)
            if key not in seen:
                seen.add(key)
                bodies.append(b)

    rows = []
    t_all = time.time()
    for bi, body in enumerate(bodies):
        crit = body["questions"]["next_skill"]["criteria"]
        names = list(crit)
        runs = []
        for label, perm in orders(names):
            b = json.loads(json.dumps(body))
            b["questions"]["next_skill"]["criteria"] = {n: crit[n] for n in perm}
            resp, ms = answer(b)
            if resp is None:
                continue
            ans = resp["answers"]
            probs = {n: float(ans["next_skill"]["probabilities"][n]) for n in names}
            action, v = verdict(b, ans)
            gates = {n: float(a["noul"]) for n, a in ans.items() if n in GUARDS}
            runs.append(dict(order=label, perm=perm, probs=probs, gates=gates, choice=action, verdict=v,
                             confidence=ans["next_skill"].get("confidence"), ms=ms))
        if not runs:
            continue
        base = runs[0]
        for r in runs:
            r["tvd"] = 0.5 * sum(abs(r["probs"][n] - base["probs"][n]) for n in names)
        rows.append(dict(index=bi, task=body["state"]["task"][:40], last=body["state"]["last_completed_skill"],
                         k=len(names), expected=expected_skill(body), runs=runs))
        flips = sum(r["choice"] != base["choice"] for r in runs)
        print(f"[{bi + 1}/{len(bodies)}] k={len(names)} last={rows[-1]['last']:<15} orig={base['choice']:<15} "
              f"flips={flips}/{len(runs) - 1} maxTVD={max(r['tvd'] for r in runs):.3f}", flush=True)

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump).write_text(json.dumps(dumped, ensure_ascii=False, indent=1))
        print(f"{len(dumped)} bodies from {len(bodies)} requests -> {args.dump}")
        return

    def summary(sel):
        sel = list(sel)
        others = [r for row in sel for r in row["runs"][1:]]
        pos = {}
        for row in sel:
            k = row["k"]
            rots = [r for r in row["runs"] if r["order"] == "original" or r["order"].startswith("rot")]
            pos.setdefault(k, {"first": [], "last": [], "n": 0})
            pos[k]["first"].append(statistics.mean(r["probs"][r["perm"][0]] for r in rots))
            pos[k]["last"].append(statistics.mean(r["probs"][r["perm"][-1]] for r in rots))
            pos[k]["n"] += 1
        return dict(
            requests=len(sel),
            calls=sum(len(row["runs"]) for row in sel),
            flip_rate=sum(any(r["choice"] != row["runs"][0]["choice"] for r in row["runs"]) for row in sel) / len(sel),
            flip_rate_per_order=sum(r["choice"] != row["runs"][0]["choice"] for row in sel for r in row["runs"][1:]) / len(others),
            verdict_change_rate=sum(any((r["choice"], r["verdict"]) != (row["runs"][0]["choice"], row["runs"][0]["verdict"])
                                        for r in row["runs"]) for row in sel) / len(sel),
            tvd_mean=statistics.mean(r["tvd"] for r in others),
            tvd_max=max(r["tvd"] for r in others),
            correct_original=sum(row["runs"][0]["choice"] == row["expected"] for row in sel) / len(sel),
            correct_all_orders=sum(r["choice"] == row["expected"] for row in sel for r in row["runs"]) / sum(len(row["runs"]) for row in sel),
            position_mass={str(k): dict(n=v["n"], uniform=round(1 / k, 4), first=round(statistics.mean(v["first"]), 4),
                                        last=round(statistics.mean(v["last"]), 4)) for k, v in sorted(pos.items())},
            median_ms=statistics.median(r["ms"] for row in sel for r in row["runs"]),
        )

    out = dict(model=args.tag, url=args.url, issue="strands-labs/strands-decider#151", seed=SEED,
               orders="original, all cyclic rotations, reversed, 3 seeded random",
               thresholds=dict(choice_confidence=MIN_CHOICE_CONFIDENCE, gate=MIN_GATE_PROBABILITY),
               summary=summary(rows), seconds=round(time.time() - t_all), rows=rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
