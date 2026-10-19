#!/usr/bin/env python3
"""Does stricter skill-order wording stop H2O-Lightning-4B from skipping secure_grasp?

The next_skill instructions are swapped between the original wording and a stricter one
that says never to skip a skill. Both Jev (hosted) and the local decider get exactly the
same request.

Part 1 (key step): the pick_place state right after lift, where H2O chose transfer
(the live H2O run) and Jev chose secure_grasp (the recorded Jev run); asked REPEAT times.
Part 2 (regression): every recorded Jev request, with one wording (argv[1], default strict).

    DECIDER_URL=http://127.0.0.1:8741 strands_so101_sim/.venv/bin/python compare_prompt_order.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from unittest.mock import patch

import strands_so101_sim.claude_jev_controller as cjc
from strands_so101_sim.claude_jev_controller import JevController
from strands_so101_sim.decider_client import DeciderClient
from compare_decider_vs_jev import SESSIONS, ROOT, gate_verdict, jev_controller, load_env, robot_state

REPEAT = 3
KEY_STATES = (
    ("H2O live run", "outputs/claude_h2o/claude_jev_pick_place_20261019_190542/session.json", "rejected", -1),
    ("Jev recorded run", "outputs/claude_jev_pick_place_20260927_185507/session.json", "decisions", 4),
)


def strict(spec):
    order = " -> ".join(("ready",) + spec.actions)
    return (f"A simulated SO-101 arm must {spec.goal} Choose the next skill. The skills must run one by one "
            f"in exactly this order: {order}. The next skill is always the one that comes immediately after "
            f"last_completed_skill in that order. Never skip a skill, even when the observation suggests its "
            f"purpose is already achieved. Repeating last_completed_skill is not allowed.")


# Wording before 2026-10-19, kept so the comparison stays reproducible after the code change.
ORIGINAL = {
    "secure_grasp": "The cube has just been lifted and is held in the air; tighten the grip before carrying it.",
    "transfer": "The cube is held tightly in the air but not above the green goal; carry it over the goal.",
}


def original(spec):
    order = " -> ".join(("ready",) + spec.actions)
    return (f"A simulated SO-101 arm must {spec.goal} Choose the next skill. Skills run in order "
            f"{order}; repeating last_completed_skill is not allowed.")


# Criteria keyed on last_completed_skill: right after lift the cube already looks "held tightly
# in the air", which matched the transfer criterion as well as secure_grasp.
ANCHORED = {
    "secure_grasp": ("last_completed_skill is lift: the cube has just been raised. Always tighten the grip "
                     "once before carrying it, even if the cube already looks firmly held."),
    "transfer": ("last_completed_skill is secure_grasp: the grip has already been tightened and the cube "
                 "is not yet above the green goal; carry it over the goal."),
}

WORDINGS = dict(
    original=lambda spec: replace(spec, instructions=original(spec), skill_criteria={**spec.skill_criteria, **ORIGINAL}),
    strict=lambda spec: replace(spec, instructions=strict(spec), skill_criteria={**spec.skill_criteria, **ORIGINAL}),
    strict_anchored=lambda spec: replace(spec, instructions=strict(spec),
                                         skill_criteria={**spec.skill_criteria, **ANCHORED}),  # now in the code
)


@dataclass(frozen=True)
class Variant(cjc.TaskSpec):
    instructions: str = ""

    @property
    def skill_instructions(self):
        return self.instructions or super().skill_instructions


def replace(spec, **changes):
    return Variant(**{**{f.name: getattr(spec, f.name) for f in fields(cjc.TaskSpec)}, **changes})


def ask(ctl, rec, wording):
    base = cjc.TASKS[ctl.spec.name]
    ctl.spec = WORDINGS[wording](base)
    try:
        with patch.object(cjc, "build_state", lambda *a, **k: rec["jev_state"]):
            d = ctl.decide(rec["claude_observation"], robot_state(rec["jev_state"]))
    finally:
        ctl.spec = base
    time.sleep(0.2)
    return dict(action=d.action, confidence=round(d.confidence, 3),
                probabilities={k: round(v, 3) for k, v in sorted(d.probabilities.items(), key=lambda kv: -kv[1])},
                gate=round(d.gate["probability"], 3) if d.gate else None,
                verdict=gate_verdict(d), latency_ms=round(d.latency_s * 1000), model=d.model)


def main():
    load_env()
    decider = DeciderClient(os.getenv("DECIDER_URL", "http://127.0.0.1:8741"))
    ctls = dict(jev=jev_controller("pick_place"), h2o=JevController(client=decider, task="pick_place"))
    pp = cjc.TASKS["pick_place"]
    out = dict(wordings={k: dict(instructions=WORDINGS[k](pp).skill_instructions, criteria={a: WORDINGS[k](pp).skill_criteria[a] for a in ANCHORED}) for k in WORDINGS},
               key_step=[], regression=[])

    print("== Part 1: right after lift (expected next skill: secure_grasp)")
    for label, path, field, idx in KEY_STATES:
        rec = json.loads((ROOT / path).read_text())[field][idx]
        assert rec["phase_before"] == "lift", rec["phase_before"]
        for wording in WORDINGS:
            for model, ctl in ctls.items():
                for i in range(REPEAT if model == "jev" else 1):
                    r = ask(ctl, rec, wording)
                    out["key_step"].append(dict(state=label, wording=wording, backend=model, trial=i, **r))
                    p = r["probabilities"]
                    print(f"  {label:16s} {wording:8s} {model:3s} #{i} -> {r['action']:12s} "
                          f"secure_grasp {p.get('secure_grasp', 0):.3f}  transfer {p.get('transfer', 0):.3f}  "
                          f"gate {r['gate']}  ({r['model']})")

    final = sys.argv[1] if len(sys.argv) > 1 else "strict"
    out["regression_wording"] = final
    print(f"\n== Part 2: every recorded request with the {final} wording")
    for name in SESSIONS:
        session = json.loads((ROOT / "outputs" / name / "session.json").read_text())
        task = session.get("task", "cube_lift")
        tctls = dict(jev=jev_controller(task), h2o=JevController(client=decider, task=task))
        for rec in list(session["decisions"]) + [r for r in session.get("rejected", []) if r]:
            row = dict(session=name, task=task, phase=rec["phase_before"], expected=rec["action"])
            for model, ctl in tctls.items():
                r = ask(ctl, rec, final)
                row[model] = dict(action=r["action"], confidence=r["confidence"], gate=r["gate"],
                                  verdict=r["verdict"], model=r["model"])
            out["regression"].append(row)
            print(f"  {task:10s} {row['phase']:15s} expected {row['expected']:15s} "
                  f"jev {row['jev']['action']:15s} {row['jev']['confidence']:.2f} {row['jev']['verdict']:4s} | "
                  f"h2o {row['h2o']['action']:15s} {row['h2o']['confidence']:.2f} {row['h2o']['verdict']}")

    success = [r for r in out["regression"] if r["session"] != SESSIONS[2]]
    out["summary"] = {
        f"{m}_{k}": v for m in ("jev", "h2o") for k, v in dict(
            same_as_recorded=sum(r[m]["action"] == r["expected"] for r in out["regression"]),
            success_steps_passed=sum(r[m]["action"] == r["expected"] and r[m]["verdict"] == "pass" for r in success),
        ).items()}
    out["summary"].update(total=len(out["regression"]), success_steps=len(success))
    print("\n== Summary\n" + json.dumps(out["summary"], indent=1))
    path = ROOT / "outputs" / f"prompt_order_comparison_{final}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
