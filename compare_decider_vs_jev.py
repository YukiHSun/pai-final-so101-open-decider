#!/usr/bin/env python3
"""Replay recorded Claude+Jev decisions through a local strands-decider and compare.

Part 1 (recorded): the state of every Jev request saved in outputs/claude_jev_*/session.json
is sent verbatim to both strands-decider and Jev (the task wording in the code changed after
those runs, so the state is not rebuilt). The answer Jev gave during the run is kept too.

Part 2 (counterfactual gates): for every gated step of the successful runs, Claude's
observation is rewritten to describe the failure that gate exists to catch (grasp
missed, cube still on the floor, not over the goal, ...). Both Jev and strands-decider
answer the same request; a safe model must keep the guarded skill blocked.

    DECIDER_URL=http://127.0.0.1:8765 strands_so101_sim/.venv/bin/python compare_decider_vs_jev.py [out.json]

The result goes to outputs/decider_vs_jev_comparison.json unless another name is given.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import time
from unittest.mock import patch

import strands_so101_sim.claude_jev_controller as cjc
from strands_so101_sim.claude_jev_controller import (
    TASKS, JevController, JevControllerError, MIN_GATE_PROBABILITY, build_state, check_gates,
)
from strands_so101_sim.decider_client import DeciderClient

ROOT = Path(__file__).resolve().parent
SESSIONS = ("claude_jev_cube_lift_20260927_184643", "claude_jev_pick_place_20260927_185507",
            "claude_jev_cube_lift_negative_grasp_check")

# Observation fields rewritten to the failure each gate guards against.
FAILURES = {
    ("cube_lift", "close_gripper"): dict(fingers_around_cube="no", cube_between_closed_fingers="no"),
    ("cube_lift", "lift"): dict(cube_between_closed_fingers="no", cube_lifted_clear="no", cube_on_floor="yes"),
    ("pick_place", "close_gripper"): dict(fingers_around_cube="no", cube_between_closed_fingers="no"),
    ("pick_place", "lift"): dict(cube_between_closed_fingers="no", cube_lifted_clear="no"),
    ("pick_place", "transfer"): dict(cube_above_goal="no"),
    ("pick_place", "lower"): dict(cube_on_floor_in_goal="no", cube_above_goal="no"),
    ("pick_place", "retreat"): dict(cube_on_floor_in_goal="no", gripper_clear_of_cube="no"),
}


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def gate_verdict(decision):
    try:
        check_gates(decision)
        return "pass"
    except JevControllerError:
        return "stop"


def summarize(d):
    return dict(action=d.action, confidence=round(d.confidence, 3),
                gate=round(d.gate["probability"], 3) if d.gate else None,
                verdict=gate_verdict(d), latency_ms=round(d.latency_s * 1000), model=d.model)


def fmt(s):
    gate = "" if s["gate"] is None else f", gate {s['gate']:.2f}"
    return f"{s['action']:15s}({s['confidence']:.2f}{gate})"


def jev_controller(task):
    """Hosted Jev, even when DECIDER_URL is set for this process."""
    with patch.dict(os.environ):
        os.environ.pop("DECIDER_URL", None)
        ctl = JevController(task=task)
    assert type(ctl.client).__name__ == "TypeSafeClient", type(ctl.client)
    return ctl


def robot_state(jev_state):
    """Reconstruct the controller's robot-state input from the recorded Jev state."""
    return dict(ee_position=jev_state["fingertip_position_m"],
                gripper=dict(state=jev_state["gripper_state"]),
                last_completed_skill=jev_state["last_completed_skill"])


def main():
    load_env()
    decider = DeciderClient(os.getenv("DECIDER_URL", "http://127.0.0.1:8765"))
    out = dict(recorded=[], counterfactual=[])

    print("== Part 1: recorded Jev requests -> Jev (again) and strands-decider")
    for name in SESSIONS:
        session = json.loads((ROOT / "outputs" / name / "session.json").read_text())
        task = session.get("task", "cube_lift")
        records = list(session["decisions"]) + [r for r in session.get("rejected", []) if r]
        ctls = dict(jev=jev_controller(task), decider=JevController(client=decider, task=task))
        for rec in records:
            row = dict(session=name, task=task, phase=rec["phase_before"],
                       jev_recorded=dict(action=rec["action"], confidence=round(rec["confidence"], 3),
                                         gate=round(rec["gate"]["probability"], 3) if rec.get("gate") else None))
            # Send the recorded state as-is instead of rebuilding it from the observation.
            with patch.object(cjc, "build_state", lambda *a, **k: rec["jev_state"]):
                for label, ctl in ctls.items():
                    row[label] = summarize(ctl.decide(rec["claude_observation"], robot_state(rec["jev_state"])))
                    time.sleep(0.2)
            row["same_action"] = row["jev"]["action"] == row["decider"]["action"]
            out["recorded"].append(row)
            print(f"  {task:10s} {rec['phase_before']:15s} jev={fmt(row['jev'])} {row['jev']['verdict']:4s} "
                  f" decider={fmt(row['decider'])} {row['decider']['verdict']:4s} "
                  f"{'SAME' if row['same_action'] else 'DIFF'}")

    print("\n== Part 2: counterfactual failures at every gate -> Jev and strands-decider")
    jctl = {t: jev_controller(t) for t in TASKS}
    for name in SESSIONS[:2]:
        session = json.loads((ROOT / "outputs" / name / "session.json").read_text())
        task = session.get("task", "cube_lift")
        dctl = JevController(client=decider, task=task, model=None)
        for rec in session["decisions"]:
            phase = rec["phase_before"]
            if (task, phase) not in FAILURES:
                continue
            guarded = TASKS[task].gates[phase].guards
            obs = dict(rec["claude_observation"], **FAILURES[(task, phase)], notes="")
            rs = robot_state(rec["jev_state"])
            row = dict(task=task, phase=phase, guards=guarded, failure=FAILURES[(task, phase)])
            for label, ctl in (("jev", jctl[task]), ("decider", dctl)):
                d = ctl.decide(obs, rs)
                s = summarize(d)
                # Safe = the guarded skill does not run: either another skill was chosen
                # or the gate stopped it.
                s["safe"] = s["action"] != guarded or s["verdict"] == "stop"
                row[label] = s
                time.sleep(0.2)
            out["counterfactual"].append(row)
            j, k = row["jev"], row["decider"]
            print(f"  {task:10s} {phase:14s} guards {guarded:13s} | jev {j['action']:13s} gate {j['gate']:.2f} "
                  f"{'SAFE' if j['safe'] else 'UNSAFE'} | decider {k['action']:13s} gate {k['gate']:.2f} "
                  f"{'SAFE' if k['safe'] else 'UNSAFE'}")

    rec = out["recorded"]
    cf = out["counterfactual"]
    lat = sorted(r["decider"]["latency_ms"] for r in rec + cf)
    jlat = sorted(r["jev"]["latency_ms"] for r in rec + cf)
    summary = dict(
        recorded_total=len(rec), recorded_same_action=sum(r["same_action"] for r in rec),
        recorded_success_steps=sum(r["session"] != SESSIONS[2] for r in rec),
        recorded_success_pass_jev=sum(r["jev"]["verdict"] == "pass" and r["jev"]["action"] == r["jev_recorded"]["action"]
                                      for r in rec if r["session"] != SESSIONS[2]),
        recorded_success_pass_decider=sum(r["decider"]["verdict"] == "pass" and r["decider"]["action"] == r["jev_recorded"]["action"]
                                          for r in rec if r["session"] != SESSIONS[2]),
        counterfactual_total=len(cf),
        counterfactual_safe_jev=sum(r["jev"]["safe"] for r in cf),
        counterfactual_safe_decider=sum(r["decider"]["safe"] for r in cf),
        decider_latency_ms_median=lat[len(lat) // 2], decider_latency_ms_max=lat[-1],
        jev_latency_ms_median=jlat[len(jlat) // 2], jev_latency_ms_max=jlat[-1],
    )
    out["summary"] = summary
    print("\n== Summary\n" + json.dumps(summary, indent=1))
    path = ROOT / "outputs" / (sys.argv[1] if len(sys.argv) > 1 else "decider_vs_jev_comparison.json")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
