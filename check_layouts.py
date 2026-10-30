#!/usr/bin/env python3
"""Which cube / goal positions can the scripted skills (the "hand") complete at all?

Runs the ten pick-and-place skills in their fixed order, without any AI, for each layout and
records success or the ControlError that stopped it. The simulator is deterministic, so an AI
run that picks the same skills ends exactly like this scripted run.

    strands_so101_sim/.venv/bin/python check_layouts.py [out.json]
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

from strands_so101_sim.pick_place_env import PickPlaceEnv

ROOT = Path(__file__).resolve().parent
# Cube-only and goal-only sweeps around the default (cube 0.22, 0 / goal 0.16, 0), then a few
# layouts that move both. Goals must stay in x 0.14-0.20, |y| <= 0.04 and >= 45 mm from the cube.
LAYOUTS = ([((x, y), (0.16, 0.0)) for x in (0.21, 0.22, 0.23, 0.24) for y in (-0.02, -0.01, 0.0, 0.01, 0.02)]
           + [((0.22, 0.0), (x, y)) for x in (0.15, 0.16, 0.17) for y in (-0.02, -0.01, 0.0, 0.01, 0.02)
              if (x, y) != (0.16, 0.0)]
           + [((0.21, 0.03), (0.16, 0.0)), ((0.24, -0.03), (0.16, 0.0)), ((0.22, 0.0), (0.16, 0.03)),
              ((0.22, 0.0), (0.15, -0.03)), ((0.21, -0.02), (0.17, 0.02)), ((0.22, 0.01), (0.16, -0.02)),
              ((0.21, 0.02), (0.17, -0.01))])


def run(cube, goal):
    env = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            env = PickPlaceEnv(cube, goal)
            for action in PickPlaceEnv.ACTIONS:
                env.active_skill = action
                env.execute(action, env.visual_target() if action == "move_above_cube" else None)
        return dict(ok=True, placement_error_mm=round(1000 * env.metrics()["xy_error"], 2))
    except Exception as exc:  # the stop reason is the result
        return dict(ok=False, stopped_after=getattr(env, "phase", None), error=f"{type(exc).__name__}: {exc}")
    finally:
        if env:
            env.close()


def main():
    rows = []
    for cube, goal in LAYOUTS:
        rows.append(dict(cube_xy=cube, goal_xy=goal, **run(cube, goal)))
        print(json.dumps(rows[-1]), flush=True)
    out = ROOT / "outputs" / (sys.argv[1] if len(sys.argv) > 1 else "layout_feasibility.json")
    out.write_text(json.dumps(dict(feasible=f"{sum(r['ok'] for r in rows)}/{len(rows)}", layouts=rows), indent=1) + "\n")
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
