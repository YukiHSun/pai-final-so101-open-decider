#!/usr/bin/env python3
"""Step-wise SO-101 Cube Lift / Pick and Place driven by Claude (vision) + Jev (skill decisions).

Claude is the Claude Code session itself, not an API. Each decision is one
process invocation so Claude can read the saved RGB frame in between:

  start  -> create a session (--task cube_lift|pick_place) and save the first image
  step   -> Claude passes its observation JSON; Jev picks the skill; it executes
  video  -> re-render the whole session to MP4 (no Jev calls)

MuJoCo and the IK restarts (fixed RNG seed) are deterministic, so every call
replays the already-executed skills from scratch and verifies the fingertip
trajectory matches the recorded one before continuing.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPLAY_TOLERANCE_M = 1e-6


def load_session(path):
    return json.loads((path / "session.json").read_text())


def save_session(path, session):
    (path / "session.json").write_text(json.dumps(session, indent=2, ensure_ascii=False) + "\n")


def quiet(run_dir):
    """Controller prints include cube truth; keep them out of Claude's view."""
    return contextlib.redirect_stdout(open(run_dir / "ctrl.log", "a"))


def task_of(session):
    return session.get("task", "cube_lift")


def build_env(session, frame_callback=None):
    from strands_so101_sim.cube_lift_env import CubeLiftEnv, ControlError
    import numpy as np
    if task_of(session) == "pick_place":
        from strands_so101_sim.pick_place_env import PickPlaceEnv
        env = PickPlaceEnv(tuple(session["cube_xy"]), tuple(session["place_xy"]), frame_callback)
    else:
        env = CubeLiftEnv(tuple(session["cube_xy"]), frame_callback)
    for i, record in enumerate(session["decisions"]):
        env.active_skill = record["action"]
        target = env.visual_target() if record["action"] == "move_above_cube" else None
        env.execute(record["action"], target)
        drift = float(np.linalg.norm(env.ee_position() - np.asarray(record["ee_after"])))
        if drift > REPLAY_TOLERANCE_M:
            env.close()
            raise ControlError(f"Replay diverged at decision {i} ({drift:.2e} m)")
    return env


def save_observation(run_dir, env, label):
    from PIL import Image
    obs = env.observe()
    Image.fromarray(obs["rgb"]).save(run_dir / f"{label}.png")
    # Debug metrics are stored locally, never sent to Claude or Jev.
    (run_dir / f"{label}.json").write_text(json.dumps({"robot_state": obs["robot_state"],
        "debug": env.metrics(), "simulation_step": env.steps}, indent=2) + "\n")
    return obs


def cmd_start(args):
    if not (0.18 <= args.cube_x <= 0.26 and abs(args.cube_y) <= 0.04):
        raise SystemExit("Cube is outside the calibrated workspace")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_dir / f"claude_jev_{args.task}_{stamp}").resolve()
    run_dir.mkdir(parents=True)
    session = dict(task=args.task, cube_xy=[args.cube_x, args.cube_y], decisions=[], status="running",
                   error=None, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if args.task == "pick_place":
        session["place_xy"] = [args.place_x, args.place_y]
    with quiet(run_dir):
        env = build_env(session)
    try:
        save_observation(run_dir, env, "00_ready")
    finally:
        env.close()
    save_session(run_dir, session)
    print(f"[SESSION] {run_dir}")
    print(f"[IMAGE] {run_dir / '00_ready.png'}  (left: oblique, right: top-down)")
    return 0


def cmd_step(args):
    from strands_so101_sim.claude_jev_controller import JevController, check_gates
    run_dir = args.session.resolve()
    session = load_session(run_dir)
    if session["status"] != "running":
        raise SystemExit(f"Session is {session['status']}")
    observation = json.loads(args.observation)
    controller = JevController(task=task_of(session))
    with quiet(run_dir):
        env = build_env(session)
    record = None
    try:
        obs = env.observe()
        decision = controller.decide(observation, obs["robot_state"])
        record = {k: v for k, v in asdict(decision).items() if k != "state"}
        record.update(claude_observation=observation, jev_state=decision.state,
                      phase_before=env.phase, simulation_step_before=env.steps)
        print(f"[JEV] action={decision.action} confidence={decision.confidence:.3f} "
              f"gate={decision.gate} latency={decision.latency_s:.2f}s model={decision.model}")
        print(f"[JEV] probabilities={json.dumps(decision.probabilities)}")
        check_gates(decision)
        target = None
        if decision.action == "move_above_cube":
            target = env.visual_target()
        env.active_skill = decision.action
        with quiet(run_dir):
            env.execute(decision.action, target)
        record["ee_after"] = env.ee_position().tolist()
        session["decisions"].append(record)
        n = len(session["decisions"])
        save_observation(run_dir, env, f"{n:02d}_{env.phase}")
        if decision.action == "done":
            # env.execute("done") raises unless the task's physical success check passed.
            session["status"] = "success"
            m = env.metrics()
            if task_of(session) == "pick_place":
                print(f"[SUCCESS] pick and place; lift={100*env.lift_record['cube_rise']:.2f} cm; "
                      f"placement error={1000*m['xy_error']:.2f} mm; stable={m['stable_steps']} steps")
            else:
                print(f"[SUCCESS] cube lifted {100*m['cube_rise']:.2f} cm; bilateral hold={m['hold_steps']} steps")
        else:
            print(f"[IMAGE] {run_dir / f'{n:02d}_{env.phase}.png'}")
    except Exception as exc:
        session["status"] = "stopped"
        session["error"] = str(exc)
        if record is not None:
            record["rejected"] = True
            session.setdefault("rejected", []).append(record)
        print(f"[STOP] {exc}", file=sys.stderr)
    finally:
        session["metrics"] = env.metrics()
        if getattr(env, "lift_record", None):
            session["lift"] = env.lift_record
        env.close()
        save_session(run_dir, session)
    return 0 if session["status"] in ("running", "success") else 1


class DecisionPanel:
    """Bottom strip that shows, for the skill being executed, what the eye saw and what the head chose."""

    FIELD_JA = {"cube_visible": "キューブが見える", "goal_visible": "ゴールが見える", "cube_on_floor": "床にある",
                "gripper_above_cube": "真上にいる", "fingers_around_cube": "指がはさむ位置",
                "cube_between_closed_fingers": "閉じた指にはさまれている", "cube_lifted_clear": "浮いている",
                "cube_above_goal": "ゴールの上", "cube_on_floor_in_goal": "ゴールの中に置かれた",
                "fingers_open": "指が開いている", "gripper_clear_of_cube": "グリッパーが離れた"}
    SKILL_JA = {"move_above_cube": "キューブの上へ", "descend": "下ろす", "close_gripper": "つかむ", "lift": "持ち上げる",
                "secure_grasp": "持ち直す", "transfer": "運ぶ", "lower": "置く", "release": "離す", "retreat": "離れる",
                "done": "完了"}
    HEIGHT = 160  # 480 + 160 keeps the height a multiple of 16, so the encoder does not rescale
    LINE = 34

    def __init__(self, session, label):
        from PIL import ImageFont
        self.label = label
        self.total = len(session["decisions"])
        # Each skill appears once per run, so the active skill identifies the decision being executed.
        self.by_action = {d["action"]: (i, d) for i, d in enumerate(session["decisions"])}
        font_path = Path("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc")
        self.font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default()

    def skill(self, action):
        return f"{self.SKILL_JA.get(action, action)}（{action}）"

    def lines(self, active_skill, width=1260):
        """(text, colour) rows; the first one is the eye's answer and may wrap onto a second row."""
        if active_skill not in self.by_action:
            return [("待機中（最初の判断の前）", (255, 214, 102))]
        i, d = self.by_action[active_skill]
        seen = [self.FIELD_JA.get(k, k) for k, v in d["claude_observation"].items() if k != "notes" and v == "yes"]
        ranked = sorted(d["probabilities"].items(), key=lambda kv: -kv[1])
        runner_up = f"（次点 {self.SKILL_JA.get(ranked[1][0], ranked[1][0])} {ranked[1][1]:.2f}）" if len(ranked) > 1 else ""
        gate = d.get("gate")
        if gate:
            ok = gate["probability"] >= 0.60
            gate_text = f"安全確認：{gate['probability']:.2f}（0.60 以上で実行）→ {'通過' if ok else '停止'}"
        else:
            gate_text = "安全確認：このスキルの前にはなし"
        eye = f"{i + 1}/{self.total} 手目　目（Claude）が「はい」と答えた項目：{'、'.join(seen)}"
        return ([(row, (255, 214, 102)) for row in self.wrap(eye, 2, width)]
                + [(f"頭（{self.label}）が選んだスキル：{self.skill(d['action'])}　確信度 {d['confidence']:.2f}{runner_up}", "white"),
                   (gate_text, "white")])

    def wrap(self, text, max_rows, width):
        rows, row = [], ""
        for ch in text:
            if self.font.getlength(row + ch) > width and row:
                rows.append(row)
                row = "　" + ch
            else:
                row += ch
        rows.append(row)
        return rows[:max_rows]

    def attach(self, frame, active_skill):
        from PIL import Image, ImageDraw
        out = Image.new("RGB", (frame.width, frame.height + self.HEIGHT), (20, 24, 36))
        out.paste(frame, (0, 0))
        draw = ImageDraw.Draw(out)
        for n, (text, colour) in enumerate(self.lines(active_skill, frame.width - 20)):
            draw.text((10, frame.height + 6 + self.LINE * n), text, font=self.font, fill=colour)
        return out


def cmd_video(args):
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    import numpy as np
    from strands_so101_sim.so101_pick_ik import grab_frame
    run_dir = args.session.resolve()
    session = load_session(run_dir)
    task = task_of(session)
    video_path = run_dir / (f"claude_jev_{task}_decisions.mp4" if args.show_decisions else f"claude_jev_{task}.mp4")
    writer = imageio.get_writer(video_path, fps=25, codec="libx264", quality=7)
    panel = DecisionPanel(session, args.label) if args.show_decisions else None

    def record_frame(e):
        if task == "pick_place":
            frame = Image.fromarray(np.concatenate([grab_frame(e.sim, "oblique"), grab_frame(e.sim, "topdown")], axis=1))
            second = f"cube z {e.cube_position()[2]*100:.2f} cm | grasp {e.contacts()} | placement stable {e.stable_steps} steps"
        else:
            frame = Image.fromarray(grab_frame(e.sim, "oblique"))
            second = f"rise {100*(e.cube_position()[2]-e.initial_z):.2f} cm | contact {e.contacts()} | hold {e.hold_steps}"
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 37), fill="black")
        draw.text((8, 3), f"{args.label} | {getattr(e, 'active_skill', 'open_gripper')} | step {e.steps}", fill="white")
        draw.text((8, 20), second, fill="white")
        if panel:
            frame = panel.attach(frame, getattr(e, "active_skill", "open_gripper"))
        writer.append_data(np.asarray(frame))

    try:
        with quiet(run_dir):
            env = build_env(session, record_frame)
        env.close()
    finally:
        writer.close()
    print(f"[VIDEO] {video_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start")
    p.add_argument("--task", choices=("cube_lift", "pick_place"), default="cube_lift")
    p.add_argument("--cube-x", type=float, default=0.22)
    p.add_argument("--cube-y", type=float, default=0.0)
    p.add_argument("--place-x", type=float, default=0.16)
    p.add_argument("--place-y", type=float, default=0.0)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p = sub.add_parser("step")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--observation", required=True, help="Claude's observation JSON")
    p = sub.add_parser("video")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--label", default="CLAUDE+JEV", help="overlay label for the decision backend")
    p.add_argument("--show-decisions", action="store_true",
                   help="add a bottom strip with the observation, chosen skill, confidence and gate per step")
    args = parser.parse_args(argv)
    if sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "glfw")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    return dict(start=cmd_start, step=cmd_step, video=cmd_video)[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
