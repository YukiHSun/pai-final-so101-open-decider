"""Jev skill selector fed by a Claude-written visual observation.

Jev (TypeSafe System One) is text-only, so the RGB frames are read by Claude
(the Claude Code session supervising the run), which writes a short structured
observation. This module validates that observation, sends it together with the
allow-listed robot state to Jev, and applies fixed probability gates.
No MuJoCo dependency and no low-level motor outputs.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

ANSWERS = ("yes", "no", "unclear")
MIN_CHOICE_CONFIDENCE = 0.45
MIN_GATE_PROBABILITY = 0.60


@dataclass(frozen=True)
class Gate:
    name: str
    question: str
    yes: str
    no: str
    guards: str  # the action that may only run when this gate passes


@dataclass(frozen=True)
class TaskSpec:
    name: str
    goal: str
    actions: tuple
    skill_criteria: dict
    observation_fields: dict
    gates: dict  # last_completed_skill -> Gate

    @property
    def skill_instructions(self):
        order = " -> ".join(("ready",) + self.actions)
        return (f"A simulated SO-101 arm must {self.goal} Choose the next skill. The skills must run one by one "
                f"in exactly this order: {order}. The next skill is always the one that comes immediately after "
                f"last_completed_skill in that order. Never skip a skill, even when the observation suggests its "
                f"purpose is already achieved. Repeating last_completed_skill is not allowed.")


CUBE_LIFT = TaskSpec(
    name="cube_lift",
    goal="grip a 3 cm red cube and lift it over 5 cm.",
    actions=("move_above_cube", "descend", "close_gripper", "lift", "done"),
    skill_criteria={
        "move_above_cube": "The gripper has not yet approached the cube; move the open gripper above the visible cube.",
        "descend": "The open gripper is hovering above the cube; lower the open fingers around it.",
        "close_gripper": "The open fingers straddle the cube at its height; close them on the cube.",
        "lift": "The cube is squeezed between the closed fingers while still on the floor; raise it.",
        "done": "The cube is already held clear of the floor after lifting, or the task cannot continue safely.",
    },
    observation_fields={
        "cube_visible": "Is the red cube visible in the images? (yes / no / unclear)",
        "cube_on_floor": "Is the cube resting on the floor? (yes / no / unclear)",
        "gripper_above_cube": "Is the open gripper hovering directly above the cube? (yes / no / unclear)",
        "fingers_around_cube": "Are both fingers straddling the cube at its height? (yes / no / unclear)",
        "cube_between_closed_fingers": "Is the cube squeezed between closed fingers? (yes / no / unclear)",
        "cube_lifted_clear": "Is the cube held clearly above the floor? (yes / no / unclear)",
    },
    gates={
        "close_gripper": Gate("grasp_secure",
                              "Is the cube firmly squeezed between both closed fingers so that lifting is safe?",
                              "Both fingers are closed on the cube.",
                              "The grasp missed, is one-sided, or the cube is not between the fingers.", "lift"),
        "lift": Gate("task_complete",
                     "Is the cube held clear of the floor by the gripper, completing the lift?",
                     "The cube is held in the air between the fingers.",
                     "The cube is still on the floor, dropped, or not visible.", "done"),
    },
)

PICK_PLACE = TaskSpec(
    name="pick_place",
    goal=("pick up a 3 cm red cube, carry it, and set it down inside the green square goal area, "
          "then release it and move the gripper away."),
    actions=("move_above_cube", "descend", "close_gripper", "lift", "secure_grasp",
             "transfer", "lower", "release", "retreat", "done"),
    skill_criteria={
        "move_above_cube": "The gripper has not yet approached the cube; move the open gripper above the visible cube.",
        "descend": "The open gripper is hovering above the cube; lower the open fingers around it.",
        "close_gripper": "The open fingers straddle the cube at its height; close them on the cube.",
        "lift": "The cube is squeezed between the closed fingers while still on the floor; raise it.",
        # Keyed on last_completed_skill: right after lift the cube already looks held tightly, so a
        # purely visual criterion let decision models skip secure_grasp.
        "secure_grasp": ("last_completed_skill is lift: the cube has just been raised. Always tighten the grip "
                         "once before carrying it, even if the cube already looks firmly held."),
        "transfer": ("last_completed_skill is secure_grasp: the grip has already been tightened and the cube "
                     "is not yet above the green goal; carry it over the goal."),
        "lower": "The held cube is above the green goal area; lower it onto the floor.",
        "release": "The held cube rests on the floor inside the green goal; open the fingers.",
        "retreat": "The fingers are open and the cube sits inside the goal; raise the gripper away.",
        "done": "The cube rests alone inside the green goal with the gripper raised away, "
                "or the task cannot continue safely.",
    },
    observation_fields={
        "cube_visible": "Is the red cube visible in the images? (yes / no / unclear)",
        "goal_visible": "Is the green goal square visible in the images? (yes / no / unclear)",
        "gripper_above_cube": "Is the gripper directly above or at the cube? (yes / no / unclear)",
        "fingers_around_cube": "Are both fingers straddling the cube at its height? (yes / no / unclear)",
        "cube_between_closed_fingers": "Is the cube squeezed between closed fingers? (yes / no / unclear)",
        "cube_lifted_clear": "Is the cube held clearly above the floor? (yes / no / unclear)",
        "cube_above_goal": "Is the cube positioned over the green goal square? (yes / no / unclear)",
        "cube_on_floor_in_goal": "Is the cube resting on the floor inside the green goal square? (yes / no / unclear)",
        "fingers_open": "Are the gripper fingers open? (yes / no / unclear)",
        "gripper_clear_of_cube": "Is the gripper raised away, not touching the cube? (yes / no / unclear)",
    },
    gates={
        "close_gripper": CUBE_LIFT.gates["close_gripper"],
        "lift": Gate("cube_held_in_air",
                     "Is the cube held in the air between the closed fingers, clear of the floor?",
                     "The cube is lifted and held between the fingers.",
                     "The cube is still on the floor, dropped, or not between the fingers.", "secure_grasp"),
        "transfer": Gate("cube_over_goal",
                         "Is the held cube now positioned over the green goal square?",
                         "The held cube is above the green goal area.",
                         "The cube is not above the goal, or it is no longer held.", "lower"),
        "lower": Gate("cube_set_in_goal",
                      "Is the cube resting on the floor inside the green goal square, still between the fingers?",
                      "The cube sits on the floor inside the goal area.",
                      "The cube is outside the goal, still in the air, or dropped.", "release"),
        "retreat": Gate("task_complete",
                        "Does the cube rest alone inside the green goal with the gripper raised away?",
                        "The cube is inside the goal and nothing touches it.",
                        "The cube is outside the goal, moving, or the gripper still touches it.", "done"),
    },
)
TASKS = {spec.name: spec for spec in (CUBE_LIFT, PICK_PLACE)}

# Backwards-compatible names used by the Cube Lift tests and README.
OBSERVATION_FIELDS = CUBE_LIFT.observation_fields


class JevControllerError(RuntimeError):
    """A decision was unavailable, invalid, or below a gate; caller must stop."""


@dataclass
class JevDecision:
    action: str
    confidence: float
    probabilities: dict
    gate: dict | None
    model: str
    request_id: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    state: dict = field(repr=False)


def validate_observation(observation, spec=CUBE_LIFT):
    if not isinstance(observation, dict):
        raise JevControllerError("Observation must be a JSON object")
    fields = spec.observation_fields
    unknown = set(observation) - set(fields) - {"notes"}
    missing = set(fields) - set(observation)
    if unknown or missing:
        raise JevControllerError(f"Observation fields mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")
    clean = {}
    for key in fields:
        if observation[key] not in ANSWERS:
            raise JevControllerError(f"{key} must be one of {ANSWERS}")
        clean[key] = observation[key]
    notes = observation.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 500:
        raise JevControllerError("notes must be a string of at most 500 characters")
    clean["notes"] = notes.strip()
    return clean


def public_robot_state(state, spec=CUBE_LIFT):
    """Allow-list copies fields instead of forwarding arbitrary observation dicts."""
    try:
        ee = np.asarray(state["ee_position"], dtype=float)
        gripper = state["gripper"]["state"]
        phase = state["last_completed_skill"]
        if ee.shape != (3,) or not np.isfinite(ee).all():
            raise ValueError("Invalid fingertip position")
        if phase not in ("ready",) + spec.actions or gripper not in ("open", "closed"):
            raise ValueError("Invalid state label")
        return dict(ee_position=ee.tolist(), gripper_state=gripper, last_completed_skill=phase)
    except (KeyError, TypeError, ValueError) as exc:
        raise JevControllerError("Invalid robot state") from exc


def build_state(observation, robot_state, spec=CUBE_LIFT):
    """Jev state: Claude's image reading plus allow-listed proprioception only."""
    state = public_robot_state(robot_state, spec)
    visual = validate_observation(observation, spec)
    return {
        "task": spec.goal[0].upper() + spec.goal[1:],
        "last_completed_skill": state["last_completed_skill"],
        "gripper_state": state["gripper_state"],
        "fingertip_position_m": [round(v, 4) for v in state["ee_position"]],
        "visual_observation": {spec.observation_fields[k]: visual[k] for k in spec.observation_fields},
        "visual_notes": visual["notes"],
    }


class JevController:
    def __init__(self, model=None, client=None, timeout=20.0, task="cube_lift"):
        self.spec = TASKS[task]
        self.model = model or os.getenv("JEV_MODEL", "jev-latest").strip()
        if client is None and os.getenv("DECIDER_URL", "").strip():
            # Local strands-decider instead of the hosted Jev API; same request shape.
            from strands_so101_sim.decider_client import DeciderClient
            client = DeciderClient(os.environ["DECIDER_URL"].strip())
            self.model = model or os.getenv("DECIDER_MODEL", "").strip() or None
        if client is None:
            if not os.getenv("TYPESAFE_API_KEY", "").strip():
                raise JevControllerError("Set TYPESAFE_API_KEY in the environment or the workspace .env")
            from typesafe_sdk import TypeSafeClient
            client = TypeSafeClient(model=self.model, timeout=timeout)
        self.client = client
        self.calls = 0

    def questions(self, phase):
        from typesafe_sdk import Choice, Noul, NoulCriteria
        spec = self.spec
        questions = {"next_skill": Choice(instructions=spec.skill_instructions, criteria=spec.skill_criteria)}
        gate = spec.gates.get(phase)
        if gate:
            questions[gate.name] = Noul(instructions=gate.question, criteria=NoulCriteria(true=gate.yes, false=gate.no))
        return questions

    def decide(self, observation, robot_state) -> JevDecision:
        state = build_state(observation, robot_state, self.spec)
        phase = state["last_completed_skill"]
        self.calls += 1
        started = time.monotonic()
        try:
            response = self.client.system_one(state, self.questions(phase), model=self.model)
        except Exception as exc:
            # Do not log request headers or API keys.
            raise JevControllerError(f"TypeSafe request failed ({type(exc).__name__})") from exc
        latency = time.monotonic() - started
        choice = response.answers["next_skill"]
        action = str(choice.choice)
        if action not in self.spec.actions:
            raise JevControllerError(f"Unknown action from Jev: {action}")
        gate = None
        if phase in self.spec.gates:
            spec_gate = self.spec.gates[phase]
            gate = {"name": spec_gate.name, "guards": spec_gate.guards,
                    "probability": float(response.answers[spec_gate.name].noul)}
        try:
            request_id = response.request_id
        except Exception:
            request_id = None
        return JevDecision(action=action, confidence=float(choice.confidence),
                           probabilities={k: float(v) for k, v in dict(choice.probabilities).items()},
                           gate=gate, model=str(response.model), request_id=request_id,
                           input_tokens=int(response.usage.input_tokens or 0),
                           output_tokens=int(response.usage.output_tokens or 0),
                           latency_s=latency, state=state)


def check_gates(decision: JevDecision):
    """Raise instead of acting when Jev is unsure. Thresholds are fixed, not tuned per run."""
    if decision.confidence < MIN_CHOICE_CONFIDENCE:
        raise JevControllerError(f"Jev skill confidence {decision.confidence:.2f} < {MIN_CHOICE_CONFIDENCE}")
    gate = decision.gate
    if gate and decision.action == gate["guards"] and gate["probability"] < MIN_GATE_PROBABILITY:
        raise JevControllerError(f"Gate {gate['name']} {gate['probability']:.2f} < {MIN_GATE_PROBABILITY}")
