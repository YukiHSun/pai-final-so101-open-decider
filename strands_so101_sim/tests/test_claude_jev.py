"""Claude+Jev controller contract tests; no TypeSafe API access."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from strands_so101_sim.claude_jev_controller import (
    JevController, JevControllerError, OBSERVATION_FIELDS, PICK_PLACE, build_state, check_gates,
)


def state(phase="ready", gripper="open"):
    return dict(joint_positions=[0.0]*6, ee_position=[0.2, 0.0, 0.1],
                gripper=dict(position_rad=0.45, command_rad=0.45, state=gripper),
                last_completed_skill=phase)


def observation(**overrides):
    value = {k: "no" for k in OBSERVATION_FIELDS}
    value.update(cube_visible="yes", cube_on_floor="yes", notes="test")
    value.update(overrides)
    return value


def response(choice, confidence=1.0, gate=None):
    answers = {"next_skill": SimpleNamespace(choice=choice, confidence=confidence, probabilities={choice: confidence})}
    if gate:
        answers[gate[0]] = SimpleNamespace(noul=gate[1])
    return SimpleNamespace(answers=answers, model="jev-test", request_id="req",
                           usage=SimpleNamespace(input_tokens=10, output_tokens=2))


class ClaudeJevTests(unittest.TestCase):
    def test_observation_is_strict(self):
        for bad in ([], observation(cube_visible="maybe"), observation(cube_xyz="0.22"),
                    {k: v for k, v in observation().items() if k != "cube_visible"}):
            with self.assertRaises(JevControllerError):
                build_state(bad, state())

    def test_state_has_no_object_truth(self):
        sent = build_state(observation(), state())
        self.assertEqual(set(sent), {"task", "last_completed_skill", "gripper_state",
                                     "fingertip_position_m", "visual_observation", "visual_notes"})

    def test_gate_question_only_after_grasp_and_lift(self):
        client = MagicMock()
        client.system_one.return_value = response("move_above_cube")
        JevController(model="jev-test", client=client).decide(observation(), state())
        self.assertEqual(set(client.system_one.call_args.args[1]), {"next_skill"})
        client.system_one.return_value = response("lift", gate=("grasp_secure", 0.9))
        decision = JevController(model="jev-test", client=client).decide(
            observation(), state("close_gripper", "closed"))
        self.assertEqual(set(client.system_one.call_args.args[1]), {"next_skill", "grasp_secure"})
        check_gates(decision)

    def test_low_confidence_and_failed_gate_stop(self):
        client = MagicMock()
        client.system_one.return_value = response("descend", confidence=0.3)
        with self.assertRaises(JevControllerError):
            check_gates(JevController(model="jev-test", client=client).decide(observation(), state("move_above_cube")))
        client.system_one.return_value = response("lift", gate=("grasp_secure", 0.2))
        with self.assertRaises(JevControllerError):
            check_gates(JevController(model="jev-test", client=client).decide(
                observation(), state("close_gripper", "closed")))

    def test_api_failure_is_reported_without_details(self):
        client = MagicMock()
        client.system_one.side_effect = RuntimeError("secret-key-in-message")
        with self.assertRaises(JevControllerError) as caught:
            JevController(model="jev-test", client=client).decide(observation(), state())
        self.assertNotIn("secret", str(caught.exception))

    def test_pick_place_spec_matches_env(self):
        from strands_so101_sim.pick_place_env import PickPlaceEnv
        self.assertEqual(PICK_PLACE.actions, PickPlaceEnv.ACTIONS)
        self.assertEqual(set(PICK_PLACE.skill_criteria), set(PICK_PLACE.actions))
        order = ("ready",) + PICK_PLACE.actions
        for phase, gate in PICK_PLACE.gates.items():
            # Each gate guards exactly the skill that follows its phase.
            self.assertEqual(order[order.index(phase) + 1], gate.guards)
        with self.assertRaises(JevControllerError):
            build_state(observation(), state(), PICK_PLACE)  # Cube Lift fields are not enough


if __name__ == "__main__":
    unittest.main()
