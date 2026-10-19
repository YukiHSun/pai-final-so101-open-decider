"""Placement acceptance and motion guard regression tests (no API or rendering)."""
import unittest

from strands_so101_sim.cube_lift_env import ControlError
from strands_so101_sim.pick_place_env import PickPlaceEnv, placement_valid


def stable_placement():
    return dict(lift_verified=True, released=True, xy_error=0.005,
                footprint_inside=True, height_error=0.0001, linear_speed=0.0,
                angular_speed=0.0, floor_supported=True, robot_contact=False, clearance=0.06)


class PlacementTests(unittest.TestCase):
    def test_rejects_incomplete_or_unstable_placements(self):
        self.assertTrue(placement_valid(**stable_placement()))
        failures = dict(lift_verified=False, released=False, xy_error=0.011,
                        footprint_inside=False, height_error=0.003, linear_speed=0.011,
                        angular_speed=0.11, floor_supported=False, robot_contact=True,
                        clearance=0.039)
        for name, value in failures.items():
            with self.subTest(condition=name):
                self.assertFalse(placement_valid(**{**stable_placement(), name: value}))
        self.assertFalse(placement_valid(**{**stable_placement(), "xy_error": float("nan")}))

    def test_stability_requires_consecutive_steps(self):
        e = PickPlaceEnv.__new__(PickPlaceEnv)
        e._placement_ready = True
        e.active_skill = "retreat"
        e.released = True
        e.stable_steps = 499
        state = stable_placement()
        e.placement_state = lambda: state
        e._after_step()
        self.assertEqual(e.stable_steps, 500)
        state["robot_contact"] = True
        e._after_step()
        self.assertEqual(e.stable_steps, 0)

    def test_transfer_stops_after_persistent_contact_loss(self):
        e = PickPlaceEnv.__new__(PickPlaceEnv)
        e._placement_ready = True
        e.active_skill = "transfer"
        e.released = False
        e.lost_contact_steps = e.max_contact_gap = 0
        e.contacts = lambda: False
        for _ in range(25):
            e._after_step()
        with self.assertRaisesRegex(ControlError, "Grasp lost"):
            e._after_step()
        self.assertEqual(e.max_contact_gap, 26)

    def test_release_cannot_skip_lift_and_transport(self):
        e = PickPlaceEnv.__new__(PickPlaceEnv)
        e.stopped = False
        e.phase = "ready"
        with self.assertRaisesRegex(ControlError, "Unsafe transition"):
            e.execute("release")

    def test_done_does_not_accept_a_single_stable_instant(self):
        e = PickPlaceEnv.__new__(PickPlaceEnv)
        e.stopped = False
        e.phase = "retreat"
        e.stable_steps, e.required_hold_steps = 499, 500
        e.placement_state = stable_placement
        with self.assertRaisesRegex(ControlError, "1 second"):
            e.execute("done")


if __name__ == "__main__":
    unittest.main()
