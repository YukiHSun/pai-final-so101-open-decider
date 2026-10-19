"""Scripted RGB pick and place, extending the verified Cube Lift controller."""
from __future__ import annotations

import mujoco
import numpy as np

from strands_so101_sim.cube_lift_env import ControlError, CubeLiftEnv
from strands_so101_sim.so101_pick_ik import make_scene


def placement_valid(*, lift_verified, released, xy_error, footprint_inside,
                    height_error, linear_speed, angular_speed, floor_supported,
                    robot_contact, clearance):
    """Placement must be released, supported and stationary, not just near a goal."""
    values = (xy_error, height_error, linear_speed, angular_speed, clearance)
    return bool(all(np.isfinite(v) for v in values) and lift_verified and released
                and xy_error <= 0.010 and footprint_inside and height_error <= 0.002
                and linear_speed <= 0.010 and angular_speed <= 0.10
                and floor_supported and not robot_contact and clearance >= 0.040)


class PickPlaceEnv(CubeLiftEnv):
    ACTIONS = ("move_above_cube", "descend", "close_gripper", "lift",
               "secure_grasp", "transfer", "lower", "release", "retreat", "done")
    GOAL_HALF_SIZE = 0.030

    def __init__(self, cube_xy=(0.22, 0.0), place_xy=(0.16, 0.0), frame_callback=None):
        self.place_xy = np.asarray(place_xy, dtype=float)
        cube_xy = np.asarray(cube_xy, dtype=float)
        if (self.place_xy.shape != (2,) or cube_xy.shape != (2,)
                or not np.isfinite(np.r_[self.place_xy, cube_xy]).all()
                or not 0.14 <= self.place_xy[0] <= 0.20 or abs(self.place_xy[1]) > 0.04
                or np.linalg.norm(self.place_xy-cube_xy) < 0.045):
            raise ValueError("Place goal must be in the calibrated workspace and >=45 mm from the cube")
        self.lift_verified = False
        self.lift_record = None
        self.released = False
        self.stable_steps = 0
        self.lost_contact_steps = 0
        self.max_contact_gap = 0
        self.active_skill = "open_gripper"
        self._placement_ready = False
        self.sim = make_scene(tuple(cube_xy), self.EDGE, 0.0)
        try:
            # A visual goal only. It must never support or constrain the cube.
            self.sim.add_object(name="place_target", shape="box", is_static=True,
                                position=[*self.place_xy, 0.0001],
                                size=[2*self.GOAL_HALF_SIZE, 2*self.GOAL_HALF_SIZE, 0.0002],
                                color=[0.10, 0.65, 0.20, 0.7])
            self._initialize(frame_callback)
            m = self.model
            marker = m.body("place_target").id
            mask = m.geom_bodyid == marker
            m.geom_contype[mask] = 0
            m.geom_conaffinity[mask] = 0
            self.robot_geoms = {g for g in range(m.ngeom)
                                if m.body(int(m.geom_bodyid[g])).name.startswith("so101/")}
            self.floor_geoms = set(np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist())
            self.cube_dof = int(m.jnt_dofadr[m.body_jntadr[self.cube_bid]])
            self.cube_corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                                          for z in (-1, 1)], float) * self.EDGE/2
            self._placement_ready = True
        except BaseException:
            self.sim.destroy()
            raise

    def placement_state(self):
        pos = self.cube_position()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[self.cube_q+3:self.cube_q+7])
        corners = self.cube_corners @ rotation.reshape(3, 3).T + pos
        footprint_inside = bool(np.all(np.abs(corners[:, :2]-self.place_xy) <= self.GOAL_HALF_SIZE))
        robot_contact = False
        floor_force = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for cube, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if self.model.geom_bodyid[cube] != self.cube_bid:
                    continue
                if other in self.robot_geoms and c.dist <= 0:
                    robot_contact = True
                if other in self.floor_geoms:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    floor_force += float(force[0])
        vel = self.data.qvel[self.cube_dof:self.cube_dof+6]
        # Geometry/proprioception and cube truth are used only for monitoring.
        ee = self.data.site_xpos[self.sid] + self.data.site_xmat[self.sid].reshape(3, 3) @ self.tcp
        state = dict(lift_verified=self.lift_verified, released=self.released,
                     xy_error=float(np.linalg.norm(pos[:2]-self.place_xy)), footprint_inside=footprint_inside,
                     height_error=float(abs(pos[2]-self.EDGE/2)),
                     linear_speed=float(np.linalg.norm(vel[:3])), angular_speed=float(np.linalg.norm(vel[3:])),
                     floor_supported=bool(floor_force > 0.1), robot_contact=robot_contact,
                     clearance=float(ee[2]-pos[2]))
        return state

    def metrics(self):
        base = CubeLiftEnv.metrics(self)
        base["lift_success_now"] = base.pop("success")
        state = self.placement_state() if self._placement_ready else {}
        return {**base, **state, "place_goal": self.place_xy.tolist(), "stable_steps": self.stable_steps,
                "max_transfer_contact_gap_steps": self.max_contact_gap,
                "success": bool(state and self.phase == "done" and placement_valid(**state)
                                and self.stable_steps >= self.required_hold_steps)}

    def _after_step(self):
        if not self._placement_ready:
            return
        if self.active_skill == "transfer":
            self.lost_contact_steps = 0 if self.contacts() else self.lost_contact_steps+1
            self.max_contact_gap = max(self.max_contact_gap, self.lost_contact_steps)
            if self.lost_contact_steps > 25:
                raise ControlError("Grasp lost during transfer for more than 50 ms")
        if self.released:
            self.stable_steps = self.stable_steps+1 if placement_valid(**self.placement_state()) else 0

    def observe(self):
        obs = super().observe()
        obs["robot_state"]["gripper"]["state"] = "closed" if self.jaw < 0.4 else "open"
        return obs

    def _cartesian_path(self, end, step=0.0025, rounds=4):
        start = self.ee_position().copy()
        count = max(1, int(np.ceil(np.linalg.norm(np.asarray(end)-start)/step)))
        for waypoint in np.linspace(start, end, count+1)[1:]:
            self._hold_tcp(waypoint, self.jaw, rounds)
        if np.linalg.norm(self.ee_position()-end) > 0.004:
            raise ControlError(f"{self.active_skill} tracking failed")

    def execute(self, action, target=None):
        allowed = dict(zip(("ready",)+self.ACTIONS[:-1], self.ACTIONS))
        if self.stopped or action != allowed.get(self.phase):
            raise ControlError(f"Unsafe transition: {self.phase} -> {action}")
        self.active_skill = action
        if action in self.ACTIONS[:4]:
            super().execute(action, target)
            if action == "lift":
                self.lift_record = CubeLiftEnv.metrics(self)
                self.lift_verified = self.lift_record["success"]
                if not self.lift_verified:
                    raise ControlError("Lift must pass the 5 cm / 1 second criterion before transport")
            return
        if action == "secure_grasp":
            # The V-shaped fingers unload during lift. Restore a modest preload
            # in air; applying this command on the floor over-squeezes the cube.
            point = self.ee_position().copy()
            for jaw in np.linspace(self.jaw, 0.24, 5)[1:]:
                goal = self.data.ctrl[self.aids].copy()
                goal[-1] = jaw
                self._drive(goal, 200, 150)
                self._hold_tcp(point, jaw, 3)
            self.jaw = 0.24
            if not self.contacts():
                raise ControlError("Grasp lost while tightening")
        elif action == "transfer":
            if not self.contacts():
                raise ControlError("No bilateral grasp before transfer")
            end = self.ee_position().copy()
            # Offset the measured TCP using the ORIGINAL RGB estimate, not live cube truth.
            end[:2] += self.place_xy-self.target[:2]
            self._cartesian_path(end, step=0.002)
        elif action == "lower":
            end = self.ee_position().copy()
            # Known floor/cube geometry plus calibrated finger-to-cube offset.
            end[2] = 0.024
            self._cartesian_path(end)
        elif action == "release":
            for jaw in np.linspace(self.jaw, 0.60, 9)[1:]:
                goal = self.data.ctrl[self.aids].copy()
                goal[-1] = jaw
                self._drive(goal, 200, 150)
            self.jaw = 0.60
            self.released = True
        elif action == "retreat":
            self._cartesian_path(self.ee_position()+[0, 0, 0.065])
            self._drive(self.data.ctrl[self.aids].copy(), 1, 1000)
        elif action == "done":
            if not placement_valid(**self.placement_state()) or self.stable_steps < self.required_hold_steps:
                raise ControlError("Placement did not settle released inside the goal for 1 second")
        self.phase = action
        print(f"[CTRL] step={self.steps} phase={action} ee={self.ee_position().round(4)} jaw={self.data.qpos[self.arm_q[-1]]:.4f} {self.metrics()}", flush=True)
