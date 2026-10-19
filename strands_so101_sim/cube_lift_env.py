"""Local SO-101 skills using the existing MuJoCo scene and mink IK.

Cube truth is used only by diagnostics and the explicit scripted baseline.
"""
from __future__ import annotations

import numpy as np
import mujoco

from strands_so101_sim.so101_pick_ik import (
    ARM_JOINTS, EE_SITE, FIXED_FINGER_MESH, MOVING_JAW_MESH,
    ArmIK, arm_addresses, grab_frame, grasp_tcp, make_scene, measure_tcp,
)


class ControlError(RuntimeError):
    pass


class CubeLiftEnv:
    EDGE = 0.030
    ACTIONS = ("move_above_cube", "descend", "close_gripper", "lift", "done")

    def __init__(self, cube_xy=(0.22, 0.0), frame_callback=None):
        self.sim = make_scene(cube_xy, self.EDGE, 0.0)
        try:
            self._initialize(frame_callback)
        except BaseException:
            self.sim.destroy()
            raise

    def _initialize(self, frame_callback):
        self.model, self.data = self.sim.mj_model, self.sim.mj_data
        m, d = self.model, self.data
        self.arm_q, self.arm_dof = arm_addresses(m)
        self.jids = np.array([m.joint(n).id for n in ARM_JOINTS])
        self.limits = m.jnt_range[self.jids].copy()
        self.aids = np.array([int(np.flatnonzero(m.actuator_trnid[:, 0] == j)[0]) for j in self.jids])
        self.sid = m.site(EE_SITE).id
        self.cube_bid = m.body("cube").id
        self.cube_q = m.jnt_qposadr[m.body_jntadr[self.cube_bid]]
        self.pad_geoms = []
        for name in (FIXED_FINGER_MESH, MOVING_JAW_MESH):
            mid = m.mesh(name).id
            self.pad_geoms.append(next(g for g in range(m.ngeom) if m.geom_dataid[g] == mid and m.geom_contype[g]))
        m.opt.iterations, m.opt.ls_iterations, m.opt.tolerance = 200, 50, 1e-10
        for g in self.pad_geoms + list(np.flatnonzero(m.geom_bodyid == self.cube_bid)):
            m.geom_solref[g] = [0.002, 1.0]
            m.geom_solimp[g] = [0.99, 0.9999, 0.0001, 0.5, 2.0]
        self.ik = ArmIK(m, self.arm_q[-1], np.setdiff1d(np.arange(m.nv), self.arm_dof))
        self.tcp = grasp_tcp(measure_tcp(m, d), self.EDGE)
        self.frame_callback = frame_callback
        self.steps = 0
        self.phase = "ready"
        self.stopped = False
        self.hold_steps = 0
        self.required_hold_steps = int(round(1.0 / m.opt.timestep))
        self.initial_z = float(self.cube_position()[2])
        self.max_rise = 0.0
        self.target = None
        self.jaw = 0.45
        self._drive(np.r_[d.qpos[self.arm_q[:5]], self.jaw], 300, 300)
        self.initial_z = float(self.cube_position()[2])

    def cube_position(self):
        return self.data.qpos[self.cube_q:self.cube_q + 3].copy()

    def ee_position(self):
        return self.ik.tcp_pos(self.data.qpos, self.tcp)

    def observe(self):
        """Public policy input. Intentionally excludes cube truth and metrics."""
        mujoco.mj_forward(self.model, self.data)
        return {
            "rgb": np.concatenate([grab_frame(self.sim, "oblique"), grab_frame(self.sim, "topdown")], axis=1),
            "robot_state": {
                "joint_positions": self.data.qpos[self.arm_q].tolist(),
                "ee_position": self.ee_position().tolist(),
                "gripper": {"position_rad": float(self.data.qpos[self.arm_q[-1]]),
                            "command_rad": float(self.data.ctrl[self.aids[-1]]),
                            "state": "closed" if self.phase in ("close_gripper", "lift", "done") else "open"},
                "last_completed_skill": self.phase,
            },
        }

    def visual_target(self):
        from strands_so101_sim.cube_vision import CameraCalibration, locate_red_cube
        cid = self.model.camera("topdown").id
        calibration = CameraCalibration(self.data.cam_xpos[cid].copy(),
                                        self.data.cam_xmat[cid].reshape(3, 3).copy(),
                                        float(self.model.cam_fovy[cid]))
        return locate_red_cube(grab_frame(self.sim, "topdown"), calibration, self.EDGE)

    def contacts(self):
        found = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for cube, pad in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if self.model.geom_bodyid[cube] == self.cube_bid and pad in self.pad_geoms:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    if force[0] > 0.01:
                        found.add(pad)
        return len(found) == 2

    def metrics(self):
        rise = float(self.cube_position()[2] - self.initial_z)
        return dict(cube_position=self.cube_position().tolist(), cube_rise=rise,
                    bilateral_contact=self.contacts(), hold_steps=self.hold_steps,
                    success=bool(rise > 0.05 and self.contacts() and self.hold_steps >= self.required_hold_steps))

    def _drive(self, goal, ramp=200, settle=150):
        if self.stopped:
            raise ControlError("Simulation is stopped")
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (6,) or not np.isfinite(goal).all():
            raise ControlError("Invalid joint command")
        if np.any(goal < self.limits[:, 0] - 1e-6) or np.any(goal > self.limits[:, 1] + 1e-6):
            raise ControlError("Joint target exceeds model limits")
        start = self.data.ctrl[self.aids].copy()
        for i in range(ramp + settle):
            a = min(1.0, (i + 1) / max(1, ramp))
            self.data.ctrl[self.aids] = (1-a)*start + a*goal
            mujoco.mj_step(self.model, self.data)
            self.steps += 1
            rise = float(self.cube_position()[2] - self.initial_z)
            self.max_rise = max(self.max_rise, rise)
            self.hold_steps = self.hold_steps + 1 if rise > 0.05 and self.contacts() else 0
            if not np.isfinite(self.data.qpos).all():
                raise ControlError("Non-finite physics state")
            self._after_step()
            if self.frame_callback and self.steps % 20 == 0:
                self.frame_callback(self)

    def _after_step(self):
        """Optional task monitoring hook; does not alter the Cube Lift policy."""
        pass

    def _hold_tcp(self, target, jaw, rounds=4):
        for _ in range(rounds):
            have = self.ee_position()
            error = np.asarray(target) - have
            direction = self.data.site_xmat[self.sid].reshape(3, 3)[:, 1]
            angular_error = np.cross(direction, [0.0, 0.0, -1.0])
            if np.linalg.norm(error) < 0.0003 and np.linalg.norm(angular_error) < 0.005:
                break
            # Jacobian at the calibrated fingertip, rather than the wrist site.
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros_like(jp)
            mujoco.mj_jac(self.model, self.data, jp, jr, have, self.model.site_bodyid[self.sid])
            J = np.vstack([jp[:, self.arm_dof[:5]], 0.05*jr[:2, self.arm_dof[:5]]])
            residual = np.r_[error, 0.05*angular_error[:2]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-7*np.eye(5), residual)
            goal = self.data.ctrl[self.aids].copy()
            goal[:5] += np.clip(dq, -0.04, 0.04)
            goal[-1] = jaw
            self._drive(goal, 200, 150)

    def _pose(self, target, seed, seeds=1):
        seed = seed.copy()
        seed[self.arm_q[-1]] = self.jaw
        q = self.ik.solve(np.asarray(target), self.tcp, seed, seeds=seeds)
        if np.linalg.norm(self.ik.tcp_pos(q, self.tcp)-target) > 0.003:
            raise ControlError("IK target unreachable")
        return q

    def execute(self, action, target=None):
        allowed = dict(zip(("ready",) + self.ACTIONS[:-1], self.ACTIONS))
        if self.stopped or action != allowed.get(self.phase):
            raise ControlError(f"Unsafe transition: {self.phase} -> {action}")
        if action == "move_above_cube":
            self.target = np.asarray(target, dtype=float)
            if self.target.shape != (3,) or not np.isfinite(self.target).all():
                raise ControlError("Missing cube estimate")
            if not (0.18 <= self.target[0] <= 0.26 and abs(self.target[1]) <= 0.04 and 0.01 <= self.target[2] <= 0.02):
                raise ControlError("Target outside the calibrated workspace")
            # Choose a top-down branch at grasp height, then retreat along it.
            grasp_q = self._pose(self.target, self.data.qpos, seeds=60)
            hover = self.target + [0, 0, 0.09]
            q = grasp_q
            for z in np.linspace(self.target[2], hover[2], 10)[1:]:
                q = self._pose(np.r_[self.target[:2], z], q)
            self._drive(np.r_[q[self.arm_q[:5]], self.jaw], 1200, 500)
            self._hold_tcp(hover, self.jaw, 8)
            if np.linalg.norm(self.ee_position()-hover) > 0.02:
                raise ControlError("Hover tracking failed")
        elif action == "descend":
            for z in np.linspace(self.target[2]+0.09, self.target[2], 37)[1:]:
                self._hold_tcp(np.r_[self.target[:2], z], self.jaw, 4)
            self._hold_tcp(self.target, self.jaw, 6)
            if np.linalg.norm(self.ee_position()-self.target) > 0.003:
                raise ControlError("Descent tracking failed")
        elif action == "close_gripper":
            # Calibrated for the actual V-shaped jaws at this grasp depth.
            # Stronger squeezing saturates the shoulder and pushes the TCP up.
            close = 0.28
            for jaw in np.linspace(self.jaw, close, 7)[1:]:
                goal = self.data.ctrl[self.aids].copy()
                goal[-1] = jaw
                self._drive(goal, 300, 200)
                self._hold_tcp(self.target, jaw, 12)
            self.jaw = close
            self.model.opt.impratio = 20
            self.model.geom_friction[self.pad_geoms] = [3.0, 0.5, 0.05]
            for g in self.pad_geoms + list(np.flatnonzero(self.model.geom_bodyid == self.cube_bid)):
                self.model.geom_condim[g] = 6
            if not self.contacts():
                raise ControlError("Cube is not touching both fingers")
        elif action == "lift":
            if not self.contacts():
                raise ControlError("Grasp lost before lift")
            start = self.ee_position()
            for dz in np.linspace(0.0, 0.065, 27)[1:]:
                self._hold_tcp(start + [0, 0, dz], self.jaw, 3)
            self._drive(self.data.ctrl[self.aids].copy(), 1, self.required_hold_steps + 100)
        elif action == "done":
            if not self.metrics()["success"]:
                raise ControlError("Premature done: physical success not verified")
        self.phase = action
        print(f"[CTRL] step={self.steps} phase={action} ee={self.ee_position().round(4)} jaw={self.data.qpos[self.arm_q[-1]]:.4f} {self.metrics()}", flush=True)

    def stop(self):
        # Paused MuJoCo advances no further. Preserve grip effort, do not open.
        self.stopped = True

    def close(self):
        self.stop()
        self.sim.destroy()
