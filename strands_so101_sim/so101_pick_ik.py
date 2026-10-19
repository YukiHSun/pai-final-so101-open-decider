#!/usr/bin/env python3
"""SO-101 pick-up in MuJoCo with mink differential IK (CPU only, no GPU).

Builds a tabletop scene via strands-robots (Robot("so101") -> MuJoCo sim),
plans a top-down grasp with mink IK on the gripper site, and executes
reach -> descend -> close -> lift with real physics (friction grip).

Success = the cube's z rises above LIFT_SUCCESS_Z.

Usage:
    MUJOCO_GL=glfw python so101_pick_ik.py [--video out.mp4] [--cube-x 0.22]
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import mujoco
import numpy as np

from strands_robots import Robot

logging.basicConfig(level=logging.WARNING)

EE_SITE = "so101/gripper"
GRIPPER_JOINT = "so101/6"
# The two meshes that form the jaws. The gripper body also carries a servo and a
# motor holder, which are bulkier than the finger itself, so the fingers must be
# picked by mesh name - "the largest geom on the body" grabs the servo and puts
# the measured pinch point ~7 cm off, back inside the wrist.
FIXED_FINGER_MESH = "so101/wrist_roll_follower_so101_v1"
MOVING_JAW_MESH = "so101/moving_jaw_so101_v1"
# Jaw targets, radians (joint range [-0.174, 1.745]). The pad-to-pad gap at the
# fingertip band, measured from the jaw meshes, is 1.9 mm fully closed, 20 mm at
# 0.056 and 38 mm at 0.30. The approach opening is deliberately only a few mm
# wider than the object: measured at the grasp pose with the jaws at 0.30, the
# moving jaw sits 16 mm clear of the cube's near face, so the close never reaches
# it - the fixed pad alone rakes across the cube's top, levers the wrist ~9 mm
# upward, and the lift comes away empty. `approach_jaw` derives this per object.
GRIPPER_OPEN = 0.30
GRIPPER_CLOSED = -0.174
# Clearance per side between jaw and object while approaching. Enough to descend
# past the cube without clipping it, small enough that the moving jaw still has
# the object within reach when it closes.
APPROACH_CLEARANCE = 0.004
# Jaw command held during the grasp and lift, deliberately past the joint's lower
# limit so the position actuator keeps pushing instead of relaxing on contact.
GRIPPER_SQUEEZE = -0.8
# 2 cm cube: measured jaw centre-to-centre gap is only ~24 mm fully closed, so a
# 3 cm cube cannot fit between the fingers at all.
CUBE_EDGE = 0.020
# Riser height under the cube. The fixed finger sits 29.5 mm below the pinch
# point, so a cube on the bare floor cannot be straddled at all (see make_scene).
PEDESTAL_H = 0.030
# How far to raise the cube, and how far it must rise to count. Kept modest on
# purpose: the wrist can only stay near-vertical for the first ~8 cm of the lift
# (it degrades to 0.93 of straight down by +10 cm), and once it tips past that
# the cube rolls out of the jaws.
LIFT_HEIGHT = 0.08
LIFT_SUCCESS_Z = 0.05
# Finger geometry, measured from the collision-mesh vertices expressed in the
# gripper site's frame (see measure_tcp). The gripper body's origin sits at site
# y = -0.098 and the free finger tips reach y = +0.006, so the fingers extend
# along the site's local +Y and that is the approach direction. Getting this
# sign wrong inverts the grasp: "most downward" then picks a pose with the jaws
# facing up and the wrist driven into the floor.
FINGER_AXIS = np.array([0.0, 1.0, 0.0])
DOWN = np.array([0.0, 0.0, -1.0])
# Site-frame Y interval covering the free finger tips, where the two jaws
# overlap and an object actually gets pinched.
TIP_BAND = (-0.010, 0.010)
# Measured mapping from jaw angle to the moving jaw's inner face in site-frame Z
# (the opening axis), taken from the tip band with the arm at qpos 0. The fixed
# pad's inner face sits at Z = 0.0, so this IS the jaw opening.
#   jaw  -0.174 (limit) | -0.034 | +0.056 | +0.146 | +0.306 (open)
#   gap    1.9 mm       | 13.1mm | 20.0mm | 26.4mm |  38.1mm
JAW_GAP_SLOPE = 0.0755  # m of gap per rad, from the table above
JAW_GAP_AT_ZERO = 0.0158  # gap at jaw = 0


def grasp_jaw(object_width: float, squeeze: float = 0.10) -> float:
    """Jaw command for pinching an object `object_width` across.

    The angle that makes the gap exactly `object_width` is where the jaw *touches*
    the object. Position actuators produce force in proportion to
    (command - actual), so the command is driven `squeeze` rad past that: the
    object blocks the jaw short of it and the residual is grip force.

    Both terms are calibrated against a jaw-command sweep with a 20 mm cube held
    between the fingers, not against the geometric table alone - the mesh table
    puts contact at +0.056 rad, but the cube is already loaded with 7.8 N of
    normal force at a command of +0.10, because it is gripped further back along
    the jaw than the fingertip band. Squeeze is small on purpose. Grip force is
    NOT the binding constraint here (the sweep shows 18-39 N of normal force);
    over-squeezing is actively harmful, because the moving jaw pushes the cube
    along the opening axis as it travels, and the opening axis is horizontal
    (world -Y at the grasp pose). At 0.19 rad the cube is walked ~4 mm sideways
    during the close and shot clean out of the gripper during the lift.
    """
    contact = (object_width - JAW_GAP_AT_ZERO) / JAW_GAP_SLOPE
    return contact - squeeze


def approach_jaw(object_width: float, clearance: float = APPROACH_CLEARANCE) -> float:
    """Jaw angle that opens `clearance` wider than the object, per side.

    Opening all the way (GRIPPER_OPEN) parks the moving jaw far outside the
    object; since only that jaw travels, the close then never reaches the object
    at all. Opening just past it keeps both pads within millimetres of the faces
    they have to pinch.
    """
    gap = object_width + 2.0 * clearance
    return min((gap - JAW_GAP_AT_ZERO) / JAW_GAP_SLOPE, GRIPPER_OPEN)


def grab_frame(sim: Robot, camera: str) -> np.ndarray:
    """Decode sim.render()'s PNG bytes into an RGB array."""
    import io

    from PIL import Image

    res = sim.render(camera_name=camera)
    for block in res["content"]:
        if "image" in block:
            png = block["image"]["source"]["bytes"]
            return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    raise RuntimeError(f"render returned no image: {res}")


def make_scene(
    cube_xy: tuple[float, float],
    cube_edge: float = CUBE_EDGE,
    pedestal_h: float = PEDESTAL_H,
) -> Robot:
    """SO-101 + red cube on a pedestal + three cameras.

    The pedestal is what makes the grasp geometrically possible at all. The fixed
    finger reaches 29.5 mm below the pinch point, so grasping a cube whose centre
    is lower than that would need the finger underneath the floor: on the bare
    ground the moving jaw can only press down on the cube until it squirts out
    sideways. Standing the cube on a ~3 cm riser puts its centre high enough for
    the fixed finger to swing in beneath it.
    """
    sim = Robot("so101", mesh=False)
    if pedestal_h > 0:
        sim.add_object(
            name="pedestal",
            shape="box",
            position=[cube_xy[0], cube_xy[1], pedestal_h / 2],
            # Same footprint as the cube. `size` here is the FULL extent, so a
            # narrower riser (the obvious "keep the jaws clear" choice) leaves the
            # cube overhanging on every side and it simply tips off the moment a
            # jaw touches it. Nothing is gained by narrowing it either: measured at
            # the grasp pose both jaw meshes bottom out at z >= 0.0707 while the
            # riser top is at 0.070, so they clear it.
            size=[cube_edge, cube_edge, pedestal_h],
            color=[0.35, 0.35, 0.4, 1.0],
            is_static=True,
        )
    sim.add_object(
        name="cube",
        shape="box",
        position=[cube_xy[0], cube_xy[1], pedestal_h + cube_edge / 2],
        size=[cube_edge] * 3,
        color=[0.85, 0.1, 0.1, 1.0],
        mass=0.03,
    )
    sim.add_camera(name="oblique", position=[0.45, -0.35, 0.35], target=[cube_xy[0], cube_xy[1], 0.05])
    sim.add_camera(name="topdown", position=[0.15, 0.0, 0.7], target=[0.15, 0.0, 0.0])
    return sim


def _mesh_cloud(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """World-space vertices of a mesh geom at the current configuration."""
    mesh = model.geom_dataid[geom]
    if mesh < 0:
        raise ValueError(f"geom {geom} is not a mesh")
    adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    verts = model.mesh_vert[adr : adr + num].reshape(-1, 3).astype(float)
    return (data.geom_xmat[geom].reshape(3, 3) @ verts.T).T + data.geom_xpos[geom]


def measure_tcp(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Offset from the gripper site to the fingertip pinch point, site frame.

    Measured from the actual jaw meshes with the jaws OPEN - this is the point
    the object must be centred on while approaching. Measuring at the CLOSED
    angle instead gives z = +0.001, which is the fixed finger's own surface: aim
    there and the descent drives that finger through the cube and shoves it
    ~13 mm out of the way before the jaws ever close.

    Only the moving jaw travels, so the open gap is not centred on the closed
    pinch point; on this gripper the open-gap centre sits ~19 mm above it.

    The two mesh convex hulls interpenetrate near the wrist, so mj_geomDistance
    reports a constant -20 mm regardless of jaw angle and is useless here. Using
    the mesh vertices directly, restricted to the fingertip band, gives the real
    numbers: 1.9 mm gap closed, 30 mm at jaw 0.20, 52 mm at 0.50.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    jaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)]
    saved = data.qpos.copy()
    data.qpos[:] = 0.0
    data.qpos[jaw_q] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    rot = data.site_xmat[sid].reshape(3, 3)
    origin = data.site_xpos[sid]
    pads = []
    for mesh_name in (FIXED_FINGER_MESH, MOVING_JAW_MESH):
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        # Prefer the physical (contype != 0) copy; visual duplicates share the mesh.
        geoms = [g for g in range(model.ngeom) if model.geom_dataid[g] == mid]
        geom = next((g for g in geoms if model.geom_contype[g]), geoms[0])
        cloud = (rot.T @ (_mesh_cloud(model, data, geom) - origin).T).T
        # Keep the fingertip band only - where the two jaws actually face off.
        pads.append(cloud[(cloud[:, 1] > TIP_BAND[0]) & (cloud[:, 1] < TIP_BAND[1])])

    fixed, moving = pads
    # Return the fixed finger's inner face, not the middle of the open gap. Only
    # the moving jaw travels, and it sweeps the entire gap: park the object in
    # the centre and the closing jaw reaches it at speed and punches it clean out
    # of the gripper (the cube ends up 30 cm away). Seating the object against
    # the stationary finger instead leaves the moving jaw only the last few
    # millimetres to cover. Callers offset by half the object's width - see
    # grasp_tcp.
    tcp = np.array([0.0, float(np.mean(TIP_BAND)), fixed[:, 2].max()])
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return tcp


def grasp_tcp(pad_tcp: np.ndarray, object_width: float) -> np.ndarray:
    """Shift the pad TCP out by half an object so the object seats on the pad.

    `measure_tcp` returns the fixed finger's inner face; aiming that AT the
    object's centre would bury the finger halfway inside it. Offsetting along the
    opening axis (site local Z) by half the width puts the object's face flush
    against the fixed finger, which is the only configuration where the moving
    jaw pinches instead of sweeping the object away.
    """
    return pad_tcp + np.array([0.0, 0.0, object_width / 2.0])


ARM_JOINTS = [f"so101/{i}" for i in range(1, 7)]


def arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """qpos and dof addresses of the 6 SO-101 joints, in actuator order.

    The scene also holds the cube's free joint, so qpos/dof are wider than the
    arm; every arm-only write must go through these index arrays.
    """
    qadr, dadr = [], []
    for name in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])
    return np.array(qadr), np.array(dadr)


class ArmIK:
    """mink differential IK restricted to the 5 arm joints (jaw excluded)."""

    def __init__(self, model: mujoco.MjModel, gripper_qadr: int, free_dofs: np.ndarray):
        import mink

        self.mink = mink
        self.model = model
        self.gripper_qadr = gripper_qadr
        # DOFs the IK must never touch (cube free joint) - zeroed in every dq.
        self.free_dofs = free_dofs
        self.config = mink.Configuration(model)
        # Orientation is left free - see solve() for why a pose task is fatal on
        # a 5-DOF arm. The weight is set to 0 there; this is just construction.
        self.task: Any = None
        self._set_orientation_cost(0.0)
        # The posture task regularizes toward the seed, which solve() re-targets
        # per call. Anchoring it at qpos0 instead makes the lift keyframe unwind
        # the grasp pose back toward the home configuration and drop the cube.
        self.posture = mink.PostureTask(model=model, cost=1e-4)
        self.posture.set_target(model.qpos0.copy())
        self.limits = [mink.ConfigurationLimit(model)]
        self.gripper_dof = model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
        ]

    def _set_orientation_cost(self, cost: float) -> None:
        self.task = self.mink.FrameTask(
            frame_name=EE_SITE,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=cost,
            lm_damping=1.0,
        )

    def finger_dir(self, q: np.ndarray) -> np.ndarray:
        """World-space direction the fingers point at configuration `q`.

        Measured on the model: the fingers run along the site's local +Y.
        """
        self.config.update(q.copy())
        return self.config.get_transform_frame_to_world(EE_SITE, "site").rotation().as_matrix() @ FINGER_AXIS

    def _descend(self, tcp_target: np.ndarray, tcp: np.ndarray, q_init: np.ndarray, iters: int) -> tuple[float, np.ndarray]:
        """Position-only descent aiming the TCP (not the site) at `tcp_target`.

        The site goal is recomputed from the current wrist rotation each
        iteration, so the constraint stays "put the finger pads on this point"
        while the orientation is left completely free.
        """
        self.config.update(q_init.copy())
        for _ in range(iters):
            tf = self.config.get_transform_frame_to_world(EE_SITE, "site")
            rot = tf.rotation().as_matrix()
            goal = np.eye(4)
            goal[:3, :3] = rot
            goal[:3, 3] = tcp_target - rot @ tcp
            self.task.set_target(self.mink.SE3.from_matrix(goal))
            dq = self.mink.solve_ik(self.config, [self.task, self.posture], 1e-2, "quadprog", 1e-3, limits=self.limits)
            # Freeze the jaw (scripted, not IK-driven) and the cube's free joint.
            dq[self.gripper_dof] = 0.0
            dq[self.free_dofs] = 0.0
            self.config.integrate_inplace(dq, 1e-2)
        q = self.config.q.copy()
        err = float(np.linalg.norm(self.tcp_pos(q, tcp) - tcp_target))
        return err, q

    def solve(
        self,
        tcp_target: np.ndarray,
        tcp: np.ndarray,
        q_init: np.ndarray,
        want_dir: np.ndarray = DOWN,
        iters: int = 300,
        seeds: int = 60,
        tol: float = 3e-3,
    ) -> np.ndarray:
        """IK for the fingertip TCP at `tcp_target`, preferring `want_dir` fingers.

        Orientation is NOT constrained. On this 5-DOF arm, adding even a weak
        top-down orientation task costs 6-9 cm of position error at every
        tabletop point, whereas position-only solves land within a millimetre.
        Instead many random restarts are run and, among the ones that hit the
        target, the pose whose fingers point closest to `want_dir` is kept -
        which recovers a near-vertical grasp for free.
        """
        jaw = float(q_init[self.gripper_qadr])
        lo, hi = self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1]
        rng = np.random.default_rng(0)
        self._set_orientation_cost(0.0)
        # Stay near the current pose. With seeds=1 (the grasp and lift keyframes)
        # this is what keeps the wrist vertical through the lift instead of
        # rolling over to a different IK branch and shedding the cube.
        self.posture.set_target(q_init.copy())

        best_hit: tuple[float, np.ndarray] | None = None  # (alignment cost, q)
        best_any: tuple[float, np.ndarray] | None = None  # (position error, q)
        for k in range(seeds):
            seed = q_init.copy()
            if k:
                seed[:6] = lo + rng.random(6) * (hi - lo)
            seed[self.gripper_qadr] = jaw
            err, q = self._descend(tcp_target, tcp, seed, iters)
            if best_any is None or err < best_any[0]:
                best_any = (err, q)
            if err < tol:
                cost = 1.0 - float(self.finger_dir(q) @ want_dir)
                if best_hit is None or cost < best_hit[0]:
                    best_hit = (cost, q)

        q = (best_hit or best_any)[1].copy()
        q[self.gripper_qadr] = jaw
        return q

    def tcp_pos(self, q: np.ndarray, tcp: np.ndarray) -> np.ndarray:
        """World position of the fingertip TCP at configuration `q`."""
        self.config.update(q.copy())
        tf = self.config.get_transform_frame_to_world(EE_SITE, "site")
        return tf.translation() + tf.rotation().as_matrix() @ tcp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube-x", type=float, default=0.22)
    ap.add_argument("--cube-y", type=float, default=0.0)
    ap.add_argument("--video", default=None, help="Write an MP4 rollout here.")
    ap.add_argument("--camera", default="oblique")
    ap.add_argument("--cube-edge", type=float, default=CUBE_EDGE)
    ap.add_argument("--pedestal", type=float, default=PEDESTAL_H, help="Riser height under the cube (m).")
    # The jaws close asymmetrically (only one finger moves), so the pinch point
    # shifts as they close; these trim the descent target to sit between them.
    ap.add_argument("--grasp-dz", type=float, default=0.0, help="Vertical trim on the grasp target (m).")
    ap.add_argument("--grasp-dy", type=float, default=0.0, help="Lateral trim along the jaw axis (m).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sim = make_scene((args.cube_x, args.cube_y), args.cube_edge, args.pedestal)
    model, data = sim.mj_model, sim.mj_data
    arm_q, arm_dof = arm_addresses(model)
    gripper_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)]
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    free_dofs = np.setdiff1d(np.arange(model.nv), arm_dof)
    ik = ArmIK(model, gripper_qadr, free_dofs)
    tcp = grasp_tcp(measure_tcp(model, data), args.cube_edge)
    if not args.quiet:
        print(f"grasp TCP in site frame: {np.round(tcp, 4)}")

    cube_xy = np.array([args.cube_x, args.cube_y])
    # Grasp at the cube's mid-height so the jaws straddle its body.
    grasp_z = args.pedestal + args.cube_edge / 2 + args.grasp_dz
    grasp_xy = cube_xy + np.array([0.0, args.grasp_dy])
    grasp = np.array([*grasp_xy, grasp_z])
    # reach: hover above the cube; grasp: descend onto it; lift: straight up.
    # Step counts are physics steps at dt=0.002 s, so 400 steps = 0.8 s of ramp.
    # The position actuators need this long to actually track: ramping in ~0.2 s
    # leaves the arm lagging its command by >10 cm and the grasp never lands.
    # `seeds` is how many random IK restarts to try. The approach keyframes get
    # many, to search out a genuinely top-down wrist. Once the jaws are around
    # the cube only the warm start is used (seeds=1): a restart lands on a
    # different IK branch, and swinging through it swipes the cube away instead
    # of lifting it.
    HOVER_H = 0.09
    jaw_open = approach_jaw(args.cube_edge)
    keyframes = [("hover", grasp + [0.0, 0.0, HOVER_H], jaw_open, 500, 300, 60)]
    # Descend in 1 cm steps for the same reason the lift is stepped. `ctrl` is
    # interpolated in JOINT space, so a single hover->grasp command sweeps the
    # wrist through an arc, not a straight line down. Measured, that arc dragged
    # the fixed finger sideways into the cube and shoved it 6.9 mm out of the
    # gripper (cube x 0.220 -> 0.227) before the jaw even started to close, so the
    # close then shut on empty air. Re-solving every centimetre keeps the path
    # vertical and the cube untouched until the jaw closes on it.
    n_down = max(1, int(round(HOVER_H / 0.01)))
    for i in range(1, n_down + 1):
        dz = HOVER_H * (1.0 - i / n_down)
        last = i == n_down
        keyframes.append((f"descend{i:02d}", grasp + [0.0, 0.0, dz], jaw_open, 300, 200, 60 if last else 1))
    # Close slowly (3 s ramp). The moving jaw still has some distance to cover,
    # and ramped over 0.8 s it arrives fast enough to knock the cube loose rather
    # than pinch it.
    jaw_grip = grasp_jaw(args.cube_edge)
    keyframes.append(("close", grasp, jaw_grip, 1500, 500, 1))
    # Lift in 2 cm steps rather than one jump to LIFT_HEIGHT. Solved from the
    # grasp pose in small increments the wrist stays within 0.8 of straight down
    # all the way up (measured); asked for the full height in one solve, the IK
    # lands on a sideways branch and swinging into it shakes the cube loose.
    n_steps = max(1, int(round(LIFT_HEIGHT / 0.01)))
    for i in range(1, n_steps + 1):
        dz = LIFT_HEIGHT * i / n_steps
        keyframes.append((f"lift{i:02d}", grasp + [0.0, 0.0, dz], jaw_grip, 600, 400, 1))

    frames: list[np.ndarray] = []
    q_seed = data.qpos.copy()
    max_cube_z = float(data.xpos[cube_bid][2])

    frame_every = 12  # physics steps between rendered frames (dt=0.002 -> ~42 fps)

    def drive(ctrl_goal: np.ndarray, ramp_steps: int, settle_steps: int) -> None:
        """Ramp `data.ctrl` from where it is to `ctrl_goal`, then hold."""
        nonlocal max_cube_z
        ctrl_start = data.qpos[arm_q].copy()
        for i in range(ramp_steps + settle_steps):
            alpha = min(1.0, (i + 1) / ramp_steps)
            data.ctrl[:] = (1 - alpha) * ctrl_start + alpha * ctrl_goal
            mujoco.mj_step(model, data)
            max_cube_z = max(max_cube_z, float(data.xpos[cube_bid][2]))
            if args.video and i % frame_every == 0:
                frames.append(grab_frame(sim, args.camera))

    # Running feed-forward correction for actuator sag, calibrated on the descent
    # and then carried by every later keyframe (see below). It is added to the IK's
    # INPUT only - `target` stays the true goal, so the reported error is the real
    # one. Adding it to `target` as well double-counts it and lifts the gripper
    # ~9 mm above the cube at the close.
    offset = np.zeros(3)

    # Arm command that was holding the cube at the end of the close. Every lift
    # keyframe is planned from THIS, warm-started from the grasp pose, so the wrist
    # keeps the orientation that is actually gripping.
    grip_ctrl: np.ndarray | None = None
    grip_seed: np.ndarray | None = None

    # Extra downward bias on the close command. The jaw's own closing torque reacts
    # back through the wrist and lifts the arm ~8 mm even with the arm command held
    # perfectly constant (measured: joints 2-3 saturate at 0.26 N). The fingers ride
    # up the cube's faces as that happens and end up pinching its top corner - the
    # contact normals are correct and carry 3-7 N, but 1 mm of engagement shears off
    # the moment the lift starts. Commanding the close this far below the grasp
    # cancels the rise so the pads stay at the cube's mid-height.
    CLOSE_PRELOAD = 0.008

    for name, xyz, jaw, ramp_steps, settle_steps, seeds in keyframes:
        target = np.asarray(xyz, dtype=float)
        # The preload has to be carried by the lift too. Applied at the close alone
        # it is undone by the first lift step, which re-solves for the un-biased
        # target and jumps the wrist back up 13 mm - the cube is left behind.
        preload = CLOSE_PRELOAD if (name == "close" or name.startswith("lift")) else 0.0
        # The jaw command deliberately over-drives past the joint limit for grip
        # force, but the IK seed must stay inside the limit to be a valid pose.
        seed = (grip_seed if grip_seed is not None else q_seed).copy()
        seed[gripper_qadr] = max(jaw, GRIPPER_CLOSED)
        q_goal = ik.solve(target + offset - [0.0, 0.0, preload], tcp, seed, seeds=seeds)
        if not args.quiet:
            print(f"  -> {name}: planned finger_dir={np.round(ik.finger_dir(q_goal), 3)}")
        ctrl_goal = q_goal[arm_q].copy()
        # Command the jaw directly rather than using the IK's value for it. These
        # are position actuators, so grip force is proportional to
        # (command - actual): the cube blocks the jaw short of `jaw` and the
        # residual is the squeeze. Letting the IK pick the jaw angle instead gives
        # ~0.03 N, far under the cube's 0.29 N weight, and the cube slides straight
        # back out during the lift.
        ctrl_goal[-1] = jaw
        # Freeze the joints that do not need to change to raise the wrist. Measured,
        # re-solving all five arm joints per lift step walks the gripper sideways by
        # fractions of a millimetre each step, and because the jaw's opening axis is
        # horizontal that shear pumps the cube out along it (cube y 0.004 -> 0.03 ->
        # 0.27 m over eight steps - it is ejected, not dropped). Joint 1 is the base
        # yaw and joint 5 the wrist roll; neither contributes to a vertical move from
        # a top-down grasp, so pinning them to the gripping pose removes the shear.
        if grip_ctrl is not None:
            ctrl_goal[0] = grip_ctrl[0]
            ctrl_goal[4] = grip_ctrl[4]
        drive(ctrl_goal, ramp_steps, settle_steps)
        actual = ik.tcp_pos(data.qpos, tcp)
        if name == "close":
            grip_ctrl = ctrl_goal.copy()
            grip_seed = data.qpos.copy()

        # The IK solution is exact, but the position actuators settle several mm
        # short of it under gravity. On the descent that matters: the grasp seats
        # the cube against the *fixed* finger, so a 3 mm shortfall leaves a gap
        # there and the closing jaw shoves the cube out of the gripper instead of
        # pinching it. A single feed-forward correction from the previous
        # keyframe's error is not enough (the sag is pose-dependent), so close the
        # loop: offset the target by the residual and re-drive until it lands.
        # Once calibrated, `offset` is carried into every later keyframe so the
        # close and the lift keep commanding the overshoot that is actually
        # holding the arm on the cube.
        # Calibrate one keyframe BEFORE the fingers reach the cube. The residual
        # has to be taken out in all three axes - a 5 mm shortfall in x leaves a
        # gap at the fixed pad and the closing jaw pushes the cube through it -
        # but correcting it at the grasp height would drag the fingers sideways
        # across the cube. One step up the arm is still in free air, so the
        # correction is free; the last descent step then comes straight down onto
        # an already-correct column.
        if name == f"descend{max(1, n_down - 1):02d}":
            for _ in range(4):
                err = target - actual
                if np.linalg.norm(err) < 1e-3:
                    break
                offset += err
                q_seed[gripper_qadr] = max(jaw, GRIPPER_CLOSED)
                q_goal = ik.solve(target + offset, tcp, q_seed, seeds=1)
                ctrl_goal = q_goal[arm_q].copy()
                ctrl_goal[-1] = jaw
                drive(ctrl_goal, 200, 200)
                actual = ik.tcp_pos(data.qpos, tcp)
                q_seed = data.qpos.copy()
        if not args.quiet:
            print(
                f"[{name:8s}] tcp_target={np.round(target, 3)} tcp_actual={np.round(actual, 3)} "
                f"err={np.linalg.norm(actual - target):.4f} jaw={data.qpos[gripper_qadr]:+.3f} "
                f"cube_z={data.xpos[cube_bid][2]:.4f}"
            )
        q_seed = data.qpos.copy()

    cube_z = float(data.xpos[cube_bid][2])
    # Success is measured as a rise above the cube's own resting height, so a
    # pedestal cannot inflate it.
    rest_z = args.pedestal + args.cube_edge / 2
    threshold = rest_z + LIFT_SUCCESS_Z
    success = max_cube_z > threshold
    print(f"\ncube final z={cube_z:.4f}  max z during rollout={max_cube_z:.4f}  (threshold {threshold:.4f})")
    print(f"RESULT: {'SUCCESS - cube lifted' if success else 'FAIL - cube not lifted'}")

    if args.video and frames:
        import imageio.v3 as iio

        iio.imwrite(args.video, np.stack(frames), fps=20)
        print(f"video: {args.video} ({len(frames)} frames)")

    sim.destroy()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
