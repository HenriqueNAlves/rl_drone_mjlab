"""Drone tilted-landing environment configuration for MJLab."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import math

import mujoco
import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import (
  Entity,
  EntityArticulationInfoCfg,
  EntityCfg,
  VariantEntityCfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  dr,
  projected_gravity,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import SiteEffortAction, SiteEffortActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
)
from mjlab.utils.noise import (
  GaussianNoiseCfg,
  NoiseModelWithAdditiveBiasCfg,
  UniformNoiseCfg,
)
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRONE_XML: Path = Path(__file__).parent / "drone.xml"
_DRONE_CFG = SceneEntityCfg("drone")

# Nominal spawn point: above the tower top so the drone approaches from above
# and descends onto it instead of flying into the stack's side. Its height is
# also the mean of the randomized spawn height (see `_SPAWN_HEIGHT_ABS`), and
# its bearing feeds `_APPROACH_YAW`, so x/y must stay on the 45 deg diagonal.
_SPAWN_HEIGHT = 0.5                                 # m, mean spawn altitude
_SPAWN = (0.0, 0.0, _SPAWN_HEIGHT)

# --- Tower geometry -------------------------------------------------------
# Three loose boxes stacked under the tilted platform. Declared up here because
# the platform height (and therefore the target) is derived from the stack.
_BOX_HALF = (0.12, 0.12, 0.07)                       # 0.24 x 0.24 footprint
_BOX_COUNT = 3
_TOWER_MASS = 0.5                                    # kg, per loose piece
_STACK_TOP = _BOX_COUNT * 2.0 * _BOX_HALF[2]         # 0.42 m

# Half-length of the platform along the slope. Identical for every tilt, so
# the pad footprint never changes with the angle. (Pinning the height instead
# would stretch a shallow ramp into a 0.9 m seesaw balanced on a 0.24 m box.)
_WEDGE_HALF_L = 0.138564                             # m
_PLATFORM_L = 2.0 * _WEDGE_HALF_L                    # m

# Platform tilts a mesh exists for. mjlab can give different worlds different
# *mesh* variants of the same entity; we deliberately do not use that spread.
# The real tower is one object with one slope, so training across all four
# spends three quarters of the experience on worlds that do not exist and
# makes the policy hedge against a tilt it will never meet.
#
# 45 deg is deliberately absent: friction is MuJoCo's default mu = 1.0 and
# tan(45 deg) = 1.0, so at 45 deg the drone is exactly at the sliding
# threshold and the outcome is luck, not skill.
_PLATFORM_TILT_DEG = (0.0, 10.0, 20.0, 30.0)
_PLATFORM_TILTS = tuple(math.radians(d) for d in _PLATFORM_TILT_DEG)

# The one slope every world gets, training and play alike. Set this to the
# measured tilt of the real pad; a different angle is a different run (and a
# different checkpoint), not a bigger batch.
_TILT_DEG = 10.0


def _platform_rise(tilt: float) -> float:
  """How far the platform's high edge rises above its low edge."""
  return _WEDGE_HALF_L * math.tan(tilt)


_PLATFORM_RISE = tuple(_platform_rise(t) for t in _PLATFORM_TILTS)

# Pad centre in the horizontal plane. The height depends on the variant: the
# wedge's low edge is pinned to the stack top, so the pad centre (the midpoint
# of the sloped top face) sits `_STACK_TOP + _PLATFORM_THICKNESS + rise` above
# the floor.
_PAD_XY = (2.0, 2.0)
_TARGET = (_PAD_XY[0], _PAD_XY[1], _STACK_TOP)

# Horizontal bearing from the spawn to the target. The drone flies in along it
# and flares (leans its top back) to brake; tilting about the perpendicular
# axis leans the target the same way, so the attitude it has to reach is close
# to the flare it already makes. The tower top is built with the same bearing,
# so its platform normal is exactly the attitude the drone has to hold.
_APPROACH_YAW = math.atan2(_TARGET[1] - _SPAWN[1], _TARGET[0] - _SPAWN[0])


_SITE_NAMES = ("rotor_fl", "rotor_fr", "rotor_rl", "rotor_rr")

_MIN_HEIGHT = 0.05
_MAX_POS_XY = 10.0
_MAX_VEL = 20.0

# Contact force (N) above which a tower-box touch counts as a collision,
# matching the threshold used for the ground sensor.
_TOWER_CONTACT_FORCE = 1.0

# Episode duration in seconds. After this, the episode ends — no reward,
# no penalty, just a restart.
_EPISODE_LENGTH_S = 10.0

# Safety net: terminate once the drone tilts past this angle from upright,
# so an inverted, unrecoverable drone doesn't spin out of control.
_MAX_TILT_RAD = math.radians(90.0)

# --- Reward weights -------------------------------------------------------
# Negative shaping (closer / slower is better) plus a positive alive bonus.
# The alive term is what makes surviving better than terminating; keep it
# >= the spawn distance (2.83 m) or a surviving step goes net-negative.
_REWARD_DISTANCE = 1.0
_REWARD_SPEED = 0.02
_REWARD_CLOSING = 0.5
_REWARD_RATE = 0.01
_REWARD_ALIVE = 3.0

_PENALTY_CRASH = 150.0
_PENALTY_TOWER = 150.0
_PENALTY_TILT = 150.0
_PENALTY_RUNAWAY = 150.0
_PENALTY_TIMEOUT = 60.0
# Terminal bonus for reaching the goal. As a one-shot it must beat what the
# drone could earn by surviving just *outside* the tolerance for a whole
# episode (~+54 at dt=0.02), or it learns to loiter there instead of finishing.
# One-shot value = weight * dt, so keep the weight above ~2200.
_REWARD_SUCCESS = 5000.0

# Target-attitude shaping: -tilt_error (rad), ramped in over the last
# `_TILT_GATE_RADIUS` metres. Applied everywhere it would fight the tilt the
# drone needs in order to fly and brake, so it is gated to the landing.
_REWARD_TILT = 2.0
_TILT_GATE_RADIUS = 1.0

# Success tolerances (used by both the `success` reward and termination)
_SUCCESS_POS_TOL = 0.20     # m
_SUCCESS_VEL_TOL = 0.15     # m/s
# Curriculum: 12 deg on the 10 deg ramp, then 10 deg on the 20 deg ramp,
# then 5 deg on the final 30 deg ramp. Keep it a bit looser than the ramp
# while the policy is still learning to arrive tilted at all.
_SUCCESS_TILT_TOL = math.radians(12.0)
# Require the drone to actually touch the platform, not just fly through the
# landing pose. Set False to let "at the pose, slow, tilted" count on its own.
_SUCCESS_REQUIRE_CONTACT = True


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------
# One block, one half-width per parameter. `_PCT` knobs are relative (0.10 =
# +-10 % of the nominal), `_ABS` knobs are in the parameter's own unit. The
# values below are the FULL spread; `_DR_SCALE` multiplies all of them at once.
# Nothing else about the aircraft is randomized. Sensor error is a separate
# story: see `_OBS_*` below, which is on by default and not scaled by
# `_DR_SCALE`.
#
# Randomization curriculum (manual, like the tilt ladder):
#   0.0 -- deterministic aircraft. Nominal mass / inertia / thrust / COM, fixed
#          spawn. The real 35 ms motor lag and 15 ms transport delay stay on,
#          because those are physics, not uncertainty.
#   0.5 -- half the spread below.
#   1.0 -- the spread below.
_DR_SCALE = 1.0

# --- Stage A: propulsion / body -------------------------------------------
_DR_MASS_PCT = 0.10        # 0.160 kg  -> 0.144 .. 0.176
_DR_INERTIA_PCT = 0.30     # extra shape spread on top of the mass scaling
_DR_COM_ABS = 0.003        # m, COM shift per axis (+-3 mm)
# Thrust-to-weight. This is the payload / battery term: mass alone cannot
# change the ratio here, because the plant writes c * m * g, so a heavier
# airframe is automatically stronger. ±30 % is a sagging 6S pack, which really
# does lose that much (thrust goes as RPM^2, and voltage sags ~20 %).
_DR_THRUST_PCT = 0.30      # c -> c * (1 +- 0.30)
_DR_TAU_PCT = 0.30         # 35 ms -> 24 .. 46
_DR_DELAY_ABS = 0.010      # s,   15 ms -> 5 .. 25 ms

# --- Stage A: contact ------------------------------------------------------
# Pad friction, absolute mu. Every geom currently sits at MuJoCo's default 1.0,
# so this only ever makes the pad slicker -- which is the direction that costs
# a landing, and the direction a real pad (foam, dust, a little flex) can go.
_DR_FRICTION = (0.60, 1.00)

# --- Stage A: initial state ------------------------------------------------
_DR_INIT_XY_ABS = 0.10     # m
_DR_INIT_Z_ABS = 0.05      # m
_DR_INIT_VEL_ABS = 0.10    # m/s, per axis
_DR_INIT_RP_ABS = math.radians(5.0)
# Kept small on purpose: `rel_pos` is world-frame, so the policy only learns
# its heading from the horizontal part of `target_up` (|sin(tilt)| = 0.17 at
# 10 deg). Full +-180 deg becomes safe only once `rel_pos` is rotated into the
# heading frame. See project_notes.md.
_DR_INIT_YAW_ABS = math.radians(15.0)
_DR_INIT_RATE_ABS = 0.20   # rad/s

# --- Nominals -------------------------------------------------------------
# The airframe nominals (mass, inertia, thrust per motor, yaw torque) live in
# `drone.xml` only; duplicating them here would let the two drift apart. The
# actuation nominals have no XML representation, so they live here.
_NOM_TAU_S = 0.035                      # s, thrust lag from the log
_NOM_DELAY_S = 0.015                    # s, transport delay from the log


# ---------------------------------------------------------------------------
# Observation corruption
# ---------------------------------------------------------------------------
# This is what the aircraft's estimator gets wrong, not how the aircraft
# responds. It is deliberately not part of `_DR_SCALE`: the plant can be
# deterministic while the sensors are still imperfect, and the real drone
# always has imperfect sensors.
#
# Three sources, three stories, and the errors are shared inside a group
# because that is how the real signals are produced:
#
#   MoCap     rel_pos, lin_vel        one camera stream: one lag, and velocity
#                                     consistent with the position it derives
#                                     from. Position noise is the accuracy of
#                                     the tracker (a few mm on a real rig).
#   gyro      ang_vel                 fast and precise, but it drifts: white
#                                     noise on top of a per-episode bias.
#   estimator gravity, target_up      both are the *same* estimated rotation
#                                     applied to different world vectors, so
#                                     they are wrong together. See
#                                     `_SharedTiltBias`.
#
# The numbers are hardware, not taste: gyro noise on these flight controllers
# is ~0.1 %, MoCap position ~2-5 mm, and the attitude estimate is good to
# 1-3 deg in steady flight. Inflating them does not make the sim more real --
# it makes the policy defend against an error that does not exist.
_OBS_NOISE_POS = 0.004                  # m,     MoCap position, 1 sigma
_OBS_NOISE_VEL = 0.03                   # m/s,   MoCap velocity, 1 sigma
_OBS_NOISE_RATE = 0.01                  # rad/s, gyro white noise, 1 sigma
_OBS_RATE_BIAS = 0.02                   # rad/s, per-episode gyro bias, +-
_OBS_NOISE_ATT = 0.01                   # unit vector, ~0.6 deg of estimator jitter
_OBS_TILT_BIAS_DEG = 1.5                # deg, per-episode attitude estimate error
_OBS_LAG_MOCAP = 1                      # control steps, 20 ms at 50 Hz
_OBS_LAG_ATT = 1                        # control steps


# ---------------------------------------------------------------------------
# Spawn area
# ---------------------------------------------------------------------------
# The drone starts anywhere in a flat annulus around the tower, at a random
# altitude, and has to land on it from whatever bearing it drew. This is a
# property of the *task*, not of the aircraft, so it is deliberately not
# scaled by `_DR_SCALE`: it must stay randomized even when the aircraft is
# deterministic.
#
# The bounds are chosen against three hard limits:
#   near -- far enough that the drone never starts on top of the stack, and
#           never inside the 1 m tilt-gate radius where the attitude shaping
#           begins.
#   far  -- `runaway` fires at 10 m from the world origin and the tower sits
#           2.83 m out, so 4 m still leaves >= 3.1 m of margin. It also keeps
#           every spawn well inside the +-5 m `rel_pos` observation clamp.
#   high -- one 10 s episode has to cover the whole descent from the top of
#           this band.
_SPAWN_DISTANCE_NEAR = 1.20                          # m, keep-out radius
_SPAWN_DISTANCE_FAR = 4.00                           # m
_SPAWN_HEIGHT_ABS = 0.35                             # +- m  (0.15 .. 0.85)
_SPAWN_SPEED_ABS = 0.10                              # +- m/s, per axis
_SPAWN_LEAN_ABS = math.radians(5.0)                  # +- rad, roll and pitch
_SPAWN_HEADING_ABS = math.radians(15.0)              # +- rad, yaw
_SPAWN_SPIN_ABS = 0.20                               # +- rad/s, per axis


# ---------------------------------------------------------------------------
# Aerodynamics (rotor drag)
# ---------------------------------------------------------------------------
# Motors only: the ARX filter prescribes the rate, so there is no rotor torque
# to cancel. (`_MAX_BODY_RATE` is reused as the ARX command scale.)
#
# The drone must not be able to spin faster than `_MAX_BODY_RATE`. Clamping the
# rate would hand that limit to the policy for free, so instead the airframe
# gets the drag that makes the limit an *equilibrium*: the drag torque cancels
# the peak rotor torque exactly at `_MAX_BODY_RATE`.
#
#   w_max = Q_peak / c   ->   c = Q_peak / w_max
#
# The rate is then only reachable with the rotors saturated, and it cannot be
# exceeded: above it the drag torque exceeds anything the rotors can produce.
#
# Q_peak is the same on roll and pitch (square frame, 0.065 m arm on both); yaw
# uses the rotor reaction torque instead of the arm.
_MAX_BODY_RATE = math.radians(500.0)    # rad/s, ceiling on every body rate
_ARM = 0.065                            # m,     mirrors drone.xml site pos
_THRUST_MAX = 1.04                      # N,     mirrors drone.xml ctrlrange
_YAW_COEFF = 0.017                      # N*m/N, mirrors drone.xml gear z

_DAMP_ROLL = 2.0 * _ARM * _THRUST_MAX / _MAX_BODY_RATE
_DAMP_PITCH = _DAMP_ROLL
_DAMP_YAW = 2.0 * _YAW_COEFF * _THRUST_MAX / _MAX_BODY_RATE


# ---------------------------------------------------------------------------
# Plant selection: four motors vs CTBR + ARX
# ---------------------------------------------------------------------------
# "motors" -- the policy drives the four rotors directly; the geometry turns
#             the forces into torque (delay, gain, lag per motor).
# "arx"    -- the policy drives [thrust, p, q, r]. A per-channel first-order
#             filter models the flight controller's *closed-loop* rate
#             response, and a rate servo turns that reference into body
#             torque, so contacts can still fight it.
_PLANT = "arx"

# Action scaling. `c = 1 + u[0]` is thrust in g, so hover is zero action.
_ARX_RATE_MAX_PQ = _MAX_BODY_RATE      # 8.73 rad/s, measured roll ceiling
_ARX_RATE_MAX_R = math.radians(505.0)  # 8.82 rad/s, measured yaw ceiling

# First-order lag per channel [p, q, r, c], seconds, identified from
# documents/btfl_001(in).csv by tools/identify_arx.py. The rates come from the
# Welch cross-spectrum of gyroSp -> gyroADC; a one-step least-squares fit does
# not work there because a smooth stick makes it fit the next sample well while
# getting the lag wrong. Thrust is a genuine one-pole, so OLS is fine:
# throttle -> mean(omega^2), R2 0.9997.
#
# The fitted values are command -> body rate, so they already include the
# flight controller's own rate loop. The servo below adds ~7 ms on top, which
# is why these sit slightly under the fit.
#   p/q  roll 38 ms, pitch 51 ms. One shared value puts the total at ~42 ms.
#   r    131 ms fitted -> 124 here plus servo.
#   c    40-43 ms fitted -> 37 here plus servo.
_ARX_TAU_PQ_S = 0.035
_ARX_TAU_R_S = 0.124
_ARX_TAU_C_S = 0.037
_ARX_DELAY_S = 0.015                   # transport delay, measured

# Rate servo: fraction of the rate error cancelled per physics substep. This
# is the discrete form of tau_servo = dt / alpha = 8 ms, comfortably inside
# the explicit-feedback stability limit (alpha must stay below 1).
_ARX_SERVO_ALPHA = 0.25
_ARX_GRAVITY = 9.81                    # m/s^2, thrust = c * m * g


# ---------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------
# Four loose bodies: three stacked boxes plus a wedge on top. Nothing is
# welded, so the drone can tumble the stack. mjlab allows at most one freejoint
# per Entity, so each loose body is its own Entity; with no reset event they
# reset to their stacked poses from the model default every episode.
#
# The wedge's sloped face is the landing platform. Its profile is a prism whose
# hypotenuse is the platform: it runs from the stack top up to the platform's
# high edge, with its normal leaning back toward the spawn (the same attitude
# the drone has to hold to land on it).
# Every world gets the same tilt, `_TILT_DEG`, as a mesh variant: one run is
# one slope, deliberately, because the real tower has one.

# Box colours.
_BOX_RGBA = (0.72, 0.52, 0.28, 1.0)
_TOP_RGBA = (0.86, 0.70, 0.34, 1.0)

# `_WEDGE_HALF_L` / `_PLATFORM_TILT_DEG` / `_STACK_TOP` live at the top of the
# file, so changing the slope range is one tuple.
# The whole tower is built in a flat local frame and yawed onto the bearing,
# so every part (boxes and wedge alike) squarely faces the incoming drone.
_TOWER_QUAT = (
  math.cos(0.5 * _APPROACH_YAW),
  0.0,
  0.0,
  math.sin(0.5 * _APPROACH_YAW),
)

# Local frame: a plate whose top face carries the slope, with the low edge at
# the body origin. Every tilt variant shares one body pose and differs only in
# the mesh, which is what mjlab's per-world mesh variants require.
#
# The plate has a constant thickness so even a flat platform is a closed solid:
# a zero-thickness sheet has coplanar vertices and MuJoCo refuses to build its
# convex hull.
_PLATFORM_THICKNESS = 0.02                           # m


def _wedge_verts(tilt: float) -> list[float]:
  rise = _platform_rise(tilt)
  half_w = _BOX_HALF[0]
  t = _PLATFORM_THICKNESS
  l = _PLATFORM_L
  return [
    0.0, -half_w, 0.0,      # 0 low near bottom
    0.0, +half_w, 0.0,      # 1 low far bottom
    l, -half_w, 0.0,        # 2 high near bottom
    l, +half_w, 0.0,        # 3 high far bottom
    0.0, -half_w, t,        # 4 low near top
    0.0, +half_w, t,        # 5 low far top
    l, -half_w, t + 2.0 * rise,   # 6 high near top
    l, +half_w, t + 2.0 * rise,   # 7 high far top
  ]


_WEDGE_FACES = [
  0, 1, 2, 1, 3, 2,     # bottom
  4, 6, 5, 5, 6, 7,     # sloped platform
  0, 5, 1, 0, 4, 5,     # low edge
  2, 3, 6, 3, 7, 6,     # high edge
  0, 2, 4, 2, 6, 4,     # near side
  1, 5, 3, 3, 5, 7,     # far side
]


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

# NOTE: the ground geom is created by MJLab's TerrainEntityCfg and is
# named "terrain" (see terrains/terrain_entity.py). Keep this pattern in
# sync if the terrain type changes.
_DRONE_GROUND_CONTACT_CFG = ContactSensorCfg(
  name="drone_ground_contact",
  primary=ContactMatch(mode="geom", pattern="drone_geom", entity="drone"),
  secondary=ContactMatch(mode="geom", pattern="terrain", entity=None),
  fields=("found", "force"),
  reduce="maxforce",
  num_slots=1,
  track_air_time=False,
)

# Drone <-> landing platform contact. Scoped to the wedge body, so touching the
# stack below does not count as a landing. Matched by body rather than geom
# because the per-world mesh variants rename the geom to a synthesized
# "mjlab/pad/..." name. `netforce` sums the whole contact patch, so a resting
# drone reads ~its own weight whatever the attitude.
_DRONE_PLATFORM_CONTACT_CFG = ContactSensorCfg(
  name="drone_platform_contact",
  primary=ContactMatch(mode="geom", pattern="drone_geom", entity="drone"),
  secondary=ContactMatch(mode="body", pattern="top", entity="tower_top"),
  fields=("found", "force", "normal", "tangent"),
  reduce="netforce",
  num_slots=1,
  track_air_time=False,
)

# Drone <-> tower boxes. The platform (`tower_top`'s `top` body) is deliberately
# NOT included: it is the landing surface, so penalising it would teach the
# drone not to touch down. The boxes are pure obstacles — a landing that needs
# them was not a landing.
_TOWER_BOX_CONTACT_CFGS = tuple(
  ContactSensorCfg(
    name=f"drone_box_{i}_contact",
    primary=ContactMatch(mode="geom", pattern="drone_geom", entity="drone"),
    secondary=ContactMatch(
      mode="geom", pattern="box_geom", entity=f"tower_box_{i}"
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    track_air_time=False,
  )
  for i in range(1, _BOX_COUNT + 1)
)
_TOWER_BOX_SENSORS = tuple(cfg.name for cfg in _TOWER_BOX_CONTACT_CFGS)


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_DRONE_XML))
  marker = spec.body("target_marker")
  marker.pos = list(_TARGET)
  # Shrink the target marker: at the platform centre the default 0.15 sphere
  # would swallow the landing surface.
  marker.geoms[0].size[:] = [0.04, 0.04, 0.04]
  return spec


_DRONE_ARTICULATION = EntityArticulationInfoCfg(
  # Motors only: the ARX term applies thrust and torque as one wrench, so these
  # actuators stay at ctrl = 0. They are kept so `_PLANT = "motors"` still runs.
  actuators=(
    XmlActuatorCfg(
      target_names_expr=_SITE_NAMES,
      transmission_type=TransmissionType.SITE,
    ),
  ),
)

_DRONE_INIT = EntityCfg.InitialStateCfg(
  pos=_SPAWN,
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)


def _get_drone_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_DRONE_ARTICULATION,
    init_state=_DRONE_INIT,
  )


def _box_spec_fn(center: tuple[float, float, float]):
  """Spec for one loose box of the tower."""

  def build() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box", pos=center, quat=_TOWER_QUAT)
    body.add_freejoint(name="root")
    body.add_geom(
      name="box_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=_BOX_HALF,
      mass=_TOWER_MASS,
      rgba=_BOX_RGBA,
    )
    return spec

  return build


def _low_edge_pos() -> tuple[float, float, float]:
  """World position of the platform's low edge, i.e. its body origin.

  The wedge is pinned here (on the stack top, half a platform back from the
  pad centre along the approach bearing) so every tilt variant can share the
  same body pose and only swap its mesh.
  """
  half = 0.5 * _PLATFORM_L
  return (
    _PAD_XY[0] - half * math.cos(_APPROACH_YAW),
    _PAD_XY[1] - half * math.sin(_APPROACH_YAW),
    _STACK_TOP,
  )


def _wedge_spec_fn(tilt: float):
  """Spec for a platform tilted by `tilt` radians.

  Body origin sits at the low edge, so the mesh alone carries the tilt and
  `VariantEntityCfg` can hand a different slope to every world.
  """

  def build() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(
      name="top", pos=_low_edge_pos(), quat=_TOWER_QUAT
    )
    body.add_freejoint(name="root")
    mesh = spec.add_mesh(
      name="wedge",
      uservert=_wedge_verts(tilt),
      userface=list(_WEDGE_FACES),
    )
    body.add_geom(
      name="top_geom",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh.name,
      mass=_TOWER_MASS,
      rgba=_TOP_RGBA,
    )
    return spec

  return build


# MuJoCo only builds a mesh's convex polytope when a *collision* geom uses it,
# and a variant entity binds just one geom (the template) at compile time. The
# other variants' meshes would then be collision-blind: the engine sees a mesh
# with no hull, which shows up as a platform that sinks into the stack and
# chatters. A hidden reference geom on a contact bit nothing else uses makes
# the compiler do the work while still generating no contacts at runtime.
_HULL_PROBE_CONTYPE = 2


def _add_variant_hull_probes(spec: mujoco.MjSpec) -> None:
  body = next(b for b in spec.bodies if b.name == "top")
  used = {
    geom.meshname
    for geom in body.geoms
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH
    and (int(geom.contype) or int(geom.conaffinity))
  }
  for mesh in spec.meshes:
    if mesh.name in used:
      continue
    probe = body.add_geom()
    probe.name = f"hull_probe_{mesh.name.replace('/', '_')}"
    probe.type = mujoco.mjtGeom.mjGEOM_MESH
    probe.meshname = mesh.name
    probe.contype = _HULL_PROBE_CONTYPE
    probe.conaffinity = 0
    probe.density = 0.0
    probe.rgba = (0.0, 0.0, 0.0, 0.0)


class _PlatformEntity(Entity):
  """Entity that compiles a collision hull for every platform variant."""

  def _build_spec(self) -> None:
    super()._build_spec()
    _add_variant_hull_probes(self._spec)


class _PlatformVariantCfg(VariantEntityCfg):
  """`VariantEntityCfg` whose entity patches in the missing variant hulls."""

  def build(self) -> Entity:
    return _PlatformEntity(self)


def _tower_entities() -> dict[str, EntityCfg]:
  entities: dict[str, EntityCfg] = {}
  for i in range(_BOX_COUNT):
    center = (_TARGET[0], _TARGET[1], (i + 0.5) * 2.0 * _BOX_HALF[2])
    entities[f"tower_box_{i + 1}"] = EntityCfg(
      spec_fn=_box_spec_fn(center),
      init_state=EntityCfg.InitialStateCfg(pos=center, rot=_TOWER_QUAT),
    )

  # One mesh per tilt exists, but every world is assigned the same one: the
  # tower is a single physical object with a single slope.
  variants = {
    f"tilt_{degree:g}": _wedge_spec_fn(math.radians(degree))
    for degree in _PLATFORM_TILT_DEG
  }
  if _TILT_DEG not in _PLATFORM_TILT_DEG:
    raise ValueError(
      f"_TILT_DEG = {_TILT_DEG} is not in _PLATFORM_TILT_DEG "
      f"{_PLATFORM_TILT_DEG}."
    )
  tilt_index = _PLATFORM_TILT_DEG.index(_TILT_DEG)
  assignment: dict[str, float] | Callable[[int], Sequence[int]] = (
    lambda n: [tilt_index] * n
  )

  entities["tower_top"] = _PlatformVariantCfg(
    variants=variants,
    assignment=assignment,
    init_state=EntityCfg.InitialStateCfg(
      pos=_low_edge_pos(), rot=_TOWER_QUAT
    ),
  )
  return entities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_pose(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """World pad centre and platform normal, one row per env.

  Every world shares the single tilt variant `_TILT_DEG` (mjlab assigns mesh
  variants when the simulation is built and never resamples them), so all rows
  are identical; the per-env shape is kept because every reward, observation
  and termination term indexes it per env.
  The pad centre is the midpoint of the sloped face; `_PLATFORM_RISE` holds
  the extra height each variant adds on top of the stack.
  """
  cached = getattr(env, "_pad_pose_cache", None)
  if cached is not None:
    return cached

  assignment = env.sim.world_to_variant.get("tower_top")
  if assignment is None:
    idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  else:
    idx = assignment.to(device=env.device, dtype=torch.long)

  dtype = torch.float32
  rise = torch.tensor(_PLATFORM_RISE, device=env.device, dtype=dtype)[idx]
  tilt = torch.tensor(_PLATFORM_TILTS, device=env.device, dtype=dtype)[idx]
  sin_tilt, cos_tilt = torch.sin(tilt), torch.cos(tilt)

  centre = torch.stack(
    (
      torch.full_like(rise, _PAD_XY[0]),
      torch.full_like(rise, _PAD_XY[1]),
      _STACK_TOP + _PLATFORM_THICKNESS + rise,
    ),
    dim=-1,
  )
  normal = torch.stack(
    (
      -sin_tilt * math.cos(_APPROACH_YAW),
      -sin_tilt * math.sin(_APPROACH_YAW),
      cos_tilt,
    ),
    dim=-1,
  )

  env._pad_pose_cache = (centre, normal)
  return centre, normal


def _target_tensor(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _pad_pose(env)[0]


def _target_up_tensor(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _pad_pose(env)[1]


def _tilt_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Angle (rad) between the drone's up axis and the target up direction.

  Yaw-invariant: only the body +Z axis has to match, which is all a landing
  on a symmetric quad needs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.root_link_quat_w                      # (N, 4) (w, x, y, z)
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

  # Third column of R(q): the body's local +Z axis in the world frame.
  up = torch.stack(
    (
      2.0 * (x * z + w * y),
      2.0 * (y * z - w * x),
      1.0 - 2.0 * (x * x + y * y),
    ),
    dim=-1,
  )
  cos = torch.clamp((up * _target_up_tensor(env)).sum(dim=-1), -1.0, 1.0)
  return torch.acos(cos)


def _platform_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "drone_platform_contact",
) -> torch.Tensor:
  """True where the drone is touching the landing platform with real force."""
  sensor: ContactSensor = env.scene.sensors[sensor_name]
  data = sensor.data
  if data.force is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  force_mag = torch.norm(data.force, dim=-1)
  return (force_mag > 1.0).any(dim=-1)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _log_half_width(pct: float) -> float:
  """Log-space half-width for a +-`pct` multiplicative randomization.

  `dr.pseudo_inertia` scales mass by ``exp(2*alpha)`` and each principal moment
  by ``exp(2*d)``, so a +-10 % change needs a +-0.5*ln(1.1) half-width.
  """
  return 0.5 * math.log1p(pct)


# Motors only: `ArxAction` above replaces this whole chain (delay, gain, lag,
# site forces, rotor drag) with one filter plus a rate servo.
@dataclass(kw_only=True)
class DrThrustActionCfg(SiteEffortActionCfg):
  """Site effort control with per-env thrust gain, motor lag and delay.

  `SiteEffortAction` writes the policy output straight to the actuator. The
  real chain has three extra pieces: the thrust coefficient is not known
  exactly, the motors spool up with a first-order lag (~35 ms), and the
  command arrives with a transport delay (~15 ms). All three need per-env
  state, which a stateless term cannot express, so they live in this term.
  """

  dr_scale: float = _DR_SCALE
  gain_pct: float = _DR_THRUST_PCT
  tau_nominal_s: float = _NOM_TAU_S
  tau_pct: float = _DR_TAU_PCT
  delay_nominal_s: float = _NOM_DELAY_S
  delay_abs_s: float = _DR_DELAY_ABS

  def build(self, env: ManagerBasedRlEnv) -> DrThrustAction:
    return DrThrustAction(self, env)


class DrThrustAction(SiteEffortAction):
  """`SiteEffortAction` plus gain, first-order lag and transport delay.

  ``apply_actions`` runs once per physics substep (2 ms), which is where the
  lag is integrated and the delay line is clocked; one policy step spans
  ``decimation`` substeps, so the filter is resolved at 500 Hz.
  """

  cfg: DrThrustActionCfg

  def __init__(self, cfg: DrThrustActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._dt = float(env.physics_dt)
    n = self.num_envs

    # Per-env actuation parameters, resampled on every episode reset.
    self._gain = torch.ones(n, 1, device=self.device)
    self._tau = torch.zeros(n, 1, device=self.device)
    self._delay_steps = torch.zeros(n, 1, dtype=torch.long, device=self.device)

    # Delay line. Length comes from the knob, not from the sample, so the
    # buffer never has to be reallocated mid-episode.
    max_delay_s = cfg.delay_nominal_s + cfg.delay_abs_s
    self._delay_len = max(1, round(max_delay_s / self._dt) + 1)
    self._history = torch.zeros(
      n, self._delay_len, self.action_dim, device=self.device
    )
    self._cursor = 0

    self._filtered = torch.zeros(n, self.action_dim, device=self.device)
    self._damp = torch.tensor(
      [_DAMP_ROLL, _DAMP_PITCH, _DAMP_YAW], device=self.device
    )
    self._sample(torch.arange(n, device=self.device))

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)
    if isinstance(env_ids, torch.Tensor):
      ids = env_ids.to(self.device)
    else:
      ids = torch.arange(self.num_envs, device=self.device)
    self._history[ids] = 0.0
    self._filtered[ids] = 0.0
    self._sample(ids)

  def _sample(self, ids: torch.Tensor) -> None:
    """Draw this episode's gain, time constant and delay.

    `dr_scale` scales the spread only: at 0 the aircraft is deterministic but
    still has the nominal motor lag and transport delay.
    """
    s = self.cfg.dr_scale
    k = ids.numel()

    u = torch.rand((k, 1), device=self.device)
    self._gain[ids] = 1.0 + s * self.cfg.gain_pct * (2.0 * u - 1.0)

    u = torch.rand((k, 1), device=self.device)
    tau = self.cfg.tau_nominal_s * (
      1.0 + s * self.cfg.tau_pct * (2.0 * u - 1.0)
    )
    self._tau[ids] = tau.clamp_min(0.0)

    u = torch.rand((k, 1), device=self.device)
    delay = self.cfg.delay_nominal_s + s * self.cfg.delay_abs_s * (2.0 * u - 1.0)
    steps = torch.round(delay.clamp_min(0.0) / self._dt)
    self._delay_steps[ids] = steps.clamp(max=self._delay_len - 1).long()

  def apply_actions(self) -> None:
    # Transport delay: push the command, read back the one from `delay_steps`
    # ago. Delay 0 reads the sample written on this very call.
    self._history[:, self._cursor] = self._processed_actions
    idx = torch.remainder(
      self._cursor - self._delay_steps[:, 0], self._delay_len
    )
    cmd = self._history[torch.arange(self.num_envs, device=self.device), idx]
    self._cursor = (self._cursor + 1) % self._delay_len

    # Thrust-coefficient uncertainty.
    cmd = cmd * self._gain

    # First-order motor lag. tau = 0 collapses to the identity (alpha = 1).
    alpha = 1.0 - torch.exp(-self._dt / self._tau.clamp_min(1e-6))
    self._filtered.add_(alpha * (cmd - self._filtered))

    self._entity.set_site_effort_target(
      self._filtered, site_ids=self._target_ids
    )
    self._apply_rotor_drag()

  def _apply_rotor_drag(self) -> None:
    """Aerodynamic torque opposing body rotation.

    This is a real torque on the body (``xfrc_applied``), not a clamp: it grows
    linearly with the body rate, so `_MAX_BODY_RATE` is the rate at which it
    exactly cancels the maximum rotor torque. Holding 500 deg/s therefore costs
    full saturation, and exceeding it is impossible.

    ``apply_actions`` is the only hook that runs once per physics *substep*,
    which is what the roll/pitch time constants (~34 ms) need -- a 50 Hz event
    would badly distort them.
    """
    entity = self._entity
    quat = entity.data.root_link_quat_w
    rate_b = quat_apply_inverse(quat, entity.data.root_link_ang_vel_w)
    torque_w = quat_apply(quat, -rate_b * self._damp)
    entity.write_external_wrench_to_sim(
      forces=None,
      torques=torque_w.unsqueeze(1),
      body_ids=[0],
    )


# ---------------------------------------------------------------------------
# CTBR + ARX action term
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class ArxActionCfg(ActionTermCfg):
  """Collective-thrust + body-rate command through a first-order ARX filter.

  Elijah's model: four decoupled channels [p, q, r, c], each a one-pole lag
  ``x <- a*x + (1-a)*u`` with ``a = exp(-dt/tau)``. That is the *closed-loop*
  response of the flight controller, which is why it transfers to hardware:
  on a real drone the FC closes this loop for you.

  The last step differs. Elijah writes the filtered rate straight onto the
  body, which discards contact torque. Here the filter produces a *reference*
  and a rate servo produces the torque to track it, clipped at what four
  rotors can deliver. Free flight then matches his model; a hard landing can
  still overpower the aircraft and tip it over.
  """

  rate_max_pq: float = _ARX_RATE_MAX_PQ
  rate_max_r: float = _ARX_RATE_MAX_R
  tau_pq_s: float = _ARX_TAU_PQ_S
  tau_r_s: float = _ARX_TAU_R_S
  tau_c_s: float = _ARX_TAU_C_S
  delay_s: float = _ARX_DELAY_S
  servo_alpha: float = _ARX_SERVO_ALPHA
  dr_scale: float = _DR_SCALE
  tau_pct: float = _DR_TAU_PCT
  thrust_pct: float = _DR_THRUST_PCT
  delay_abs_s: float = _DR_DELAY_ABS

  def build(self, env: ManagerBasedRlEnv) -> ArxAction:
    return ArxAction(self, env)


class ArxAction(ActionTerm):
  """``[thrust, p, q, r]`` -> delay -> ARX lag -> rate reference -> torque.

  The action is FRD, matching the flight controller's axes: +p is
  right-wing-down, +q is nose-up, +r is nose-right (right-hand rule about
  front/right/down), and thrust is a scalar along the body's up axis (-z in
  FRD). The plant below runs in the simulator's FLU frame; ``process_actions``
  converts at the boundary.
  """

  cfg: ArxActionCfg

  def __init__(self, cfg: ArxActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._dt = float(env.physics_dt)
    n = self.num_envs

    # Raw action, physical command [p, q, r, c], filter state [p, q, r, c].
    self._raw = torch.zeros(n, 4, device=self.device)
    self._cmd = torch.zeros(n, 4, device=self.device)
    self._state = torch.zeros(n, 4, device=self.device)

    self._tau = torch.zeros(n, 4, device=self.device)
    self._thrust = torch.ones(n, device=self.device)
    self._delay_steps = torch.zeros(n, 4, dtype=torch.long, device=self.device)

    max_delay_s = cfg.delay_s + cfg.delay_abs_s
    self._delay_len = max(1, round(max_delay_s / self._dt) + 1)
    self._history = torch.zeros(n, self._delay_len, 4, device=self.device)
    self._cursor = 0

    # Peak torque available from four rotors at full differential thrust.
    self._torque_max = torch.tensor(
      [
        2.0 * _ARM * _THRUST_MAX,
        2.0 * _ARM * _THRUST_MAX,
        2.0 * _YAW_COEFF * _THRUST_MAX,
      ],
      device=self.device,
    )
    self._body_id = int(self._entity.indexing.root_body_id)

    all_ids = torch.arange(n, device=self.device)
    self._sample(all_ids)
    self._prime_hover(all_ids)

  @property
  def action_dim(self) -> int:
    return 4

  @property
  def channel_names(self) -> tuple[str, ...]:
    """Action-channel names, in Elijah's order: ``[thrust, p, q, r]``."""
    return ("thrust", "p", "q", "r")

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions.clamp(-1.0, 1.0)
    # The policy speaks FRD (x forward, y right, z down), the servo compares
    # against FLU body rates, so pitch and yaw flip sign. Roll and thrust do
    # not: +p is right-wing-down in both, and thrust is a scalar along the
    # body's up axis (-z in FRD).
    self._cmd[:, 0] = self._raw[:, 1] * self.cfg.rate_max_pq    # p
    self._cmd[:, 1] = -self._raw[:, 2] * self.cfg.rate_max_pq   # q: + is nose-up
    self._cmd[:, 2] = -self._raw[:, 3] * self.cfg.rate_max_r    # r: + is nose-right
    self._cmd[:, 3] = (1.0 + self._raw[:, 0]) * self._thrust    # c, in g

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if isinstance(env_ids, torch.Tensor):
      ids = env_ids.to(self.device)
    else:
      ids = torch.arange(self.num_envs, device=self.device)
    self._sample(ids)
    self._prime_hover(ids)

  def _prime_hover(self, ids: torch.Tensor) -> None:
    """Start at rest with the collective already holding hover.

    Without this every episode begins in free fall for one thrust time
    constant, which is not what the aircraft does when it is already airborne.
    """
    self._state[ids] = 0.0
    self._state[ids, 3] = 1.0
    self._history[ids] = 0.0
    # The filter runs on the *command*, so the command that produces a hover
    # output is 1/thrust_scale on a weak aircraft.
    self._history[ids, :, 3] = (1.0 / self._thrust[ids]).unsqueeze(-1)

  def _sample(self, ids: torch.Tensor) -> None:
    s = self.cfg.dr_scale
    k = ids.numel()
    nominal = torch.tensor(
      [
        self.cfg.tau_pq_s,
        self.cfg.tau_pq_s,
        self.cfg.tau_r_s,
        self.cfg.tau_c_s,
      ],
      device=self.device,
    )

    u = torch.rand((k, 4), device=self.device)
    tau = nominal * (1.0 + s * self.cfg.tau_pct * (2.0 * u - 1.0))
    self._tau[ids] = tau.clamp_min(1e-4)

    u = torch.rand((k, 4), device=self.device)
    delay = self.cfg.delay_s + s * self.cfg.delay_abs_s * (2.0 * u - 1.0)
    steps = torch.round(delay.clamp_min(0.0) / self._dt)
    self._delay_steps[ids] = steps.clamp(max=self._delay_len - 1).long()

    # Thrust-to-weight: the payload / battery draw. Per episode, not per step.
    u = torch.rand((k,), device=self.device)
    self._thrust[ids] = 1.0 + s * self.cfg.thrust_pct * (2.0 * u - 1.0)

  def apply_actions(self) -> None:
    # Transport delay, per channel and per env. Delay 0 reads the sample
    # written on this very call.
    self._history[:, self._cursor] = self._cmd
    idx = torch.remainder(
      self._cursor - self._delay_steps, self._delay_len
    ).unsqueeze(1)
    delayed = self._history.gather(1, idx).squeeze(1)
    self._cursor = (self._cursor + 1) % self._delay_len

    # One-pole ARX per channel: x <- a*x + (1-a)*u, unit DC gain.
    a = torch.exp(-self._dt / self._tau)
    self._state = a * self._state + (1.0 - a) * delayed

    quat = self._entity.data.root_link_quat_w
    omega_b = quat_apply_inverse(quat, self._entity.data.root_link_ang_vel_w)

    # Rate servo. The gain is I / tau_servo, so the closed-loop rate error
    # decays by `servo_alpha` per substep. Clipped at the airframe's peak
    # differential torque, which is what lets a propeller strike win.
    inertia = self._entity.data.model.body_inertia[:, self._body_id]
    torque_b = (self.cfg.servo_alpha / self._dt) * inertia * (
      self._state[:, :3] - omega_b
    )
    torque_b = torch.clamp(torque_b, -self._torque_max, self._torque_max)
    torque_w = quat_apply(quat, torque_b)

    # Thrust along body +z, in g, at the body origin (which is the COM here).
    mass = self._entity.data.model.body_mass[:, self._body_id]
    force_b = torch.zeros(self.num_envs, 3, device=self.device)
    force_b[:, 2] = self._state[:, 3] * mass * _ARX_GRAVITY
    force_w = quat_apply(quat, force_b)

    self._entity.write_external_wrench_to_sim(
      forces=force_w.unsqueeze(1),
      torques=torque_w.unsqueeze(1),
      body_ids=[0],
    )


# ---------------------------------------------------------------------------
# Domain randomization events
# ---------------------------------------------------------------------------


def _reset_drone_around_platform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  distance_near: float = _SPAWN_DISTANCE_NEAR,
  distance_far: float = _SPAWN_DISTANCE_FAR,
  height_abs: float = _SPAWN_HEIGHT_ABS,
  speed_abs: float = _SPAWN_SPEED_ABS,
  lean_abs: float = _SPAWN_LEAN_ABS,
  heading_abs: float = _SPAWN_HEADING_ABS,
  spin_abs: float = _SPAWN_SPIN_ABS,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> None:
  """Drop the drone at a random point in the spawn annulus."""
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  device = env.device

  def symmetric(count: int, half: float) -> torch.Tensor:
    """Uniform in [-half, half]; exactly 0.0 when `half` is 0.0."""
    return half * (2.0 * torch.rand(count, device=device) - 1.0)

  # Uniform over *area*, not over radius: taking sqrt keeps the density
  # constant per square metre, so a far spawn is as likely as a near one. A
  # plain uniform radius would pile the samples onto the near edge.
  u = torch.rand(n, device=device)
  dist = torch.sqrt(
    distance_near**2 + u * (distance_far**2 - distance_near**2)
  )
  angle = 2.0 * math.pi * torch.rand(n, device=device)
  pad_xy = torch.tensor(_TARGET[:2], device=device, dtype=torch.float32)
  xy = pad_xy + dist.unsqueeze(-1) * torch.stack(
    (torch.cos(angle), torch.sin(angle)), dim=-1
  )
  pos = torch.cat(
    (xy, (_SPAWN[2] + symmetric(n, height_abs)).unsqueeze(-1)), dim=-1
  )
  # Deliberately NOT offset by env_origins. The tower is one object per world,
  # built at world `_TARGET` by `_tower_entities()`, and every term (reward,
  # observation, success) reads world coordinates. Adding the origin would move
  # the spawn area to `origin + _TARGET` while the tower stays at `_TARGET`, so
  # envs would spawn 0.4-34 m from the pad (measured) instead of around it.
  # `env_origins` is a grid bookkeeping offset that nothing else in this scene
  # applies.

  quat = quat_from_euler_xyz(
    symmetric(n, lean_abs), symmetric(n, lean_abs), symmetric(n, heading_abs)
  )

  asset.write_root_link_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=env_ids)
  vel = speed_abs * (2.0 * torch.rand(n, 3, device=device) - 1.0)
  spin = spin_abs * (2.0 * torch.rand(n, 3, device=device) - 1.0)
  asset.write_root_link_velocity_to_sim(
    torch.cat((vel, spin), dim=-1), env_ids=env_ids
  )


def _dr_events() -> dict[str, EventTermCfg]:
  """Startup / reset randomization.

  `reset_base` is always present: the spawn ring belongs to the task, not to
  the aircraft, so it does not scale with `_DR_SCALE`. Everything below it is
  empty when `_DR_SCALE` is 0.
  """
  events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
      func=_reset_drone_around_platform,
      mode="reset",
      params={
        "distance_near": _SPAWN_DISTANCE_NEAR,
        "distance_far": _SPAWN_DISTANCE_FAR,
        "height_abs": _SPAWN_HEIGHT_ABS,
        "speed_abs": _SPAWN_SPEED_ABS,
        "lean_abs": _SPAWN_LEAN_ABS,
        "heading_abs": _SPAWN_HEADING_ABS,
        "spin_abs": _SPAWN_SPIN_ABS,
        "asset_cfg": _DRONE_CFG,
      },
    ),
    # Sensor error is per episode, not per step, so it is drawn here rather
    # than by the noise pipeline (which has no reset hook of its own).
    "reset_obs_bias": EventTermCfg(func=_reset_obs_corruption, mode="reset"),
  }
  if _DR_SCALE <= 0.0:
    return events

  half_mass = _log_half_width(_DR_SCALE * _DR_MASS_PCT)
  half_inertia = _log_half_width(_DR_SCALE * _DR_INERTIA_PCT)

  # Mass, inertia and COM in one physically consistent draw: `alpha` is a
  # density change (mass and inertia move together), `d` is a shape change
  # that leaves mass alone, `t` shifts the COM.
  events["dr_body"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("drone", body_names=("drone",)),
      "alpha_range": (-half_mass, half_mass),
      "d_range": (-half_inertia, half_inertia),
      "t_range": (-_DR_SCALE * _DR_COM_ABS, _DR_SCALE * _DR_COM_ABS),
    },
  )
  # Pad friction. Only the landing geoms: the stack below is never touched, and
  # randomizing it too would just add a knob with no effect.
  lo, hi = _DR_FRICTION
  mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
  events["dr_friction"] = EventTermCfg(
    func=dr.geom_friction,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("tower_top", geom_names=(".*",)),
      "ranges": (mid - _DR_SCALE * half, mid + _DR_SCALE * half),
      "operation": "abs",
    },
  )
  return events


# ---------------------------------------------------------------------------
# Frame convention
# ---------------------------------------------------------------------------
# The simulator is authored FLU: x forward, y left, z up, because MuJoCo's
# gravity and the drone model are z-up. The real aircraft is FRD: x forward,
# y right, z down -- that is the frame every flight controller uses, and it is
# the frame the log this plant was identified from is logged in (`gyroADC`).
#
# So the *policy interface* is FRD and the *physics* stays FLU. The boundary
# converts. FLU -> FRD is the fixed rotation diag(1, -1, -1): 180 degrees about
# the body x axis, which leaves the forward axis and every roll quantity alone
# and flips the lateral and vertical ones.
#
# Converting here rather than in the deployment bridge means an identified tau
# transfers to the real aircraft without a sign table, and that a sign bug is
# a bug in one place instead of two.


def _flu_to_frd(v: torch.Tensor) -> torch.Tensor:
  """Express a body-frame vector (FLU) in FRD. Forward/roll are unchanged."""
  return torch.stack((v[..., 0], -v[..., 1], -v[..., 2]), dim=-1)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class _SharedTiltBias:
  """Per-episode attitude error, worn by `gravity` and `target_up` alike.

  Both are the same estimated rotation applied to different world vectors, so
  on a real aircraft they are wrong *together*. Sampling them separately would
  let the two disagree about which way is up, which no estimator can do, and
  the policy would spend capacity on a distinction that does not exist.

  Applied as a small rotation (``v + delta x v``), not an additive offset: a
  component along ``v`` would change the length of a unit vector, and no
  attitude filter outputs a 1.03-long "unit" vector.
  """

  def __init__(self, std_deg: float = _OBS_TILT_BIAS_DEG) -> None:
    self._std = math.radians(std_deg)
    self._delta: torch.Tensor | None = None

  def resample(
    self,
    env_ids: torch.Tensor | slice | None,
    num_envs: int,
    device: torch.device | str,
  ) -> None:
    if self._delta is None or self._delta.shape[0] != num_envs:
      self._delta = torch.zeros((num_envs, 3), device=device)
    ids = slice(None) if env_ids is None else env_ids
    n = self._delta[ids].shape[0]
    self._delta[ids] = torch.randn((n, 3), device=device) * self._std

  def apply(self, v: torch.Tensor) -> torch.Tensor:
    if self._delta is None:  # first frame, before any reset event ran
      return v
    return v + torch.linalg.cross(self._delta[: v.shape[0]], v, dim=-1)


_tilt_bias = _SharedTiltBias()


def _reset_obs_corruption(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None
) -> None:
  """Give every new episode a fresh set of per-episode sensor errors."""
  _tilt_bias.resample(env_ids, env.num_envs, env.device)


def drone_rel_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(
    asset.data.root_link_pos_w - _target_tensor(env), -5.0, 5.0
  )


def drone_lin_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Linear velocity in the body frame, FRD: (forward, right, down)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(_flu_to_frd(asset.data.root_link_lin_vel_b), -5.0, 5.0)


def drone_ang_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Body rates in FRD: (p, q, r) = right-wing-down, nose-up, nose-right."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(_flu_to_frd(asset.data.root_link_ang_vel_b), -5.0, 5.0)


def target_up_body(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
  enable_tilt_bias: bool = True,
) -> torch.Tensor:
  """Target up direction expressed in the body frame, FRD (3,).

  Lets the policy see which way it has to tilt to reach the target attitude.
  On a level drone over a level platform this reads (0, 0, -1): up points
  against the down-axis.
  """
  asset: Entity = env.scene[asset_cfg.name]
  n = _target_up_tensor(env)
  v = _flu_to_frd(quat_apply_inverse(asset.data.root_link_quat_w, n))
  return _tilt_bias.apply(v) if enable_tilt_bias else v


def projected_gravity_frd(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
  enable_tilt_bias: bool = True,
) -> torch.Tensor:
  """Gravity direction in the body frame, FRD. Level flight reads (0, 0, 1)."""
  v = _flu_to_frd(projected_gravity(env, asset_cfg))
  return _tilt_bias.apply(v) if enable_tilt_bias else v


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def distance_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """-distance to the target: a straight line, closer is better.

  Clamped at ``_REWARD_ALIVE`` so that ``alive + distance`` is never
  negative: there is no reachable state where ending the episode beats
  surviving.
  """
  asset: Entity = env.scene[asset_cfg.name]
  dist = torch.norm(
    asset.data.root_link_pos_w - _target_tensor(env), dim=-1
  )
  return -torch.clamp(dist, max=_REWARD_ALIVE)


def speed_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  return -torch.clamp(speed, max=20.0)


def closing_speed(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Velocity component pointing at the target: positive when closing in.

  Rewards flying *toward* the goal (and so a straight approach), rather
  than merely being close to it.
  """
  asset: Entity = env.scene[asset_cfg.name]
  to_goal = _target_tensor(env) - asset.data.root_link_pos_w
  direction = to_goal / torch.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
  closing = (asset.data.root_link_lin_vel_w * direction).sum(dim=-1)
  return torch.clamp(closing, -10.0, 10.0)


def alive_bonus(
  env: ManagerBasedRlEnv,
  minimum_height: float = _MIN_HEIGHT,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """+1 while the drone stays above the minimum height."""
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.root_link_pos_w[:, 2] > minimum_height).float()


def crash_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str = "drone_ground_contact",
) -> torch.Tensor:
  """-1 on the step the drone contacts the ground.

  Reads the same contact sensor as the ``ground_collision`` termination,
  so the penalty and the reset fire on the same event.
  """
  sensor: ContactSensor = env.scene.sensors[sensor_name]
  data = sensor.data
  if data.force is None:
    return torch.zeros(env.num_envs, device=env.device)
  force_mag = torch.norm(data.force, dim=-1)
  return -(force_mag > 1.0).any(dim=-1).float()


def tower_collision_penalty(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...] = _TOWER_BOX_SENSORS,
  min_force: float = _TOWER_CONTACT_FORCE,
  platform_sensor: str = "drone_platform_contact",
) -> torch.Tensor:
  """-1 on the step the drone hits a tower box.

  Mirrors the ``tower_collision`` termination. The platform is not watched,
  so a genuine landing is unaffected.
  """
  return -tower_collision(env, sensor_names, min_force, platform_sensor).float()


def tilt_penalty(
  env: ManagerBasedRlEnv,
  max_tilt: float = _MAX_TILT_RAD,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """-1 on the step the drone tips past the tilt limit.

  Mirrors the ``tilted_over`` termination so the penalty and the reset
  fire on the same event.
  """
  return -tilted_over(env, max_tilt=max_tilt, asset_cfg=asset_cfg).float()


def attitude_penalty(
  env: ManagerBasedRlEnv,
  gate_radius: float = _TILT_GATE_RADIUS,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """-tilt_error (rad), ramped in over the last ``gate_radius`` metres.

  Gated so it only shapes the landing: applied everywhere it would fight the
  tilt the drone needs to accelerate and brake on the way in.
  """
  asset: Entity = env.scene[asset_cfg.name]
  err = _tilt_error(env, asset_cfg)
  dist = torch.norm(
    asset.data.root_link_pos_w - _target_tensor(env), dim=-1
  )
  gate = torch.clamp((gate_radius - dist) / gate_radius, 0.0, 1.0)
  return -err * gate


def runaway_penalty(
  env: ManagerBasedRlEnv,
  max_xy: float = _MAX_POS_XY,
  max_vel: float = _MAX_VEL,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """-1 on the step the drone flies out of bounds.

  Mirrors the ``runaway`` termination so the penalty and the reset fire
  on the same event.
  """
  return -runaway(env, max_xy=max_xy, max_vel=max_vel, asset_cfg=asset_cfg).float()


def timeout_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """-1 on the final step of an episode that reached the time limit.

  Mirrors the ``time_out`` termination. This is the mildest penalty:
  every other early exit (crash / tilt / runaway) costs more, so it only
  nudges the policy toward reaching the goal rather than merely surviving.
  """
  return -(env.episode_length_buf >= env.max_episode_length).float()


def rate_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  rate = torch.norm(asset.data.root_link_ang_vel_w, dim=-1)
  return -torch.clamp(rate, max=20.0)


def _at_goal(
  env: ManagerBasedRlEnv,
  pos_tol: float = _SUCCESS_POS_TOL,
  vel_tol: float = _SUCCESS_VEL_TOL,
  tilt_tol: float = _SUCCESS_TILT_TOL,
  require_contact: bool = _SUCCESS_REQUIRE_CONTACT,
  sensor_name: str = "drone_platform_contact",
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """True where the drone is at the target, stopped, and at the target tilt.

  With ``require_contact`` it must also be resting on the landing platform.
  Shared by the ``success`` reward and termination so both fire on the same
  step."""
  asset: Entity = env.scene[asset_cfg.name]
  pos_err = torch.norm(
    asset.data.root_link_pos_w - _target_tensor(env), dim=-1
  )
  speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  ok = (
    (pos_err < pos_tol)
    & (speed < vel_tol)
    & (_tilt_error(env, asset_cfg) < tilt_tol)
  )
  if require_contact:
    ok = ok & _platform_contact(env, sensor_name)
  return ok


def success_bonus(
  env: ManagerBasedRlEnv,
  pos_tol: float = _SUCCESS_POS_TOL,
  vel_tol: float = _SUCCESS_VEL_TOL,
  tilt_tol: float = _SUCCESS_TILT_TOL,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """+1 on the step the drone lands on the platform in the target attitude.

  Terminal bonus: the matching ``success`` termination ends the episode
  on the same step.
  """
  return _at_goal(env, pos_tol, vel_tol, tilt_tol, asset_cfg=asset_cfg).float()


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

def drone_ground_collision(
  env: ManagerBasedRlEnv,
  sensor_name: str = "drone_ground_contact",
) -> torch.Tensor:
  """Terminate when the drone contacts the ground with significant force."""
  sensor: ContactSensor = env.scene.sensors[sensor_name]
  data = sensor.data
  if data.force is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  force_mag = torch.norm(data.force, dim=-1)
  return (force_mag > 1.0).any(dim=-1)


def tower_collision(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...] = _TOWER_BOX_SENSORS,
  min_force: float = _TOWER_CONTACT_FORCE,
  platform_sensor: str = "drone_platform_contact",
) -> torch.Tensor:
  """Terminate when the drone hits a tower box while not landing on the platform.

  Only the boxes are watched; the tilted platform is the landing surface, so
  contact with it stays free of any penalty.

  The platform's low edge sits directly on top of box 3, so a firm touchdown
  briefly presses the drone's frame into that box. A box contact that happens
  while the drone is already supported by the platform is therefore treated as
  part of the landing -- only a box hit *without* platform support counts.
  """
  hit = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for name in sensor_names:
    data = env.scene.sensors[name].data
    if data.force is None:
      continue
    force_mag = torch.norm(data.force, dim=-1).reshape(env.num_envs, -1)
    hit |= (force_mag > min_force).any(dim=-1)
  return hit & ~_platform_contact(env, platform_sensor).reshape(env.num_envs)


def runaway(
  env: ManagerBasedRlEnv,
  max_xy: float = _MAX_POS_XY,
  max_vel: float = _MAX_VEL,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w
  vel = asset.data.root_link_lin_vel_w
  far = torch.norm(pos[:, :2], dim=-1) > max_xy
  fast = torch.norm(vel, dim=-1) > max_vel
  return far | fast


def tilted_over(
  env: ManagerBasedRlEnv,
  max_tilt: float = _MAX_TILT_RAD,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Safety net: terminate once the drone tilts past `max_tilt` from
  upright, so an inverted, unrecoverable drone doesn't spin out of
  control."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.root_link_quat_w                      # (N, 4) (w, x, y, z)
  x, y = q[:, 1], q[:, 2]

  # World-frame z component of the body's local +Z axis:
  # R[2, 2] = 1 - 2*(x*x + y*y), which equals cos(tilt from upright).
  up_z = 1.0 - 2.0 * (x * x + y * y)
  return up_z < math.cos(max_tilt)


def success(
  env: ManagerBasedRlEnv,
  pos_tol: float = _SUCCESS_POS_TOL,
  vel_tol: float = _SUCCESS_VEL_TOL,
  tilt_tol: float = _SUCCESS_TILT_TOL,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Terminate once the drone has landed on the platform in the target attitude.

  Goal reached — a true terminal state (not a time-out), so the value is
  not bootstrapped past it.
  """
  return _at_goal(env, pos_tol, vel_tol, tilt_tol, asset_cfg=asset_cfg)


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

def _make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "rel_pos": ObservationTermCfg(
      func=drone_rel_pos,
      noise=GaussianNoiseCfg(std=_OBS_NOISE_POS),
      delay_min_lag=_OBS_LAG_MOCAP,
      delay_max_lag=_OBS_LAG_MOCAP,
    ),
    "lin_vel": ObservationTermCfg(
      func=drone_lin_vel,
      params={"asset_cfg": _DRONE_CFG},
      noise=GaussianNoiseCfg(std=_OBS_NOISE_VEL),
      delay_min_lag=_OBS_LAG_MOCAP,
      delay_max_lag=_OBS_LAG_MOCAP,
    ),
    "ang_vel": ObservationTermCfg(
      func=drone_ang_vel,
      params={"asset_cfg": _DRONE_CFG},
      # White noise plus a bias that holds for the whole episode: a real gyro
      # is precise but never perfectly zeroed, and the bias is what a
      # calibration step is for.
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(std=_OBS_NOISE_RATE),
        bias_noise_cfg=UniformNoiseCfg(
          n_min=-_OBS_RATE_BIAS, n_max=_OBS_RATE_BIAS
        ),
      ),
    ),
    "gravity": ObservationTermCfg(
      func=projected_gravity_frd,
      params={"asset_cfg": _DRONE_CFG, "enable_tilt_bias": not play},
      noise=GaussianNoiseCfg(std=_OBS_NOISE_ATT),
      delay_min_lag=_OBS_LAG_ATT,
      delay_max_lag=_OBS_LAG_ATT,
    ),
    "target_up": ObservationTermCfg(
      func=target_up_body,
      params={"asset_cfg": _DRONE_CFG, "enable_tilt_bias": not play},
      noise=GaussianNoiseCfg(std=_OBS_NOISE_ATT),
      delay_min_lag=_OBS_LAG_ATT,
      delay_max_lag=_OBS_LAG_ATT,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = (
    {"arx": ArxActionCfg(entity_name="drone")}
    if _PLANT == "arx"
    else {
      "rotor_thrusts": DrThrustActionCfg(
        entity_name="drone",
        actuator_names=_SITE_NAMES,
        scale=1.0,
      )
    }
  )

  rewards = {
    "distance": RewardTermCfg(
      func=distance_penalty, weight=_REWARD_DISTANCE
    ),
    "speed": RewardTermCfg(func=speed_penalty, weight=_REWARD_SPEED),
    "closing": RewardTermCfg(func=closing_speed, weight=_REWARD_CLOSING),
    "rate": RewardTermCfg(func=rate_penalty, weight=_REWARD_RATE),
    "alive": RewardTermCfg(func=alive_bonus, weight=_REWARD_ALIVE),
    "attitude": RewardTermCfg(func=attitude_penalty, weight=_REWARD_TILT),
    "crash": RewardTermCfg(func=crash_penalty, weight=_PENALTY_CRASH),
    "tower_collision": RewardTermCfg(
      func=tower_collision_penalty, weight=_PENALTY_TOWER
    ),
    "tilt": RewardTermCfg(func=tilt_penalty, weight=_PENALTY_TILT),
    "runaway": RewardTermCfg(func=runaway_penalty, weight=_PENALTY_RUNAWAY),
    "timeout": RewardTermCfg(func=timeout_penalty, weight=_PENALTY_TIMEOUT),
    "success": RewardTermCfg(func=success_bonus, weight=_REWARD_SUCCESS),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "ground_collision": TerminationTermCfg(func=drone_ground_collision),
    "tower_collision": TerminationTermCfg(func=tower_collision),
    "runaway": TerminationTermCfg(func=runaway),
    "tilted_over": TerminationTermCfg(func=tilted_over),
    "success": TerminationTermCfg(func=success),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"drone": _get_drone_cfg(), **_tower_entities()},
      sensors=(
        _DRONE_GROUND_CONTACT_CFG,
        _DRONE_PLATFORM_CONTACT_CFG,
        *_TOWER_BOX_CONTACT_CFGS,
      ),
      num_envs=1,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    events=_dr_events(),
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="drone",
      body_name="drone",
      distance=3.0,
      elevation=-15.0,
      azimuth=0.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.002),
      # The tower adds several simultaneous contacts (stack + drone), which
      # overflows the default constraint heuristic.
      njmax=300,
      nconmax=100,
    ),
    decimation=10,
    episode_length_s=_EPISODE_LENGTH_S,
  )


def drone_tilted_landing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(play=play)
  if play:
    cfg.observations["actor"].enable_corruption = False
    # Domain randomization must never leak into an evaluation: no randomized
    # body and no randomized actuation. The nominal motor lag and transport
    # delay stay on -- the policy is trained with them. The spawn stays
    # randomized, because an evaluation should show the policy on fresh
    # bearings. The episode keeps its training length (10 s).
    cfg.events = {"reset_base": cfg.events["reset_base"]}
    if _PLANT == "arx":
      arx_cfg = cfg.actions["arx"]
      assert isinstance(arx_cfg, ArxActionCfg)
      arx_cfg.dr_scale = 0.0
    else:
      action_cfg = cfg.actions["rotor_thrusts"]
      assert isinstance(action_cfg, DrThrustActionCfg)
      action_cfg.dr_scale = 0.0
  return cfg


def drone_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      # PPO plus a penalty on the change in the mean action between two
      # consecutive control steps -- the jitter is addressed in the loss, not
      # in the reward. See `smooth_ppo.py`; tune with ARX_SMOOTH_COEF.
      class_name="mjlab.tasks.tilted_landing_arx.smooth_ppo:SmoothPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.002,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.995,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="drone_tilted_landing_arx",
    save_interval=50,
    num_steps_per_env=64,
    max_iterations=500,
    logger="tensorboard",
  )
