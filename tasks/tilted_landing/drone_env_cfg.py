"""Drone tilted-landing environment configuration for MJLab."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import math

import mujoco
import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  projected_gravity,
  time_out,
)
from mjlab.envs.mdp.actions import SiteEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
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
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRONE_XML: Path = Path(__file__).parent / "drone.xml"
_DRONE_CFG = SceneEntityCfg("drone")

# Landing point: the centre of the tower's tilted top face. Below the spawn
# altitude so the drone approaches *above* the tower and descends onto it,
# instead of flying into the stack's side.
_SPAWN = (0.0, 0.0, 1.0)

# --- Tower geometry -------------------------------------------------------
# Three loose boxes stacked under the tilted platform. Declared up here because
# the platform height (and therefore the target) is derived from the stack.
_BOX_HALF = (0.12, 0.12, 0.07)                       # 0.24 x 0.24 footprint
_BOX_COUNT = 3
_TOWER_MASS = 0.5                                    # kg, per loose piece
_STACK_TOP = _BOX_COUNT * 2.0 * _BOX_HALF[2]         # 0.42 m

# Half-length of the platform along the slope. Held fixed across curriculum
# stages so the pad keeps its footprint and the tilt only changes its height.
# (Pinning the height instead would stretch a shallow ramp into a 0.9 m
# seesaw balanced on a 0.24 m box.)
_WEDGE_HALF_L = 0.138564                             # m

# Curriculum: stage 1 = 10 deg, stage 2 = 20 deg, stage 3 = 30 deg.
_TARGET_TILT_RAD = math.radians(10.0)

# Half-height of the wedge, derived so the platform's low edge stays pinned to
# the stack top while the high edge rises by 2 * _WEDGE_HALF_H.
_WEDGE_HALF_H = _WEDGE_HALF_L * math.tan(_TARGET_TILT_RAD)

# Platform midpoint. The wedge body pose *is* the landing pose.
_TARGET = (2.0, 2.0, _STACK_TOP + _WEDGE_HALF_H)

# Horizontal bearing from the spawn to the target. The drone flies in along it
# and flares (leans its top back) to brake; tilting about the perpendicular
# axis leans the target the same way, so the attitude it has to reach is close
# to the flare it already makes. The tower top is built with the same bearing,
# so its platform normal is exactly the attitude the drone has to hold.
_APPROACH_YAW = math.atan2(_TARGET[1] - _SPAWN[1], _TARGET[0] - _SPAWN[0])

# World-frame unit vector the drone's body +Z should point along: up, leaning
# back toward the spawn. Doubles as the tower platform's normal.
_TARGET_UP = (
  -math.sin(_TARGET_TILT_RAD) * math.cos(_APPROACH_YAW),
  -math.sin(_TARGET_TILT_RAD) * math.sin(_APPROACH_YAW),
  math.cos(_TARGET_TILT_RAD),
)

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
_SUCCESS_POS_TOL = 0.25     # m
_SUCCESS_VEL_TOL = 0.15     # m/s
# Curriculum: 12 deg on the 10 deg ramp, then 10 deg on the 20 deg ramp,
# then 5 deg on the final 30 deg ramp. Keep it a bit looser than the ramp
# while the policy is still learning to arrive tilted at all.
_SUCCESS_TILT_TOL = math.radians(12.0)
# Require the drone to actually touch the platform, not just fly through the
# landing pose. Set False to let "at the pose, slow, tilted" count on its own.
_SUCCESS_REQUIRE_CONTACT = True


# ---------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------
# Four loose bodies: three stacked boxes plus a wedge on top. Nothing is
# welded, so the drone can tumble the stack. mjlab allows at most one freejoint
# per Entity, so each loose body is its own Entity; with no reset event they
# reset to their stacked poses from the model default every episode.
#
# The wedge's sloped face is the landing platform. Its profile is a right
# triangle whose hypotenuse is the platform: it runs from the stack top up to
# the platform's high edge, with the platform's midpoint at `_TARGET` and a
# normal of exactly `_TARGET_UP`.

# Box colours.
_BOX_RGBA = (0.72, 0.52, 0.28, 1.0)
_TOP_RGBA = (0.86, 0.70, 0.34, 1.0)

# `_WEDGE_HALF_L` / `_WEDGE_HALF_H` / `_STACK_TOP` are derived at the top of
# the file, next to `_TARGET_TILT_RAD`, so a curriculum stage is one constant.
# The whole tower is built in a flat local frame and yawed onto the bearing,
# so every part (boxes and wedge alike) squarely faces the incoming drone.
_TOWER_QUAT = (
  math.cos(0.5 * _APPROACH_YAW),
  0.0,
  0.0,
  math.sin(0.5 * _APPROACH_YAW),
)

# Local frame: right triangle in x-z, extruded +-0.12 along y.
# 0/1 = near-bottom edge, 2/3 = far-bottom edge, 4/5 = far-top edge.
_WEDGE_VERTS = [
  -_WEDGE_HALF_L, -_BOX_HALF[0], -_WEDGE_HALF_H,
  -_WEDGE_HALF_L, +_BOX_HALF[0], -_WEDGE_HALF_H,
  +_WEDGE_HALF_L, -_BOX_HALF[0], -_WEDGE_HALF_H,
  +_WEDGE_HALF_L, +_BOX_HALF[0], -_WEDGE_HALF_H,
  +_WEDGE_HALF_L, -_BOX_HALF[0], +_WEDGE_HALF_H,
  +_WEDGE_HALF_L, +_BOX_HALF[0], +_WEDGE_HALF_H,
]
_WEDGE_FACES = [
  0, 3, 2, 0, 1, 3,     # bottom
  2, 5, 4, 2, 3, 5,     # upright far face
  0, 5, 1, 0, 4, 5,     # sloped platform
  0, 2, 4,              # end cap
  1, 5, 3,              # end cap
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

# Drone <-> landing platform contact. Scoped to the wedge, so touching the
# stack below does not count as a landing. `netforce` sums the whole contact
# patch, so a resting drone reads ~its own weight whatever the attitude.
_DRONE_PLATFORM_CONTACT_CFG = ContactSensorCfg(
  name="drone_platform_contact",
  primary=ContactMatch(mode="geom", pattern="drone_geom", entity="drone"),
  secondary=ContactMatch(mode="geom", pattern="top_geom", entity="tower_top"),
  fields=("found", "force", "normal", "tangent"),
  reduce="netforce",
  num_slots=1,
  track_air_time=False,
)

# Drone <-> tower boxes. The platform (`tower_top` / `top_geom`) is deliberately
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


def _wedge_spec_fn():
  """Spec for the tilted platform capping the tower.

  Body origin sits at the platform's midpoint, so the body pose *is* the
  landing pose.
  """

  def build() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="top", pos=_TARGET, quat=_TOWER_QUAT)
    body.add_freejoint(name="root")
    mesh = spec.add_mesh(
      name="wedge",
      uservert=list(_WEDGE_VERTS),
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


def _tower_entities() -> dict[str, EntityCfg]:
  entities: dict[str, EntityCfg] = {}
  for i in range(_BOX_COUNT):
    center = (_TARGET[0], _TARGET[1], (i + 0.5) * 2.0 * _BOX_HALF[2])
    entities[f"tower_box_{i + 1}"] = EntityCfg(
      spec_fn=_box_spec_fn(center),
      init_state=EntityCfg.InitialStateCfg(pos=center, rot=_TOWER_QUAT),
    )
  entities["tower_top"] = EntityCfg(
    spec_fn=_wedge_spec_fn(),
    init_state=EntityCfg.InitialStateCfg(pos=_TARGET, rot=_TOWER_QUAT),
  )
  return entities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_tensor(asset: Entity) -> torch.Tensor:
  return torch.tensor(
    _TARGET,
    device=asset.data.root_link_pos_w.device,
    dtype=asset.data.root_link_pos_w.dtype,
  )


def _target_up_tensor(asset: Entity) -> torch.Tensor:
  return torch.tensor(
    _TARGET_UP,
    device=asset.data.root_link_pos_w.device,
    dtype=asset.data.root_link_pos_w.dtype,
  )


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
  cos = torch.clamp((up * _target_up_tensor(asset)).sum(dim=-1), -1.0, 1.0)
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
# Observations
# ---------------------------------------------------------------------------

def drone_rel_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(
    asset.data.root_link_pos_w - _target_tensor(asset), -5.0, 5.0
  )


def drone_lin_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(asset.data.root_link_lin_vel_b, -5.0, 5.0)


def drone_ang_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(asset.data.root_link_ang_vel_b, -5.0, 5.0)


def target_up_body(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Target up direction expressed in the body frame (3,).

  Lets the policy see which way it has to tilt to reach the target attitude.
  """
  asset: Entity = env.scene[asset_cfg.name]
  n = _target_up_tensor(asset).unsqueeze(0).expand(env.num_envs, -1)
  return quat_apply_inverse(asset.data.root_link_quat_w, n)


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
    asset.data.root_link_pos_w - _target_tensor(asset), dim=-1
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
  to_goal = _target_tensor(asset) - asset.data.root_link_pos_w
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
    asset.data.root_link_pos_w - _target_tensor(asset), dim=-1
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
    asset.data.root_link_pos_w - _target_tensor(asset), dim=-1
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

def _make_env_cfg() -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "rel_pos": ObservationTermCfg(func=drone_rel_pos),
    "lin_vel": ObservationTermCfg(
      func=drone_lin_vel,
      params={"asset_cfg": _DRONE_CFG},
    ),
    "ang_vel": ObservationTermCfg(
      func=drone_ang_vel,
      params={"asset_cfg": _DRONE_CFG},
    ),
    "gravity": ObservationTermCfg(
      func=projected_gravity,
      params={"asset_cfg": _DRONE_CFG},
    ),
    "target_up": ObservationTermCfg(
      func=target_up_body,
      params={"asset_cfg": _DRONE_CFG},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "rotor_thrusts": SiteEffortActionCfg(
      entity_name="drone",
      actuator_names=_SITE_NAMES,
      scale=1.0,
    ),
  }

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
    events={},
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
  cfg = _make_env_cfg()
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
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
    experiment_name="drone_tilted_landing",
    save_interval=50,
    num_steps_per_env=64,
    max_iterations=500,
    logger="tensorboard",
  )