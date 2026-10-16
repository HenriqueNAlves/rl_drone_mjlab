"""Drone hover environment configuration for MJLab."""

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
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRONE_XML: Path = Path(__file__).parent / "drone.xml"
_DRONE_CFG = SceneEntityCfg("drone")

_TARGET = (2.0, 2.0, 1.0)
_SPAWN = (0.0, 0.0, 1.0)

_SITE_NAMES = ("rotor_fl", "rotor_fr", "rotor_rl", "rotor_rr")

_MIN_HEIGHT = 0.05
_MAX_POS_XY = 10.0
_MAX_VEL = 20.0

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
_PENALTY_TILT = 150.0
_PENALTY_RUNAWAY = 150.0
_PENALTY_TIMEOUT = 60.0
# Terminal bonus for reaching the goal. As a one-shot it must beat what the
# drone could earn by surviving just *outside* the tolerance for a whole
# episode (~+54 at dt=0.02), or it learns to loiter there instead of finishing.
# One-shot value = weight * dt, so keep the weight above ~2200.
_REWARD_SUCCESS = 5000.0

# Success tolerances (used by both the `success` reward and termination)
_SUCCESS_POS_TOL = 0.25     # m
_SUCCESS_VEL_TOL = 0.15     # m/s


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


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_DRONE_XML))
  marker = spec.body("target_marker")
  marker.pos = list(_TARGET)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_tensor(asset: Entity) -> torch.Tensor:
  return torch.tensor(
    _TARGET,
    device=asset.data.root_link_pos_w.device,
    dtype=asset.data.root_link_pos_w.dtype,
  )


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
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """True where the drone is within ``pos_tol`` of the target and slower
  than ``vel_tol``. Shared by the ``success`` reward and termination so
  both fire on the same step."""
  asset: Entity = env.scene[asset_cfg.name]
  pos_err = torch.norm(
    asset.data.root_link_pos_w - _target_tensor(asset), dim=-1
  )
  speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  return (pos_err < pos_tol) & (speed < vel_tol)


def success_bonus(
  env: ManagerBasedRlEnv,
  pos_tol: float = _SUCCESS_POS_TOL,
  vel_tol: float = _SUCCESS_VEL_TOL,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """+1 on the step the drone reaches the target and stops.

  Terminal bonus: the matching ``success`` termination ends the episode
  on the same step.
  """
  return _at_goal(env, pos_tol, vel_tol, asset_cfg).float()


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
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Terminate once the drone reaches the target and stops.

  Goal reached — a true terminal state (not a time-out), so the value is
  not bootstrapped past it.
  """
  return _at_goal(env, pos_tol, vel_tol, asset_cfg)


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
    "crash": RewardTermCfg(func=crash_penalty, weight=_PENALTY_CRASH),
    "tilt": RewardTermCfg(func=tilt_penalty, weight=_PENALTY_TILT),
    "runaway": RewardTermCfg(func=runaway_penalty, weight=_PENALTY_RUNAWAY),
    "timeout": RewardTermCfg(func=timeout_penalty, weight=_PENALTY_TIMEOUT),
    "success": RewardTermCfg(func=success_bonus, weight=_REWARD_SUCCESS),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "ground_collision": TerminationTermCfg(func=drone_ground_collision),
    "runaway": TerminationTermCfg(func=runaway),
    "tilted_over": TerminationTermCfg(func=tilted_over),
    "success": TerminationTermCfg(func=success),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"drone": _get_drone_cfg()},
      sensors=(_DRONE_GROUND_CONTACT_CFG,),
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
    ),
    decimation=10,
    episode_length_s=_EPISODE_LENGTH_S,
  )


def drone_hover_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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
    experiment_name="drone_hover",
    save_interval=50,
    num_steps_per_env=64,
    max_iterations=500,
    logger="tensorboard",
  )