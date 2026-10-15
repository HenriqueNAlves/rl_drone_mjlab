"""Drone hover environment configuration for MJLab."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch
import math

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
from mjlab.utils.lab_api.math import quat_error_magnitude, quat_from_euler_xyz
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
_MAX_TILT_COS = math.cos(math.radians(45))

_REWARD_DISTANCE = 1.0
_REWARD_SPEED = 0.1
_REWARD_RATE = 0.01
_REWARD_ALIVE = 0.0
_REWARD_HEIGHT = 2.0        # Gaussian reward for staying near target z
_PENALTY_LOW_HEIGHT = 5.0   # Per-step penalty when below safe height
_PENALTY_CRASH = 200.0      # Contact-based crash penalty (dt-scaled → ~4 effective)
_PENALTY_TILT = 150.0
_REWARD_SUCCESS = 5000.0     # Very large bonus for success
_REWARD_GOAL = 20.0          # Hover reward at target
_REWARD_ANGLE = 100.0        # Strong angle reward during approach (position-gated)
_REWARD_APPROACH = 10.0      # Approach reward (pitch while flying toward target)

_SAFE_HEIGHT = 0.3          # Height below which penalty applies
_TARGET_Z = 1.0             # Target hover height
_HEIGHT_STD = 0.3           # Gaussian std for height reward
_ANGLE_STD = math.radians(15)  # Gaussian std for angle reward (15 deg)

_SUCCESS_POS_TOL = 0.25      # Position tolerance for success (m)
_SUCCESS_VEL_TOL = 0.25      # Velocity tolerance for success (m/s)
_SUCCESS_ANG_TOL = math.radians(2)  # Angle tolerance for success (2 deg)

# Target orientation for tilted platform landing (40 deg pitch, 0 roll, 0 yaw)
# Note: quat_from_euler_xyz returns [4], we'll expand to [1, 4] in functions
_TARGET_EULER = (0.0, math.radians(40), 0.0)


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

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
# Observations
# ---------------------------------------------------------------------------

def drone_rel_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(
    _TARGET,
    device=asset.data.root_link_pos_w.device,
    dtype=asset.data.root_link_pos_w.dtype,
  )
  return torch.clamp(asset.data.root_link_pos_w - target, -5.0, 5.0)


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
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(
    _TARGET,
    device=asset.data.root_link_pos_w.device,
    dtype=asset.data.root_link_pos_w.dtype,
  )
  dist = torch.norm(asset.data.root_link_pos_w - target, dim=-1)
  return -dist


def speed_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  return -speed


def rate_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  rate = torch.norm(asset.data.root_link_ang_vel_w, dim=-1)
  return -rate


def alive_bonus(
  env: ManagerBasedRlEnv,
  minimum_height: float = _MIN_HEIGHT,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.root_link_pos_w[:, 2] > minimum_height).float()


def height_reward(
  env: ManagerBasedRlEnv,
  target_z: float = _TARGET_Z,
  std: float = _HEIGHT_STD,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Gaussian reward for staying near target height."""
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.root_link_pos_w[:, 2]
  return torch.exp(-0.5 * ((z - target_z) / std) ** 2)


def low_height_penalty(
  env: ManagerBasedRlEnv,
  safe_height: float = _SAFE_HEIGHT,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Linear penalty per step when below safe height."""
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.root_link_pos_w[:, 2]
  below = torch.clamp(safe_height - z, min=0.0)
  return -below / safe_height  # -1 at ground, 0 at safe_height


def crash_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str = "drone_ground_contact",
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalty when contact sensor detects significant ground impact."""
  sensor: ContactSensor = env.scene.sensors[sensor_name]
  data = sensor.data
  if data.force is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
  force_mag = torch.norm(data.force, dim=-1)  # [B, N]
  return -(force_mag > force_threshold).any(dim=-1).float()


def tilt_penalty(
  env: ManagerBasedRlEnv,
  max_tilt_cos: float = _MAX_TILT_COS,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """-1 on the step the drone tips past the tilt threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  gz = asset.data.projected_gravity_b[:, 2]
  return -(gz > max_tilt_cos).float()


def _get_target_quat(device, dtype, batch_size=1) -> torch.Tensor:
  """Create target quaternion on the correct device with batch dimension."""
  return quat_from_euler_xyz(
    torch.tensor(_TARGET_EULER[0], device=device, dtype=dtype),
    torch.tensor(_TARGET_EULER[1], device=device, dtype=dtype),
    torch.tensor(_TARGET_EULER[2], device=device, dtype=dtype),
  ).unsqueeze(0).expand(batch_size, 4)


def success_termination(
  env: ManagerBasedRlEnv,
  pos_tol: float = _SUCCESS_POS_TOL,
  vel_tol: float = _SUCCESS_VEL_TOL,
  ang_tol: float = _SUCCESS_ANG_TOL,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Terminate when drone reaches target position, velocity, and orientation."""
  asset: Entity = env.scene[asset_cfg.name]
  device = asset.data.root_link_pos_w.device
  dtype = asset.data.root_link_pos_w.dtype
  n_envs = asset.data.root_link_pos_w.shape[0]
  
  target = torch.tensor(_TARGET, device=device, dtype=dtype)
  target_quat = _get_target_quat(device, dtype, n_envs)
  
  pos_err = torch.norm(asset.data.root_link_pos_w - target, dim=-1)
  speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  ang_err = quat_error_magnitude(asset.data.root_link_quat_w, target_quat)
  return (pos_err < pos_tol) & (speed < vel_tol) & (ang_err < ang_tol)


def success_bonus(
  env: ManagerBasedRlEnv,
  pos_tol: float = _SUCCESS_POS_TOL,
  vel_tol: float = _SUCCESS_VEL_TOL,
  ang_tol: float = _SUCCESS_ANG_TOL,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Bonus reward for reaching target with low velocity and correct orientation."""
  asset: Entity = env.scene[asset_cfg.name]
  device = asset.data.root_link_pos_w.device
  dtype = asset.data.root_link_pos_w.dtype
  n_envs = asset.data.root_link_pos_w.shape[0]
  
  target = torch.tensor(_TARGET, device=device, dtype=dtype)
  target_quat = _get_target_quat(device, dtype, n_envs)
  
  pos_err = torch.norm(asset.data.root_link_pos_w - target, dim=-1)
  speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  ang_err = quat_error_magnitude(asset.data.root_link_quat_w, target_quat)
  return ((pos_err < pos_tol) & (speed < vel_tol) & (ang_err < ang_tol)).float()


def goal_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Hover reward at target: position + velocity + orientation all good."""
  asset: Entity = env.scene[asset_cfg.name]
  device = asset.data.root_link_pos_w.device
  dtype = asset.data.root_link_pos_w.dtype
  n_envs = asset.data.root_link_pos_w.shape[0]
  
  target = torch.tensor(_TARGET, device=device, dtype=dtype)
  target_quat = _get_target_quat(device, dtype, n_envs)
  
  pos_err = torch.norm(asset.data.root_link_pos_w - target, dim=-1)
  speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  ang_err = quat_error_magnitude(asset.data.root_link_quat_w, target_quat)
  
  # Tight tolerances for hover at target
  pos_reward = torch.exp(-0.5 * (pos_err / 0.3) ** 2)
  vel_reward = torch.exp(-0.5 * (speed / 0.3) ** 2)
  ang_reward = torch.exp(-0.5 * (ang_err / math.radians(5)) ** 2)
  
  return pos_reward * vel_reward * ang_reward


def angle_error_reward(
  env: ManagerBasedRlEnv,
  std: float = _ANGLE_STD,
  pos_gate: float = 2.5,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Strong angle reward during approach (position-gated, wider tolerance)."""
  asset: Entity = env.scene[asset_cfg.name]
  device = asset.data.root_link_pos_w.device
  dtype = asset.data.root_link_pos_w.dtype
  n_envs = asset.data.root_link_pos_w.shape[0]
  
  target = torch.tensor(_TARGET, device=device, dtype=dtype)
  target_quat = _get_target_quat(device, dtype, n_envs)
  
  pos_err = torch.norm(asset.data.root_link_pos_w - target, dim=-1)
  ang_err = quat_error_magnitude(asset.data.root_link_quat_w, target_quat)
  
  gate = (pos_err < pos_gate).float()
  return gate * torch.exp(-0.5 * (ang_err / math.radians(15)) ** 2)


def approach_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Reward for moving toward target with correct pitch during approach."""
  asset: Entity = env.scene[asset_cfg.name]
  device = asset.data.root_link_pos_w.device
  dtype = asset.data.root_link_pos_w.dtype
  n_envs = asset.data.root_link_pos_w.shape[0]
  
  target = torch.tensor(_TARGET, device=device, dtype=dtype)
  target_quat = _get_target_quat(device, dtype, n_envs)
  
  to_target = target - asset.data.root_link_pos_w
  to_target_xy = to_target[:, :2]
  target_dist_xy = torch.norm(to_target_xy, dim=-1)
  target_dir_xy = to_target_xy / (target_dist_xy.unsqueeze(-1) + 1e-6)
  
  vel_xy = asset.data.root_link_lin_vel_w[:, :2]
  vel_mag_xy = torch.norm(vel_xy, dim=-1)
  vel_dir_xy = vel_xy / (vel_mag_xy.unsqueeze(-1) + 1e-6)
  
  ang_err = quat_error_magnitude(asset.data.root_link_quat_w, target_quat)
  angle_good = torch.exp(-0.5 * (ang_err / math.radians(20)) ** 2)
  
  alignment = (vel_dir_xy * target_dir_xy).sum(dim=-1)
  alignment = torch.clamp(alignment, -1.0, 1.0)
  vel_toward = torch.clamp(alignment, 0.0, 1.0)
  
  moving = (vel_mag_xy > 0.2).float()
  not_at_target = (target_dist_xy > 0.8).float()
  
  return angle_good * vel_toward * moving * not_at_target


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

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
  max_tilt_cos: float = _MAX_TILT_COS,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  gz = asset.data.projected_gravity_b[:, 2]
  return gz > max_tilt_cos


def drone_ground_collision(
  env: ManagerBasedRlEnv,
  sensor_name: str = "drone_ground_contact",
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Terminate when the drone contacts the ground with significant force."""
  sensor: ContactSensor = env.scene.sensors[sensor_name]
  data = sensor.data
  if data.force is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  force_mag = torch.norm(data.force, dim=-1)  # [B, N]
  return (force_mag > force_threshold).any(dim=-1)


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
    "distance": RewardTermCfg(func=distance_penalty, weight=_REWARD_DISTANCE),
    "speed": RewardTermCfg(func=speed_penalty, weight=_REWARD_SPEED),
    "rate": RewardTermCfg(func=rate_penalty, weight=_REWARD_RATE),
    "alive": RewardTermCfg(func=alive_bonus, weight=_REWARD_ALIVE),
    "height": RewardTermCfg(func=height_reward, weight=_REWARD_HEIGHT),
    "low_height": RewardTermCfg(func=low_height_penalty, weight=_PENALTY_LOW_HEIGHT),
    "crash": RewardTermCfg(func=crash_penalty, weight=_PENALTY_CRASH),
    "tilt": RewardTermCfg(func=tilt_penalty, weight=_PENALTY_TILT),
    "success": RewardTermCfg(func=success_bonus, weight=_REWARD_SUCCESS),
    "goal": RewardTermCfg(func=goal_reward, weight=20.0),
    "angle": RewardTermCfg(func=angle_error_reward, weight=_REWARD_ANGLE),
    "approach": RewardTermCfg(func=approach_reward, weight=_REWARD_APPROACH),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "ground_collision": TerminationTermCfg(func=drone_ground_collision),
    "runaway": TerminationTermCfg(func=runaway),
    "tilted_over": TerminationTermCfg(func=tilted_over),
    "success": TerminationTermCfg(func=success_termination),
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
    episode_length_s=20.0,
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
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="drone_hover",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
    logger="tensorboard",
  )