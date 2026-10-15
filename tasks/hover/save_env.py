"""Drone hover environment configuration for MJLab."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
_MAX_TILT_COS = 0.707   # terminate at ~45 deg tilt

_REWARD_DISTANCE = 1.0
_REWARD_SPEED = 0.1
_REWARD_RATE = 0.01
_REWARD_ALIVE = 1.0
_PENALTY_CRASH = 150.0
_PENALTY_TILT = 150.0


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
  return -torch.clamp(dist, max=10.0)


def speed_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  return -torch.clamp(speed, max=20.0)


def rate_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  rate = torch.norm(asset.data.root_link_ang_vel_w, dim=-1)
  return -torch.clamp(rate, max=20.0)


def alive_bonus(
  env: ManagerBasedRlEnv,
  minimum_height: float = _MIN_HEIGHT,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.root_link_pos_w[:, 2] > minimum_height).float()


def crash_penalty(
  env: ManagerBasedRlEnv,
  minimum_height: float = _MIN_HEIGHT,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return -(asset.data.root_link_pos_w[:, 2] < minimum_height).float()


def tilt_penalty(
  env: ManagerBasedRlEnv,
  max_tilt_cos: float = _MAX_TILT_COS,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """-1 on the step the drone tips past the tilt threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  gz = asset.data.projected_gravity_b[:, 2]
  return -(gz > max_tilt_cos).float()


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

def ground_collision(
  env: ManagerBasedRlEnv,
  minimum_height: float = _MIN_HEIGHT,
  asset_cfg: SceneEntityCfg = _DRONE_CFG,
) -> torch.Tensor:
  """Fallback height check. Fires when the drone's root drops below
  minimum_height. Kept for the case where contact-based detection
  isn't available."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < minimum_height


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
    "crash": RewardTermCfg(func=crash_penalty, weight=_PENALTY_CRASH),
    "tilt": RewardTermCfg(func=tilt_penalty, weight=_PENALTY_TILT),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "ground_collision": TerminationTermCfg(func=ground_collision),
    "runaway": TerminationTermCfg(func=runaway),
    "tilted_over": TerminationTermCfg(func=tilted_over),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"drone": _get_drone_cfg()},
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