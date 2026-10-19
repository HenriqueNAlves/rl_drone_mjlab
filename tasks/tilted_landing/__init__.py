from mjlab.tasks.tilted_landing.drone_env_cfg import (
  drone_tilted_landing_env_cfg,
  drone_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Drone-TiltedLanding",
  env_cfg=drone_tilted_landing_env_cfg(),
  play_env_cfg=drone_tilted_landing_env_cfg(play=True),
  rl_cfg=drone_ppo_runner_cfg(),
)