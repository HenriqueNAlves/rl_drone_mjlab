from mjlab.tasks.hover.drone_env_cfg import (
  drone_hover_env_cfg,
  drone_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Drone-Hover",
  env_cfg=drone_hover_env_cfg(),
  play_env_cfg=drone_hover_env_cfg(play=True),
  rl_cfg=drone_ppo_runner_cfg(),
)
