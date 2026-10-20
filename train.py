"""Convenience launcher for MJLab training."""

from __future__ import annotations

import os
import argparse
import subprocess
import sys

DEFAULT_TASK = "Drone-TiltedLanding"
DEFAULT_ITERATIONS = 500
DEFAULT_ENVS = 256
DEFAULT_LOGGER = "tensorboard"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task_id", nargs="?", default=DEFAULT_TASK)
  parser.add_argument("--iterations", "-i", type=int,
                      default=DEFAULT_ITERATIONS)
  parser.add_argument("--envs", "-e", type=int, default=DEFAULT_ENVS)
  parser.add_argument("--logger", default=DEFAULT_LOGGER,
                      choices=["tensorboard", "wandb"])
  parser.add_argument("--experiment", default=None)
  parser.add_argument("--extra", default="")
  args, _ = parser.parse_known_args()

  cmd = [
    os.path.expanduser("~/mjlab/.venv/bin/train"),
    args.task_id,
    "--agent.logger", args.logger,
    "--agent.max_iterations", str(args.iterations),
    "--env.scene.num-envs", str(args.envs),
  ]
  if args.experiment is not None:
    cmd += ["--agent.experiment_name", args.experiment]
  if args.extra:
    cmd += args.extra.split()

  print(f"\n  Task:       {args.task_id}")
  print(f"  Iterations: {args.iterations}")
  print(f"  Envs:       {args.envs}")
  print(f"  Logger:     {args.logger}\n")

  sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
  main()




