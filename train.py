"""Convenience launcher for MJLab training."""

from __future__ import annotations

import os
import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_TASK = "Drone-TiltedLandingArx"
DEFAULT_ITERATIONS = 1000
DEFAULT_ENVS = 256
DEFAULT_LOGGER = "tensorboard"
DEFAULT_EXPERIMENT = "drone_tilted_landing_arx"
LOG_ROOT = Path(__file__).resolve().parent / "logs" / "rsl_rl"


def _str2bool(value: str) -> bool:
  if isinstance(value, bool):
    return value
  if value.strip().lower() in {"1", "true", "yes", "on", "y"}:
    return True
  if value.strip().lower() in {"0", "false", "no", "off", "n"}:
    return False
  raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _checkpoint_iter(path: Path) -> int:
  try:
    return int(path.stem.rsplit("_", 1)[1])
  except (IndexError, ValueError):
    return -1


def _resolve_warm_start(parser, spec: str, experiment: str) -> tuple[str, str]:
  """Split ``RUN/CHECKPOINT.pt`` and fail here if it is not on disk.

  Checking before launching means a typo costs a second instead of the time it
  takes mjlab to build the scene only to die inside the loader.
  """
  run, sep, checkpoint = spec.strip().rstrip("/").rpartition("/")
  if not sep or not run or not checkpoint:
    parser.error(f"--warm-start-model must be RUN/CHECKPOINT.pt, got {spec!r}")

  run_dir = LOG_ROOT / experiment / run
  path = run_dir / checkpoint
  if not path.is_file():
    found = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_iter)
    if found:
      hint = f"{len(found)} checkpoints there, {found[0].name} .. {found[-1].name}"
    else:
      hint = "no such run, or it has no model_*.pt checkpoints"
    parser.error(f"no checkpoint at {path}\n  ({hint})")
  return run, checkpoint


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task_id", nargs="?", default=DEFAULT_TASK)
  parser.add_argument("--iterations", "-i", type=int,
                      default=DEFAULT_ITERATIONS)
  parser.add_argument("--envs", "-e", type=int, default=DEFAULT_ENVS)
  parser.add_argument("--logger", default=DEFAULT_LOGGER,
                      choices=["tensorboard", "wandb"])
  parser.add_argument("--experiment", default=None)
  parser.add_argument("--warm-start", type=_str2bool, nargs="?", const=True,
                      default=False, metavar="true|false",
                      help="start from an already-trained checkpoint instead of "
                           "random weights (default: false). The checkpoint "
                           "carries the weights, the observation normaliser and "
                           "the exploration noise; the environment is unchanged.")
  parser.add_argument("--warm-start-model", default=None,
                      metavar="RUN/CHECKPOINT.pt",
                      help="checkpoint to warm start from, e.g. "
                           "2026-10-20_13-21-16/model_400.pt (runs are under "
                           "logs/rsl_rl/<experiment>/)")
  parser.add_argument("--extra", default="")
  args, _ = parser.parse_known_args()
  experiment = args.experiment or DEFAULT_EXPERIMENT

  cmd = [
    os.path.expanduser("~/mjlab/.venv/bin/train"),
    args.task_id,
    "--agent.logger", args.logger,
    "--agent.max_iterations", str(args.iterations),
    "--env.scene.num-envs", str(args.envs),
  ]
  if args.experiment is not None:
    cmd += ["--agent.experiment_name", args.experiment]
  if args.warm_start:
    if args.warm_start_model is None:
      parser.error("--warm-start needs --warm-start-model RUN/CHECKPOINT.pt")
    run, checkpoint = _resolve_warm_start(parser, args.warm_start_model, experiment)
    cmd += [
      # mjlab runs tyro with FlagConversionOff: booleans take an explicit
      # value, so a bare `--agent.resume` is a parse error.
      "--agent.resume", "True",
      "--agent.load-run", run,
      "--agent.load-checkpoint", checkpoint,
    ]
  if args.extra:
    cmd += args.extra.split()

  print(f"\n  Task:       {args.task_id}")
  total = " (total, not extra)" if args.warm_start else ""
  print(f"  Iterations: {args.iterations}{total}")
  print(f"  Envs:       {args.envs}")
  print(f"  Logger:     {args.logger}")
  if args.warm_start:
    print(f"  Warm start: {experiment}/{run}/{checkpoint}")
  else:
    print("  Warm start: no (random weights)")
  print()

  sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
  main()




