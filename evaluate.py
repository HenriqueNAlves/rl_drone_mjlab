"""Interactive checkpoint picker for MJLab play.

Scans logs/rsl_rl/<experiment>/ for checkpoint directories and lets
you pick one by number. Then launches `play` with that checkpoint, printing
the reason each episode terminated (success / ground_collision / tilted_over /
runaway) and a summary on exit.

The task id is derived from the run's experiment name (e.g.
`drone_tilted_landing` -> `Drone-TiltedLanding`), so you never have to
remember which task a checkpoint came from.

Usage:
    uv run python evaluate.py                     # all runs
    uv run python evaluate.py Drone-TiltedLanding # only that task's runs
    uv run python evaluate.py --no-check          # skip the obs-dim preflight
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_LOGS_ROOT = "logs/rsl_rl"


def find_launcher() -> Path:
  """Path to the `play` wrapper that reports why each episode ended."""
  launcher = Path(__file__).resolve().parent / "eval_play.py"
  if not launcher.is_file():
    raise SystemExit(
      f"Missing {launcher}.\n"
      f"It ships next to evaluate.py and adds the episode-termination report."
    )
  return launcher


def find_runs(logs_root: str):
  """Scan logs_root for <experiment>/<timestamp>/ dirs containing checkpoints.

  Returns a list of dicts:
      {"experiment": str, "timestamp": str, "path": str,
       "checkpoints": [(int, path), ...]}
  """
  runs = []
  if not os.path.isdir(logs_root):
    return runs

  for experiment in sorted(os.listdir(logs_root)):
    exp_dir = os.path.join(logs_root, experiment)
    if not os.path.isdir(exp_dir):
      continue

    for timestamp in sorted(os.listdir(exp_dir)):
      run_dir = os.path.join(exp_dir, timestamp)
      if not os.path.isdir(run_dir):
        continue

      checkpoints = []
      for fname in os.listdir(run_dir):
        m = re.match(r"model_(\d+)\.pt$", fname)
        if m:
          checkpoints.append((int(m.group(1)), os.path.join(run_dir, fname)))

      if checkpoints:
        checkpoints.sort()
        runs.append({
          "experiment": experiment,
          "timestamp": timestamp,
          "path": run_dir,
          "checkpoints": checkpoints,
        })

  return runs


def load_task_index():
  """Map experiment_name -> registered task ids."""
  import mjlab.tasks  # noqa: F401  (populates the registry)
  from mjlab.tasks.registry import list_tasks, load_rl_cfg

  index: dict[str, list[str]] = {}
  for task_id in list_tasks():
    experiment = load_rl_cfg(task_id).experiment_name
    index.setdefault(experiment, []).append(task_id)
  return index


def checkpoint_obs_dim(path: str) -> int | None:
  """Observation width a checkpoint's actor was built for."""
  import torch

  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  state = ckpt.get("actor_state_dict", {})
  for key in ("obs_normalizer._mean", "mlp.0.weight"):
    tensor = state.get(key)
    if tensor is not None and tensor.ndim == 2:
      return int(tensor.shape[1])
  return None


def env_obs_dim(task_id: str) -> int | None:
  """Observation width the current env config produces, or None on failure.

  Built in a subprocess so the environment's startup logging doesn't pollute
  the picker.
  """
  code = (
    "import mjlab.tasks;"
    "from mjlab.tasks.registry import load_env_cfg;"
    "from mjlab.envs import ManagerBasedRlEnv;"
    f"cfg = load_env_cfg({task_id!r}, play=True);"
    "cfg.scene.num_envs = 1;"
    "env = ManagerBasedRlEnv(cfg, device='cpu');"
    "env.reset();"
    "print('OBS_DIM', env.obs_buf['actor'].shape[-1])"
  )
  try:
    proc = subprocess.run(
      [sys.executable, "-c", code],
      capture_output=True,
      text=True,
      timeout=300,
    )
  except (OSError, subprocess.SubprocessError):
    return None
  if proc.returncode != 0:
    return None
  for line in proc.stdout.splitlines():
    if line.startswith("OBS_DIM "):
      return int(line.split()[1])
  return None


def prompt_choice(prompt: str, n_options: int) -> int:
  while True:
    try:
      raw = input(f"{prompt} (1-{n_options}, or 'q'): ").strip()
    except EOFError:
      print("\nExiting.")
      raise SystemExit(1)
    if raw.lower() == "q":
      print("Exiting.")
      raise SystemExit
    try:
      c = int(raw)
      if 1 <= c <= n_options:
        return c - 1
      print(f"Please enter a number between 1 and {n_options}.")
    except ValueError:
      print("Please enter a number, or 'q' to quit.")


def main():
  parser = argparse.ArgumentParser(
    description="Interactively pick a checkpoint and launch play."
  )
  parser.add_argument(
    "task_id", nargs="?", default=None,
    help="only list runs for this task id (default: every run)"
  )
  parser.add_argument(
    "--logs_root", default=DEFAULT_LOGS_ROOT,
    help=f"directory to scan (default: {DEFAULT_LOGS_ROOT})"
  )
  parser.add_argument(
    "--no-check", action="store_true",
    help="skip the observation-dimension preflight before launching"
  )
  parser.add_argument(
    "--plot", choices=("episode", "live", "none"), default="episode",
    help="telemetry plots: when an episode ends (default), live, or none"
  )
  parser.add_argument(
    "--plot-save", default=None,
    help="write each finished episode to this directory as a PNG"
  )
  parser.add_argument(
    "--plot-env", type=int, default=0,
    help="which environment to record when playing several (default 0)"
  )
  args = parser.parse_args()

  task_index = load_task_index()

  runs = find_runs(args.logs_root)

  # Optional filter: show only the runs belonging to one task.
  if args.task_id is not None:
    from mjlab.tasks.registry import load_rl_cfg

    if args.task_id not in task_index[load_rl_cfg(args.task_id).experiment_name]:
      known = sorted(t for ts in task_index.values() for t in ts)
      raise SystemExit(
        f"Unknown task id {args.task_id!r}.\nAvailable: {', '.join(known)}"
      )
    wanted = load_rl_cfg(args.task_id).experiment_name
    runs = [r for r in runs if r["experiment"] == wanted]
    if not runs:
      raise SystemExit(
        f"No checkpoints found for {args.task_id} under {args.logs_root}/"
        f"{wanted}/. Run training first with train.py."
      )
  elif not runs:
    raise SystemExit(
      f"No checkpoints found under {args.logs_root}/. "
      f"Run training first with train.py."
    )

  for run in runs:
    run["tasks"] = task_index.get(run["experiment"], [])

  # ---- Level 1: choose a run ----
  print()
  print("=" * 70)
  print("Available runs:")
  print("=" * 70)

  for i, run in enumerate(runs, start=1):
    latest_iter = run["checkpoints"][-1][0]
    n_ckpts = len(run["checkpoints"])
    mtime = time.strftime(
      "%Y-%m-%d %H:%M",
      time.localtime(os.path.getmtime(run["path"])),
    )
    tasks = run["tasks"]
    if len(tasks) == 1:
      task_label = tasks[0]
    elif len(tasks) > 1:
      task_label = f"ambiguous ({len(tasks)} tasks)"
    else:
      task_label = "unknown task"
    print(
      f"  {i:2d}. {run['experiment']:<24} "
      f"{run['timestamp']:<22} "
      f"({n_ckpts:3d} checkpoints, latest iter {latest_iter:>5}) "
      f"{mtime}  [{task_label}]"
    )

  print()
  run_idx = prompt_choice("Choose a run", len(runs))
  chosen_run = runs[run_idx]

  # ---- Resolve which task this run belongs to ----
  if args.task_id is not None:
    task_id = args.task_id
  elif len(chosen_run["tasks"]) == 1:
    task_id = chosen_run["tasks"][0]
  elif len(chosen_run["tasks"]) > 1:
    raise SystemExit(
      f"Experiment {chosen_run['experiment']!r} maps to several tasks: "
      f"{', '.join(chosen_run['tasks'])}.\n"
      f"Pass the task id, e.g. `python evaluate.py {chosen_run['tasks'][0]}`."
    )
  else:
    raise SystemExit(
      f"Experiment {chosen_run['experiment']!r} does not match any registered "
      f"task.\nThis happens when training used a custom --experiment name.\n"
      f"Pass the task id explicitly, e.g. `python evaluate.py Drone-Hover`."
    )

  # ---- Level 2: choose a checkpoint ----
  print()
  print("=" * 70)
  print(f"Checkpoints in {chosen_run['experiment']}/{chosen_run['timestamp']}:")
  print(f"Task: {task_id}")
  print("=" * 70)

  checkpoints = chosen_run["checkpoints"]

  # If many, group them; else show all
  if len(checkpoints) <= 20:
    display = checkpoints
  else:
    # show first few, then every Nth, then last few
    step = max(1, len(checkpoints) // 15)
    display = checkpoints[::step]
    if checkpoints[-1] not in display:
      display.append(checkpoints[-1])

  for i, (it, path) in enumerate(display, start=1):
    size_kb = os.path.getsize(path) / 1024
    mtime = time.strftime(
      "%Y-%m-%d %H:%M",
      time.localtime(os.path.getmtime(path)),
    )
    print(f"  {i:2d}. iter {it:>5}   {size_kb:8.1f} KB   {mtime}")

  print()
  ckpt_idx = prompt_choice("Choose a checkpoint", len(display))
  _, chosen_ckpt = display[ckpt_idx]

  # ---- Preflight: does this checkpoint match the current env config? ----
  # Guards against replaying a run trained before the observation set changed,
  # which otherwise fails deep inside torch.load_state_dict.
  if not args.no_check:
    print()
    print("Checking checkpoint against the current env config...")
    want = checkpoint_obs_dim(chosen_ckpt)
    have = env_obs_dim(task_id)
    if want is not None and have is not None and want != have:
      raise SystemExit(
        f"\nCheckpoint expects {want} observation values but {task_id} now "
        f"produces {have}.\n"
        f"This run was trained with an older env config and cannot be "
        f"replayed with the current task.\n"
        f"Pick a newer run, or re-run with --no-check to try anyway."
      )
    if want is not None and have is not None:
      print(f"  OK: {want} observation values, matches {task_id}.")

  print()
  print(f"Launching play with: {chosen_ckpt}")
  print("Episode outcomes are printed as they happen; a summary is shown on exit.")
  print()

  cmd = [
    sys.executable, str(find_launcher()), task_id,
    "--checkpoint_file", chosen_ckpt,
    "--plot", args.plot,
    "--plot-env", str(args.plot_env),
  ]
  if args.plot_save:
    cmd += ["--plot-save", args.plot_save]
  sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
  main()
