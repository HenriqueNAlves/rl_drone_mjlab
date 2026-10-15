"""Interactive checkpoint picker for MJLab play.

Scans logs/rsl_rl/<experiment>/ for checkpoint directories and lets
you pick one by number. Then launches `play` with that checkpoint.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py Drone-Hover
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time


DEFAULT_TASK = "Drone-Hover"
DEFAULT_LOGS_ROOT = "logs/rsl_rl"


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


def prompt_choice(prompt: str, n_options: int) -> int:
  while True:
    raw = input(f"{prompt} (1-{n_options}, or 'q'): ").strip()
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
    "task_id", nargs="?", default=DEFAULT_TASK,
    help=f"MJLab task id (default: {DEFAULT_TASK})"
  )
  parser.add_argument(
    "--logs_root", default=DEFAULT_LOGS_ROOT,
    help=f"directory to scan (default: {DEFAULT_LOGS_ROOT})"
  )
  args = parser.parse_args()

  runs = find_runs(args.logs_root)
  if not runs:
    raise SystemExit(
      f"No checkpoints found under {args.logs_root}/. "
      f"Run training first with train.py."
    )

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
    print(
      f"  {i:2d}. {run['experiment']:<24} "
      f"{run['timestamp']:<22} "
      f"({n_ckpts:3d} checkpoints, latest iter {latest_iter:>5}) "
      f"{mtime}"
    )

  print()
  run_idx = prompt_choice("Choose a run", len(runs))
  chosen_run = runs[run_idx]

  # ---- Level 2: choose a checkpoint ----
  print()
  print("=" * 70)
  print(f"Checkpoints in {chosen_run['experiment']}/{chosen_run['timestamp']}:")
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

  print()
  print(f"Launching play with: {chosen_ckpt}")
  print()

  cmd = [
    "play", args.task_id,
    "--checkpoint_file", chosen_ckpt,
  ]
  sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
  main()    