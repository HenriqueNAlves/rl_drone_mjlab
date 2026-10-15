#!/usr/bin/env python3
"""Headless drone evaluation with detailed metrics for assessment.

Usage:
    uv run python eval_drone.py [checkpoint_path] [--episodes N] [--device cpu|cuda]
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add tasks/hover to path for drone_env_cfg
sys.path.insert(0, 'tasks/hover')

import drone_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.models.mlp_model import MLPModel


def load_actor(checkpoint_path: str, device: str = 'cpu'):
    """Load trained actor from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    actor_state = ckpt['actor_state_dict']
    
    # The model was trained with concatenated obs under 'actor' and 'critic' keys
    from tensordict import TensorDict
    obs = TensorDict({
        'actor': torch.zeros(1, 12),
        'critic': torch.zeros(1, 12),
    }, batch_size=[1])
    
    actor = MLPModel(
        obs=obs,
        obs_groups={'actor': ['actor'], 'critic': ['critic']},
        obs_set='actor',
        output_dim=4,
        hidden_dims=[64, 64],
        activation='elu',
        obs_normalization=True,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'scalar'}
    )
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    print(f"[Load] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    actor.eval()
    actor.to(device)
    return actor


def run_episode(env, actor, device, max_steps=1000, verbose=False):
    """Run one episode, return detailed metrics."""
    obs, _ = env.reset()
    drone = env.scene['drone']
    
    # Episode tracking
    episode_data = {
        'steps': 0,
        'total_reward': 0.0,
        'reward_breakdown': {k: 0.0 for k in env.reward_manager.active_terms},
        'pos_errors': [],
        'angle_errors': [],
        'speeds': [],
        'heights': [],
        'success': False,
        'termination_reason': None,
        'final_pos': None,
        'final_vel': None,
        'final_quat': None,
    }
    
    for step in range(max_steps):
        # Actor expects obs dict with 'actor' key (concatenated tensor)
        actor_obs = {'actor': obs['actor'].to(device)}
        with torch.no_grad():
            action = actor(actor_obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get state
        pos_w = drone.data.root_link_pos_w[0]
        vel_w = drone.data.root_link_lin_vel_w[0]
        quat_w = drone.data.root_link_quat_w[0]
        
        pos_err = torch.norm(pos_w - torch.tensor([2.0, 2.0, 1.0], device=device)).item()
        speed = torch.norm(vel_w).item()
        height = pos_w[2].item()
        
        # Angle error
        target_quat = drone_env_cfg._get_target_quat(device, quat_w.dtype, 1)  # [1, 4]
        ang_err = drone_env_cfg.quat_error_magnitude(
            quat_w.unsqueeze(0),
            target_quat
        ).item()
        
        # Track
        episode_data['steps'] += 1
        episode_data['total_reward'] += reward[0].item()
        episode_data['pos_errors'].append(pos_err)
        episode_data['angle_errors'].append(ang_err)
        episode_data['speeds'].append(speed)
        episode_data['heights'].append(height)
        episode_data['final_pos'] = pos_w.cpu().numpy()
        episode_data['final_vel'] = vel_w.cpu().numpy()
        episode_data['final_quat'] = quat_w.cpu().numpy()
        
        # Reward breakdown (approximate from manager)
        for term_name in env.reward_manager.active_terms:
            term_idx = env.reward_manager.active_terms.index(term_name)
            episode_data['reward_breakdown'][term_name] += env.reward_manager._step_reward[0, term_idx].item()
        
        # Check success
        success = drone_env_cfg.success_termination(env)[0].item()
        if success:
            episode_data['success'] = True
            episode_data['termination_reason'] = 'success'
            break
        
        if terminated[0]:
            # Determine what caused termination
            episode_data['termination_reason'] = 'terminated'
            break
    
    return episode_data


def print_eval_data_check(ep_idx, data):
    """#EVAL DATA CHECK - Detailed per-episode analysis."""
    print(f"\n{'='*70}")
    print(f"#EVAL DATA CHECK - Episode {ep_idx + 1}")
    print(f"{'='*70}")
    
    status = "SUCCESS ✓" if data['success'] else f"FAIL ({data['termination_reason'] or 'timeout'}) ✗"
    print(f"Status: {status}")
    print(f"Steps: {data['steps']}")
    print(f"Total Reward: {data['total_reward']:.2f}")
    
    # Position & Angle
    print(f"\n--- Position & Angle ---")
    print(f"  Final pos: {data['final_pos']}")
    print(f"  Final vel: {data['final_vel']}")
    print(f"  Final quat: {data['final_quat']}")
    print(f"  Final pos_err: {data['pos_errors'][-1]:.4f} m")
    print(f"  Final ang_err: {data['angle_errors'][-1]*180/np.pi:.2f}°")
    print(f"  Max pos_err: {max(data['pos_errors']):.4f} m")
    print(f"  Max ang_err: {max(data['angle_errors'])*180/np.pi:.2f}°")
    
    # Dynamics
    print(f"\n--- Dynamics ---")
    print(f"  Max speed: {max(data['speeds']):.3f} m/s")
    print(f"  Avg speed: {np.mean(data['speeds']):.3f} m/s")
    print(f"  Min/Max height: {min(data['heights']):.3f} / {max(data['heights']):.3f} m")
    
    # Reward breakdown
    print(f"\n--- Reward Breakdown (cumulative) ---")
    for name, val in data['reward_breakdown'].items():
        print(f"  {name:15s}: {val:8.2f}")
    
    # Trajectory pattern analysis
    pos_errs = data['pos_errors']
    if len(pos_errs) > 10:
        # Check if approaching or oscillating
        first_half = pos_errs[:len(pos_errs)//2]
        second_half = pos_errs[len(pos_errs)//2:]
        trend = np.mean(second_half) - np.mean(first_half)
        print(f"\n--- Trajectory Pattern ---")
        print(f"  Pos err trend: {'improving' if trend < 0 else 'worsening'} ({trend:+.4f})")
        
        # Check for oscillations
        diffs = np.diff(pos_errs)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        print(f"  Oscillations: {sign_changes} direction changes")


def main():
    parser = argparse.ArgumentParser(description="Headless drone evaluation with detailed metrics")
    parser.add_argument('checkpoint', nargs='?', help='Path to checkpoint .pt file')
    parser.add_argument('--episodes', '-n', type=int, default=10, help='Number of episodes')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
    args = parser.parse_args()
    
    # Find latest checkpoint if not specified
    if args.checkpoint is None:
        log_dir = Path('logs/rsl_rl/drone_hover')
        runs = sorted(log_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("No runs found in logs/rsl_rl/drone_hover/")
            return 1
        latest_run = runs[0]
        checkpoints = sorted(latest_run.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
        if not checkpoints:
            print(f"No checkpoints in {latest_run}")
            return 1
        args.checkpoint = str(checkpoints[-1])
        print(f"Using latest checkpoint: {args.checkpoint}")
    
    # Load environment and model
    cfg = drone_env_cfg.drone_hover_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    actor = load_actor(args.checkpoint, args.device)
    
    print(f"\n{'#'*70}")
    print(f"# DRONE EVALUATION - {args.episodes} episodes")
    print(f"# Checkpoint: {args.checkpoint}")
    print(f"# Device: {args.device}")
    print(f"{'#'*70}")
    
    # Run episodes
    all_episodes = []
    for ep in range(args.episodes):
        data = run_episode(env, actor, args.device, args.max_steps)
        all_episodes.append(data)
        print_eval_data_check(ep, data)
    
    env.close()
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# SUMMARY STATISTICS ({len(all_episodes)} episodes)")
    print(f"{'#'*70}")
    
    n = len(all_episodes)
    successes = sum(1 for d in all_episodes if d['success'])
    crashes = sum(1 for d in all_episodes if d['termination_reason'] == 'terminated')
    
    print(f"Success rate: {successes}/{n} ({100*successes/n:.1f}%)")
    print(f"Crash rate: {crashes}/{n} ({100*crashes/n:.1f}%)")
    print(f"Timeout rate: {n - successes - crashes}/{n} ({100*(n - successes - crashes)/n:.1f}%)")
    
    if successes > 0:
        succ_data = [d for d in all_episodes if d['success']]
        print(f"\nSuccessful episodes:")
        print(f"  Avg steps: {np.mean([d['steps'] for d in succ_data]):.1f}")
        print(f"  Avg final pos_err: {np.mean([d['pos_errors'][-1] for d in succ_data]):.4f} m")
        print(f"  Avg final ang_err: {np.mean([d['angle_errors'][-1] for d in succ_data])*180/np.pi:.2f}°")
        print(f"  Avg total reward: {np.mean([d['total_reward'] for d in succ_data]):.2f}")
    
    # Position error stats
    all_final_pos_errs = [d['pos_errors'][-1] for d in all_episodes]
    print(f"\nAll episodes - Final pos_err: mean={np.mean(all_final_pos_errs):.4f}, std={np.std(all_final_pos_errs):.4f}, min={np.min(all_final_pos_errs):.4f}, max={np.max(all_final_pos_errs):.4f}")
    
    all_final_ang_errs = [d['angle_errors'][-1] for d in all_episodes]
    print(f"All episodes - Final ang_err: mean={np.mean(all_final_ang_errs)*180/np.pi:.2f}°, std={np.std(all_final_ang_errs)*180/np.pi:.2f}°")
    
    # Reward component averages
    print(f"\nAvg reward components per episode:")
    comp_names = all_episodes[0]['reward_breakdown'].keys()
    for name in comp_names:
        vals = [d['reward_breakdown'][name] for d in all_episodes]
        print(f"  {name:15s}: {np.mean(vals):8.2f} ± {np.std(vals):6.2f}")
    
    return 0


if __name__ == '__main__':
    import numpy as np
    sys.exit(main())