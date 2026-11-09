"""
Problems 2-3: Offline RL with TD3 (+ BC regularization, + CQL) on Minari hopper-medium-v0.

Usage (from the code/ directory):
    python run_offline.py                                    # 2.2: vanilla TD3
    python run_offline.py --bc_regularization_weight <alpha> # 2.3: TD3 + BC
    python run_offline.py --cql_alpha 5                      # 3.1: TD3 + CQL
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from common.td3 import TD3
from offline.data import get_buffer_and_environments_for_task
from offline.evaluation import evaluate_policy, plot_curves, record_eval_video


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    offline_buffer, eval_env = get_buffer_and_environments_for_task(args.dataset, device=device)
    obs_dim = int(np.prod(eval_env.observation_space.shape))
    act_dim = int(np.prod(eval_env.action_space.shape))
    print(f"Dataset: {args.dataset} ({offline_buffer.size} transitions), obs dim: {obs_dim}, action dim: {act_dim}")

    agent = TD3(
        obs_dim, act_dim, eval_env.action_space.low, eval_env.action_space.high, device=device,
        hidden=(128, 128), lr=args.lr, gamma=args.gamma, tau=args.tau,
        policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_delay=args.delay,
        bc_regularization_weight=args.bc_regularization_weight,
        cql_alpha=args.cql_alpha, cql_n_actions=args.cql_n_actions, cql_temp=args.cql_temp,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best.pt")
    plot_path = os.path.join(args.out_dir, "training_curves.png")

    log = {"steps": [], "eval_mean": [], "eval_std": [], "eval_steps": []}
    best_eval = -np.inf
    eval_env.reset(seed=args.seed)

    for step in range(1, args.total_steps + 1):
        stats = agent.update(offline_buffer, args.batch_size)
        log["steps"].append(step)
        for k, v in stats.items():
            log.setdefault(k, []).append(v)

        if step % args.log_every == 0:
            print(f"[Step {step:>8d}] " + " | ".join(f"{k}: {v:.4f}" for k, v in stats.items()))

        if args.eval_every > 0 and step % args.eval_every == 0:
            mean_r, std_r = evaluate_policy(agent, eval_env, episodes=args.eval_episodes, seed=1000)
            log["eval_mean"].append(mean_r)
            log["eval_std"].append(std_r)
            log["eval_steps"].append(step)
            print(f"--- Eval at step {step}: {mean_r:.1f} ± {std_r:.1f}")
            if mean_r > best_eval:
                best_eval = mean_r
                torch.save(agent.actor.state_dict(), best_path)

        if args.video_every > 0 and step % args.video_every == 0:
            try:
                path = record_eval_video(agent, eval_env, os.path.join(args.out_dir, "videos"), f"eval_t{step}")
                print(f"Video saved: {path}")
            except Exception as e:  # rendering is not available on every machine
                print(f"Video recording failed: {e}")

        if args.plot_every > 0 and step % args.plot_every == 0:
            plot_curves(log, plot_path)

    # Final evaluation with the best checkpoint
    if os.path.exists(best_path):
        agent.actor.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model from {best_path} for final evaluation")
    if args.eval_episodes > 0:
        mean_r, std_r = evaluate_policy(agent, eval_env, episodes=args.eval_episodes * 2, seed=4200)
        print(f"Final performance: {mean_r:.1f} ± {std_r:.1f}")
        log["eval_mean"].append(mean_r)
        log["eval_std"].append(std_r)
        log["eval_steps"].append(args.total_steps)

    plot_curves(log, plot_path)
    torch.save(agent.actor.state_dict(), os.path.join(args.out_dir, "final.pt"))
    with open(os.path.join(args.out_dir, "training_log.json"), "w") as f:
        json.dump(log, f)
    print(f"All artifacts saved to: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mujoco/hopper/medium-v0", help="Minari dataset id")
    p.add_argument("--total_steps", type=int, default=100_000, help="Number of gradient updates")
    p.add_argument("--cpu", action="store_true", help="Force CPU usage")
    p.add_argument("--seed", type=int, default=42000)

    # TD3
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005, help="Target network soft update rate")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing noise std")
    p.add_argument("--noise_clip", type=float, default=0.5, help="Target policy smoothing noise clip")
    p.add_argument("--delay", type=int, default=2, help="Actor / target update delay")

    # Offline RL regularizers
    p.add_argument("--bc_regularization_weight", type=float, default=0.0, help="alpha_reg (Problem 2.3)")
    p.add_argument("--cql_alpha", type=float, default=0.0, help="CQL alpha (Problem 3.1)")
    p.add_argument("--cql_n_actions", type=int, default=4, help="Number of sampled actions for CQL")
    p.add_argument("--cql_temp", type=float, default=1.0, help="CQL temperature")

    # Logging and evaluation
    p.add_argument("--log_every", type=int, default=10_000, help="Print training stats every N steps")
    p.add_argument("--plot_every", type=int, default=10_000, help="Update plots every N steps")
    p.add_argument("--eval_every", type=int, default=50_000, help="Evaluate every N steps (0 to disable)")
    p.add_argument("--eval_episodes", type=int, default=100, help="Episodes per evaluation")
    p.add_argument("--video_every", type=int, default=100_000, help="Record video every N steps (0 to disable)")
    p.add_argument("--out_dir", default=None, help="Output directory (default: runs/<config>/<timestamp>)")
    args = p.parse_args()

    if args.out_dir is None:
        name = f"td3_bc{args.bc_regularization_weight:g}_cql{args.cql_alpha:g}"
        args.out_dir = os.path.join("runs", name, datetime.now().strftime("%Y%m%d-%H%M%S"))

    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"{k:28s}: {v}")
    print("=" * 60)
    run(args)
