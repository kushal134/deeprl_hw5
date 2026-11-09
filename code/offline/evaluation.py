"""Evaluation, video recording, and plotting for Problems 2-3. You do not need to modify this file."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def evaluate_policy(agent, env, episodes=10, seed=42):
    """Mean and std of the undiscounted return of the deterministic policy."""
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_r = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(agent.act(obs))
            done = terminated or truncated
            ep_r += r
        scores.append(ep_r)
    return float(np.mean(scores)), float(np.std(scores))


def record_eval_video(agent, env, video_dir, video_name, episodes=1):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    os.makedirs(video_dir, exist_ok=True)
    frames = []
    for _ in range(episodes):
        obs, _ = env.reset()
        frames.append(env.render())
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(agent.act(obs))
            done = terminated or truncated
            frames.append(env.render())

    video_path = os.path.join(video_dir, f"{video_name}.mp4")
    ImageSequenceClip(frames, fps=30).write_videofile(video_path, codec="libx264", logger=None)
    return video_path


# (log key(s), title, y label); x axis is the number of gradient updates
_PANELS = [
    (["actor_loss"], "Actor Loss", "Loss"),
    (["critic1_loss", "critic2_loss"], "Critic Losses", "Loss"),
    (["q1", "q2"], "Q-Values (dataset actions)", "Q-Value"),
    (["ddpg_policy_loss"], "DDPG Policy Loss", "Loss"),
    (["bc_regularization_loss"], "BC Regularization Loss", "Loss"),
    (["cql_loss"], "CQL Loss", "Loss"),
    (["cql_current_actions", "cql_random_actions"], "CQL Action Q-Values", "Q-Value"),
    (["in_distribution_q_pred"], "In-Distribution Q-Value Pred", "Q-Value"),
]


def plot_curves(log, out_path="training_curves.png"):
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle("TD3 Training Progress", fontsize=16)
    axes = axes.flatten()

    ax = axes[0]
    if log["eval_mean"]:
        ax.errorbar(log["eval_steps"], log["eval_mean"], yerr=log["eval_std"], marker="o", linewidth=2, alpha=0.8)
    ax.set_title("Evaluation Returns")
    ax.set_xlabel("Gradient Updates")
    ax.set_ylabel("Mean Return")
    ax.grid(True, alpha=0.3)

    for ax, (keys, title, ylabel) in zip(axes[1:], _PANELS):
        for key in keys:
            if log.get(key):
                ax.plot(log["steps"][-len(log[key]):], log[key], linewidth=1.5, alpha=0.8, label=key)
        ax.set_title(title)
        ax.set_xlabel("Gradient Updates")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if len(keys) > 1:
            ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
