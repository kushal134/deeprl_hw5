"""Rollout helper for Problem 1. You do not need to modify this file."""

import numpy as np


class Agent:
    def __init__(self, env):
        self.env = env

    def sample(self, horizon, policy):
        """
        Sample a rollout of at most `horizon` steps using `policy`
        (any object with `reset()` and `act(state, t)`).
        """
        state, _ = self.env.reset()
        states, actions, rewards = [state], [], []

        policy.reset()
        for t in range(horizon):
            actions.append(policy.act(states[t], t))
            state, reward, terminated, truncated, _ = self.env.step(actions[t])
            states.append(state)
            rewards.append(reward)
            if terminated or truncated:
                break

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": float(np.sum(rewards)),
            "rewards": np.array(rewards),
        }
