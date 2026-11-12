"""
Problem 1.4: Training TD3 with synthetic data from a learned dynamics model.

Download the pretrained ensemble `model.pt` (link in the handout) into code/ first.

Usage (from the code/ directory):
    python run_synthetic.py --question 1.4.1
    python run_synthetic.py --question 1.4.2
    python run_synthetic.py --question 1.4.3
"""

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import pets  # noqa: F401  (registers the Pushing2D environments)
from common.replay_buffer import ReplayBuffer
from common.td3 import TD3
from pets.model import PENN
from run_pets import LR, OUT_DIR, STATE_DIM


class TrainerTD3:
    def __init__(self, env_name, seed, start_timesteps, eval_freq, max_timesteps,
                 expl_noise, batch_size, num_nets, device, model_path="model.pt"):
        self.env = gym.make(env_name)
        self.sim = self.env.unwrapped  # gives access to set_state / step_state / elapsed_steps
        obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.model = PENN(num_nets, STATE_DIM, self.action_dim, LR, device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        # Set seeds
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.policy = TD3(
            obs_dim, self.action_dim, self.env.action_space.low, self.env.action_space.high,
            device=device, hidden=(256, 256), lr=1e-3, orthogonal_init=False,
        )
        self.memory = ReplayBuffer(obs_dim, self.action_dim, max_size=10_000, device=device)
        self.real_memory = ReplayBuffer(obs_dim, self.action_dim, max_size=5_000, device=device)
        self.start_timesteps = start_timesteps
        self.eval_freq = eval_freq
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps

    def explore(self, state):
        noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
        return (self.policy.act(state) + noise).clip(-self.max_action, self.max_action)

    def eval_policy(self, eval_episodes=10):
        avg_reward, successes = 0.0, 0
        for _ in range(eval_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                state, reward, terminated, truncated, info = self.env.step(self.policy.act(state))
                done = terminated or truncated
                avg_reward += reward
                successes += info.get("done") == "goal reached"
        avg_reward /= eval_episodes
        success_rate = successes / eval_episodes
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: Reward {avg_reward:.3f}, Success Rate {success_rate:.2f}")
        print("---------------------------------------")
        return avg_reward, success_rate

    def get_synthetic_transition(self, state, action):
        """
        Predict the next state with the learned model and pass it to the env
        to get the full transition. Returns (next_state, reward, terminated, truncated, info).
        """
        # Randomly choose a network from the ensemble and use it to predict the new state.
        # Remember: the model takes the first STATE_DIM entries of the state (no goal) and
        # predicts the *difference* s' - s.
        ### BEGIN STUDENT SOLUTION - 1.4 ###
        state_np = np.asarray(state, dtype=np.float32)

        state_t = torch.as_tensor(state_np[:STATE_DIM],dtype=torch.float32,device=self.model.device,).unsqueeze(0)

        action_t = torch.as_tensor(action,dtype=torch.float32,device=self.model.device,).unsqueeze(0)

        model_input = torch.cat([state_t, action_t], dim=-1)

        net_idx = np.random.randint(self.model.num_nets)

        with torch.no_grad():
            mean, logvar = self.model(model_input)[net_idx]

            delta = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

            next_physical_state = state_t + delta

        next_physical_state = next_physical_state.squeeze(0).cpu().numpy()

        new_state = np.concatenate([next_physical_state, state_np[STATE_DIM:],])

        return self.sim.step_state(new_state)
       
        ### END STUDENT SOLUTION - 1.4 ###
      

    def train_td3_with_mix(self, synthetic_ratio):
        """Problem 1.4.1: each step is synthetic with probability synthetic_ratio, else real."""
        state, _ = self.env.reset()
        episode_reward, episode_timesteps, episode_num = 0, 0, 0
        eval_rewards, eval_success_rates = [], []

        for t in range(1, int(self.max_timesteps) + 1):
            episode_timesteps += 1

            if t < self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = self.explore(state)

            if np.random.rand() < synthetic_ratio:
                next_state, reward, terminated, truncated, _ = self.get_synthetic_transition(state, action)
            else:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.memory.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if t >= self.start_timesteps:
                self.policy.update(self.memory, self.batch_size)

            if done:
                print(f"Total T: {t} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                      f"Reward: {episode_reward:.3f}")
                state, _ = self.env.reset()
                episode_reward, episode_timesteps = 0, 0
                episode_num += 1

            if t % self.eval_freq == 0:
                avg_rew, success_rate = self.eval_policy()
                eval_rewards.append(avg_rew)
                eval_success_rates.append(success_rate)

        return eval_rewards, eval_success_rates

    def _real_step(self, state, action):
        """Step the real env and store the transition (with its timestep) in real_memory."""
        timestep = self.sim.elapsed_steps
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.real_memory.add(state, action, reward, next_state, done, timestep=timestep)
        return next_state, done

    def train_with_model_rollouts(self, rollout_length):
        """Problem 1.4.2: branch short model rollouts off states visited in the real env."""
        eval_rewards, eval_success_rates = [], []

        # Warm-up data with random actions
        state, _ = self.env.reset()
        for _ in range(self.start_timesteps * 2):
            state, done = self._real_step(state, self.env.action_space.sample())
            if done:
                state, _ = self.env.reset()

        t = self.start_timesteps
        while t < self.max_timesteps:
            # Do a real environment rollout
            state, _ = self.env.reset()
            done = False
            while not done:
                state, done = self._real_step(state, self.explore(state))

            # Start a model rollout from a random previously visited state
            batch = self.real_memory.sample(1)
            model_state = batch["obs"][0].cpu().numpy()
            self.sim.set_state(model_state, elapsed_steps=batch["timesteps"][0])

            for _ in range(rollout_length):
                model_action = self.explore(model_state)
                next_model_state, model_reward, terminated, truncated, _ = (
                    self.get_synthetic_transition(model_state, model_action)
                )
                model_done = terminated or truncated
                self.memory.add(model_state, model_action, model_reward, next_model_state, model_done)
                model_state = next_model_state
                t += 1

                self.policy.update(self.memory, self.batch_size)

                if t % self.eval_freq == 0:
                    print(f"T: {t}")
                    avg_rew, success_rate = self.eval_policy()
                    eval_rewards.append(avg_rew)
                    eval_success_rates.append(success_rate)

                if model_done:
                    break

        return eval_rewards, eval_success_rates

    def compute_and_plot_model_error(self, max_rollout_length=15, num_trajectories=50):
        """
        Problem 1.4.3: roll out the model open-loop along real trajectories (using the
        real actions) and plot the L2 error between predicted and real states vs. rollout step.
        """
        all_errors = []
        while len(all_errors) < num_trajectories:
            # Real rollout: states[k], actions[k] -> states[k + 1]
            state, _ = self.env.reset()
            states, actions, timesteps = [state], [], []
            done = False
            while not done:
                action = self.policy.act(state)
                timesteps.append(self.sim.elapsed_steps)
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                states.append(state)
                actions.append(action)
            if len(actions) < max_rollout_length:
                continue  # too short to branch a full-length model rollout from

            start = np.random.randint(0, len(actions) - max_rollout_length + 1)
            model_state = states[start]
            self.sim.set_state(model_state, elapsed_steps=timesteps[start])

            errors = []
            for k in range(max_rollout_length):
                model_state = self.get_synthetic_transition(model_state, actions[start + k])[0]
                errors.append(np.linalg.norm(states[start + k + 1][:STATE_DIM] - model_state[:STATE_DIM]))
            all_errors.append(errors)

        err = np.asarray(all_errors)
        mean_err = err.mean(axis=0)
        sem_err = err.std(axis=0) / np.sqrt(len(err))
        steps = np.arange(1, max_rollout_length + 1)

        plt.figure()
        plt.plot(steps, mean_err, linewidth=2, label="Mean error")
        plt.fill_between(steps, mean_err - sem_err, mean_err + sem_err, alpha=0.25, label="Std. error")
        plt.xlabel("Model rollout step")
        plt.ylabel("Prediction error (L2)")
        plt.title("Model prediction error across trajectories")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "1.4.3-l2-error-rollout.png"))
        plt.close()


def plot_success(curves, label_fmt, eval_iterations, fpath):
    plt.figure(figsize=(12, 8))
    for key, rates in curves.items():
        plt.plot(eval_iterations[: len(rates)], rates, linestyle="-", label=label_fmt.format(key))
    plt.title("Success Rate vs. Total Timesteps", fontsize=16)
    plt.xlabel("Total Timesteps", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--question", choices=["1.4.1", "1.4.2", "1.4.3", "all"], default="all")
    p.add_argument("--model_path", default="model.pt")
    p.add_argument("--seeds", type=int, default=3, help="Number of seeds to average over")
    args = p.parse_args()

    config = dict(
        env_name="Pushing2D-v1",
        start_timesteps=5000,  # how many steps to take random actions for
        eval_freq=2500,  # how often to run evaluation
        max_timesteps=75000,  # max time steps to run environment
        expl_noise=0.1,  # std of Gaussian exploration noise
        batch_size=256,
        num_nets=2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_path=args.model_path,
    )
    seeds = list(range(args.seeds))
    eval_iterations = np.arange(1, config["max_timesteps"] // config["eval_freq"] + 1) * config["eval_freq"]

    if args.question in ("1.4.1", "all"):
        curves = {}
        for ratio in [0.0, 0.2, 0.5, 0.8, 1.0]:
            runs = [TrainerTD3(seed=s, **config).train_td3_with_mix(ratio)[1] for s in seeds]
            curves[ratio] = np.mean(runs, axis=0)
        plot_success(curves, "Ratio = {}", eval_iterations, os.path.join(OUT_DIR, "1.4.1-synthetic-ratio.png"))

    if args.question in ("1.4.2", "all"):
        curves = {}
        for length in [2, 4, 8]:
            print(f"\nTraining with Rollout Length = {length}")
            runs = [TrainerTD3(seed=s, **config).train_with_model_rollouts(length)[1] for s in seeds]
            min_len = min(len(r) for r in runs)
            curves[length] = np.mean([r[:min_len] for r in runs], axis=0)
        plot_success(curves, "Rollout Length = {}", eval_iterations,
                     os.path.join(OUT_DIR, "1.4.2-rollout-length.png"))

    if args.question in ("1.4.3", "all"):
        TrainerTD3(seed=0, **config).compute_and_plot_model_error(max_rollout_length=15, num_trajectories=50)
