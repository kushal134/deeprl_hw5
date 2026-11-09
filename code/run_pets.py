"""
Problem 1.1-1.3: MPC with ground-truth dynamics, a single probabilistic network, and PETS.

Usage (from the code/ directory):
    python run_pets.py --question 1.1
    python run_pets.py --question 1.2
    python run_pets.py --question 1.3
"""

import argparse
import logging
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import pets  # noqa: F401  (registers the Pushing2D environments)
from pets.agent import Agent
from pets.model import PENN
from pets.mpc import MPC

OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)-10s %(levelname)-8s: %(message)s",
    datefmt="%Y-%m-%d %T",
    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(OUT_DIR, "log.txt"))],
)
log = logging.getLogger("run_pets")

# Training params
TASK_HORIZON = 40
PLAN_HORIZON = 5

# CEM params
POPSIZE = 200
NUM_ELITES = 20
MAX_ITERS = 5

# Model params
LR = 1e-3
STATE_DIM = 8

LOG_EVERY = 10


class Experiment:
    """
    Holds a CEM-based and a random-shooting-based MPC policy.
    If num_nets is None, both plan with the ground-truth dynamics; otherwise they
    share a learned probabilistic ensemble (PENN) with num_nets networks.
    """

    def __init__(self, env_name="Pushing2D-v1", num_nets=None, mpc_params=None, device=None):
        self.env = gym.make(env_name)
        self.agent = Agent(self.env)
        self.warmup = False

        mpc_params = dict(mpc_params)
        mpc_params["use_gt_dynamics"] = num_nets is None
        self.model = None
        if num_nets is not None:
            self.model = PENN(num_nets, STATE_DIM, self.env.action_space.shape[0], LR, device=device)

        self.cem_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS,
                              **mpc_params, use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS,
                                 **mpc_params, use_random_optimizer=True)

    def test(self, num_episodes, optimizer="cem"):
        policy = self.cem_policy if optimizer == "cem" else self.random_policy
        samples = []
        for j in range(num_episodes):
            if j == 0 or (j + 1) % LOG_EVERY == 0:
                log.info("Test episode {}".format(j))
            samples.append(self.agent.sample(TASK_HORIZON, policy))
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success

    def _train_on(self, samples, num_train_itrs):
        losses = self.cem_policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples],
            num_train_itrs=num_train_itrs,
        )
        self.random_policy.has_been_trained = True
        return losses

    def model_warmup(self, num_episodes, num_train_itrs):
        """
        Train the model on episodes collected with a random policy.
            :param num_episodes: number of randomly sampled episodes
            :param num_train_itrs: number of iterations to train the ensemble for
        """
        samples = []
        for i in range(num_episodes):
            if i == 0 or (i + 1) % 100 == 0:
                log.info("Warm up episode %d" % i)
            # The policy acts randomly until the model has been trained once
            samples.append(self.agent.sample(TASK_HORIZON, self.random_policy))
        self.warmup = True
        return self._train_on(samples, num_train_itrs)

    def train(self, num_train_itrs, num_episodes_per_itr, evaluation_interval):
        """MBRL with PETS (Algorithm 1)."""
        cem_test, rnd_test, loss = [], [], []
        for i in range(num_train_itrs):
            log.info("#" * 68)
            log.info("Starting training iteration %d." % (i + 1))

            samples = [self.agent.sample(TASK_HORIZON, self.cem_policy) for _ in range(num_episodes_per_itr)]
            log.info("Rewards obtained: {}, Rollout Length: {}".format(
                [sample["reward_sum"] for sample in samples], len(samples[-1]["obs"])))
            loss.append(self._train_on(samples, num_train_itrs=5)[-1])

            # Q1.3.2: Test with both CEM + MPC & Random Policy + MPC
            if i == 0 or (i + 1) % evaluation_interval == 0:
                avg_return, avg_success = self.test(20, optimizer="cem")
                log.info("Test success CEM + MPC: {}".format(avg_success))
                cem_test.append([avg_return, avg_success])
                avg_return, avg_success = self.test(20, optimizer="random")
                log.info("Test success Random + MPC: {}".format(avg_success))
                rnd_test.append([avg_return, avg_success])

        return cem_test, rnd_test, loss


def report(name, exp, num_episodes, optimizer):
    avg_reward, avg_success = exp.test(num_episodes, optimizer=optimizer)
    log.info("{}: avg_reward: {}, avg_success: {}".format(name, avg_reward, avg_success))


def test_cem_gt_dynamics(num_episodes=50):
    """Problem 1.1"""
    no_mpc = {"use_mpc": False, "num_particles": 1}
    mpc = {"use_mpc": True, "num_particles": 1}

    log.info("### Q1.1.1: CEM (without MPC)")
    report("CEM PushingEnv", Experiment(mpc_params=no_mpc), num_episodes, "cem")

    log.info("### Q1.1.2: Random Policy (without MPC)")
    report("Random PushingEnv", Experiment(mpc_params=no_mpc), num_episodes, "random")

    log.info("### Q1.1.2: Random Policy + MPC")
    report("Random + MPC PushingEnv", Experiment(mpc_params=mpc), num_episodes, "random")

    log.info("### Q1.1.3: CEM with and without MPC, with and without control noise")
    for env_name in ["Pushing2D-v1", "Pushing2DNoisyControl-v1"]:
        report("CEM {}".format(env_name), Experiment(env_name, mpc_params=no_mpc), num_episodes, "cem")
        report("CEM + MPC {}".format(env_name), Experiment(env_name, mpc_params=mpc), num_episodes, "cem")


def plot_loss(losses, title, fpath):
    plt.figure(dpi=100)
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Negative Log Likelihood")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


def train_single_dynamics(num_test_episodes=50, device=None):
    """Problem 1.2"""
    log.info("### Q1.2.1: Train a single network on 1000 random episodes for 100 iterations")
    exp = Experiment(num_nets=1, mpc_params={"use_mpc": True, "num_particles": 6}, device=device)
    losses = exp.model_warmup(num_episodes=1000, num_train_itrs=100)
    plot_loss(losses, "1.2.1: Single Network Training", os.path.join(OUT_DIR, "1.2.1-loss.png"))

    log.info("### Q1.2.2: Test with Random Policy for %d episodes" % num_test_episodes)
    report("Single + Random", exp, num_test_episodes, "random")

    log.info("### Q1.2.3: Test with CEM for %d episodes" % num_test_episodes)
    report("Single + CEM", exp, num_test_episodes, "cem")


def train_pets(device=None):
    """Problem 1.3"""
    num_nets = 2
    evaluation_interval = 50
    log.info("### Q1.3.1: Train PETS from 100 random episodes + 500 MPC iterations")
    exp = Experiment(num_nets=num_nets, mpc_params={"use_mpc": True, "num_particles": 6}, device=device)
    exp.model_warmup(num_episodes=100, num_train_itrs=10)
    cem_test, rnd_test, loss = exp.train(
        num_train_itrs=500, num_episodes_per_itr=1, evaluation_interval=evaluation_interval
    )
    plot_loss(loss, "1.3.1: PETS model loss", os.path.join(OUT_DIR, "1.3.1-loss.png"))

    log.info("### Q1.3.2: Plot CEM + MPC vs. Random Policy + MPC over training")
    iters = np.arange(len(cem_test)) * evaluation_interval
    for i, name in enumerate(["Returns", "Success Rate"]):
        plt.figure(dpi=100)
        plt.plot(iters, np.array(cem_test)[:, i], label="CEM + MPC")
        plt.plot(iters, np.array(rnd_test)[:, i], label="Random + MPC")
        plt.xlabel("Training Iterations")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(os.path.join(OUT_DIR, "1.3.2-cem-vs-random-%s.png" % name.lower().replace(" ", "-")),
                    bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--question", choices=["1.1", "1.2", "1.3", "all"], default="all")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.question in ("1.1", "all"):
        test_cem_gt_dynamics(50)
    if args.question in ("1.2", "all"):
        train_single_dynamics(50, device=device)
    if args.question in ("1.3", "all"):
        train_pets(device=device)
