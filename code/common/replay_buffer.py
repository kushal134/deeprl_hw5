import numpy as np
import torch


class ReplayBuffer:
    """
    Ring buffer of transitions (s, a, r, s', done), shared by Problem 1.4
    (online TD3 with synthetic data) and Problems 2-3 (offline TD3).

    Data is stored in numpy arrays; `sample` returns torch tensors on `device`.
    Each transition can optionally store the environment timestep at which it
    was collected (used in Problem 1.4.2 to restart model rollouts mid-episode).
    """

    def __init__(self, obs_dim, act_dim, max_size=10_000, device="cpu"):
        self.max_size = int(max_size)
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size,), dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size,), dtype=np.float32)
        self.timesteps = np.zeros((self.max_size,), dtype=np.int64)

    def add(self, obs, action, reward, next_obs, done, timestep=0):
        """Add a single transition."""
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.timesteps[i] = timestep

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        """
        Add N transitions at once (much faster than calling `add` N times).
        All arguments are array-likes with leading dimension N.
        """
        n = len(obs)
        idx = (self.ptr + np.arange(n)) % self.max_size
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = np.asarray(rewards).reshape(-1)
        self.next_obs[idx] = next_obs
        self.dones[idx] = np.asarray(dones, dtype=np.float32).reshape(-1)
        self.timesteps[idx] = 0

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        """Uniformly sample (with replacement) a batch of transitions."""
        if self.size == 0:
            raise ValueError("The buffer is empty")
        idx = np.random.randint(0, self.size, size=batch_size)

        def to_t(x):
            return torch.as_tensor(x[idx], device=self.device)

        return dict(
            obs=to_t(self.obs),
            actions=to_t(self.actions),
            rewards=to_t(self.rewards),
            next_obs=to_t(self.next_obs),
            dones=to_t(self.dones),
            timesteps=self.timesteps[idx],
        )
