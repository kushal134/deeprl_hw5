"""
Twin Delayed DDPG (TD3), shared by Problem 1.4 (online, with synthetic data)
and Problems 2-3 (offline, with optional BC regularization and CQL).

You only need to edit this file for Problem 2.3 (BC regularization) and
Problem 3.1 (CQL loss). Look for "BEGIN STUDENT SOLUTION".
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, orthogonal_init=True):
    """MLP with ReLU between layers: sizes = [in, h1, ..., out]."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    if orthogonal_init:
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)
    return net


class Actor(nn.Module):
    """Deterministic policy mu(s) = tanh(f(s)), rescaled to the action bounds."""

    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden, orthogonal_init):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], orthogonal_init)
        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("scale", (act_high - act_low) / 2.0)
        self.register_buffer("bias", (act_high + act_low) / 2.0)

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * self.scale + self.bias


class Critic(nn.Module):
    """Q-network Q(s, a) -> scalar, returns shape [B]."""

    def __init__(self, obs_dim, act_dim, hidden, orthogonal_init):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1], orthogonal_init)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class TD3:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_low,
        act_high,
        device="cpu",
        hidden=(256, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        orthogonal_init=True,
        bc_regularization_weight=0.0,
        cql_alpha=0.0,
        cql_n_actions=4,
        cql_temp=1.0,
    ):
        self.device = torch.device(device)
        self.act_dim = act_dim
        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.bc_regularization_weight = bc_regularization_weight
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.cql_temp = cql_temp

        self.actor = Actor(obs_dim, act_dim, act_low, act_high, hidden, orthogonal_init).to(self.device)
        self.critic1 = Critic(obs_dim, act_dim, hidden, orthogonal_init).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, hidden, orthogonal_init).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        self.total_updates = 0

    @torch.no_grad()
    def act(self, obs):
        """Deterministic action for a single (unbatched) numpy observation."""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        return self.actor(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()

    def update(self, buffer, batch_size):
        """Sample a batch from `buffer` and perform one TD3 update. Returns a stats dict."""
        batch = buffer.sample(batch_size)
        obs, actions, rewards = batch["obs"], batch["actions"], batch["rewards"]
        next_obs, dones = batch["next_obs"], batch["dones"]
        self.total_updates += 1

        # ---- Critic update (with target policy smoothing) ----
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(
                self.target_actor(next_obs) + noise, self.act_low, self.act_high
            )
            target_q = torch.min(
                self.target_critic1(next_obs, next_actions),
                self.target_critic2(next_obs, next_actions),
            )
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        cql_stats = {}
        if self.cql_alpha > 0.0:
            cql_loss, cql_stats = self._get_cql_loss(batch)
            critic_loss = critic_loss + self.cql_alpha * cql_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Delayed actor update ----
        ddpg_policy_loss = torch.zeros((), device=self.device)
        bc_loss = torch.zeros((), device=self.device)
        actor_loss = torch.zeros((), device=self.device)
        if self.total_updates % self.policy_delay == 0:
            actions_pi = self.actor(obs)
            ddpg_policy_loss = -self.critic1(obs, actions_pi).mean()
            actor_loss = ddpg_policy_loss

            if self.bc_regularization_weight > 0:
                # ------ Problem 2.3: BC regularization loss ------
                # Compute bc_loss = ||a - mu(s)||^2 (averaged over the batch) and
                # add bc_regularization_weight * bc_loss to actor_loss.
                # Tip: put a `breakpoint()` here to inspect tensors in pdb.
                ### BEGIN STUDENT SOLUTION - 2.3 ###
                raise NotImplementedError("Problem 2.3: BC regularization")
                ### END STUDENT SOLUTION - 2.3 ###

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

        stats = {
            "actor_loss": actor_loss.item(),
            "ddpg_policy_loss": ddpg_policy_loss.item(),
            "bc_regularization_loss": bc_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "q1": current_q1.mean().item(),
            "q2": current_q2.mean().item(),
        }
        stats.update(cql_stats)
        return stats

    def _get_cql_loss(self, batch):
        """Returns the CQL regularizer (before multiplying by cql_alpha) and stats for logging."""
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        batch_size = obs.shape[0]

        # Use these variable names so that the plots pick up on the values.
        cql_loss = torch.zeros((), device=self.device)
        current_action_q1_values = torch.zeros((batch_size, self.cql_n_actions), device=self.device)
        random_action_q1_values = torch.zeros((batch_size, self.cql_n_actions), device=self.device)
        q_pred = torch.zeros((batch_size,), device=self.device)

        # ------ Problem 3.1: CQL loss ------
        # Steps to compute the CQL loss:
        # 1. Compute Q-values for the in-distribution actions (actions from the offline dataset).
        # 2. Sample random actions uniformly from the action space [act_low, act_high].
        # 3. Also compute actions from the current policy at both the current and next observations.
        # 4. Compute Q-values for all the sampled actions (random, current, next).
        # 5. Compute the CQL loss using torch.logsumexp (scaled by self.cql_temp).
        # Note: self.cql_n_actions is how many random actions to sample per batch element.
        # Tip: put a `breakpoint()` here to inspect tensors in pdb.
        ### BEGIN STUDENT SOLUTION - 3.1 ###
        raise NotImplementedError("Problem 3.1: CQL loss")
        ### END STUDENT SOLUTION - 3.1 ###

        return cql_loss, {
            "cql_loss": cql_loss.item(),
            "cql_current_actions": current_action_q1_values.mean().item(),
            "cql_random_actions": random_action_q1_values.mean().item(),
            "in_distribution_q_pred": q_pred.mean().item(),
        }

    def _soft_update(self, net, target):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
