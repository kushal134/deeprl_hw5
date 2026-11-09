"""Action-sequence optimizers used by MPC. You do not need to modify this file."""

import numpy as np


class _Optimizer:
    def __init__(self, cost_fn, out_dim, popsize, num_elites, max_iters,
                 upper_bound, lower_bound, epsilon=0.001):
        """
        Arguments:
            cost_fn: maps action sequences [N, out_dim] to costs [N]
            out_dim (int): dimension of the search space (action_dim * plan_horizon)
            popsize (int): number of candidates sampled per iteration
            num_elites (int): number of top candidates used to refit the distribution (CEM only)
            max_iters (int): number of iterations
            epsilon (float): CEM stops early once the max variance drops below epsilon
        """
        assert num_elites <= popsize, "Number of elites must be <= population size!"
        self.cost_fn = cost_fn
        self.out_dim = out_dim
        self.popsize = popsize
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.ub = upper_bound.reshape([1, out_dim])
        self.lb = lower_bound.reshape([1, out_dim])
        self.epsilon = epsilon

    def _sample(self, n, mean, var):
        samples = np.random.normal(size=[n, self.out_dim]) * np.sqrt(var) + mean
        return np.clip(samples, self.lb, self.ub)


class CEMOptimizer(_Optimizer):
    """Cross Entropy Method (Algorithm 4)."""

    def solve(self, init_mean, init_var):
        """
        Arguments:
            init_mean (np.ndarray): shape (action_dim * plan_horizon,)
            init_var (np.ndarray): shape (action_dim * plan_horizon,)
        Returns: the optimized mean action sequence.
        """
        mean, var = init_mean, init_var
        for _ in range(self.max_iters):
            if np.max(var) <= self.epsilon:
                break
            samples = self._sample(self.popsize, mean, var)
            elites = samples[np.argsort(self.cost_fn(samples))[: self.num_elites]]
            mean, var = np.mean(elites, axis=0), np.var(elites, axis=0)
        return mean


class RandomOptimizer(_Optimizer):
    """Random shooting: sample popsize * max_iters sequences from N(init_mean, init_var), keep the best."""

    def solve(self, init_mean, init_var):
        samples = self._sample(self.popsize * self.max_iters, init_mean, init_var)
        return samples[np.argmin(self.cost_fn(samples))]
