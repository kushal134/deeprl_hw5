import logging

import numpy as np
import torch
import torch.nn as nn

HIDDEN_UNITS = (400, 400, 400)

log = logging.getLogger(__name__)


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks.
    Each network maps (s, a) to the mean and log-variance of a Gaussian over s' - s.
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.full((1, state_dim), -3.0, device=device)
        self.min_logvar = torch.full((1, state_dim), -7.0, device=device)

        self.networks = nn.ModuleList(
            [self.create_network() for _ in range(num_nets)]
        ).to(device=device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def create_network(self):
        sizes = [self.state_dim + self.action_dim, *HIDDEN_UNITS]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], 2 * self.state_dim))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        """Returns a list (one entry per network) of (mean, logvar) tuples."""
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(net(inputs)) for net in self.networks]

    def get_output(self, output):
        """Split the raw output of one network into mean and (soft-bounded) log variance."""
        mean = output[:, : self.state_dim]
        raw_v = output[:, self.state_dim :]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, targ, mean, logvar):
        """Gaussian negative log likelihood of `targ` under N(mean, exp(logvar))."""
        ### BEGIN STUDENT SOLUTION - 1.2 ###
        ## omit the 2pi constant
        loss = 0.5 * (logvar + (targ - mean) ** 2 / torch.exp(logvar))
        return loss.mean()
        ### END STUDENT SOLUTION - 1.2 ###

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2).
        Arguments:
          inputs: state and action inputs, shape [N, state_dim + action_dim]
          targets: state differences s' - s, shape [N, state_dim]
        Returns:
          list with the average loss over all networks at each train iteration
        """
        ### BEGIN STUDENT SOLUTION - 1.2 ###
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        losses = []
        n = inputs.shape[0]

        for _ in range(num_train_itrs):
            batch_idx = torch.randint(
                0, n, (batch_size,), device=self.device
            )

            batch_inputs = inputs[batch_idx]
            batch_targets = targets[batch_idx]

            predictions = self.forward(batch_inputs)

            loss = sum(self.get_loss(batch_targets, mean, logvar) for mean, logvar in predictions) / self.num_nets

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            losses.append(loss.item())

        return losses
        ### END STUDENT SOLUTION - 1.2 ###
