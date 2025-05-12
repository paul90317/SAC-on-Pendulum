import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch import Tensor

from replay_buffer import Batch

device = torch.device('cuda')

def tensor(x):
    return torch.tensor(x, device=device, dtype=torch.float32)

class NetWork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def post_init(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers:int=5,
        num_neurons_per_hidden_layer:int=64
    ) -> nn.Sequential:

    layers = []

    layers.extend([
        nn.Linear(num_in, num_neurons_per_hidden_layer),
        nn.ReLU(),
    ])

    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
            nn.ReLU(),
        ])

    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)

class Actor(NetWork):

    """Outputs a distribution with parameters learnable by gradient descent."""

    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.shared_net   = get_net(num_in=input_dim, num_out=64, final_activation=nn.ReLU())
        self.means_net    = nn.Linear(64, action_dim)
        self.log_stds_net = nn.Linear(64, action_dim)
        self.post_init(1e-3)
        

    def forward(self, obs: Tensor):
        out = self.shared_net(obs)
        means, log_stds = self.means_net(out), self.log_stds_net(out)

        stds = torch.exp(torch.clamp(log_stds, -20, 2))
        dist = Normal(loc=means, scale=stds)

        u = dist.rsample()
        a = 2 * torch.tanh(u)
        logp = dist.log_prob(u) - 2 * (np.log(2) - u - F.softplus(-2 * u)) - np.log(2)
        return a, logp
    
    def act(self, obs) -> tuple[Tensor, Tensor]:
        return self(obs)

class QNet(NetWork):

    """Has little quirks; just a wrapper so that I don't need to call concat many times"""

    def __init__(self, input_dim, action_dim):
        super(QNet, self).__init__()
        self.net = get_net(num_in=input_dim+action_dim, num_out=1, final_activation=None)
        self.post_init(1e-3)

    def forward(self, states: Tensor, actions: Tensor):
        return self.net(torch.cat([states, actions], dim=1))

class Trainer:
    def __init__(self, input_dim, action_dim):

        self.actor   = Actor(input_dim=input_dim, action_dim=action_dim)

        self.Q1       = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q1_targ  = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q1_targ.load_state_dict(self.Q1.state_dict())

        self.Q2       = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q2_targ  = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q2_targ.load_state_dict(self.Q2.state_dict())

        self.gamma = 0.99
        self.alpha = 0.1
        self.polyak = 0.995

    # ==================================================================================================================
    # Helper methods (it is generally not my style of using helper methods but here they improve readability)
    # ==================================================================================================================

    def step(self, obs: np.ndarray) -> tuple:
        with torch.no_grad():
            obs = tensor(obs).unsqueeze(0)
            act, _ = self.actor.act(obs)
            return act[0].cpu().numpy()

    def clip_gradient(self, net: nn.Module) -> None:
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    # ==================================================================================================================
    # Methods for learning
    # ==================================================================================================================

    def update_networks(self, b: Batch) -> None:
        b_s = b.s.to(device)
        b_ns = b.ns.to(device)
        b_d = b.d.to(device)
        b_r = b.r.to(device)
        b_a = b.a.to(device)
        # ========================================
        # Step 12: calculating targets
        # ========================================

        with torch.no_grad():
            na, logp = self.actor.act(b_ns)
            targets = b_r + self.gamma * (1 - b_d) * \
                      (torch.min(self.Q1_targ(b_ns, na), self.Q2_targ(b_ns, na)) - self.alpha * logp)

        # ========================================
        # Step 13: learning the Q functions
        # ========================================

        Q1_predictions = self.Q1(b_s, b_a)
        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)

        self.Q1.optimizer.zero_grad()
        Q1_loss.backward()
        self.clip_gradient(net=self.Q1)
        self.Q1.optimizer.step()

        Q2_predictions = self.Q2(b_s, b_a)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)

        self.Q2.optimizer.zero_grad()
        Q2_loss.backward()
        self.clip_gradient(net=self.Q2)
        self.Q2.optimizer.step()

        # ========================================
        # Step 14: learning the policy
        # ========================================

        for param in self.Q1.parameters():
            param.requires_grad = False
        for param in self.Q2.parameters():
            param.requires_grad = False

        a, logp = self.actor.act(b_s)
        policy_loss = - torch.mean(torch.min(self.Q1(b_s, a), self.Q2(b_s, a)) - self.alpha * logp)

        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.clip_gradient(net=self.actor)
        self.actor.optimizer.step()

        for param in self.Q1.parameters():
            param.requires_grad = True
        for param in self.Q2.parameters():
            param.requires_grad = True

        # ========================================
        # Step 15: update target networks
        # ========================================

        with torch.no_grad():
            self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
            self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

    def save_actor(self, save_dir: str, filename: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, filename))

    def load_actor(self, save_dir: str, filename: str) -> None:
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, filename)))







