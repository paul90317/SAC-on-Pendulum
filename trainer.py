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

    def forward(self, obs: Tensor, actions: Tensor):
        return self.net(torch.cat([obs, actions], dim=1))
    
    def val(self, obs: Tensor, actions: Tensor) -> Tensor:
        return self(obs, actions)
    
class VNet(NetWork):
    """Has little quirks; just a wrapper so that I don't need to call concat many times"""

    def __init__(self, input_dim):
        super(VNet, self).__init__()
        self.net = get_net(num_in=input_dim, num_out=1, final_activation=None)
        self.post_init(1e-3)

    def forward(self, obs: Tensor):
        return self.net(obs)
    
    def val(self, obs: Tensor):
        return self(obs)


class Trainer:
    def __init__(self, input_dim, action_dim):

        self.actor   = Actor(input_dim=input_dim, action_dim=action_dim)

        self.critic = QNet(input_dim=input_dim, action_dim=action_dim)
        self.target = QNet(input_dim=input_dim, action_dim=action_dim)
        self.target.load_state_dict(self.critic.state_dict())
        self.value = VNet(input_dim)

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

        with torch.no_grad():
            a, logp = self.actor.act(b_ns)
            v_next = self.target.val(b_ns, a) - self.alpha * logp
            qtarget = b_r + self.gamma * (1 - b_d) * v_next

        qvalue = self.critic(b_s, b_a)
        qloss = torch.mean((qvalue - qtarget) ** 2)

        self.critic.optimizer.zero_grad()
        qloss.backward()
        self.clip_gradient(self.critic)
        self.critic.optimizer.step()

        # ========================================
        # Step 14: learning the policy
        # ========================================

        a, logp = self.actor.act(b_s)
        aloss = - self.critic.val(b_s, a).mean()

        self.actor.optimizer.zero_grad()
        aloss.backward()
        self.clip_gradient(self.actor)
        self.actor.optimizer.step()

        # ========================================
        # Step 15: update target networks
        # ========================================

        with torch.no_grad():
            self.polyak_update(old_net=self.target, new_net=self.critic)

    def save_actor(self, save_dir: str, filename: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, filename))

    def load_actor(self, save_dir: str, filename: str) -> None:
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, filename)))







