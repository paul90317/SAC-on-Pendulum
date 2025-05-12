import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from torch import Tensor
import random
from collections import deque

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

class Trainer:
    def __init__(self, env: gym.Env):
        self.n_critics = 2
        self.gamma = 0.99
        self.alpha = 0.1
        self.polyak = 0.995
        self.replay_buf_size = 10000
        self.batch_size = 64
        self.warmup = 200

        self.env = env
        self.actor   = Actor(input_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
        self.critics = [QNet(input_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]) for _ in range(self.n_critics)]
        self.targets = [QNet(input_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]) for _ in range(self.n_critics)]

        for t, c in zip(self.critics, self.targets):
            t.load_state_dict(c.state_dict())
        
        self.replay_buf = deque(maxlen=self.replay_buf_size)

    # ==================================================================================================================
    # Helper methods (it is generally not my style of using helper methods but here they improve readability)
    # ==================================================================================================================

    def step(self) -> tuple:
        with torch.no_grad():
            obs = tensor(self.obs).unsqueeze(0)
            act, _ = self.actor.act(obs)
            a = act[0].cpu().numpy()
            step_info = self.env.step(a)
            obs_next, reward, ter, _, _ = step_info
            self.replay_buf.append([
                self.obs, a, reward, obs_next, ter
            ])
            self.obs = obs_next
            return step_info

    def reset(self, seed = None, options = None):
        reset_info = self.env.reset(seed=seed, options=options)
        self.obs = reset_info[0]
        return reset_info

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    # ==================================================================================================================
    # Methods for learning
    # ==================================================================================================================

    def update_networks(self) -> None:
        if len(self.replay_buf) < self.warmup:
            return

        obs, action, reward, obs_next, done = zip(*random.sample(self.replay_buf, self.batch_size))
        
        obs = tensor(np.array(obs))
        action = tensor(np.array(action))
        reward = tensor(reward).unsqueeze(1)
        obs_next = tensor(np.array(obs_next))
        done = tensor(done).unsqueeze(1)

        with torch.no_grad():
            a, logp = self.actor.act(obs_next)
            v_next = torch.stack([t.val(obs_next, a) for t in self.targets]).min(0).values
            v_next -= self.alpha * logp
            qtarget = reward + self.gamma * (1 - done) * v_next

        for c in self.critics:
            qvalue = c.val(obs, action)
            qloss = torch.mean((qvalue - qtarget) ** 2)
            c.optimizer.zero_grad()
            qloss.backward()
            c.optimizer.step()

        # ========================================
        # Step 14: learning the policy
        # ========================================

        a, logp = self.actor.act(obs)
        aloss = - torch.stack([c.val(obs, a) for c in self.critics]).min(0).values
        aloss += self.alpha * logp
        aloss = aloss.mean()

        self.actor.optimizer.zero_grad()
        aloss.backward()
        self.actor.optimizer.step()

        # ========================================
        # Step 15: update target networks
        # ========================================

        with torch.no_grad():
            for t, c in zip(self.targets, self.critics):
                self.polyak_update(t, c)

    def save_actor(self, save_dir: str, filename: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, filename))

    def load_actor(self, save_dir: str, filename: str) -> None:
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, filename)))







