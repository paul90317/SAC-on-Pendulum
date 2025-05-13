import time
import gymnasium as gym
from torch import Tensor
import torch.nn as nn
import torch
from torch.distributions import Normal
from SAC import ActorInterface, CriticInterface, Agent
import argparse
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int)
args = parser.parse_args()

env = gym.make('Pendulum-v1')

class NetBuilder(nn.Module):
    def __init__(self, num_in: int, num_out: int, final_activation=None, num_hidden_layers: int = 5, num_neurons_per_hidden_layer: int = 64):
        super(NetBuilder, self).__init__()
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(num_in, num_neurons_per_hidden_layer),
            nn.ReLU(),
        ])

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
                nn.ReLU(),
            ])

        # Output layer
        layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))

        # Final activation (if provided)
        if final_activation is not None:
            layers.append(final_activation)

        # Register the layers as a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class Actor(ActorInterface):

    """Outputs a distribution with parameters learnable by gradient descent."""

    def __init__(self):
        super(Actor, self).__init__()
        self.shared_net   = NetBuilder(num_in=env.observation_space.shape[0], num_out=64, final_activation=nn.ReLU())
        self.means_net    = nn.Linear(64, env.action_space.shape[0])
        self.log_stds_net = nn.Linear(64, env.action_space.shape[0])
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

class Critic(CriticInterface):

    """Has little quirks; just a wrapper so that I don't need to call concat many times"""

    def __init__(self):
        super(Critic, self).__init__()
        self.net = NetBuilder(num_in=env.observation_space.shape[0]+env.action_space.shape[0], num_out=1, final_activation=None)
        self.post_init(1e-3)

    def forward(self, obs: Tensor, actions: Tensor):
        return self.net(torch.cat([obs, actions], dim=1))

agent = Agent(
    actor=Actor, 
    critic=Critic,
    n_critics=2,
    gamma=0.99,
    alpha=0.1,
    polyak=0.995,
    replay_buf_size=10000,
    batch_size=64,
    warmup=200
)

num_episodes = 200

start_time = time.perf_counter()

for e in range(num_episodes):

    obs, _ = env.reset()

    total_reward = 0
    steps = 0
    done = False
    while not done:
        a = agent.predict(obs)
        obs_next, reward, ter, tru, _ = env.step(a)
        done = ter or tru
        total_reward += reward

        agent.push(
            obs, a, reward, obs_next, ter
        )
        
        obs = obs_next

        steps += 1
        if steps % 1 == 0: 
            agent.update_networks()

    after_episode_time = time.perf_counter()
    time_elapsed = after_episode_time - start_time
    time_remaining = time_elapsed / (e + 1) * (num_episodes - (e + 1))

    print(f'Episode {e:4.0f} | Return {total_reward:9.3f} | Steps {steps:4.0f} | Remaining time {round(time_remaining / 3600, 2):5.2f} hours')

torch.save(agent.state_dict(), f'results/trained_policies_pth/{args.run_id}.pth')

env.close()