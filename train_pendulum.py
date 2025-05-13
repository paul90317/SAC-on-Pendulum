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

class MutiLinear(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, n_hidden):
        super(MutiLinear, self).__init__()
        models = []
        models.append(nn.Linear(in_features, hidden_features))
        models.append(nn.ReLU())
        for _ in range(n_hidden):
            models.append(nn.Linear(hidden_features, hidden_features))
            models.append(nn.ReLU())
        models.append(nn.Linear(hidden_features, out_features))
        models.append(nn.ReLU())

        self.model = nn.Sequential(
            *models
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class Actor(ActorInterface):

    """Outputs a distribution with parameters learnable by gradient descent."""

    def __init__(self):
        super(Actor, self).__init__()
        self.shared_net   = nn.Sequential(
            MutiLinear(3, 8, 16, 6),
            nn.ReLU()
        )
        self.means_net    = nn.Linear(8, env.action_space.shape[0])
        self.log_stds_net = nn.Linear(8, env.action_space.shape[0])
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
        self.net = self.shared_net   = nn.Sequential(
            MutiLinear(4, 8, 16, 6),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
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
        agent.update_networks()
            

    after_episode_time = time.perf_counter()
    time_elapsed = after_episode_time - start_time
    time_remaining = time_elapsed / (e + 1) * (num_episodes - (e + 1))

    print(f'Episode {e:4.0f} | Return {total_reward:9.3f} | Steps {steps:4.0f} | Remaining time {round(time_remaining / 3600, 2):5.2f} hours')

torch.save(agent.state_dict(), f'results/trained_policies_pth/{args.run_id}.pth')

env.close()