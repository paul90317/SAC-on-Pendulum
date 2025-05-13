import time
import gymnasium as gym
from torch import Tensor
import torch.nn as nn
import torch
from torch.distributions import Normal
from SAC import ActorInterface, CriticInterface, Agent, tensor
import numpy as np
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStackObservation
import numpy as np
import pandas as pd
import sys
from evaluation import evaluate, show_statistics

stack_size = 8
_, run_id = sys.argv

class CustomEnv(gym.Env):
    def __init__(self, render_mode=True):
        self.env = gym.make('CarRacing-v3', render_mode="rgb_array")
        self.render_mode = render_mode  # Add a flag to control rendering
        self.action_space = self.env.action_space
        self.observation_space = Box(0, 255, (96, 96), np.ubyte)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        gray_obs = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
        gray_obs = gray_obs.astype(np.ubyte)  # Convert to ubyte

        if reward > 0:
            self.tolerate = 0
        else:
            self.tolerate += 1
            if self.tolerate > 20:
                done = True
                reward = -1
        return gray_obs, reward, done, truncated, info

    def reset(self, *argv, **kargv):
        obs, info = self.env.reset(*argv, **kargv)
        
        self.tolerate = 0
        
        gray_obs = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
        gray_obs = gray_obs.astype(np.ubyte)  # Convert to ubyte
        
        return gray_obs, info

    def render(self):
        return self.env.render()

env = FrameStackObservation(CustomEnv(), stack_size)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc_cnn = nn.Sequential(  ## input size:[96, 96]
            nn.Conv2d(stack_size, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            nn.Flatten(start_dim=1),  ## output: 512
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc_cnn(x)

class Actor(ActorInterface):
    def __init__(self):
        super(Actor, self).__init__()
        self.shared_net   = nn.Sequential(
            CNN(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.means_net    = nn.Linear(32, env.action_space.shape[0])
        self.log_stds_net = nn.Linear(32, env.action_space.shape[0])
        self.low = tensor([0, 0.5, 0.15])
        self.scale = tensor([1, 0.5, 0.15])
        self.post_init(1e-3)
        

    def forward(self, obs: Tensor):
        out = self.shared_net(obs)
        means, log_stds = self.means_net(out), self.log_stds_net(out)

        stds = torch.exp(torch.clamp(log_stds, -20, 2))
        dist = Normal(loc=means, scale=stds)

        u = dist.rsample()
        a = torch.tanh(u) * self.scale + self.low
        
        logp = dist.log_prob(u) - 2 * (np.log(2) - u - F.softplus(-2 * u))
        return a, logp

class Critic(CriticInterface):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc_obs =  nn.Sequential(
            CNN(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.fc_action = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        self.fc_aggr = nn.Linear(64, 1)
        self.post_init(1e-3)

    def forward(self, obs: Tensor, actions: Tensor):
        obs = self.fc_obs(obs)
        actions = self.fc_action(actions)
        return self.fc_aggr(torch.cat([obs, actions], dim=1))

agent = Agent(
    actor=Actor, 
    critic=Critic,
    n_critics=2,
    gamma=0.99,
    alpha=0.2,
    polyak=0.995,
    replay_buf_size=4000,
    batch_size=256,
    warmup=500
)

steps = 0

num_episodes = 20000

start_time = time.perf_counter()
for e in range(num_episodes):
    agent.train()
    obs, _ = env.reset()

    total_reward = 0
    done = False
    loss_infos = []
    while not done:
        a = agent.predict(obs)
        obs_next, reward, ter, tru, _ = env.step(a)
        done = ter or tru
        total_reward += reward

        agent.push(
            obs, a, reward, obs_next, ter
        )
        
        obs_next = obs

        steps += 1
        loss_infos.append(agent.update_networks()) 

        if steps % 4000 == 0:
            evaluate(env, agent, run_id, steps, 3)

    show_statistics(loss_infos)

    after_episode_time = time.perf_counter()
    time_elapsed = after_episode_time - start_time
    time_remaining = time_elapsed / (e + 1) * (num_episodes - (e + 1))

    print(f'Episode {e:4.0f} | Return {total_reward:9.3f} | Steps {steps:4.0f} | Remaining time {round(time_remaining / 3600, 2):5.2f} hours')

evaluate(env, agent, run_id, steps, 3)

env.close()