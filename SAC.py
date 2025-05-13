import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import gymnasium as gym
from torch import Tensor
import random
from collections import deque
from typing import Type

device = torch.device('cuda')

def tensor(x):
    return torch.tensor(x, device=device, dtype=torch.float32)

class NetWork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def post_init(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

class ActorInterface(NetWork):
    def act(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        return self(obs)

class CriticInterface(NetWork):
    def val(self, obs: Tensor, actions: Tensor) -> Tensor:
        return self(obs, actions)

class Agent(nn.Module):
    def __init__(self, 
        actor: Type[ActorInterface], 
        critic: Type[CriticInterface],
        n_critics: int,
        gamma: float,
        alpha: float,
        polyak: float,
        replay_buf_size: int,
        batch_size: int,
        warmup: int
    ):
        super(Agent, self).__init__()
        self.n_critics = n_critics
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.batch_size = batch_size
        self.warmup = warmup

        self.actor   = actor()
        self.critics = nn.ModuleList([critic() for _ in range(self.n_critics)])
        self.targets = nn.ModuleList([critic() for _ in range(self.n_critics)])

        for t, c in zip(self.critics, self.targets):
            t.load_state_dict(c.state_dict())
        
        self.replay_buf = deque(maxlen=replay_buf_size)

    def predict(self, obs:np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = tensor(obs).unsqueeze(0)
            act, _ = self.actor.act(obs)
            return act[0].cpu().numpy()
        
    def push(self, 
        obs:np.ndarray,
        a:np.ndarray,
        reward:float,
        obs_next:np.ndarray,
        ter:bool
    ):
        self.replay_buf.append([
            obs, a, reward, obs_next, ter
        ])

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    def update_networks(self):
        if len(self.replay_buf) < self.warmup:
            return {
            'qloss': 0,
            'aloss': 0,
            'entropy': 0
        }

        obs, action, reward, obs_next, done = zip(*random.sample(self.replay_buf, self.batch_size))
        
        obs = tensor(np.array(obs))
        action = tensor(np.array(action))
        reward = tensor(reward).unsqueeze(1)
        obs_next = tensor(np.array(obs_next))
        done = tensor(done).unsqueeze(1)

        with torch.no_grad():
            a, logp = self.actor.act(obs_next)
            v_next = torch.stack([t.val(obs_next, a) for t in self.targets]).min(0).values
            v_next -= self.alpha * logp.sum(-1, keepdim=True)
            qtarget = reward + self.gamma * (1 - done) * v_next

        for c in self.critics:
            qvalue = c.val(obs, action)
            qloss = torch.mean((qvalue - qtarget) ** 2)
            c.optimizer.zero_grad()
            qloss.backward()
            c.optimizer.step()

        a, logp = self.actor.act(obs)
        logp = logp.sum(-1, keepdim=True)
        aloss = - torch.stack([c.val(obs, a) for c in self.critics]).min(0).values
        aloss += self.alpha * logp
        aloss = aloss.mean()

        self.actor.optimizer.zero_grad()
        aloss.backward()
        self.actor.optimizer.step()

        with torch.no_grad():
            for t, c in zip(self.targets, self.critics):
                self.polyak_update(t, c)

        return {
            'qloss': qloss.item(),
            'aloss': aloss.item(),
            'entropy': -logp.mean().item()
        }
