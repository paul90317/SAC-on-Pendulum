from torch.distributions import Categorical
import numpy as np
import torch
import torch.nn as nn
from torch import optim
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
    def dist(self, obs: Tensor) -> Categorical:
        return self(obs)

class CriticInterface(NetWork):
    def val(self, obs: Tensor) -> Tensor:
        return self(obs)

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
        self.critics = [critic() for _ in range(self.n_critics)]
        self.targets = [critic() for _ in range(self.n_critics)]
        self.qnets = nn.ModuleList(self.critics + self.targets)

        for t, c in zip(self.critics, self.targets):
            t.load_state_dict(c.state_dict())
        
        self.replay_buf = deque(maxlen=replay_buf_size)

    def predict(self, obs:np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = tensor(obs).unsqueeze(0)
            dist = self.actor.dist(obs)
            if self.training:
                return dist.sample()[0].item()
            return dist.probs[0].max(-1).indices.item()
        
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
        action = torch.tensor(action, device=device, dtype=torch.int64).unsqueeze(-1)
        obs_next = tensor(np.array(obs_next))
        reward = tensor(reward)
        done = tensor(done)

        with torch.no_grad():
            dist = self.actor.dist(obs_next)
            v_next = torch.stack([(t.val(obs_next) * dist.probs).sum(1) for t in self.targets]).min(0).values
            v_next -= self.alpha * dist.entropy()
            qtarget = reward + self.gamma * (1 - done) * v_next

        for c in self.critics:
            qvalue = c.val(obs).gather(-1, action).squeeze(-1)
            qloss = torch.mean((qvalue - qtarget) ** 2)
            c.optimizer.zero_grad()
            qloss.backward()
            c.optimizer.step()

        dist = self.actor.dist(obs)
        entropy = dist.entropy()
        aloss = - torch.stack([(c.val(obs) * dist.probs).sum(1) for c in self.critics]).min(0).values
        aloss -= self.alpha * entropy
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
            'entropy': entropy.mean().item()
        }
