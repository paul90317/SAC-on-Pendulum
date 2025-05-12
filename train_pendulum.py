import time
import gymnasium as gym
from trainer import Trainer

import argparse

# =================================================================================
# arguments

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int)
args = parser.parse_args()

env = gym.make('Pendulum-v1')
trainer = Trainer(env)

batch_size = 64
num_episodes = 1000

start_time = time.perf_counter()

for e in range(num_episodes):

    trainer.reset()

    total_reward = 0
    steps = 0
    done = False
    while not done:
        _, reward, ter, tru, _ = trainer.step()

        done = ter or tru
        total_reward += reward

        steps += 1
        if steps % 2 == 0: 
            trainer.update_networks()

    after_episode_time = time.perf_counter()
    time_elapsed = after_episode_time - start_time
    time_remaining = time_elapsed / (e + 1) * (num_episodes - (e + 1))

    print(f'Episode {e:4.0f} | Return {total_reward:9.3f} | Steps {steps:4.0f} | Remaining time {round(time_remaining / 3600, 2):5.2f} hours')

trainer.save_actor(
    save_dir='results/trained_policies_pth/',
    filename=f'{args.run_id}.pth'
)

env.close()