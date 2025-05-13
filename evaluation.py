import os
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import torch
import pandas as pd

def evaluate(env :gym.Env, agent, run_id, steps, num_eval_episodes=3):
    video_dir = f'results/{run_id}/{steps}/'
    os.makedirs(video_dir, exist_ok=True)
    video = RecordVideo(env, video_dir, episode_trigger=lambda x: x % 4 == 0)  # Record all episodes in one video
    
    agent.eval()
    eval_rewards = []

    for eval_episode in range(num_eval_episodes):
        obs, _ = video.reset()
        video.close()
        video.start_recording(f'{eval_episode}')
        
        total_reward = 0
        done = False
        while not done:
            a = agent.predict(obs)
            obs, reward, ter, tru, _ = video.step(a)
            done = ter or tru
            total_reward += reward

        eval_rewards.append(total_reward)
        video.stop_recording()
        print(f"Evaluation Episode {eval_episode + 1}: Total Reward = {total_reward:.3f}")

    avg_reward = sum(eval_rewards) / num_eval_episodes
    print(f"Average Reward over {num_eval_episodes} Evaluation Episodes: {avg_reward:.3f}")
    torch.save(agent.state_dict(), f'results/{run_id}/{steps}/weight_{avg_reward}.pth')

    return avg_reward

def show_statistics(loss_infos):
    df = pd.DataFrame(loss_infos)
    mean_stats = pd.DataFrame({
        'mean': df.mean()
    })
    print(mean_stats)