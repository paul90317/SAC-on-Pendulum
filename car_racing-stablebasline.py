import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Create the environment
env = gym.make('CarRacing-v3', render_mode='rgb_array')
env = Monitor(env)  # Monitor to log statistics
env = DummyVecEnv([lambda: env])  # Vectorized environment for Stable-Baselines3

# Create the SAC model
model = SAC('CnnPolicy', env, verbose=1, buffer_size=100000, learning_rate=3e-4, batch_size=256, gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1, ent_coef='auto')

# Train the model
model.learn(total_timesteps=100000, log_interval=10)

# Save the model
model.save('sac_car_racing')

# Evaluate the model
def evaluate_model(env, model, num_episodes=5):
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Evaluate the trained model
evaluate_model(env, model)