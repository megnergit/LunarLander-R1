import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
import time

# create the environment
env = gym.make('LunarLander-v2')

# create the model and the training loop
start_time = time.time()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e5))
cpu_time = time.time() - start_time

# repeat the experiment using GPU if available
if torch.cuda.is_available():
    start_time = time.time()
    env = DummyVecEnv([lambda: gym.make('LunarLander-v2')])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    env = VecFrameStack(env, n_stack=4)
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=int(1e5))
    gpu_time = time.time() - start_time
    print(f"Training time with GPU: {gpu_time:.2f} seconds")
else:
    print("No GPU available to test.")

# print the results
print(f"Training time with CPU: {cpu_time:.2f} seconds")

#===========================================================