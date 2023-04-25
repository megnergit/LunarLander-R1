import gymnasium as gym
import gym.envs.box2d

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
#    BaseCallback, 
    EvalCallback, 
#    EventCallback,
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

import logging

from tqdm.rich import tqdm
# from tqdm import rich
# from timeit import timeit
from time import time
import pretty_errors
import pdb
# import pyglet
#=========================================================
# initialize environment
#---------------------------------------------------------
# env = gym.make("LunarLander-v2", render_mode='human')
def make_env(env_id: str, rank: int, seed: int=0):

    def _init():

#        env = gym.make("LunarLander-v2")
        env = Monitor(gym.make(env_id))
#        pdb.set_trace()

        env.reset()
        return env
    
    set_random_seed(seed)
    return _init

#---------------------------------------------------------
def train2(): 

    n_env = 12
    env_id = "LunarLander-v2"

    env = SubprocVecEnv([make_env(env_id, i) for i in range(n_env)])
    eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(n_env)])

#    eval_env = Monitor(eval_env)


    # env.action_space.seed(42)
    # observaiton, info = env.reset(seed=42)
    #---------------------------------------------------------
    # define callbacks
    #---------------------------------------------------------
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=11, 
                                                           min_evals=51, verbose=1)
    eval_callback = EvalCallback(eval_env, 
                                callback_after_eval=stop_train_callback, 
                                eval_freq = 1_000,
                                n_eval_episodes=11, 
                                render=False, verbose=1)
    # default evaluation every 10_000 tiemsteps (?)
    #---------------------------------------------------------
    # training
    #---------------------------------------------------------
    # model = DQN('MlpPolicy', env, verbose=0)
    model = PPO('MlpPolicy', env, verbose=0)
#    model.learn(total_timesteps=int(1_000_000), 
    model.learn(total_timesteps=int(10_000), 
                callback=eval_callback,
                progress_bar=True) 
    # model.learn(total_timesteps=int(1_000))
    model.save('models/ppo_200004_lunar')
    del model

    return 0

#---------------------------------------------------------
def eval2():
    env = gym.make("LunarLander-v2")    
    model = PPO.load('models/ppo_200004_lunar')
#    model = DQN.load('models/dqn_lunar')
    obs = env.reset()

    for _ in range(1_000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        env.render()

#---------------------------------------------------------
if __name__ == "__main__":
    start_time = time()
    train2()
#    eval2()
    end_time = time()
    print(f"{end_time - start_time:4.2f} s")

#---------------------------------------------------------


# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# obs = env.reset()
# # action, _states = model.predict(obs, deterministic=True)
# # env.render()
# #---------------------------------------------------------
# # evaluation only
# for _ in range(1_000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, dones, info = env.step(action)
#     env.render()
#---------------------------------------------------------
#    print(action) 
    # 0 : do nothing
    # 1 : fire left 
    # 2 : fire down
    # 3 : fire right

#    observation, reward, terminated, truncated, info = env.step(action)

#    print(observation)
    # 8 dimensional
    # [x, y] (coordinate) 
    # [vx, vy] (velocities)
    # [angle]
    # [angular velocity]
    # [left leg touch groud?]
    # [right leg touch groud?]

#    print(reward)
    # moves away from pad : no reward
    # crash : -100
    # rest + 100

#    print(info)
#    print(terminated, truncated)
#    print('-----------------')

#    if terminated or truncated: 
#        obseration, info = env.reset()

#env.close()



#=========================================================
# END
#=========================================================
