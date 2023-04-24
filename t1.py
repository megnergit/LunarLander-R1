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
import logging

from tqdm.rich import tqdm
# import pyglet
#=========================================================
# initialize environment
#---------------------------------------------------------
# env = gym.make("LunarLander-v2", render_mode='human')
def train(): 
    env = gym.make("LunarLander-v2")
#    env = SubprocVecEnv(env)

    eval_env = Monitor(gym.make("LunarLander-v2")) # separate environemnt for evaluation only
#    eval_env = SubprocVecEnv(eval_env)
    # env.action_space.seed(42)
    # observaiton, info = env.reset(seed=42)
    #---------------------------------------------------------
    # define callbacks
    #---------------------------------------------------------
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=11, 
                                                           min_evals=51, verbose=1)
    eval_callback = EvalCallback(eval_env, 
                                callback_after_eval=stop_train_callback, 
                                eval_freq = 1000,           # for ppo
                                n_eval_episodes=11, 
                                render=True, verbose=1)
#                                render=True, verbose=1)
    # default evaluation every 10_000 tiemsteps (?)
#                                eval_freq = 100_000, # for dqn
    #---------------------------------------------------------
    # training
    #---------------------------------------------------------
    model = DQN('MlpPolicy', env, verbose=0)
#    model = PPO('MlpPolicy', env, verbose=0)
    # model.learn(total_timesteps=int(1_000_000), 
    model.learn(total_timesteps=int(10_000), 
                callback=[eval_callback],
                progress_bar=True) 
    # model.learn(total_timesteps=int(1_000))
    model.save('dqn_10001_lunar')
    del model

    return 0

#---------------------------------------------------------
if __name__ == "__main__":
    train()


#---------------------------------------------------------

# model = DQN.load('dqn_1000000_lunar', env=env)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# obs = env.reset()
# # # action, _states = model.predict(obs, deterministic=True)
# # # env.render()
# # #---------------------------------------------------------
# # # evaluation only

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
