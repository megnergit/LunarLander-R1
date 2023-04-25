import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from time import time

from tqdm.rich import tqdm
import pretty_errors
#===========================================================
# create the environment

def train3():
        
    env = gym.make('LunarLander-v2')
    eval_env = gym.make('LunarLander-v2')
    eval_env = Monitor(eval_env)

    #---------------------------------------------------------
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=11, 
                                                           min_evals=51, verbose=1)
    eval_callback = EvalCallback(eval_env, 
                                callback_after_eval=stop_train_callback, 
                                eval_freq = 1_000,
                                n_eval_episodes=11, 
                                render=False, verbose=1)
    #---------------------------------------------------------
    # create the model and the training loop
    start_time = time()
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=int(1e5), 
                callback=eval_callback,
                progress_bar=True) 

    cpu_time = time() - start_time

    # repeat the experiment using GPU if available
    if torch.cuda.is_available():
        start_time = time()
        env = DummyVecEnv([lambda: gym.make('LunarLander-v2')])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        env = VecFrameStack(env, n_stack=4)
        model = PPO('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=int(1e5), 
                    callback=eval_callback,
                    progress_bar=True) 

        gpu_time = time() - start_time
        print(f"Training time with GPU: {gpu_time:.2f} seconds")
    else:
        print("No GPU available to test.")

    # print the results
    print(f"Training time with CPU: {cpu_time:.2f} seconds")


#---------------------------------------------------------
if __name__ == "__main__":
#    start_time = time()
    train3()
#    eval2()
#    end_time = time()
#    print(f"{end_time - start_time:4.2f} s")

#===========================================================