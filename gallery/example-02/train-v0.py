# =============================================================================
# The roiginal version for training an agent in ColAvoidEnv-v0
# =============================================================================
import gym
import multiprocessing
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import time
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy


# start timer
tic = time.time()

# clear out outdated registered envs
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'ColAvoid-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    if 'ColAvoid-v1' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

# parallel environments
num_cpu = multiprocessing.cpu_count()

# make env
env = make_vec_env('gym_colavoid:ColAvoid-v0', 
                   n_envs=num_cpu, 
                   vec_env_cls=SubprocVecEnv, 
                   vec_env_kwargs=dict(start_method='fork'))
# for more info, see 
# https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb#scrollTo=AvO5BGrVv2Rk

# check env
# if the environment don't follow the interface, an error will be thrown
# this is not functional here since env has already been wrapped

# from stable_baselines3.common.env_checker import check_env
# check_env(env, warn=True)

# learning
model = A2C(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=400000) # 200s per 100000 steps
model.save("a2c_colavoid")

# stop timer
toc = time.time()
print('%.2f mins taken.' % ((toc-tic)/60))

    