# =============================================================================
# This is the code that learned to hide the agent into the corner.
# =============================================================================
import gym
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
import numpy as np


class SimulationWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env):
    # Retrieve the action space
    env.t = 0.02
    env.num_intruders = 0
    
    # Call the parent constructor, so we can access self.env later
    super(SimulationWrapper, self).__init__(env)
  
  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    print(str(self.reward) + '\t' + str(self.penalty_1) + '\t' + 
          str(self.penalty_2) + '\t' + str(self.penalty_3))
    observation, reward, done, info = self.env.step(action)
    return observation, reward, done, info


# clear out outdated registered envs
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'ColAvoid-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    if 'ColAvoid-v1' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

# without wrapping
# env = make_vec_env('gym_colavoid:ColAvoid-Test-v0', n_envs=1)

# wrapping
env = SimulationWrapper(gym.make('gym_colavoid:ColAvoid-v0'))
model = A2C.load("a2c_colavoid")

# start rendering
obs = env.reset()
try:
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
except KeyboardInterrupt:
    env.close()
    pass