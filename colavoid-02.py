#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Simple Collision Avoidance Environment
# discrete action space & (fake) continuous observation space

# changelog:
#   created: 04/03/2021
#       this code is created from the 2nd version of ColAvoidEnvDiscrete
#       located in gym-colavoid/gym_colavoid/envs/colavoid_env_v0
#       which is modified from the 1st version of ColAvoidEnvDiscrete
#       located in gym-colavoid/gym_colavoid-01/envs/colavoid_env_v0
#       copy the change log from 2nd ver. (differences from 1st ver.):
#           changed the way for escaping detection
#           changed penalty
#           changed r range_detection to 8
#           num_intruders = 1
#           experimental reward
#   

# Observation (states):
#   beams of distance
#   agent location (x, y)
# Action:
#   direction: 0-7 directions, 8 hovering
#   speed: max_speed
# Normalization:
#   none

import gym
from gym.utils import seeding
# from gym import spaces, logger
import random
import numpy as np
from gym.envs.classic_control import rendering

mode = 'learn' # stop any output in the env
# mode = 'eval' # output when using policy_evaluate
# mode = 'test' # output when using custom simulation code


class ColAvoidEnvDiscrete(gym.Env):
    # metadata = {'render.modes': ['human']}
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # SIMULATION PARAMETERS
        self.map_size = 20                      # length of side (m)
        self.num_dimensions = 2                 # number of dimensions 
        self.t = 0.20                           # time interval (s)

        # VEHICLE PARAMETERS
        self.R = 0.4                            # radius (m)
        self.S = 1                              # speed (m/s)
        self.num_agents = 1                     # number of vehicles
        self.num_directions = 8                 # number of moving directions
        self.num_states = 9                     # number of states in total
                                                # (it can also stop, etc)

        # LiDAR PARAMETERS
        self.range_detection = np.array([0,8])       
                                                # detection range (m)
                                                # (radius of vehicle ~ 5)
        self.range_view = np.array([0,2*np.pi]) # field of view
        self.num_bins = 40                      # number of bins
        self.resolution = 20                    # resolution

        # INTRUDER PARAMETERS
        self.r = 0.4                            # radius (m)
        self.s = 1.5                            # speed (m/s)
        self.num_intruders = 1                  # number of intruders at the 
                                                # same time

        # STATUS
        # self.EAST, self.NE, self.NORTH, self.NW, 
        # self.WEST, self.SW, self.SOUTH, self.SE, self.STOP = range(9)
        
        self.vel_agents = None                  # one of the 9 actions above
        self.pos_agents = None                  # position of agents
        self.vel_intruders = None               # moving direction of intruders
        self.pos_intruders = None               # position of intruders
        self.observation = None                 # 40 bins + 2 dimensions
        
        # lower bound and upper bound
        min_detection = self.range_detection[0] * np.ones(self.num_bins)
        max_detection = self.range_detection[1] * np.ones(self.num_bins)
        min_location = -self.map_size / 2 * np.ones(self.num_dimensions)
        max_location =  self.map_size / 2 * np.ones(self.num_dimensions)

        self.action_space = gym.spaces.Discrete(self.num_states)
        self.observation_space = gym.spaces.Box(
            np.concatenate((min_detection, min_location)),
            np.concatenate((max_detection, max_location)),
            dtype=np.float32)
        
        self.seed()
        self.viewer = None
        self.steps_beyond_done = None
        
        self.reward = None
        
        # self.tries = 0 # how many intruders generated
        # self.fails = 0 # how many collisions up to now
        # self.n_steps = 0 # how many steps (frames) up to now

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        # update the scene
        self.vel_agents = action
        self.update_map()

        # observation
        self.observation = self.observe()

        # done
        done = bool(self.check_collision()
                    or self.check_escape())
        
        # reward
        reward = [0] * 4
        # reward: survive
        reward[0] = 1.0 if not done else -100.0
        
        # penalty: distance to (0, 0)
        reward[1] = self.dist_to_position() ** 3
        
        # penalty: distance to closest intruder
        reward[2] = (self.range_detection[1] - self.dist_to_intruder()) ** 2
        
        # penalty: large changes on speed and direction
        reward[3] = 1.0 if self.vel_agents != 8 else 0.0
        
        self.reward = 1 * reward[0] - 1 * reward[1] - 0 * reward[2] - 0 * reward[3]
        
        if mode == 'eval':
            print("[I] Eval mode: step. vel: %.1f; rewards: %.1f, %.1f" % (self.vel_agents, 
                                                                           reward[0], 
                                                                           reward[1]))
        elif mode == 'test':
            print("[I] Test mode: step. vel: %.1f; rewards: %.1f, %.1f" % (self.vel_agents, 
                                                                           reward[0], 
                                                                           reward[1]))
        # if not done:
        #     pass
        #     # reward = reward_1 - 10 * penalty_1 - 1.0 * penalty_2 - 1.0 * penalty_3
        # elif self.steps_beyond_done is None:
        #     # just failed
        #     self.steps_beyond_done = 0
        #     # reward = reward_1 - 10 * penalty_1 - 1.0 * penalty_2 - 1.0 * penalty_3
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     # reward = reward_1 - 10 * penalty_1 - 1.0 * penalty_2 - 1.0 * penalty_3
        #     self.reset()

        info = {}
        # self.reward, self.reward_1, self.penalty_1, self.penalty_2, self.penalty_3 = \
        #     reward, reward_1, penalty_1, penalty_2, penalty_3
        
        # self.n_steps += 1                                         # stat: steps
        if done:
            # self.fails += 1                                     # stat: fails
            self.reset()
            
        # only for training
        # print("%d fails in %d tries and %d steps" % (self.fails, self.tries, self.steps))
        # print("%d %d %d" % (self.fails, self.tries, self.steps))
        
        # self.render()
        
            
        return self.observation, self.reward, done, info


    def reset(self):

        self.vel_agents = np.zeros(self.num_intruders)
        self.pos_agents = np.zeros([self.num_agents, self.num_dimensions])
        self.init_intruders()

        # self.observation = self.range_detection[0] * np.ones(42)
        self.observation = self.observe()

        return self.observation
    
    
    def update_map(self, ):
        self.move_agents()
        self.move_intruders()
        self.init_intruders()


    def move_agents(self):
        for i in range(self.num_agents):
            if not self.vel_agents == 9:
                angle = 2 * np.pi / self.num_directions * self.vel_agents
                dx = self.t * self.S * np.cos(angle)
                dy = self.t * self.S * np.sin(angle)
                self.pos_agents[i] = self.pos_agents[i] + np.array([dx, dy])


    def move_intruders(self):
        for i in range(self.num_intruders):
            angle = self.vel_intruders[i]
            dx = self.t * self.s * np.cos(angle)
            dy = self.t * self.s * np.sin(angle)
            self.pos_intruders[i] = self.pos_intruders[i] + np.array([dx, dy])


    def init_intruders(self):
        if self.vel_intruders is None or self.pos_intruders is None:
            self.vel_intruders = np.zeros(self.num_intruders)
            self.pos_intruders = self.map_size * np.ones([self.num_intruders, self.num_dimensions])
            # place them outside the map so that they can be re-initialized

        for i in range(self.num_intruders):
            # check if they lie outside a circle of r > 10
            if np.linalg.norm(self.pos_intruders[i]) > 8:
                # the place to initialize
                # place them on a circle with radius = 10m
                a = 2 * np.pi * random.random()
                # x = self.map_size / 2 * np.cos(angle)
                # y = self.map_size / 2 * np.sin(angle)
                x = 8 * np.cos(a)
                y = 8 * np.sin(a)
                self.pos_intruders[i] = np.array([x, y])
                
                # print(str(angle/np.pi)) ########################## unsolved
                
                # # strategy 1: head to a region around (0, 0)
                # # randomly select an angle for each intruder
                # opp_angle = - (np.pi - angle)
                # range_angle = np.arcsin(self.range_detection[1] / (self.map_size / 2))
                # rand_angle = opp_angle - range_angle + (2 * range_angle * random.random())
                # self.vel_intruders[i] = rand_angle

                # strategy 2: head to the agent directly
                dx = self.pos_agents[0][0] - self.pos_intruders[i][0]
                dy = self.pos_agents[0][1] - self.pos_intruders[i][1]
                if dx == 0:
                    dx = 0.000000000001
                if dx > 0:
                    self.vel_intruders[i] = np.arctan(dy/dx)
                else:
                    self.vel_intruders[i] = np.arctan(dy/dx) + np.pi
                    
                
        # new intruder generated
        # self.tries += 1                                         # stat: tries
                

    def observe(self):
        # initialize observation
        if self.observation is None:
            self.observation = np.zeros(self.num_bins + self.num_dimensions)

        for i in range(self.num_bins):
            angle = 2 * np.pi / self.num_bins * i
            for j in range(self.resolution):
                distance = self.range_detection[0] + (self.range_detection[1] 
                         - self.range_detection[0]) / self.resolution * j
                dx = distance * np.cos(angle)
                dy = distance * np.sin(angle)
                point = self.pos_agents[0] + np.array([dx, dy])
                
                touch_obstacle = False
                self.observation[i] = distance
                for k in range(self.num_intruders):
                    if np.linalg.norm(point - self.pos_intruders[k]) < self.r:
                        touch_obstacle = True
                        break
                    
                if touch_obstacle:
                    break
                
        self.observation[-2:] = self.pos_agents[0]
        return self.observation
                    
                    
    def check_collision(self):
        collision = False
        for i in range(self.num_intruders):
            distance = np.linalg.norm(self.pos_agents - self.pos_intruders[i])
            if distance < (self.R + self.r):
                collision = True
                break
        return collision
    
    
    def check_escape(self):
        escape = False
        # if np.max(np.abs(self.pos_agents[0])) > self.map_size / 2:
        #     escape = True
        if np.linalg.norm(self.pos_agents[0]) > 8:
            escape = True
        return escape
    
    
    def dist_to_position(self):
        distance = np.linalg.norm(self.pos_agents)
        return distance
    
    
    def dist_to_intruder(self):
        distance = np.min(self.observation[:-2])
        return distance
    

    def render(self, mode='human'):
        screen_size = 500
        scale = screen_size / self.map_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_size, screen_size)
            
            self.ranges = [0] * self.num_agents
            self.agents = [0] * self.num_agents
            self.agents_trans = [0] * self.num_agents
            self.intruders = [0] * self.num_intruders
            self.intruders_trans = [0] * self.num_intruders
    
            for i in range(self.num_agents):
                self.agents_trans[i] = rendering.Transform()
                        
                self.ranges[i] = rendering.make_circle(self.range_detection[1] * scale)
                self.ranges[i].set_color(.9, .9, .6)
                self.ranges[i].add_attr(self.agents_trans[i])
                self.viewer.add_geom(self.ranges[i])
                
                self.agents[i] = rendering.make_circle(self.R * scale)
                self.agents[i].set_color(.5, .5, .8)
                self.agents[i].add_attr(self.agents_trans[i])
                self.viewer.add_geom(self.agents[i])
                
            for i in range(self.num_intruders):
                self.intruders[i] = rendering.make_circle(self.r * scale)
                self.intruders_trans[i] = rendering.Transform()
                self.intruders[i].set_color(.8, .5, .5)
                self.intruders[i].add_attr(self.intruders_trans[i])
                self.viewer.add_geom(self.intruders[i])
                
        if self.observation is None:
            return None
        
        for i in range(self.num_agents):
            x = self.pos_agents[i][0] * scale + screen_size / 2.0
            y = self.pos_agents[i][1] * scale + screen_size / 2.0
            self.agents_trans[i].set_translation(x, y)
            
        for i in range(self.num_intruders):
            x = self.pos_intruders[i][0] * scale + screen_size / 2.0
            y = self.pos_intruders[i][1] * scale + screen_size / 2.0
            self.intruders_trans[i].set_translation(x, y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """
    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high
    
        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)
    
        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)
    
    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high -  self.low))
  
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
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        
        # manually added
        if done:
            if mode == 'eval':
                print('[I] Eval mode: NormActWrapper. done == True, call self.env.reset().')
            elif mode == 'test':
                print('[I] Test mode: NormActWrapper. done == True, call self.env.reset().')
            self.env.reset()
                
        return obs, reward, done, info


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
# from gym.wrappers import Monitor
from gym.wrappers import RescaleAction
from stable_baselines3.common.monitor import Monitor
import multiprocessing
from stable_baselines3.common.vec_env import SubprocVecEnv
    
import sys
import time
import os


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


if __name__ == '__main__':
    
    
    '''
    Basic validation and wrapping with Monitor and RescaleAction.
    '''
    env = ColAvoidEnvDiscrete()
    
    # if the environment don't follow the interface, an error will be thrown
    # must be used before being wrapped
    print("[I] Checking: original environment.")
    check_env(env, warn=True)               # this gives Monitor and asymmetric action space warning
    
    monitor_env = Monitor(env)
    print("[I] Checking: monitored environment.")
    check_env(monitor_env, warn=True)       # this gives asymmetric action space warning
    
    # this wrapper only works with continuous action space
    
    # # norm_monitor_env = RescaleAction(monitor_env, -1.0, 1.0)
    # norm_monitor_env = NormalizeActionWrapper(monitor_env)
    # print("[I] Checking: normalized environment.")
    # check_env(norm_monitor_env, warn=True)  # this will pass with no warning
    
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Wrapping with SubprocVecEnv.
    Default is DummyVecEnv. Use SubprocVecEnv for multiprocessing.
    Not sure what start_method is. Might cause error. Wrap everything in __main__.
    '''
    num_cpu = multiprocessing.cpu_count()
    
    file_path = os.path.dirname(os.path.realpath(__file__))
    file_name = time.strftime("a2c_colavoid_%Y_%m_%d_%H_%M_%S")
    log_dir   = os.path.join(file_path, file_name)
    # print("[I] File: directory '%s' created." % file_name)
    
    # used in train-v0 (not normalized) =======================================
    vec_monitor_env = make_vec_env(lambda: env, 
                                   n_envs=num_cpu, 
                                   # monitor_dir=log_dir, 
                                   vec_env_cls=SubprocVecEnv, 
                                   vec_env_kwargs=dict(start_method='fork'))
    # for more info, see 
    # https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb#scrollTo=AvO5BGrVv2Rk

    # it is equivalent to:
    # env = gym.make('CartPole-v1')
    # env = Monitor(env, log_dir)
    # env = DummyVecEnv([lambda: env])
    
    
    # print("[I] Checking: vectorized environment.")
    # check_env(vec_monitor_env, warn=True) # this will fail due to vectorization
    
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Build model with vec_norm_monitor_env.
    '''
    # model = A2C("MlpPolicy", vec_norm_monitor_env, verbose=1, n_steps=5, 
    #             tensorboard_log="./a2c_cartpole_tensorboard/")
    
    # used in train-v0 ========================================================
    model = A2C("MlpPolicy", vec_monitor_env, verbose=1, 
                tensorboard_log="./a2c_cartpole_tensorboard/")

    # # eval_env = monitor_env
    # eval_env = norm_monitor_env
    
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    # '''
    # Evaluate with monitor_env (before learning).
    # '''
    # mode = 'eval'
    # try:
    #     print("[I] Evaluating: monitored environment before learning.")
    #     mean_reward, std_reward = evaluate_policy(model, 
    #                                               eval_env, 
    #                                               n_eval_episodes=2, 
    #                                               deterministic=False, 
    #                                               render=True)
    # except Exception as e:
    #     print("\n[E] %s\n[E] %s" % (e, e.__doc__))
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #     input("[I] Press Enter to continue...")
    #     eval_env.close()
    # else:
    #     print(f"[I] mean_reward = {mean_reward:.2f} +/- {std_reward}")
    #     input("[I] Press Enter to continue...")
    #     eval_env.close() # close figure
    
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Training.
    '''
    mode = 'learn'
    # should stop any debugging output due to multiprocessing
    tic = time.time()
    
    print("[I] Training started.")
    
    # model.learn(total_timesteps=2e7) # 200s per 100000 steps
    
    # used in train-v0 ========================================================
    model.learn(total_timesteps=400000) # 200s per 100000 steps
    model.save("a2c_colavoid02_400k_reproduce")
    
    # file_name = time.strftime("a2c_colavoid_%Y_%m_%d_%H_%M_%S")
    # model.save(file_name)
    # print("[I] File: model '%s' saved." % file_name)
    
    toc = time.time()
    print('[I] Training finished in %.2f mins.' % ((toc-tic) / 60))
    input("[I] Press Enter to continue...")
    
    
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    # '''
    # Re-evaluate with monitor_env (after learning).
    # '''
    # mode = 'eval'
    # try:
    #     print("[I] Evaluating: monitored environment after learning.")
    #     mean_reward, std_reward = evaluate_policy(model, 
    #                                               eval_env, 
    #                                               n_eval_episodes=10, 
    #                                               deterministic=False, 
    #                                               render=True)
    # except Exception as e:
    #     print("\n[E] %s\n[E] %s" % (e, e.__doc__))
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #     input("[I] Press Enter to continue...")
    #     eval_env.close()
    # else:
    #     print(f"[I] mean_reward = {mean_reward:.2f} +/- {std_reward}")
    #     input("[I] Press Enter to continue...")
    #     eval_env.close() # close figure
        
        
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Testing code for test-v0.
    '''
    class SimulationWrapper(gym.Wrapper):
        """
        :param env: (gym.Env) Gym environment that will be wrapped
        """
        def __init__(self, env):
            # retrieve the action space
            env.t = 0.02
            
            # call the parent constructor, so we can access self.env later
            super(SimulationWrapper, self).__init__(env)
            
            self.n_steps = 0
            self.avg_steps = 1
            self.n_episodes = 0
            
            # debug
            if mode == 'test': print('[I] Test mode: init.')
        
        def reset(self):
            self.n_steps = 0
            self.avg_steps = 1
            
            # debug
            if mode == 'test':
                print('[I] Test mode, SimWrapper: reset.')
            
            return self.env.reset()

        def step(self, action):
            self.n_steps += 1
            
            print(['[I] Test mode, SimWrapper: step.',
                   'n_episodes:', self.n_episodes, 
                   'n_steps:', self.n_steps, 
                   'avg_steps:', self.avg_steps,
                   'reward:', self.reward, 
                   'action', action])
            
            observation, reward, done, info = self.env.step(action)
            
            # debug
            if mode == 'test':
                print('[I] Test mode, SimWrapper: step.')
            
            if done:
                if mode == 'test':
                    print('[I] Test mode, SimWrapper: done.')
                self.env.reset()
                self.avg_steps = (self.avg_steps * self.n_episodes + self.n_steps) / (self.n_episodes + 1)
                self.n_steps = 0
                self.n_episodes += 1
                
            
            return observation, reward, done, info
        
        
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    # model = A2C.load("a2c_colavoid02_400k")
    
    # '''
    # Testing without warpping.
    # '''
    # mode = 'test'
    # test_env = eval_env
    # obs = test_env.reset()
    # n_steps = 0
    # while n_steps <= 30:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = test_env.step(action)
    #     test_env.render()
    #     print("[I] Test mode. n_steps: %d" % n_steps)
    #     n_steps += 1
    # input("Press Enter to continue...")
    # test_env.close()

    
    '''
    Testing with warpping.
    '''
    # test_env = SimulationWrapper(eval_env)
    # used in test-v0 =========================================================
    test_env = SimulationWrapper(env)
    mode = 'test'
    obs = test_env.reset()
    while test_env.n_episodes <= 20:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = test_env.step(action)
        print("[I] Test mode. n_episodes: %d" % test_env.n_episodes)
        test_env.render()
    input("Press Enter to continue...")
    test_env.close()

    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Base code for evaluation.
    '''
    # # evaluation using eval_env and build model using vectorized env
    # # separate env for evaluation
    # eval_env = env
    # # wrap it to get vectorized env
    # env = make_vec_env(lambda: env, n_envs=2)
    # # check_env(env, warn=True)                 # this will fail
    # # check_env(env.unwrapped, warn=True)       # this will pass
    # # generate model using vectorized env
    # model = A2C("MlpPolicy", env, verbose=1)
    # # random agent before training, evaluation using original env
    # try:
    #     mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)
    # except Exception as e:
    #     print("\n[E] %s" % (e))
    #     print("\n[E] %s" % (e.__doc__))
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #     input("[I] Press Enter to continue...")
    #     eval_env.close()
    # else:
    #     print(f"[I]mean_reward={mean_reward:.2f} +/- {std_reward}")
    #     # eval_env.close() # close figure