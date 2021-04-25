#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Simple Collision Avoidance Environment 2
# continuous action space & continuous observation space

# changelog:
#   created: 03/31/2021
#       modified from the untested version with continuous action space
#       adopted changes from the latest version with discrete action space
#       please refer to the change log from the latest discrete version
#       to-do: change strategy 2 in init_intruders
#   updated: 04/01/2021
#       added ray casting in render
#       corrected observation
#   updated: 04/03/2021
#       changed reward
#   updated: 04/11/2021
#       change actions so that normalization is not needed

# Observation (states):
#   X distance to N obstacles (direction + distance)
#   X distance to M-1 agents (direction + distance)
#   beams of distance
#   agent location (x, y)
# Action:
#   direction: 0~2*pi
#   speed: 0~max_speed
# Normalization:
#   to 0~1

import gym
from gym.utils import seeding
# from gym import spaces, logger
import random
import numpy as np
from gym.envs.classic_control import rendering

mode = 'learn' # stop any output in the env
# mode = 'eval' # output when using policy_evaluate
# mode = 'test' # output when using custom simulation code

class ColAvoidEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # SIMULATION PARAMETERS
        self.map_size = 20.0                    # length of side (m)
        self.t = 0.20                           # time interval (s)

        # VEHICLE PARAMETERS
        self.R = 0.4                            # radius (m)
        # self.S = np.array([0.0, 1.0])           # range of speed (m/s)
        # self.DIR = np.array([-np.pi, np.pi])    # range of direction
        self.num_agents = 1                     # number of vehicles

        # LiDAR PARAMETERS
        self.max_lidar = 5.0                    # detection range (m)
        self.num_beams = 40                     # number of LiDAR beams
        # self.resolution = 
        self.buffer_size = 1

        # INTRUDER PARAMETERS
        self.r = 0.4                            # radius (m)
        self.s = 1.5                            # speed (m/s)
        self.num_intruders = 1                  # number of intruders at the 
                                                # same time

        # ACTION SPACE
        # speed
        self.action_space = gym.spaces.Box(
            np.array([-1, -1]), 
            np.array([ 1,  1]),
            dtype=np.float32)
        # min_speed   = np.ones(self.num_agents) * self.S[0]
        # max_speed   = np.ones(self.num_agents) * self.S[1]
        # # moving direction
        # min_dir     = np.ones(self.num_agents) * self.DIR[0]
        # max_dir     = np.ones(self.num_agents) * self.DIR[1]
        
        # self.action_space = gym.spaces.Box(
        #     np.concatenate((min_speed, min_dir)), 
        #     np.concatenate((max_speed, max_dir)),
        #     dtype=np.float32)


        # OBSERVATION SPACE
        # lower bound and upper bound
        min_lidar   = np.ones(self.num_beams) * 0.0
        max_lidar   = np.ones(self.num_beams) * self.max_lidar
        min_pos     = np.ones(2) * self.map_size / 2 * -1
        max_pos     = np.ones(2) * self.map_size / 2

        self.observation_space = gym.spaces.Box(
            np.repeat(np.concatenate((min_lidar, min_pos)), self.buffer_size),
            np.repeat(np.concatenate((max_lidar, max_pos)), self.buffer_size),
            dtype=np.float32)
        
        
        # GLOBAL VARIABLES
        self.vel_agents     = None              # [speed, direction]
        self.pos_agents     = None              # [x, y]
        self.vel_intruders  = None              # [speed, direction]
        self.pos_intruders  = None              # [x, y]
        self.buffer         = None              # [buffer_0; buffer_1]
        self.observation    = None              # [buffer(:), pos_agents(:)]
        
        self.reward         = None
        self.done           = None
        self.info           = None

        self.seed()
        self.viewer         = None
        
        # experimental
        # self.step_counter = None
        # self.intruder_escape = None
        # self.steps_beyond_done = None
        
        # debug
        if mode == 'eval': print('[I] Eval mode: init.')
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def reset(self):
        '''
        The function to initialize the env and variables.
        
        '''
        self.vel_agents     = np.zeros([self.num_agents, 2])
        self.pos_agents     = np.zeros([self.num_agents, 2])
        self.vel_intruders  = np.zeros([self.num_intruders, 2])
        self.pos_intruders  = np.zeros([self.num_intruders, 2])
        self.buffer         = np.zeros([self.buffer_size, self.num_beams + 2])
        # self.observation    = self.buffer.flatten('F')

        self.init_agents()
        self.init_intruders(reset=True)

        self.observation = self.observe()
        
        # experimental
        # self.step_counter = 0
        # self.intruder_escape = False
        # self.steps_beyond_done = None
        
        # debug
        if mode == 'eval': 
            print('[I] Eval mode: reset.')
        elif mode == 'test':
            print('[I] Test mode: reset.')
        
        return self.observation
    
    
    def init_agents(self):
        '''
        Place the agents on their origin. Run with reset().

        Returns
        -------
        None.

        '''
        if self.num_agents == 1:
            pass
        elif self.num_agents == 3:
            for i in range(self.num_agents):
                angle = np.pi/2 + i * 2 * np.pi / self.num_agents
                radius = 2.0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                self.pos_agents[i] = np.array([x, y])
        else:
            pass
    
    
    def init_intruders(self, reset=False):
        '''
        Place the intruders. Run every time an intruder escapes or reset().
        
        Params
        ------
        escape range
        
        Styles for pos & Strategies for vel
        -----------------------------------
        Style 1: generate on a square where r = a/2
        Style 2: generate on a circle where r = radius
        Strategy 1: head toward a circle of r = radius when R = born radius
        Strategy 2: head toward a circle of r = radius (with multi-agent support)
        Strategy 3: head toward the agent directly (without multi-agent support)

        Returns
        -------
        None.

        '''
        for i in range(self.num_intruders):
            # check if they lie outside the map:
            # if np.any(np.abs(self.pos_intruders[i]) >= self.map_size / 2):
            if np.linalg.norm(self.pos_intruders[i]) > 8 or reset is True:
                # style 1: born on the edges of the map
                # angle = 2 * np.pi * random.random()
                # r = self.map_size / 2
                # if angle >= 7/4 * np.pi or angle < 1/4 * np.pi:
                #     x, y = r, r*np.tan(angle)
                # elif angle >= 1/4 * np.pi and angle < 3/4 * np.pi:
                #     x, y = r/np.tan(angle), r
                # elif angle >= 3/4 * np.pi and angle < 5/4 * np.pi:
                #     x, y = -r, -r*np.tan(angle)
                # elif angle >= 5/4 * np.pi and angle < 7/4 * np.pi:
                #     x, y = -r/np.tan(angle), -r
                # self.pos_intruders[i] = np.array([x, y])
                
                # style 2: born on a circle of radius r
                angle = 2 * np.pi * random.random()
                r = 8.0
                x, y = r*np.cos(angle), r*np.sin(angle)
                self.pos_intruders[i] = np.array([x, y])
                
                # strategy 1: head to a region aroud (0, 0)
                # rel_dir = -(np.pi - angle)
                # R, r = 8.0, 2.0 # R is where it's born; r is the target circle
                # dir_range = np.arcsin(r / R)
                # dir_rand = rel_dir - dir_range + 2 * dir_range * random.random()
                # self.vel_intruders[i] = np.array([self.s, dir_rand])
                             
                # strategy 2: head to a region aroud the centre of mass of the agents
                # centre = np.mean(self.pos_agents, axis=0)
                # rel_pos = centre - self.pos_intruders[i]
                # rel_dir = np.arctan(rel_pos[1] / rel_pos[0])
                # rel_dist = np.linalg.norm(rel_pos)
                # r = 2.0
                # dir_range = np.arcsin(r / rel_dist)
                # dir_rand = rel_dir - dir_range + 2 * dir_range * random.random()
                # self.vel_intruders[i] = np.array([self.s, dir_rand])
                # # not tested yet! need to modify like strategy 3.
                
                # strategy 3: head to the agent directly (corrected)
                dx = self.pos_agents[0][0] - self.pos_intruders[i][0]
                dy = self.pos_agents[0][1] - self.pos_intruders[i][1]
                if dx == 0:
                    dx = 0.00000001
                if dx > 0:
                    direction = np.arctan(dy/dx)
                else:
                    direction = np.arctan(dy/dx) + np.pi
                self.vel_intruders[i] = np.array([self.s, direction])
    
    
    def observe(self):
        '''
        Return observation using a new method for ray casting.
        No multi-agent support.
        Corrected angle computation. 04/01/2021
        
        '''
        for i in range(self.num_agents):
            buffer = self.max_lidar * np.ones(self.num_beams)
            for j in range(self.num_intruders):
                dx = self.pos_intruders[j][0] - self.pos_agents[i][0]
                dy = self.pos_intruders[j][1] - self.pos_agents[i][1]
                if dx > 0:
                    angle = np.arctan(dy / dx)
                else:
                    angle = np.arctan(dy / dx) + np.pi
                angles = 2 * np.pi / self.num_beams * np.array(range(self.num_beams))
                k = np.tan(angles)
                d = np.abs(k * dx - dy) / np.sqrt(k**2 + 1)
                temp_buffer = self.max_lidar * np.ones(self.num_beams)
                for n in range(self.num_beams):
                    if d[n] <= self.r:
                        a = angle - angles[n]
                        L = np.linalg.norm([dx, dy]) * np.cos(a)
                        l = np.sqrt(self.r**2 - d[n]**2)
                        if L - l < self.max_lidar and L - l > 0:
                            temp_buffer[n] = L - l
                # print(temp_buffer)
                # element-wise minimum
                buffer = np.minimum(buffer, temp_buffer)
            
        # add agent position
        buffer = np.concatenate((buffer, self.pos_agents.flatten('F')))
        
        # update buffer
        for i in range(self.buffer_size):
            if i < self.buffer_size - 1:
                self.buffer[i] = self.buffer[i+1]
            else:
                self.buffer[i] = buffer
        self.observation = self.buffer.flatten('F').astype('float32')
        # print(self.observation)
        return self.observation
    
    
    def step(self, action):
        '''
        Function step()
        Added episodic training.
        '''        
        # # check if the actions are legitimate
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg
        
        # action space to velocity matrix (speed, direction)
        self.vel_agents = np.reshape(action, (self.num_agents, 2), order='F')
        
        # update the scene
        self.move_agents()
        self.move_intruders()
        
        # reset the outdated intruders
        self.init_intruders()

        # observation
        self.observation = self.observe()
        
        
        # done
        # strategy 1: continuous training
        self.done = bool(self.check_collision() or 
                         self.check_escape())
        # strategy 2: episodic training
        # self.step_counter += 1 # not using this
        # if np.any(np.linalg.norm(self.pos_intruders) > 8):
        #     self.intruder_escape = True
        # self.done = bool(self.check_collision() or 
        #                  self.check_escape() or
        #                  self.intruder_escape)
        
        # reward
        reward = [0] * 2
        
        # penalyze short distance to intruders
        shift_param = 1e-6 # the smaller, the smaller reward(R) is
        shape_param = 1 + 1e-2 # the smaller, the larger dynamic range log(x) is
        d = np.clip(self.dist_to_intruder(), a_min=self.R, a_max=self.max_lidar)
        # reward[0] = np.log((d-self.R+shift_param)/(5-self.R+shift_param)) / np.log(shape_param)
        reward[0] = 1.0 if not self.done else 0.0
        
        # penalyze long distance to origins
        shape_param = 3
        d = self.dist_to_origin()
        reward[1] = - d ** shape_param

        self.reward = 1 * reward[0] + 0.3 * reward[1]
        
        if mode == 'eval':
            print("[I] Eval mode: step. vel: %.1f, %.1f; rewards: %.1f, %.1f" % (self.vel_agents[0,0], 
                                                                                 self.vel_agents[0,1], 
                                                                                 reward[0], 
                                                                                 reward[1]))
        elif mode == 'test':
            print("[I] Test mode: step. vel: %.1f, %.1f; rewards: %.1f, %.1f" % (self.vel_agents[0,0], 
                                                                                 self.vel_agents[0,1], 
                                                                                 reward[0], 
                                                                                 reward[1]))
        
        # info
        self.info = {}
        
        if self.done:
            if mode == 'eval': 
                print('[I] Eval mode: step. done == True, not call self.reset().')
            elif mode == 'test':
                print('[I] Test mode: step. done == True, not call self.reset().')
            # self.reset()

        return self.observation, self.reward, self.done, self.info


    def move_agents(self):
        for i in range(self.num_agents):
            # dx = self.t * self.vel_agents[i, 0] * np.cos(self.vel_agents[i, 1])
            # dy = self.t * self.vel_agents[i, 0] * np.sin(self.vel_agents[i, 1])
            dx = self.t * self.vel_agents[i, 0]
            dy = self.t * self.vel_agents[i, 1]
            self.pos_agents[i] = self.pos_agents[i] + np.array([dx, dy])


    def move_intruders(self):
        for i in range(self.num_intruders):
            dx = self.t * self.vel_intruders[i, 0] * np.cos(self.vel_intruders[i, 1])
            dy = self.t * self.vel_intruders[i, 0] * np.sin(self.vel_intruders[i, 1])
            self.pos_intruders[i] = self.pos_intruders[i] + np.array([dx, dy])

    
    def check_collision(self):
        '''
        Colliison checker.
        No multi-agent support.
        
        '''
        collision = False
        for i in range(self.num_intruders):
            distance = np.linalg.norm(self.pos_agents[0] - self.pos_intruders[i])
            if distance < (self.R + self.r):
                collision = True
                break
        return collision
    
    
    def check_escape(self):
        '''
        Escape checker.
        No multi-agent support.
        
        '''
        escape = False
        # if np.max(np.abs(self.pos_agents[0])) > self.map_size / 2:
        #     escape = True
        if np.linalg.norm(self.pos_agents[0]) > 8:
            escape = True
        return escape

    
    def dist_to_origin(self):
        distance = np.linalg.norm(self.pos_agents)
        return distance
    
    
    def dist_to_intruder(self):
        distance = np.min(self.buffer[-1, :-2])
        return distance
    
    
    def render(self, mode='human'):
        '''
        Rendering function. Added ray casting.

        Parameters
        ----------
        mode : TYPE, optional
            DESCRIPTION. The default is 'human'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        screen_size = 500
        scale = screen_size / self.map_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_size, screen_size)

            self.ranges = [0] * self.num_agents
            self.agents = [0] * self.num_agents
            self.agents_trans = [0] * self.num_agents
            self.intruders = [0] * self.num_intruders
            self.intruders_trans = [0] * self.num_intruders
            
            self.beams = [0] * self.num_beams
    
            for i in range(self.num_agents):
                self.agents_trans[i] = rendering.Transform()
                
                # lidar circle
                self.ranges[i] = rendering.make_circle(self.max_lidar * scale)
                self.ranges[i].set_color(.9, .9, .6)
                self.ranges[i].add_attr(self.agents_trans[i])
                self.viewer.add_geom(self.ranges[i])
                
                # ray casting
                angles = 2 * np.pi / self.num_beams * np.array(range(self.num_beams))
                x = self.buffer[-1, :-2] * np.cos(angles) * scale
                y = self.buffer[-1, :-2] * np.sin(angles) * scale
                for m in range(self.num_beams):
                    vertices = (self.pos_agents[i][0], self.pos_agents[i][1]), (x[m], y[m])
                    self.beams[m] = rendering.make_polyline(vertices)
                    self.beams[m].set_color(1.0, .0, .0)
                    self.beams[m].add_attr(self.agents_trans[i])
                    self.beams[m].set_linewidth(2)
                    self.viewer.add_geom(self.beams[m])
                    
                # agent circle
                self.agents[i] = rendering.make_circle(self.R * scale)
                self.agents[i].set_color(.5, .5, .8)
                self.agents[i].add_attr(self.agents_trans[i])
                self.viewer.add_geom(self.agents[i])
                
            for i in range(self.num_intruders):
                self.intruders_trans[i] = rendering.Transform()
                
                self.intruders[i] = rendering.make_circle(self.r * scale)
                self.intruders[i].set_color(.8, .5, .5)
                self.intruders[i].add_attr(self.intruders_trans[i])
                self.viewer.add_geom(self.intruders[i])
        
        # run every step:
        for i in range(self.num_agents):
            # beams
            angles = 2 * np.pi / self.num_beams * np.array(range(self.num_beams))
            x = self.buffer[self.buffer_size-1, 0:-2] * np.cos(angles) * scale
            y = self.buffer[self.buffer_size-1, 0:-2] * np.sin(angles) * scale
            for m in range(self.num_beams):
                vertices = (self.pos_agents[i][0], self.pos_agents[i][1]), (x[m], y[m])
                self.beams[m].v = vertices
            
            x = self.pos_agents[i, 0] * scale + screen_size / 2.0
            y = self.pos_agents[i, 1] * scale + screen_size / 2.0
            self.agents_trans[i].set_translation(x, y)
            
        for i in range(self.num_intruders):
            x = self.pos_intruders[i, 0] * scale + screen_size / 2.0
            y = self.pos_intruders[i, 1] * scale + screen_size / 2.0
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
    env = ColAvoidEnv()
    
    # if the environment don't follow the interface, an error will be thrown
    # must be used before being wrapped
    print("[I] Checking: original environment.")
    check_env(env, warn=True)               # this gives Monitor and asymmetric action space warning
    
    monitor_env = Monitor(env)
    print("[I] Checking: monitored environment.")
    check_env(monitor_env, warn=True)       # this gives asymmetric action space warning
    
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
    
    vec_env = make_vec_env(lambda: env, 
                           n_envs=num_cpu, 
                           # monitor_dir=log_dir, 
                           # wrapper_class=NormalizeActionWrapper, 
                           vec_env_cls=SubprocVecEnv, 
                           vec_env_kwargs=dict(start_method='fork'))
    # it is equivalent to:
    # env = gym.make('CartPole-v1')
    # env = Monitor(env, log_dir)
    # env = DummyVecEnv([lambda: env])
    # print("[I] Checking: vectorized environment.")
    # check_env(vec_norm_monitor_env, warn=True) # this will fail
    
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Build model with vec_norm_monitor_env.
    '''
    model = A2C("MlpPolicy", vec_env, verbose=1, n_steps=5)
                # tensorboard_log="./tb-colavoid-03/")
    # eval_env = monitor_env
    eval_env = monitor_env
    
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Evaluate with monitor_env (before learning).
    '''
    mode = 'eval'
    try:
        print("[I] Evaluating: monitored environment before learning.")
        mean_reward, std_reward = evaluate_policy(model, 
                                                  eval_env, 
                                                  n_eval_episodes=2, 
                                                  deterministic=False, 
                                                  render=True)
    except Exception as e:
        print("\n[E] %s\n[E] %s" % (e, e.__doc__))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        input("[I] Press Enter to continue...")
        eval_env.close()
    else:
        print(f"[I] mean_reward = {mean_reward:.2f} +/- {std_reward}")
        input("[I] Press Enter to continue...")
        eval_env.close() # close figure
    
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Training.
    '''
    mode = 'learn'
    # should stop any debugging output due to multiprocessing
    tic = time.time()
    
    print("[I] Training started.")
    # os.system("tensorboard --logdir=./tb-colavoid-01/")
    model.learn(total_timesteps=1e6) # 200s per 100000 steps
    
    # file_name = time.strftime("a2c_colavoid_%Y_%m_%d_%H_%M_%S")
    # model.save(file_name)
    # print("[I] File: model '%s' saved." % file_name)
    
    toc = time.time()
    print('[I] Training finished in %.2f mins.' % ((toc-tic) / 60))
    input("[I] Press Enter to continue...")
    
    
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    '''
    Re-evaluate with monitor_env (after learning).
    '''
    mode = 'eval'
    try:
        print("[I] Evaluating: monitored environment after learning.")
        mean_reward, std_reward = evaluate_policy(model, 
                                                  eval_env, 
                                                  n_eval_episodes=10, 
                                                  deterministic=False, 
                                                  render=True)
    except Exception as e:
        print("\n[E] %s\n[E] %s" % (e, e.__doc__))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        input("[I] Press Enter to continue...")
        eval_env.close()
    else:
        print(f"[I] mean_reward = {mean_reward:.2f} +/- {std_reward}")
        input("[I] Press Enter to continue...")
        eval_env.close() # close figure
        
        
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
            env.t = 0.001
            
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
                print('[I] Test mode: reset.')
            
            return self.env.reset()

        def step(self, action):
            self.n_steps += 1
            print("%.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f"
                  % (self.n_episodes, self.n_steps, self.avg_steps,
                     self.reward, action[0], action[1]))
            
            observation, reward, done, info = self.env.step(action)
            
            # debug
            if mode == 'test': print('[I] Test mode: reset.')
            
            if done:
                # print('done!')
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
    test_env = SimulationWrapper(eval_env)
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