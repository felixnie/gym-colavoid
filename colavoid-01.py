#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Simple Collision Avoidance Environment
# continuous action space & continuous observation space

# changelog:
#   updated: 04/01/2021
#   added ray casting in render
#   corrected observation
#   updated: 03/31/2021
#   modified from the untested version with continuous action space
#   adopted changes from the latest version with discrete action space
#   please refer to the change log from the latest discrete version
#   to-do: change strategy 2 in init_intruders

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
from gym import spaces, logger
import random
import time
import numpy as np
from gym.envs.classic_control import rendering


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
        self.S = np.array([0.0, 1.0])           # range of speed (m/s)
        self.DIR = np.array([-np.pi, np.pi])    # range of direction
        self.num_agents = 1                     # number of vehicles

        # LiDAR PARAMETERS
        self.max_lidar = 5.0                    # detection range (m)
        self.num_beams = 90                     # number of LiDAR beams
        # self.resolution = 
        self.buffer_size = 2

        # INTRUDER PARAMETERS
        self.r = 0.4                            # radius (m)
        self.s = 2.0                            # speed (m/s)
        self.num_intruders = 1                  # number of intruders at the 
                                                # same time

        # ACTION SPACE
        # speed
        min_speed   = np.ones(self.num_agents) * self.S[0]
        max_speed   = np.ones(self.num_agents) * self.S[1]
        # moving direction
        min_dir     = np.ones(self.num_agents) * self.DIR[0]
        max_dir     = np.ones(self.num_agents) * self.DIR[1]
        
        self.action_space = gym.spaces.Box(
            np.concatenate((min_speed, min_dir)), 
            np.concatenate((max_speed, max_dir)),
            dtype=np.float32)


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
        self.observation    = self.buffer.flatten('F')

        self.init_agents()
        self.init_intruders(reset=True)

        self.observation = self.observe()
        
        # experimental
        self.step_counter = 0
        self.intruder_escape = False
        
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
        print(self.vel_agents)
        
        # update the scene
        self.move_agents()
        self.move_intruders()
        
        # reset the outdated intruders
        self.init_intruders()

        # observation
        self.observation = self.observe()
        
        
        # done
        # strategy 1: continuous training
        # self.done = bool(self.check_collision() or 
        #                  self.check_escape())
        # strategy 2: episodic training
        self.step_counter += 1 # not using this
        if np.any(np.linalg.norm(self.pos_intruders) > 8):
            self.intruder_escape = True
            
        self.done = bool(self.check_collision() or 
                         self.check_escape() or
                         self.intruder_escape)
        
        # reward
        reward = [0] * 1
        reward[0] = 1.0
        
        penalty = [0] * 3
        penalty[0] = 1e5 if self.done else 0.0
        penalty[1] = self.dist_to_origin()**2
        penalty[2] = (self.max_lidar - self.dist_to_intruder())**2
        self.reward = np.sum(reward) - np.sum(penalty[0:2])
        
        # info
        self.info = {}

        return self.observation, self.reward, self.done, self.info


    def move_agents(self):
        for i in range(self.num_agents):
            dx = self.t * self.vel_agents[i, 0] * np.cos(self.vel_agents[i, 1])
            dy = self.t * self.vel_agents[i, 0] * np.sin(self.vel_agents[i, 1])
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



from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import sys
import os


'''
Basic validation and wrapping with RescaleAction.
'''
from stable_baselines3.common.env_checker import check_env
env = ColAvoidEnv()
# if the environment don't follow the interface, an error will be thrown
# must be used before being wrapped
# check_env(env, warn=True) # this gives asymmetric action space warning
from gym.wrappers import RescaleAction
rescaled_env = RescaleAction(env, -1, 1)
check_env(rescaled_env, warn=True) # this will pass with no warning


'''
Wrapping with SubprocVecEnv.
Default is DummyVecEnv. Use SubprocVecEnv for multiprocessing.
Not sure what start_method is. Might cause error. Wrap everything in __main__.
'''
import multiprocessing
from stable_baselines3.common.vec_env import SubprocVecEnv
num_cpu = multiprocessing.cpu_count()
vec_rescaled_env = make_vec_env(lambda: rescaled_env, 
                                n_envs=num_cpu, 
                                vec_env_cls=SubprocVecEnv, 
                                vec_env_kwargs=dict(start_method='fork'))


'''
Build model with vec_rescaled_env and evaluate with rescaled_env.
(before training)
'''
model = A2C("MlpPolicy", vec_rescaled_env, verbose=1)
eval_env = rescaled_env
try:
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)
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


'''
Learning.
'''
import time
tic = time.time()
model.learn(total_timesteps=1e3) # 200s per 100000 steps
# model.save("a2c_colavoid")

# stop timer
toc = time.time()
print('[I] %.2f mins taken.' % ((toc-tic) / 60))


'''
Build model with vec_rescaled_env and evaluate with rescaled_env.
(after training)
'''
eval_env = rescaled_env
try:
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)
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

    
# train the agent
# tic = time.time()                           # start timer
# model.learn(total_timesteps=int(1e3))


# # # Save the agent
# # model.save("dqn_lunar")
# # del model  # delete trained model to demonstrate loading

# # model = DQN.load("dqn_lunar")

# # Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

# # after training
# try:
#     mean_reward, std_reward = evaluate_policy(model, eval_env, 
#                                               n_eval_episodes=10, 
#                                               deterministic=True,
#                                               render=True)
# except Exception as e:
#     print("\n")
#     print("[E] %s" % (e))
#     print("\n")
#     print("[E] %s" % (e.__doc__))
#     input("[I] Press Enter to continue...")
#     eval_env.close()
# else:
#     print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")