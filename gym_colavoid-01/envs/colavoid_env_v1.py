#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Simple Collision Avoidance Environment - Continuous & Distance

# Observation (states): 
#     distance to N obstacles (direction + distance)
#     distance to M-1 agents (direction + distance)
#     agent location (x, y)
# Action:
#     direction: 0~pi
#     speed: 0~max_speed
# Normalization:
#     to 0~1

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
        self.map_size = 20                      # length of side (m)
        self.t = 0.20                           # time interval (s)

        # VEHICLE PARAMETERS
        self.R = 0.4                            # radius (m)
        self.S = np.array([0.0,1.0])            # range of speed (m/s)
        self.DIR = np.array([0.0,np.pi])        # range of direction
        self.num_agents = 1                     # number of vehicles

        # LiDAR PARAMETERS
        self.max_detection = 5                  # detection range (m)

        # INTRUDER PARAMETERS
        self.r = 0.4                            # radius (m)
        self.s = 2.0                            # speed (m/s)
        self.num_intruders = 3                  # number of intruders at the 
                                                # same time

        # ACTION SPACE
        # speed
        min_speed       = np.ones(self.num_agents) * self.S[0]
        max_speed       = np.ones(self.num_agents) * self.S[1]
        # moving direction
        min_dir         = np.ones(self.num_agents) * self.DIR[0]
        max_dir         = np.ones(self.num_agents) * self.DIR[1]
        
        self.action_space = gym.spaces.Box(
            np.concatenate((min_speed, min_dir)), 
            np.concatenate((max_speed, max_dir)),
            dtype=np.float32)
        print(type(self.action_space))

        # OBSERVATION SPACE
        self.num_values = (self.num_intruders + self.num_agents - 1) * self.num_agents
        # relative distance
        min_rel_dist    = np.zeros(self.num_values)
        max_rel_dist    = np.ones(self.num_values) * self.map_size * np.sqrt(2)
        # relative direction
        min_rel_dir     = np.ones(self.num_values) * self.DIR[0]
        max_rel_dir     = np.ones(self.num_values) * self.DIR[1]
        # position of agents
        min_pos_agents  = np.ones(self.num_agents * 2) * self.map_size / 2 * -1
        max_pos_agents  = np.ones(self.num_agents * 2) * self.map_size / 2
        
        self.observation_space = gym.spaces.Box(
            np.concatenate((min_rel_dist, min_rel_dir, min_pos_agents)),
            np.concatenate((max_rel_dist, max_rel_dir, max_pos_agents)),
            dtype=np.float32)
        print(type(self.observation_space))
        
        # GLOBAL VARIABLES
        self.rel_pos        = None              # relative distance & direction
        self.vel_agents     = None
        self.pos_agents     = None              # position of agents
        self.vel_intruders  = None
        self.pos_intruders  = None
        self.observation    = None              # (N+M-1)*M + (N+M-1)*M + M*2
                
        self.seed()
        self.viewer = None
        self.steps_beyond_done = None
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def reset(self):
        self.rel_pos        = np.zeros([self.num_values, 2])
        self.vel_agents     = np.zeros([self.num_agents, 2])
        self.pos_agents     = np.zeros([self.num_agents, 2])
        self.vel_intruders  = np.zeros([self.num_intruders, 2])
        self.pos_intruders  = np.zeros([self.num_intruders * 2])
        self.observation    = np.concatenate((self.rel_pos.faltten('F'), 
                                              self.pos_agents.flatten('F')))
        self.init_agents()
        self.init_intruders()

        # self.observation = self.range_detection[0] * np.ones(42)
        self.observation = self.observe()

        return self.observation
    
    
    def init_agents(self):
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
            logger.warn(
                "The formation for num_agent = %d "
                "has not been defined yet. "
                "Please go to init_agents() to edit.",
                self.num_agents
            )
                
    
    def init_intruders(self):
        for i in range(self.num_intruders):
            # check if they lie outside the map
            if np.any(np.abs(self.pos_intruders[i]) >= self.map_size / 2):
                angle = 2 * np.pi * random.random()
                if angle >= 7/4 * np.pi or angle < 1/4 * np.pi:
                    x = self.map_size/2
                    y = self.map_size/2 * np.tan(angle)
                elif angle >= 1/4 * np.pi and angle < 3/4 * np.pi:
                    x = self.map_size/2 / np.tan(angle)
                    y = self.map_size/2
                elif angle >= 3/4 * np.pi and angle < 5/4 * np.pi:
                    x = - self.map_size/2
                    y = - self.map_size/2 * np.tan(angle)
                elif angle >= 5/4 * np.pi and angle < 7/4 * np.pi:
                    x = - self.map_size/2 / np.tan(angle)
                    y = - self.map_size/2
                # place them on the boundary of the map
                self.pos_intruders[i] = np.array([x, y])
                
                # move towards the centre of mass of the agents
                centre = np.mean(self.pos_agents, axis=0)
                rel_pos = centre - self.pos_intruders[i];
                rel_dir = np.arctan(rel_pos[1] / rel_pos[0])
                rel_dist = np.linalg.norm(rel_pos)
                # randomly select an angle for each intruder
                radius = 2.0
                dir_range = np.arcsin(radius / rel_dist)
                dir_rand = rel_dir - dir_range + (2 * dir_range * random.random())
                self.vel_intruders[i] = np.array([self.s, dir_rand])
    
    
    def step(self, action):
        # check if the actions are legitimate
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        # action space to velocity matrix (speed, direction)
        self.vel_agents = np.reshape(action, (self.num_agents, 2), order='F')
        
        # update the scene
        self.move_agents()
        self.move_intruders()
        
        # reset the outdated intruders
        self.init_intruders()

        # observation
        self.observation = self.observe()
        
        # reward
        reward = self.reward()

        # done
        done = bool(self.check_collision() or self.check_escape())
        
        # info
        info = {}

        return self.observation, reward, done, info


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


    def observe(self):
        # self.rel_pos        = np.zeros([self.num_values, 2])
        # self.vel_agents     = np.zeros([self.num_agents, 2])
        # self.pos_agents     = np.zeros([self.num_agents, 2])
        # self.vel_intruders  = np.zeros([self.num_intruders, 2])
        # self.pos_intruders  = np.zeros([self.num_intruders * 2])
        # self.observation    = np.concatenate((self.rel_pos.faltten('F'), 
        #                                       self.pos_agents.flatten('F')))
                                              
        for i in range(self.num_agents):
            for j in range(self.num_intruders):
                # agent to intruder distance
                rel_pos = self.pos_intruders[j] - self.pos_agents[i]
                rel_dir = np.arctan(rel_pos[1] / rel_pos[0])
                rel_dist = np.linalg.norm(rel_pos)
                
                idx = (self.num_intruders + self.num_agents - 1) * i + j
                self.rel_pos[idx] = np.array([rel_dist, rel_dir])
                
            for k in range(self.num_agents):
                # agent to other agents distance
                if k == i:
                    continue
                rel_pos = self.pos_agents[k] - self.pos_agents[i]
                rel_dir = np.arctan(rel_pos[1] / rel_pos[0])
                rel_dist = np.linalg.norm(rel_pos)
                
                idx = idx + 1
                self.rel_pos[idx] = np.array([rel_dist, rel_dir])

        self.observation = np.concatenate((self.rel_pos.faltten('F'), 
                                           self.pos_agents.flatten('F')))
        return self.observation
                    
                    
    def check_collision(self):
        collision = np.any(self.rel_pos[:, 0] <= self.R + self.r)
        return collision
    
    
    def check_escape(self):
        escape = np.any(np.abs(self.pos_agents) >= self.map_size / 2)
        return escape
    
    
    def reward(self):
        # reward
        done = bool(self.check_collision() or self.check_escape())
        if not done:
            distance = np.linalg.norm(self.pos_agents)
            reward = 1.0 - 0.01 * distance ** 2
        elif self.steps_beyond_done is None:
            # just failed
            self.steps_beyond_done = 0
            distance = np.linalg.norm(self.pos_agents)
            reward = 1.0 - 0.01 * distance ** 2
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0
            self.reset()
        return reward
    

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
            x = self.pos_agents[i, 0] * scale + screen_size / 2.0
            y = self.pos_agents[i, 1] * scale + screen_size / 2.0
            self.agents_trans[i].set_translation(x, y)
            print([x, y])
            
        for i in range(self.num_intruders):
            x = self.pos_intruders[i, 0] * scale + screen_size / 2.0
            y = self.pos_intruders[i, 1] * scale + screen_size / 2.0
            self.intruders_trans[i].set_translation(x, y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
