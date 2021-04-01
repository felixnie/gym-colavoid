#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:20:41 2021

@author: felixnie
"""

import gym
import numpy as np

from stable_baselines3 import DQN

model = DQN('MlpPolicy', 'LunarLander-v2', 
            verbose=1, exploration_final_eps=0.1, target_update_interval=250)

from stable_baselines3.common.evaluation import evaluate_policy

# Separate env for evaluation
eval_env = gym.make('LunarLander-v2')

# Random Agent, before training
try:
    mean_reward, std_reward = evaluate_policy(model, eval_env, 
                                              n_eval_episodes=10, 
                                              deterministic=True,
                                              render=True)
except Exception as e:
    print("\n")
    print("[E] %s" % (e))
    print("\n")
    print("[E] %s" % (e.__doc__))
    input("[I] Press Enter to continue...")
    eval_env.close()
else:
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    
# Train the agent
model.learn(total_timesteps=int(1e5))
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

model = DQN.load("dqn_lunar")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

# after training
try:
    mean_reward, std_reward = evaluate_policy(model, eval_env, 
                                              n_eval_episodes=10, 
                                              deterministic=True,
                                              render=True)
except Exception as e:
    print("\n")
    print("[E] %s" % (e))
    print("\n")
    print("[E] %s" % (e.__doc__))
    input("[I] Press Enter to continue...")
    eval_env.close()
else:
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")