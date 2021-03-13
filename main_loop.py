"""
Main Loop of simulation.
Here a problem and a solution are defined.
"""

import gym
import numpy as np

from tools import *
from base_solution import *

# TODO: uncomment if there are no gym warnings
#gym.logger.set_level(40)

# Parameters
n_episodes = 5
problem = 'CarRacing-v0'

# Initialize simulation
env = gym.make(problem)
solution = BaseSolution(env.observation_space, env.action_space)

# Loop of episodes
for ie in range(n_episodes):
    # noinspection PyRedeclaration
    state = env.reset()
    solution.reset()
    done = False

    # One-step-loop
    while not done:
        env.render()

        action = solution.get_action(state)
        new_state, reward, done, info = env.step(action)

        solution.learn(state, action, reward, new_state)
        state = new_state
